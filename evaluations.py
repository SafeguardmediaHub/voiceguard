#!/usr/bin/env python3
"""Persistent, labelled evaluation batches for the internal drift dashboard.

Evaluation uploads are intentionally separate from customer detection jobs.  They
are scored by the same worker with ``audit=False`` and live under /data so they
survive container recreation, but they never mutate the frozen drift baseline.
"""
import hashlib
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
DATASET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
CURATION_STATUSES = {"needs_label_review", "approved_for_training", "holdout_only", "rejected"}
DATASET_SPLITS = {"train", "validation", "holdout"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _db_path():
    configured = os.environ.get("VOICEGUARD_EVALUATION_DB")
    if configured:
        return configured
    jobs_db = os.environ.get("VOICEGUARD_JOBS_DB", "jobs.db")
    return os.path.join(os.path.dirname(os.path.abspath(jobs_db)), "evaluations.db")


def _root_dir():
    configured = os.environ.get("VOICEGUARD_EVALUATION_DIR")
    if configured:
        return Path(configured)
    return Path(_db_path()).parent / "evaluations"


def _connect():
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


_initialized = set()


def init_db():
    path = _db_path()
    if path in _initialized:
        return
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_batches (
                batch_id TEXT PRIMARY KEY,
                owner_key_id TEXT NOT NULL,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                label INTEGER NOT NULL CHECK(label IN (0,1)),
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_samples (
                sample_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES evaluation_batches(batch_id),
                original_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                result_json TEXT,
                error TEXT
            )""")
        # The first dashboard release already created this table in production.
        # Keep upgrades additive so deploying the dashboard never discards past
        # evaluation evidence.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(evaluation_samples)")}
        if "curation_status" not in columns:
            conn.execute("ALTER TABLE evaluation_samples ADD COLUMN curation_status TEXT NOT NULL DEFAULT 'needs_label_review'")
        if "curation_note" not in columns:
            conn.execute("ALTER TABLE evaluation_samples ADD COLUMN curation_note TEXT NOT NULL DEFAULT ''")
        if "dataset_split" not in columns:
            conn.execute("ALTER TABLE evaluation_samples ADD COLUMN dataset_split TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_dataset_revisions (
                revision_id TEXT PRIMARY KEY,
                owner_key_id TEXT NOT NULL,
                name TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_dataset_samples (
                revision_id TEXT NOT NULL REFERENCES training_dataset_revisions(revision_id),
                sample_id TEXT NOT NULL REFERENCES evaluation_samples(sample_id),
                dataset_split TEXT NOT NULL CHECK(dataset_split IN ('train','validation','holdout')),
                original_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                label INTEGER NOT NULL CHECK(label IN (0,1)),
                source TEXT NOT NULL,
                curation_note TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (revision_id, sample_id)
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_samples_status "
                     "ON evaluation_samples(status, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_samples_batch "
                     "ON evaluation_samples(batch_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_samples_revision "
                     "ON training_dataset_samples(revision_id, dataset_split)")
        conn.commit()
    finally:
        conn.close()
    _initialized.add(path)


def _clean_source(source):
    source = (source or "").strip().lower()
    if not SOURCE_RE.fullmatch(source):
        raise ValueError("source must be 1-80 lowercase characters: a-z, 0-9, '.', '_' or '-'")
    return source


def _clean_name(name):
    name = " ".join((name or "").strip().split())
    if not name or len(name) > 120:
        raise ValueError("batch name is required and must be at most 120 characters")
    return name


def create_batch(owner_key_id, name, source, label, notes=""):
    """Create an upload batch. A source has one immutable ground-truth label."""
    init_db()
    name = _clean_name(name)
    source = _clean_source(source)
    try:
        label = int(label)
    except (TypeError, ValueError):
        raise ValueError("label must be 0 (real) or 1 (fake)")
    if label not in (0, 1):
        raise ValueError("label must be 0 (real) or 1 (fake)")
    notes = (notes or "").strip()[:2000]
    batch_id = "eb_" + secrets.token_hex(8)
    conn = _connect()
    try:
        existing = conn.execute(
            "SELECT label FROM evaluation_batches WHERE source=? LIMIT 1", (source,)).fetchone()
        if existing is not None and existing["label"] != label:
            raise ValueError(
                f"source '{source}' already exists as label={existing['label']}; "
                "use a distinct source name rather than mixing real and fake labels")
        conn.execute("""
            INSERT INTO evaluation_batches
            (batch_id, owner_key_id, name, source, label, notes, status, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (batch_id, owner_key_id, name, source, label, notes, "uploading", _now()))
        conn.commit()
    finally:
        conn.close()
    (_root_dir() / batch_id / "input").mkdir(parents=True, exist_ok=True)
    return get_batch(batch_id)


def _safe_filename(name):
    base = os.path.basename(name or "audio")
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", base).strip("._") or "audio"
    suffix = Path(base).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise ValueError("unsupported audio type; use mp3, wav, flac, m4a, ogg, or opus")
    return base, suffix


def add_upload(batch_id, filename, fileobj, max_bytes):
    """Stream an uploaded audio file to durable storage and register its hash."""
    init_db()
    batch = get_batch(batch_id)
    if batch is None:
        raise KeyError("batch not found")
    if batch["status"] != "uploading":
        raise ValueError("batch is no longer accepting uploads")
    original_name, suffix = _safe_filename(filename)
    sample_id = "es_" + secrets.token_hex(8)
    dest = _root_dir() / batch_id / "input" / f"{sample_id}{suffix}"
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    digest = hashlib.sha256()
    written = 0
    try:
        with open(tmp, "wb") as out:
            while True:
                chunk = fileobj.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError("file too large")
                digest.update(chunk)
                out.write(chunk)
        if written == 0:
            raise ValueError("empty upload")
        sha256 = digest.hexdigest()
        conn = _connect()
        try:
            duplicate = conn.execute(
                "SELECT sample_id, batch_id FROM evaluation_samples WHERE sha256=?", (sha256,)).fetchone()
            if duplicate is not None:
                return {"status": "duplicate", "sample_id": duplicate["sample_id"],
                        "existing_batch_id": duplicate["batch_id"], "sha256": sha256}
            os.replace(tmp, dest)
            conn.execute("""
                INSERT INTO evaluation_samples
                (sample_id, batch_id, original_name, stored_path, sha256, status, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (sample_id, batch_id, original_name, str(dest), sha256, "uploaded", _now()))
            conn.commit()
            return {"status": "uploaded", "sample_id": sample_id, "sha256": sha256,
                    "original_name": original_name}
        finally:
            conn.close()
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def start_batch(batch_id, owner_key_id=None):
    init_db()
    conn = _connect()
    try:
        batch = conn.execute("SELECT * FROM evaluation_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if batch is None:
            raise KeyError("batch not found")
        if owner_key_id and batch["owner_key_id"] != owner_key_id:
            raise PermissionError("batch not found")
        if batch["status"] not in ("uploading", "complete", "complete_with_errors", "error"):
            raise ValueError("batch is already running")
        count = conn.execute("SELECT COUNT(*) AS n FROM evaluation_samples WHERE batch_id=?", (batch_id,)).fetchone()["n"]
        if count == 0:
            raise ValueError("upload at least one audio file before starting evaluation")
        conn.execute("UPDATE evaluation_samples SET status='queued', started_at=NULL, finished_at=NULL, result_json=NULL, error=NULL "
                     "WHERE batch_id=?", (batch_id,))
        conn.execute("UPDATE evaluation_batches SET status='queued', started_at=NULL, finished_at=NULL, error=NULL "
                     "WHERE batch_id=?", (batch_id,))
        conn.commit()
    finally:
        conn.close()
    return get_batch(batch_id, include_samples=True)


def claim_next_sample():
    """Claim an evaluation item atomically. Customer jobs retain worker priority."""
    init_db()
    conn = _connect()
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("""
            SELECT s.sample_id FROM evaluation_samples s
            JOIN evaluation_batches b ON b.batch_id=s.batch_id
            WHERE s.status='queued' AND b.status IN ('queued','running')
            ORDER BY b.created_at, s.created_at LIMIT 1
        """).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        sample_id = row["sample_id"]
        now = _now()
        conn.execute("UPDATE evaluation_samples SET status='running', started_at=? WHERE sample_id=?", (now, sample_id))
        conn.execute("UPDATE evaluation_batches SET status='running', started_at=COALESCE(started_at, ?) "
                     "WHERE batch_id=(SELECT batch_id FROM evaluation_samples WHERE sample_id=?)", (now, sample_id))
        conn.execute("COMMIT")
        return dict(conn.execute("SELECT * FROM evaluation_samples WHERE sample_id=?", (sample_id,)).fetchone())
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _finish_sample(sample_id, result=None, error=None):
    conn = _connect()
    try:
        row = conn.execute("SELECT batch_id FROM evaluation_samples WHERE sample_id=?", (sample_id,)).fetchone()
        if row is None:
            return
        status = "done" if error is None else "error"
        conn.execute("UPDATE evaluation_samples SET status=?, result_json=?, error=?, finished_at=? WHERE sample_id=?",
                     (status, json.dumps(result) if result is not None else None, str(error) if error else None,
                      _now(), sample_id))
        batch_id = row["batch_id"]
        remaining = conn.execute("SELECT COUNT(*) AS n FROM evaluation_samples "
                                 "WHERE batch_id=? AND status IN ('uploaded','queued','running')", (batch_id,)).fetchone()["n"]
        if remaining == 0:
            n_errors = conn.execute("SELECT COUNT(*) AS n FROM evaluation_samples WHERE batch_id=? AND status='error'",
                                    (batch_id,)).fetchone()["n"]
            conn.execute("UPDATE evaluation_batches SET status=?, finished_at=? WHERE batch_id=?",
                         ("complete" if n_errors == 0 else "complete_with_errors", _now(), batch_id))
        conn.commit()
    finally:
        conn.close()


def complete_sample(sample_id, result):
    _finish_sample(sample_id, result=result)


def fail_sample(sample_id, error):
    _finish_sample(sample_id, error=error)


def _result_summary(result):
    """The useful, bounded part of a detector result for the browser.

    Persist the complete result for reproducibility, but never send Grad-CAM or
    other large explainability payloads to the batch list.  These fields answer
    the operator's actual review questions: what was predicted, how confident
    was it, which cascade stage ran, and what each acoustic model contributed.
    """
    if not isinstance(result, dict):
        return None
    fields = ("verdict", "score", "pct", "duration", "chunks", "codec",
              "silence_ratio", "policy", "model_version", "sha256",
              "elapsed", "timestamp", "confidence")
    out = {name: result.get(name) for name in fields if name in result}
    out["model_scores"] = {name: result.get(name) for name in
                           ("lcnn", "aasist", "w2v", "rawnet", "ensemble")}
    out["cascade"] = result.get("cascade") or {}
    shap = result.get("shap")
    if isinstance(shap, dict):
        out["fusion_contributions"] = {name: shap.get(name) for name in
                                        ("aasist", "wav2vec", "rawnet", "base")}
        if shap.get("chunk_range_sec") is not None:
            out["fusion_chunk_range_sec"] = shap["chunk_range_sec"]
    adversarial = result.get("adversarial")
    if isinstance(adversarial, dict):
        # The monitor is advisory. Its confidence is attack-risk evidence, not a
        # fakeness probability, so retain the threshold and flag for clear UI copy.
        out["adversarial_risk"] = {name: adversarial.get(name) for name in
                                   ("flag", "confidence", "threshold", "latency_ms")}
    return out


def sample_outcome(label, status, result):
    """Classify one labelled evaluation without treating REVIEW as a real.

    The deployed policy flags REVIEW/LIKELY_FAKE/AUTO_FAKE.  Therefore a fake is
    missed only when it receives AUTO_REAL; a known-real clip is a false positive
    whenever it is flagged by that same production policy.
    """
    if status != "done" or not isinstance(result, dict):
        return {"state": status, "correct": None, "kind": "error" if status == "error" else "pending"}
    verdict = result.get("verdict")
    predicted_label = 0 if verdict == "AUTO_REAL" else 1
    correct = predicted_label == int(label)
    kind = "correct"
    if not correct:
        kind = "false_negative" if int(label) == 1 else "false_positive"
    return {"state": "scored", "correct": correct, "kind": kind,
            "predicted_label": predicted_label, "predicted": "fake" if predicted_label else "real",
            "verdict": verdict}


def hard_case_guidance(label, status, result):
    """Give bounded, evidence-based review guidance for a labelled clip.

    This deliberately recommends the *next investigation*, not an automatic model
    update. A single clip is never enough evidence to choose a fine-tuning target;
    the reviewer still needs a balanced dataset and an untouched held-out set.
    """
    outcome = sample_outcome(label, status, result)
    if outcome["correct"] is not False:
        return {"kind": "none", "summary": "No hard-case training recommendation for this correctly handled clip."}
    if not isinstance(result, dict):
        return {"kind": "unavailable", "summary": "No model evidence is available; review the processing error before curating this clip."}

    cascade = result.get("cascade") or {}
    if cascade.get("stage2_chunks") == 0:
        return {
            "kind": "stage1", "summary": "This hard case was resolved at cascade stage 1, so AASIST, Wav2Vec2, RawNet3, and fusion were not run.",
            "next_step": "First review LCNN screening thresholds and collect a balanced set from this category; do not attribute this clip to a stage-2 model.",
        }

    scores = {"AASIST": result.get("aasist"), "Wav2Vec2": result.get("w2v"), "RawNet3": result.get("rawnet")}
    usable = {}
    for name, value in scores.items():
        try:
            usable[name] = float(value)
        except (TypeError, ValueError):
            continue
    if not usable:
        return {
            "kind": "incomplete", "summary": "The clip reached stage 2, but component-level scores were unavailable.",
            "next_step": "Re-run this category and investigate the execution record before choosing any fine-tuning work.",
        }

    strongest_name, strongest_score = max(usable.items(), key=lambda item: item[1])
    weakest_name, weakest_score = min(usable.items(), key=lambda item: item[1])
    if label == 1 and strongest_score >= 50:
        return {
            "kind": "fusion_disagreement",
            "summary": f"Known fake was missed, although {strongest_name} gave the strongest fake signal ({strongest_score:.1f}%).",
            "next_step": "Check fusion calibration and decision thresholds across a balanced category-level set before fine-tuning one component.",
        }
    if label == 0 and weakest_score < 50:
        return {
            "kind": "component_disagreement",
            "summary": f"Known real was flagged while {weakest_name} was closest to real ({weakest_score:.1f}% fake probability).",
            "next_step": "Inspect component disagreement and fusion calibration on more representative real recordings before retraining a single model.",
        }
    if label == 1:
        return {
            "kind": "category_gap",
            "summary": f"Known fake was missed; executed components were consistently low (strongest: {strongest_name} {strongest_score:.1f}%).",
            "next_step": "Treat this as a category-coverage candidate: collect balanced confirmed examples, then compare component fine-tuning against fusion-only recalibration on held-out data.",
        }
    return {
        "kind": "real_domain_shift",
        "summary": f"Known real was flagged; executed components leaned fake (weakest: {weakest_name} {weakest_score:.1f}%).",
        "next_step": "Treat this as a real-domain coverage issue first: add comparable real recordings and assess false positives before changing model weights or thresholds.",
    }


def _row_to_sample(row, label=None, include_result=True):
    d = dict(row)
    result = None
    if d.get("result_json"):
        try:
            result = json.loads(d["result_json"])
        except json.JSONDecodeError:
            result = None
    # `stored_path` is a host-only implementation detail.  Administrators get a
    # protected download endpoint instead of a filesystem path in browser JSON.
    d.pop("stored_path", None)
    d.pop("result_json", None)
    if label is not None:
        d["outcome"] = sample_outcome(label, d.get("status"), result)
        d["guidance"] = hard_case_guidance(label, d.get("status"), result)
    if include_result:
        d["result"] = _result_summary(result)
    return d


def _summary(conn, batch_id):
    rows = conn.execute("SELECT status, result_json FROM evaluation_samples WHERE batch_id=?", (batch_id,)).fetchall()
    out = {"total": len(rows), "uploaded": 0, "queued": 0, "running": 0, "done": 0, "errors": 0}
    for row in rows:
        status = row["status"]
        if status in out:
            out[status] += 1
        if status == "error":
            out["errors"] += 1
    return out


def get_batch(batch_id, include_samples=False):
    init_db()
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM evaluation_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if row is None:
            return None
        batch = dict(row)
        batch["summary"] = _summary(conn, batch_id)
        if include_samples:
            rows = conn.execute("SELECT * FROM evaluation_samples WHERE batch_id=? ORDER BY created_at", (batch_id,)).fetchall()
            batch["samples"] = [_row_to_sample(r, label=batch["label"]) for r in rows]
        return batch
    finally:
        conn.close()


def list_batches(limit=50):
    init_db()
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM evaluation_batches ORDER BY created_at DESC LIMIT ?", (max(1, min(int(limit), 200)),)).fetchall()
        batches = []
        for row in rows:
            b = dict(row)
            b["summary"] = _summary(conn, b["batch_id"])
            batches.append(b)
        return batches
    finally:
        conn.close()


def get_sample(sample_id):
    """Internal lookup for the protected audio-download endpoint."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute("""
            SELECT s.*, b.label, b.source, b.name AS batch_name
            FROM evaluation_samples s JOIN evaluation_batches b ON b.batch_id=s.batch_id
            WHERE s.sample_id=?
        """, (sample_id,)).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def _clean_dataset_name(name):
    name = (name or "").strip().lower()
    if not DATASET_NAME_RE.fullmatch(name):
        raise ValueError("dataset name must be 1-80 lowercase characters: a-z, 0-9, '.', '_' or '-'")
    return name


def update_curation(sample_id, curation_status, dataset_split=None, note=""):
    """Record an explicit human decision about a completed evaluation clip.

    Nothing is implicitly selected because it was a mistake.  A reviewer decides
    whether a clip is fit for training, held out for future certification, or
    rejected.  This protects against accidental leakage from the dashboard into a
    model update.
    """
    init_db()
    curation_status = (curation_status or "").strip()
    if curation_status not in CURATION_STATUSES:
        raise ValueError("unknown curation status")
    note = (note or "").strip()[:2000]
    dataset_split = (dataset_split or "").strip() or None
    if curation_status == "approved_for_training":
        if dataset_split not in {"train", "validation"}:
            raise ValueError("approved training clips must be assigned to train or validation")
    elif curation_status == "holdout_only":
        dataset_split = "holdout"
    else:
        dataset_split = None
    conn = _connect()
    try:
        row = conn.execute("SELECT status FROM evaluation_samples WHERE sample_id=?", (sample_id,)).fetchone()
        if row is None:
            raise KeyError("sample not found")
        if row["status"] != "done":
            raise ValueError("only successfully scored clips can be curated")
        conn.execute("""
            UPDATE evaluation_samples
            SET curation_status=?, dataset_split=?, curation_note=?
            WHERE sample_id=?
        """, (curation_status, dataset_split, note, sample_id))
        conn.commit()
    finally:
        conn.close()
    return get_sample_public(sample_id)


def get_sample_public(sample_id):
    """Dashboard-safe single-sample representation, without host file paths."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute("""
            SELECT s.*, b.label FROM evaluation_samples s
            JOIN evaluation_batches b ON b.batch_id=s.batch_id
            WHERE s.sample_id=?
        """, (sample_id,)).fetchone()
        if row is None:
            return None
        return _row_to_sample(row, label=row["label"])
    finally:
        conn.close()


def curation_summary():
    """Counts reviewers can use before sealing a reproducible dataset revision."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute("""
            SELECT curation_status, dataset_split, COUNT(*) AS n
            FROM evaluation_samples WHERE status='done'
            GROUP BY curation_status, dataset_split
        """).fetchall()
        counts = {"needs_label_review": 0, "approved_for_training": 0,
                  "holdout_only": 0, "rejected": 0,
                  "train": 0, "validation": 0, "holdout": 0}
        for row in rows:
            counts[row["curation_status"]] += row["n"]
            if row["dataset_split"]:
                counts[row["dataset_split"]] += row["n"]
        return counts
    finally:
        conn.close()


def create_dataset_revision(owner_key_id, name, notes=""):
    """Snapshot explicitly curated clips for a repeatable offline training run."""
    init_db()
    name = _clean_dataset_name(name)
    notes = (notes or "").strip()[:2000]
    revision_id = "ds_" + secrets.token_hex(8)
    conn = _connect()
    try:
        selected = conn.execute("""
            SELECT s.sample_id, s.original_name, s.stored_path, s.sha256,
                   s.dataset_split, s.curation_note, b.label, b.source
            FROM evaluation_samples s JOIN evaluation_batches b ON b.batch_id=s.batch_id
            WHERE s.status='done'
              AND s.curation_status IN ('approved_for_training','holdout_only')
              AND s.dataset_split IN ('train','validation','holdout')
            ORDER BY s.dataset_split, b.source, s.created_at
        """).fetchall()
        if not selected:
            raise ValueError("approve at least one scored clip for training or holdout before creating a dataset")
        conn.execute("""
            INSERT INTO training_dataset_revisions (revision_id, owner_key_id, name, notes, created_at)
            VALUES (?,?,?,?,?)
        """, (revision_id, owner_key_id, name, notes, _now()))
        conn.executemany("""
            INSERT INTO training_dataset_samples
            (revision_id, sample_id, dataset_split, original_name, stored_path, sha256, label, source, curation_note)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, [(revision_id, row["sample_id"], row["dataset_split"], row["original_name"],
                row["stored_path"], row["sha256"], row["label"], row["source"], row["curation_note"])
               for row in selected])
        conn.commit()
    finally:
        conn.close()
    return get_dataset_revision(revision_id)


def _dataset_summary(conn, revision_id):
    rows = conn.execute("SELECT dataset_split, label, COUNT(*) AS n FROM training_dataset_samples "
                        "WHERE revision_id=? GROUP BY dataset_split, label", (revision_id,)).fetchall()
    summary = {"total": 0, "train": 0, "validation": 0, "holdout": 0,
               "real": 0, "fake": 0}
    for row in rows:
        summary["total"] += row["n"]
        summary[row["dataset_split"]] += row["n"]
        summary["fake" if row["label"] else "real"] += row["n"]
    return summary


def get_dataset_revision(revision_id, include_samples=False):
    init_db()
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM training_dataset_revisions WHERE revision_id=?", (revision_id,)).fetchone()
        if row is None:
            return None
        revision = dict(row)
        revision["summary"] = _dataset_summary(conn, revision_id)
        if include_samples:
            rows = conn.execute("""
                SELECT sample_id, dataset_split, original_name, sha256, label, source, curation_note
                FROM training_dataset_samples WHERE revision_id=?
                ORDER BY dataset_split, source, original_name
            """, (revision_id,)).fetchall()
            revision["samples"] = [dict(r) for r in rows]
        return revision
    finally:
        conn.close()


def list_dataset_revisions(limit=50):
    init_db()
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM training_dataset_revisions ORDER BY created_at DESC LIMIT ?",
                            (max(1, min(int(limit), 200)),)).fetchall()
        return [dict(row, summary=_dataset_summary(conn, row["revision_id"])) for row in rows]
    finally:
        conn.close()


def dataset_archive_entries(revision_id):
    """Internal archive inputs.  Paths never cross the browser API boundary."""
    init_db()
    conn = _connect()
    try:
        revision = get_dataset_revision(revision_id)
        if revision is None:
            return None, []
        rows = conn.execute("""
            SELECT sample_id, dataset_split, original_name, stored_path, sha256, label, source, curation_note
            FROM training_dataset_samples WHERE revision_id=?
            ORDER BY dataset_split, source, original_name
        """, (revision_id,)).fetchall()
        return revision, [dict(row) for row in rows]
    finally:
        conn.close()


def metrics(limit=100):
    """Return per-source, per-batch deployed catch/false-positive rates."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute("""
            SELECT b.batch_id, b.name, b.source, b.label, b.created_at, b.finished_at,
                   s.status, s.result_json
            FROM evaluation_batches b JOIN evaluation_samples s ON s.batch_id=b.batch_id
            WHERE b.status IN ('complete','complete_with_errors')
            ORDER BY b.finished_at DESC, b.created_at DESC
            LIMIT ?
        """, (max(1, min(int(limit) * 100, 10000)),)).fetchall()
        grouped = {}
        for row in rows:
            key = row["batch_id"]
            g = grouped.setdefault(key, {"batch_id": key, "name": row["name"], "source": row["source"],
                                         "label": row["label"], "date": row["finished_at"] or row["created_at"],
                                         "n": 0, "errors": 0, "flagged": 0, "correct": 0,
                                         "incorrect": 0, "items": []})
            if row["status"] != "done":
                g["errors"] += 1
                continue
            try:
                result = json.loads(row["result_json"])
            except Exception:
                g["errors"] += 1
                continue
            g["n"] += 1
            outcome = sample_outcome(row["label"], row["status"], result)
            flagged = outcome.get("predicted_label") == 1
            g["flagged"] += int(flagged)
            g["correct"] += int(outcome.get("correct") is True)
            g["incorrect"] += int(outcome.get("correct") is False)
            g["items"].append({"verdict": result.get("verdict"), "score": result.get("score"),
                               "flagged": flagged, "outcome": outcome.get("kind")})
        result = []
        for group in grouped.values():
            if group["n"]:
                rate = group["flagged"] / group["n"]
                if group["label"] == 1:
                    group["catch_rate"] = rate
                else:
                    group["false_positive_rate"] = rate
            group["small_sample"] = group["n"] < 30
            group.pop("items", None)
            result.append(group)
        return result[:limit]
    finally:
        conn.close()


def requeue_stale(older_than_seconds=1800):
    """Recover an evaluation item left running by a worker/container crash."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)).isoformat()
    init_db()
    conn = _connect()
    try:
        cur = conn.execute("UPDATE evaluation_samples SET status='queued', started_at=NULL "
                           "WHERE status='running' AND started_at < ?", (cutoff,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()

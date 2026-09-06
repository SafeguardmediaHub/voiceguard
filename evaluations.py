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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_samples_status "
                     "ON evaluation_samples(status, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_samples_batch "
                     "ON evaluation_samples(batch_id, created_at)")
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


def _row_to_sample(row, include_result=True):
    d = dict(row)
    if include_result and d.get("result_json"):
        try:
            d["result"] = json.loads(d["result_json"])
        except json.JSONDecodeError:
            d["result"] = None
    d.pop("result_json", None)
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
            batch["samples"] = [_row_to_sample(r) for r in rows]
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
                                         "n": 0, "errors": 0, "flagged": 0, "items": []})
            if row["status"] != "done":
                g["errors"] += 1
                continue
            try:
                result = json.loads(row["result_json"])
            except Exception:
                g["errors"] += 1
                continue
            g["n"] += 1
            verdict = result.get("verdict")
            flagged = verdict != "AUTO_REAL"
            g["flagged"] += int(flagged)
            g["items"].append({"verdict": verdict, "score": result.get("score"),
                               "flagged": flagged})
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

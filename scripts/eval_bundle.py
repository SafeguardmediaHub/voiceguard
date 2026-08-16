#!/usr/bin/env python3
"""eval_bundle.py — measure a registered bundle through the REAL detector path.

WHY THIS EXISTS
---------------
Every published VoiceGuard number (EER 2.43%, catch 99.2%, studio FP 12.0%) was
measured on V9. The deployed bundle is v9h. They differ in AASIST, the sub-model
later found inert, so no performance claim currently describes what ships
(REMEDIATION_PLAN H1 / GRC R5).

The previous measurement code, cascade_bias_audit_eval_cell.py, cannot close that
gap: it is a SECOND implementation of the cascade and has drifted from detector.py
in seven ways -- the AASIST class itself, three checkpoint paths, the cascade band,
peak-normalization, and chunking. Repointing its paths would load V8 weights into
the V9 architecture and crash (the M7 signature).

So this harness does not reimplement anything. It calls detector.detect(), the
exact public path a paying customer hits. Divergence from production is
structurally impossible rather than a maintenance obligation.

audit=False is the established idiom for non-customer detections (built for the H6
startup smoke-check), so evaluation runs do not enter the tamper-evident chain of
custody.

USAGE
-----
    VOICEGUARD_FORCE_BUNDLE=v9h python scripts/eval_bundle.py --bundle v9h --set bias_audit
    ... --set studio
    ... --set studio_fake
    ... --resume            # continue an interrupted run
    ... --deep-leakage      # add content-hash overlap (hashes ~1100 files)

Exit 0 = the run completed. Non-zero = the provenance gate refused, or the set
failed to decode. There is deliberately NO flag to skip the gate: a measurement
whose weights cannot be identified has no value.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Evaluation corpora ───────────────────────────────────────────────────────
# Repo-relative by default (the dev machine, where all four are .dockerignore'd),
# but every path is env-overridable because the run does NOT happen here: scoring
# all three sets costs ~86-115 h on CPU, dominated by studio_clips' long-form
# audio, so it belongs on a GPU box where the corpora sit at different paths
# (/kaggle/input/..., a mounted volume). Hardcoding a machine-specific path is the
# D9 mistake -- sweep_cascade._load pinned an absolute Windows ffmpeg path and
# silently decoded nothing everywhere else.
BIAS_DIR        = os.environ.get("VOICEGUARD_BIAS_DIR",
                                 os.path.join(REPO_ROOT, "bias_audit_fakes"))
STUDIO_DIR      = os.environ.get("VOICEGUARD_STUDIO_DIR",
                                 os.path.join(REPO_ROOT, "studio_clips"))
STUDIO_FAKE_DIR = os.environ.get("VOICEGUARD_STUDIO_FAKE_DIR",
                                 os.path.join(REPO_ROOT, "studio_fake_test",
                                              "processed_fakes"))

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

BIAS_LANGUAGES = ["arabic", "english", "french", "hausa", "igbo", "pidgin", "yoruba"]

# The original bias audit used the first 50 Hausa fakes alphabetically; 100 exist.
# Kept identical so the resulting figure stays comparable to the published one.
# 25 of these 50 are known corrupt (REMEDIATION_PLAN M3) and will surface as
# counted decode failures, not as a silently smaller denominator.
HAUSA_FAKE_LIMIT = 50

# A set that will not decode is an ENVIRONMENT fault, not a result. Above this
# fraction the harness aborts rather than rendering a metric over the remnant.
MAX_DECODE_FAILURE_RATE = 0.20

# The operating points detector.verdict_from_score actually uses. The previous
# notebook cell reported FP/catch at >=0.5, which corresponds to nothing deployed.
OP_LIKELY_FAKE = 0.55   # the customer was told it is fake
OP_REVIEW      = 0.30   # a human was made to look


class EvalError(RuntimeError):
    """Any condition that makes the run's output untrustworthy."""


class ProvenanceError(EvalError):
    """The harness cannot prove which weights it is about to measure."""


def _import_detector():
    """Import detector lazily.

    `import detector` loads ~380 MB of weights and runs the D5 integrity gate at
    import time. Doing it at module level would push tests/test_eval_bundle.py into
    the weights tier, where tests/conftest.py skips it entirely under
    $VOICEGUARD_CI_FAST -- so the manifest, leakage and metrics tests would stop
    running in CI without anyone noticing. Keep this lazy.
    """
    import detector
    return detector


# ══════════════════════════════════════════════════════════════════════════════
#  Manifest builders
#
#  Each returns a list of {"path", "label", "language", "source"}. Pure filesystem
#  work -- no weights, no network -- which is what keeps them in the fast test tier.
# ══════════════════════════════════════════════════════════════════════════════
def _audio_files(d):
    """Absolute paths of audio files directly inside `d`, sorted. [] if absent."""
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(AUDIO_EXTS)]


def build_bias_audit_manifest(root=BIAS_DIR):
    """The 549-sample bias-audit set: 299 real + 250 fake across 7 languages.

    THERE IS DELIBERATELY NO FALLBACK for a language with an empty real/ directory.
    bias_audit_fakes/english/real/ is empty, and the original notebook cell filled it
    with 50 clips drawn from val_v8_fresh.json -- the tuning set. Measuring on your
    own tuning data inflates the result, so English here contributes catch/FNR only,
    with no FP and no EER, and the report says so.
    """
    if not os.path.isdir(root):
        raise EvalError(f"bias audit corpus not found: {root}")
    entries = []
    for lang in BIAS_LANGUAGES:
        for p in _audio_files(os.path.join(root, lang, "real")):
            entries.append({"path": p, "label": 0,
                            "language": lang, "source": f"real_{lang}"})
        fakes = _audio_files(os.path.join(root, lang, "fake"))
        if lang == "hausa":
            fakes = fakes[:HAUSA_FAKE_LIMIT]
        for p in fakes:
            entries.append({"path": p, "label": 1,
                            "language": lang, "source": f"fake_{lang}"})
    return entries


def build_studio_manifest(root=STUDIO_DIR):
    """494 real studio clips across 7 subdirectories -> false-positive rate only.

    `_downloads/` (19 clips) is included so the count stays comparable to the
    published 494-clip figure. scan_leakage's content-hash pass reports duplicates,
    so any overlap between it and the other subdirectories is visible rather than
    assumed absent.
    """
    if not os.path.isdir(root):
        raise EvalError(f"studio corpus not found: {root}")
    entries = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for dirpath, _dirnames, filenames in os.walk(d):
            for f in sorted(filenames):
                if f.lower().endswith(AUDIO_EXTS):
                    entries.append({"path": os.path.join(dirpath, f), "label": 0,
                                    "language": sub, "source": f"studio_{sub}"})
    return entries


def build_studio_fake_manifest(root=STUDIO_FAKE_DIR):
    """50 studio-processed fakes -> catch rate only.

    All are Edge-TTS English (studio_edge_en_*). Edge appears among the training
    fake engines, so scan_leakage will flag this set as in-distribution. Any catch
    rate derived from it must carry that caveat.
    """
    if not os.path.isdir(root):
        raise EvalError(f"studio fake corpus not found: {root}")
    return [{"path": p, "label": 1, "language": "english", "source": "studio_edge_en"}
            for p in _audio_files(root)]


MANIFEST_BUILDERS = {
    "bias_audit":  build_bias_audit_manifest,
    "studio":      build_studio_manifest,
    "studio_fake": build_studio_fake_manifest,
}


def build_manifest(name):
    if name not in MANIFEST_BUILDERS:
        raise EvalError(f"unknown set {name!r}; expected one of "
                        f"{', '.join(sorted(MANIFEST_BUILDERS))}")
    return MANIFEST_BUILDERS[name]()


def manifest_composition(entries):
    """{language: {"real": n, "fake": n}} -- printed before scoring so the coverage
    gaps that bound what the numbers can mean are visible up front."""
    comp = {}
    for e in entries:
        c = comp.setdefault(e["language"], {"real": 0, "fake": 0})
        c["fake" if e["label"] else "real"] += 1
    return comp


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
#
#  Reported at the operating points detector.verdict_from_score actually uses
#  (0.85 / 0.55 / 0.30). The previous notebook cell used >=0.5, which corresponds
#  to no deployed decision.
# ══════════════════════════════════════════════════════════════════════════════
def compute_eer(scores, labels):
    """Equal error rate as a percentage, or None when the data cannot support one.

    Returns None -- never nan -- if either class is absent. An EER computed over a
    single class is not a small number, it is no number, and a nan renders as a
    value in every JSON consumer downstream.
    """
    labels = list(labels)
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_curve
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    fpr, tpr, _ = roc_curve(labels, list(scores), pos_label=1)
    return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)) * 100


def rate_at(rows, threshold, label_value):
    """Percentage of rows with the given label whose score is >= threshold.

    label_value=0 -> false-positive rate; label_value=1 -> catch rate.
    None when no row carries that label.
    """
    sel = [r for r in rows if r["label"] == label_value]
    if not sel:
        return None
    return 100.0 * sum(1 for r in sel if r["score"] >= threshold) / len(sel)


def compute_metrics(rows):
    """Aggregate scored rows. Error rows are excluded from every rate but counted."""
    ok = [r for r in rows if r.get("status") == "ok"]
    n_error = sum(1 for r in rows if r.get("status") == "error")
    scores = [r["score"] for r in ok]
    labels = [r["label"] for r in ok]
    lat = [r["latency_ms"] for r in ok]
    s1 = sum(r["stage1_chunks"] for r in ok)
    s2 = sum(r["stage2_chunks"] for r in ok)
    return {
        "n_scored": len(ok),
        "n_error": n_error,
        "n_real": sum(1 for l in labels if l == 0),
        "n_fake": sum(1 for l in labels if l == 1),
        "eer": compute_eer(scores, labels),
        "fp_at_likely_fake":    rate_at(ok, OP_LIKELY_FAKE, 0),
        "fp_at_review":         rate_at(ok, OP_REVIEW, 0),
        "catch_at_likely_fake": rate_at(ok, OP_LIKELY_FAKE, 1),
        "catch_at_review":      rate_at(ok, OP_REVIEW, 1),
        "stage1_chunks": s1,
        "stage2_chunks": s2,
        "stage1_resolution_pct": (100.0 * s1 / (s1 + s2)) if (s1 + s2) else None,
        "latency_ms_mean": float(np.mean(lat)) if lat else None,
        "latency_ms_p95":  float(np.percentile(lat, 95)) if lat else None,
    }


def fmt(v, nd=2):
    """Render a metric for the human report: None becomes 'n/a', never 'nan'."""
    return "n/a" if v is None else f"{v:.{nd}f}"


# ══════════════════════════════════════════════════════════════════════════════
#  Leakage scan
#
#  Ported from kaggle_provenance_leakage_cell.py section B, against the local
#  manifests. Runs before scoring so every reported number ships with a statement
#  about whether its test set was contaminated -- which is how the 99.2% catch
#  figure got loose without one.
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR = os.environ.get("VOICEGUARD_MODELS_DIR",
                            os.path.join(REPO_ROOT, "models"))
TRAIN_MANIFESTS = [
    os.path.join(MODELS_DIR, "train_v8_fresh.json"),
    os.path.join(MODELS_DIR, "teacher_scores_v9_train.json"),
]
VAL_MANIFESTS = [
    os.path.join(MODELS_DIR, "val_v8_fresh.json"),
    os.path.join(MODELS_DIR, "teacher_scores_v9_val.json"),
]

# Engines seen in fake `source` strings. Substring match, longest concern first.
_ENGINES = ("edge", "openai", "openvoice", "xtts", "coqui", "elevenlabs",
            "wavefake", "in_the_wild", "conference")


def _basename(p):
    return os.path.basename(str(p).replace("\\", "/")).lower()


def norm_engine(s):
    """Reduce a `source` string to its TTS engine, or return it unchanged."""
    s = (s or "").lower()
    for eng in _ENGINES:
        if eng in s:
            return eng
    return s


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _load_reference_manifest(fp):
    """Return [{path, label, source}] from a reference manifest.

    Handles both shapes in this repo: train/val_v8_fresh.json is a list of dicts;
    teacher_scores_v9_*.json is {path: score}.
    """
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        return [{"path": k} for k in d]
    return [e for e in d if isinstance(e, dict) and e.get("path")]


def scan_leakage(entries, train_files=None, val_files=None, deep=False):
    """Overlap between an evaluation manifest and the sets the models were fit on."""
    train_files = TRAIN_MANIFESTS if train_files is None else train_files
    val_files   = VAL_MANIFESTS   if val_files   is None else val_files

    found, missing = [], []

    def _collect(files):
        out = []
        for fp in files:
            if os.path.exists(fp):
                found.append(fp)
                out.extend(_load_reference_manifest(fp))
            else:
                missing.append(fp)
        return out

    train = _collect(train_files)
    val   = _collect(val_files)

    train_paths = {str(e["path"]) for e in train}
    val_paths   = {str(e["path"]) for e in val}
    train_bn    = {_basename(e["path"]) for e in train}
    val_bn      = {_basename(e["path"]) for e in val}

    def _bn_overlap(ref_bn):
        return {
            "real": sum(1 for e in entries
                        if e["label"] == 0 and _basename(e["path"]) in ref_bn),
            "fake": sum(1 for e in entries
                        if e["label"] == 1 and _basename(e["path"]) in ref_bn),
        }

    train_eng = Counter(norm_engine(e.get("source"))
                        for e in train if e.get("label") == 1)
    test_eng  = Counter(norm_engine(e.get("source"))
                        for e in entries if e["label"] == 1)

    content_overlap = None
    if deep:
        ref_hashes = set()
        for e in train + val:
            try:
                ref_hashes.add(sha256_file(str(e["path"])))
            except OSError:
                pass
        content_overlap = 0
        for e in entries:
            try:
                if sha256_file(e["path"]) in ref_hashes:
                    content_overlap += 1
            except OSError:
                pass

    return {
        "exact_path_in_train": sum(1 for e in entries if str(e["path"]) in train_paths),
        "exact_path_in_val":   sum(1 for e in entries if str(e["path"]) in val_paths),
        "basename_in_train": _bn_overlap(train_bn),
        "basename_in_val":   _bn_overlap(val_bn),
        "fake_engines_test":  dict(test_eng),
        "fake_engines_train": dict(train_eng),
        # In BOTH -> catch on these may reflect memorised engine artifacts rather
        # than generalization. In test only -> genuinely held out.
        "fake_engines_shared":   sorted(set(test_eng) & set(train_eng)),
        "fake_engines_held_out": sorted(set(test_eng) - set(train_eng)),
        "content_hash_overlap": content_overlap,
        "reference_manifests_found":   found,
        "reference_manifests_missing": missing,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Provenance gate
# ══════════════════════════════════════════════════════════════════════════════
def assert_bundle_provenance(version, detector_module=None):
    """Refuse to measure anything unless we can prove WHICH weights are loaded.

    Importing detector already runs the D5 gate (detector._verify_bundle_before_load,
    called at detector.py:522), which fails closed unless every artifact matches the
    SHA-256 in registry.jsonl. This does not reimplement that. It closes the one hole
    that D5 deliberately leaves open for serving:

      _verify_bundle_before_load returns early with a WARNING when manifest is None,
      and _resolve_bundle_paths falls back to the legacy models/ layout under the
      version string 'v9-hybrid-legacy' when the registry is unavailable. A serving
      process should still start. A measurement must not -- it would attribute
      numbers produced by loose files in models/ to the registered bundle.

    Returns the verified {artifact: sha256} map, which the caller writes into the
    results header so every number carries the fingerprint of the weights behind it.
    """
    D = detector_module or _import_detector()

    active = getattr(D, "ACTIVE_VERSION", None)
    if active != version:
        raise ProvenanceError(
            f"asked to measure {version!r} but detector loaded {active!r}. "
            f"Set VOICEGUARD_FORCE_BUNDLE={version} before running, so the bundle is "
            f"pinned at import without touching the live ACTIVE.json pointer.")

    if getattr(D, "_ACTIVE_MANIFEST", None) is None:
        raise ProvenanceError(
            f"{version!r} resolved to the unverified legacy models/ layout, which has "
            "no recorded hashes. Register and pull the bundle "
            "(`python bundle_registry.py pull --active`) before measuring.")

    problems = D._Registry().integrity_problems(version)
    if problems is None:
        raise ProvenanceError(
            f"bundle {version!r} is not registered, so its artifacts cannot be "
            "verified; refusing to measure.")
    if problems:
        raise ProvenanceError(
            f"bundle {version!r} failed integrity verification; refusing to measure:\n  "
            + "\n  ".join(problems))

    entry = D._Registry().get_bundle(version)
    return {"bundle": version,
            "active_version": active,
            "artifacts": dict(entry["files"])}


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring loop
# ══════════════════════════════════════════════════════════════════════════════
def _row_from_result(entry, result):
    """Flatten one detect() response into an evaluation row."""
    casc = result.get("cascade") or {}
    return {
        "path": entry["path"],
        "label": entry["label"],
        "language": entry["language"],
        "source": entry["source"],
        "score": float(result["score"]),
        "verdict": result["verdict"],
        "n_chunks": int(result.get("chunks", 0)),
        "stage1_chunks": int(casc.get("stage1_chunks", 0)),
        "stage2_chunks": int(casc.get("stage2_chunks", 0)),
        "latency_ms": float(result.get("elapsed", 0.0)) * 1000.0,
        "status": "ok",
    }


def _read_scored_paths(rows_path):
    """Paths already present in a previous run's rows.jsonl (for --resume)."""
    done = {}
    if not os.path.exists(rows_path):
        return done
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn final line from an interrupted run
            done[r["path"]] = r
    return done


def score_manifest(entries, out_dir, detector_module=None, resume=False,
                   progress_every=25):
    """Score every entry through detector.detect(path, audit=False).

    audit=False matters: H6 makes every live detection append to the tamper-evident
    chain of custody, and evaluation detections must not enter it.

    Rows are appended to <out_dir>/rows.jsonl AS THEY ARE PRODUCED, so a crash 900
    clips into a 1,093-clip run loses nothing. resume=True replays what is already
    there and scores only the remainder.
    """
    D = detector_module or _import_detector()
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "rows.jsonl")

    done = _read_scored_paths(rows_path) if resume else {}
    if not resume and os.path.exists(rows_path):
        os.remove(rows_path)      # a fresh run must not append to an older one

    rows = []
    total = len(entries)
    with open(rows_path, "a", encoding="utf-8") as sink:
        for i, entry in enumerate(entries, 1):
            if entry["path"] in done:
                rows.append(done[entry["path"]])
                continue
            try:
                result = D.detect(entry["path"], audit=False)
                row = _row_from_result(entry, result)
            except Exception as e:
                # Counted, never skipped. Silently dropping undecodable clips is how
                # sweep_cascade._load hid a total decode failure (REMEDIATION_PLAN D9).
                row = {"path": entry["path"], "label": entry["label"],
                       "language": entry["language"], "source": entry["source"],
                       "status": "error", "error": f"{type(e).__name__}: {e}"}
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if progress_every and i % progress_every == 0:
                n_err = sum(1 for r in rows if r["status"] == "error")
                print(f"  [{i}/{total}] {n_err} decode failures so far", flush=True)

    n_err = sum(1 for r in rows if r["status"] == "error")
    if total and (n_err / total) > MAX_DECODE_FAILURE_RATE:
        raise EvalError(
            f"{n_err} of {total} clips ({100.0*n_err/total:.1f}%) failed to decode, "
            f"above the {MAX_DECODE_FAILURE_RATE:.0%} ceiling. This is an environment "
            "fault, not a result -- check $VOICEGUARD_FFMPEG and the corpus, and do "
            "not treat any metric from this run as meaningful.")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Orchestration + CLI
# ══════════════════════════════════════════════════════════════════════════════
EVAL_OUT = os.path.join(REPO_ROOT, "eval_out")


def per_language_metrics(rows):
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["language"], []).append(r)
    return {lang: compute_metrics(rs) for lang, rs in sorted(by_lang.items())}


def print_report(summary):
    m = summary["metrics"]
    print("\n" + "=" * 78)
    print(f"  {summary['set']} @ bundle {summary['provenance']['bundle']}")
    print("=" * 78)
    print(f"  scored {m['n_scored']}  ({m['n_real']} real, {m['n_fake']} fake)  "
          f"decode failures: {m['n_error']}")
    print(f"    EER                     {fmt(m['eer'])}%")
    print(f"    FP    @ LIKELY_FAKE     {fmt(m['fp_at_likely_fake'])}%")
    print(f"    FP    @ REVIEW          {fmt(m['fp_at_review'])}%")
    print(f"    catch @ LIKELY_FAKE     {fmt(m['catch_at_likely_fake'])}%")
    print(f"    catch @ REVIEW          {fmt(m['catch_at_review'])}%")
    print(f"    stage-1 resolution      {fmt(m['stage1_resolution_pct'], 1)}%")
    lat_note = "" if summary.get("latency_representative", True) else \
        f"   [NOT production-representative: ran on {summary.get('device')}]"
    print(f"    latency mean / p95      {fmt(m['latency_ms_mean'], 0)} / "
          f"{fmt(m['latency_ms_p95'], 0)} ms{lat_note}")

    print(f"\n  {'language':12s} {'real':>5s} {'fake':>5s} {'EER':>8s} "
          f"{'FP@.55':>8s} {'catch@.55':>10s}")
    print("  " + "-" * 52)
    for lang, lm in summary["per_language"].items():
        print(f"  {lang:12s} {lm['n_real']:5d} {lm['n_fake']:5d} "
              f"{fmt(lm['eer']):>8s} {fmt(lm['fp_at_likely_fake']):>8s} "
              f"{fmt(lm['catch_at_likely_fake']):>10s}")

    lk = summary["leakage"]
    print(f"\n  leakage: {lk['exact_path_in_train']} test paths in TRAIN, "
          f"{lk['exact_path_in_val']} in VAL (val = tuning set)")
    print(f"    fake engines shared with training: {lk['fake_engines_shared'] or 'none'}")
    print(f"    fake engines genuinely held out:   {lk['fake_engines_held_out'] or 'NONE'}")
    if lk["reference_manifests_missing"]:
        print(f"    [WARN] reference manifests missing, so the scan is PARTIAL: "
              f"{lk['reference_manifests_missing']}")


def run_eval(bundle, set_name, out_root=EVAL_OUT, detector_module=None,
             resume=False, deep_leakage=False):
    """Gate -> manifest -> leakage -> score -> metrics -> summary.json."""
    D = detector_module or _import_detector()

    # The gate runs FIRST. Scoring a single clip before knowing which weights are
    # loaded would produce numbers that cannot be attributed to anything.
    provenance = assert_bundle_provenance(bundle, detector_module=D)

    entries = build_manifest(set_name)
    composition = manifest_composition(entries)
    print(f"\n{set_name}: {len(entries)} samples")
    for lang, c in sorted(composition.items()):
        note = ""
        if c["fake"] == 0:
            note = "   <- no fakes: no catch rate, no EER"
        elif c["real"] == 0:
            note = "   <- no reals: no FP rate, no EER"
        print(f"  {lang:12s} {c['real']:3d} real, {c['fake']:3d} fake{note}")

    leakage = scan_leakage(entries, deep=deep_leakage)

    out_dir = os.path.join(out_root, bundle, set_name)
    rows = score_manifest(entries, out_dir, detector_module=D, resume=resume)

    # Which device produced these numbers. Accuracy (EER/FP/catch) is
    # device-insensitive to ~1e-6; LATENCY IS NOT. Production is a CPU droplet, so
    # a cuda run's ms figures do not describe what a customer experiences and are
    # marked non-representative rather than quietly published.
    device = str(getattr(D, "DEVICE", "unknown"))
    latency_representative = device.startswith("cpu")

    summary = {
        "set": set_name,
        "device": device,
        "latency_representative": latency_representative,
        "provenance": provenance,
        "composition": composition,
        "leakage": leakage,
        "metrics": compute_metrics(rows),
        "per_language": per_language_metrics([r for r in rows if r["status"] == "ok"]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_report(summary)
    print(f"\n  -> {os.path.join(out_dir, 'summary.json')}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure a registered bundle through the real detector path (H1)")
    ap.add_argument("--bundle", required=True,
                    help="registered bundle version to measure, e.g. v9h. Must match "
                         "$VOICEGUARD_FORCE_BUNDLE.")
    ap.add_argument("--set", dest="set_name", required=True,
                    choices=sorted(MANIFEST_BUILDERS),
                    help="which evaluation corpus to run")
    ap.add_argument("--out", default=EVAL_OUT, help=f"output root (default {EVAL_OUT})")
    ap.add_argument("--resume", action="store_true",
                    help="replay rows.jsonl and score only what is missing")
    ap.add_argument("--deep-leakage", action="store_true",
                    help="also hash every clip for byte-identical overlap (slow)")
    args = ap.parse_args(argv)

    try:
        run_eval(args.bundle, args.set_name, out_root=args.out,
                 resume=args.resume, deep_leakage=args.deep_leakage)
    except ProvenanceError as e:
        print(f"\nPROVENANCE GATE REFUSED: {e}", file=sys.stderr)
        return 2
    except EvalError as e:
        print(f"\nEVALUATION ABORTED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

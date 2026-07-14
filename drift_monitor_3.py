#!/usr/bin/env python3
"""
drift_monitor.py — Living Test Set + Drift Monitor (Phase 5 Task 5 / Phase 8)

Production-hardened drift monitor. Scores the DEPLOYED model by importing
server.py (which loads whatever bundle is ACTIVE in the registry — V9 today),
so it always measures exactly what production serves. Computes metrics with
statistical confidence, checks for confirmed
drift across consecutive runs, sends email alerts on confirmed drift,
and logs everything append-only for trend analysis.

USAGE
-----
    python drift_monitor.py --init-baseline    # First-time setup
    python drift_monitor.py                    # Normal run (full suite)
    python drift_monitor.py --quick            # Faster run (subsample)
    python drift_monitor.py --diff             # Compare latest vs baseline
    python drift_monitor.py --trend            # Show EER trend across last N runs
    python drift_monitor.py --schedule         # Print crontab line
    python drift_monitor.py --add-samples DIR LABEL SOURCE
                                               # Ingest new samples into living test set

OUTPUTS
-------
    drift_baseline.json         — saved baseline metrics (set once, manually updated)
    drift_log.jsonl             — one line per run, append-only (trend source)
    drift_alert_state.json      — consecutive-breach counter for confirmation logic
    drift_report_<utc>.txt      — human-readable text report per run
    drift_report_<utc>.json     — structured per-run JSON for downstream analysis

EMAIL ALERTS
------------
    Set environment variables before running:
        DRIFT_SMTP_HOST     — e.g. smtp.gmail.com
        DRIFT_SMTP_PORT     — e.g. 587
        DRIFT_SMTP_USER     — sender email address
        DRIFT_SMTP_PASS     — sender password or app password
        DRIFT_ALERT_TO      — recipient email address(es), comma-separated

    Alerts fire only after DRIFT_CONFIRM_RUNS consecutive breaching runs
    (default 2) to suppress single-run noise.

LIVING TEST SET
---------------
    Use --add-samples to continuously fold new AI-generated audio into the
    val manifest without rebuilding from scratch. Samples are hashed so
    duplicates are skipped automatically.

DEPLOYMENT (Digital Ocean)
--------------------------
    1. Copy this file and models/ to your droplet
    2. Set environment variables in /etc/environment or a .env file
    3. Run --init-baseline once
    4. Add crontab entries via --schedule
"""

import os, sys, json, time, argparse, hashlib, gc, smtplib, logging
from pathlib import Path
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.io import wavfile
from scipy import stats as scipy_stats
from sklearn.metrics import roc_curve

# ─── Logging setup ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ',
)
log = logging.getLogger('drift_monitor')

# Reports use Unicode box-drawing / warning glyphs. On a Windows cp1252 console
# printing them raises UnicodeEncodeError — which, in a normal run, would abort
# BEFORE the per-run report/log files are written and lose the results. Force
# UTF-8 stdout/stderr so the report always prints (no-op on Linux/DO).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ─── Configuration ──────────────────────────────────────────────────────────
# All path / credential config lives here. On Digital Ocean, override with
# environment variables so secrets never live in source code.

# Paths — resolve from env vars first, fall back to local defaults
NOIZAI_DIR   = os.environ.get('DRIFT_NOIZAI_DIR',  'data/noiz_phone')
OUTPUT_DIR   = Path(os.environ.get('DRIFT_OUTPUT_DIR', 'output'))

VAL_MANIFEST = os.environ.get('DRIFT_VAL_MANIFEST', 'models/val_v8_fresh.json')

# Output files
BASELINE_FILE    = OUTPUT_DIR / 'drift_baseline.json'
LOG_FILE         = OUTPUT_DIR / 'drift_log.jsonl'
ALERT_STATE_FILE = OUTPUT_DIR / 'drift_alert_state.json'
# Continual-learning retrain trigger (Phase 5): confirmed drift raises a machine-readable
# retrain signal + optional command hook. Actual retraining runs externally (GPU/Kaggle).
RETRAIN_TRIGGER_FILE = OUTPUT_DIR / 'retrain_trigger.json'
RETRAIN_LOG_FILE     = OUTPUT_DIR / 'retrain_trigger_log.jsonl'
RETRAIN_CMD          = os.environ.get('VOICEGUARD_RETRAIN_CMD', '')

# Email config — loaded from env, never hardcoded
EMAIL_CFG = {
    'smtp_host': os.environ.get('DRIFT_SMTP_HOST', ''),
    'smtp_port': int(os.environ.get('DRIFT_SMTP_PORT', 587)),
    'smtp_user': os.environ.get('DRIFT_SMTP_USER', ''),
    'smtp_pass': os.environ.get('DRIFT_SMTP_PASS', ''),
    'alert_to':  [a.strip() for a in os.environ.get('DRIFT_ALERT_TO', '').split(',') if a.strip()],
}

# How many consecutive breaching runs before firing an email alert.
# Set to 1 to alert on every breach (noisy). 2 is the recommended minimum.
DRIFT_CONFIRM_RUNS = int(os.environ.get('DRIFT_CONFIRM_RUNS', 2))

# Production thresholds (match deployed server.py)
T_AUTO_FAKE = 0.85
T_LIKELY    = 0.55
T_REVIEW    = 0.30

# Drift detection thresholds
ALERT_THRESHOLDS = {
    'clean_ensemble_eer_pp':    3.0,   # +3pp ensemble EER → alert
    'per_source_catch_rate_pp': 10.0,  # any source drops 10pp → alert
    'noizai_catch_rate_pp':     15.0,  # Noiz.ai catch drops 15pp → alert
    'phone_catch_rate_pp':      15.0,  # phone-class catch drops 15pp → alert
}

# Statistical significance: minimum p-value to treat an EER shift as real signal.
# Uses a two-proportion z-test on misclassification counts.
EER_SIGNIFICANCE_ALPHA = 0.05

# Trend: number of past runs to include in trend analysis
TREND_WINDOW = 10

# Sample sizes
N_CLEAN_FULL  = 200
N_CLEAN_QUICK = 60

# Audio
SR         = 16000
SAMPLE_LEN = 4 * SR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED   = 42

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.opus', '.ogg'}

# ─── Models ─────────────────────────────────────────────────────────────────
# Model definitions live in server.py and are loaded from the ACTIVE registry
# bundle; this monitor scores through them via load_models() (see below).

# ─── Audio loading ──────────────────────────────────────────────────────────

def load_audio(path, max_samples=SAMPLE_LEN):
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t:
        wav_path = t.name
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', str(path), '-ar', str(SR), '-ac', '1',
             '-acodec', 'pcm_s16le', wav_path],
            capture_output=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (code {result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        _, data = wavfile.read(wav_path)
    finally:
        try: os.unlink(wav_path)
        except: pass
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if len(data) > max_samples:
        start = (len(data) - max_samples) // 2
        data = data[start:start + max_samples]
    elif len(data) < max_samples:
        data = np.pad(data, (0, max_samples - len(data)))
    return data

# ─── V8 loading & scoring ──────────────────────────────────────────────────

def _validate_manifest_schema(df):
    """Raise early if the manifest is missing required columns."""
    required = {'path', 'label', 'source'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Val manifest is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    if df['label'].isnull().any():
        raise ValueError("Val manifest contains null labels.")
    if not set(df['label'].unique()).issubset({0, 1}):
        raise ValueError(f"Val manifest labels must be 0 or 1. Found: {df['label'].unique()}")

def load_models():
    """Reuse the DEPLOYED scoring path. Importing server.py loads whatever model
    bundle is currently ACTIVE in the registry (V9 today), so the drift monitor
    measures exactly what production serves — no duplicated model code here, and
    it automatically tracks any future promoted bundle.

    Returns the imported `server` module, which exposes `ensemble_score_variants`,
    `lcnn_score`, `cascade_score_chunk`, `thresholds`, `ACTIVE_VERSION`, `CHUNK`.
    """
    global T_AUTO_FAKE, T_LIKELY, T_REVIEW
    log.info("Importing deployed scoring (server active bundle)...")
    import detector as server
    # Sync verdict thresholds to the active bundle so drift verdicts match production.
    T_AUTO_FAKE = float(server.thresholds['auto_fake'])
    T_LIKELY    = float(server.thresholds['likely_fake'])
    T_REVIEW    = float(server.thresholds['to_review'])
    log.info(f"Active bundle: {server.ACTIVE_VERSION} | thresholds "
             f"auto_fake>={T_AUTO_FAKE} likely_fake>={T_LIKELY} to_review>={T_REVIEW}")
    return server

def calibrate(p_raw, coef, intercept, eps=1e-6):
    p_raw = np.clip(p_raw, eps, 1 - eps)
    logit = np.log(p_raw / (1 - p_raw))
    return 1.0 / (1.0 + np.exp(-(coef * logit + intercept)))

def verdict_from_score(score):
    if score >= T_AUTO_FAKE: return 'auto_fake'
    if score >= T_LIKELY:    return 'likely_fake'
    if score >= T_REVIEW:    return 'to_review'
    return 'auto_real'

def score_one(server, audio_np):
    """Score one clip through the deployed model, exactly as server.py serves it.
    Returns a dict:
      ensemble — full 3-model ensemble prob (peak-normed; ensemble health / localization)
      aasist / wav2vec / rawnet — per-component fake probs (localization)
      lcnn     — stage-1 LCNN screener prob (Platt-calibrated)
      deployed — the CASCADE final prob users actually get (lcnn if the screener is
                 confident, else the ensemble), following the production routing
      stage    — 1 if resolved by the screener, 2 if escalated to the ensemble
    The full ensemble is always computed here (for per-component drift localization);
    the deployed value follows server's cascade band (server.CASCADE_LOW/HIGH)."""
    wav_1d = torch.from_numpy(audio_np).float().to(DEVICE)
    n = wav_1d.shape[0]
    if n < server.CHUNK:
        wav_1d = F.pad(wav_1d, (0, server.CHUNK - n))
    elif n > server.CHUNK:
        wav_1d = wav_1d[:server.CHUNK]
    lcnn = server.lcnn_score(wav_1d)
    # Ensemble path is peak-normalized in production (matches xgb_v9/cal_v9).
    peak = wav_1d.abs().max()
    ens_wav = wav_1d / peak if peak > 1e-8 else wav_1d
    chunk_3d = ens_wav.unsqueeze(0).unsqueeze(0)
    s_a, s_w, s_r, ensemble = server.ensemble_score_variants(chunk_3d)
    if lcnn <= server.CASCADE_LOW or lcnn >= server.CASCADE_HIGH:
        deployed, stage = lcnn, 1
    else:
        deployed, stage = ensemble, 2
    return {'ensemble': ensemble, 'aasist': s_a, 'wav2vec': s_w, 'rawnet': s_r,
            'lcnn': lcnn, 'deployed': deployed, 'stage': stage}

def compute_eer(labels, scores):
    """Compute Equal Error Rate. Returns (eer, n_errors_at_eer)."""
    if len(set(labels)) < 2:
        return float('nan'), 0
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    # Approximate error count at EER threshold for significance testing
    n_errors = int(round(eer * len(labels)))
    return eer, n_errors

def eer_significance_test(b_eer, c_eer, b_n, c_n):
    """
    Two-proportion z-test: is the shift in error rate statistically significant?
    Treats EER as a proportion of misclassified samples.
    Returns p-value. Low p (< alpha) means the shift is likely real, not noise.
    """
    if b_n == 0 or c_n == 0 or b_eer is None or c_eer is None \
            or np.isnan(b_eer) or np.isnan(c_eer):
        return 1.0  # not enough data / degenerate EER — treat as "no signal"
    b_errors = int(round(b_eer * b_n))
    c_errors = int(round(c_eer * c_n))
    p_pool = (b_errors + c_errors) / (b_n + c_n)
    if p_pool in (0.0, 1.0):
        return 1.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1/b_n + 1/c_n))
    if se == 0:
        return 1.0
    z = (c_eer - b_eer) / se
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))  # two-tailed
    return float(p_value)

# ─── Evaluation suite ──────────────────────────────────────────────────────

def evaluate_clean(models, val_df, n_samples, seed=SEED):
    """Compute clean ensemble EER and per-source verdict distributions."""
    n_per_class = n_samples // 2
    real_rows = val_df[val_df['label'] == 0]
    fake_rows = val_df[val_df['label'] == 1]

    if len(real_rows) < n_per_class or len(fake_rows) < n_per_class:
        log.warning(
            f"Requested {n_per_class} samples per class but manifest has "
            f"{len(real_rows)} real / {len(fake_rows)} fake. Capping to available."
        )
        n_per_class = min(len(real_rows), len(fake_rows))

    real_sub = real_rows.sample(n_per_class, random_state=seed)
    fake_sub = fake_rows.sample(n_per_class, random_state=seed)
    sub = pd.concat([real_sub, fake_sub]).reset_index(drop=True)

    scores, labels, sources = [], [], []
    deployed_scores, stages = [], []
    per_component = {'aasist': [], 'wav2vec': [], 'rawnet': [], 'lcnn': []}
    n_skipped = 0

    for _, row in sub.iterrows():
        try:
            audio = load_audio(row['path'])
            r = score_one(models, audio)
            scores.append(r['ensemble'])
            deployed_scores.append(r['deployed'])
            stages.append(r['stage'])
            labels.append(int(row['label']))
            sources.append(row['source'])
            per_component['aasist'].append(r['aasist'])
            per_component['wav2vec'].append(r['wav2vec'])
            per_component['rawnet'].append(r['rawnet'])
            per_component['lcnn'].append(r['lcnn'])
        except Exception as e:
            log.warning(f"Skipping {row['path']}: {e}")
            n_skipped += 1

    scores   = np.array(scores)
    deployed = np.array(deployed_scores)
    labels   = np.array(labels)
    sources  = np.array(sources)
    stages   = np.array(stages)

    # Per-source breakdown — both the raw ensemble verdict AND the DEPLOYED cascade
    # verdict (what production actually returns), so drift is visible for both.
    per_source = {}
    for src in sorted(set(sources)):
        mask = sources == src
        src_labels   = labels[mask]
        src_verdicts = [verdict_from_score(s) for s in scores[mask]]
        src_dep_verd = [verdict_from_score(s) for s in deployed[mask]]

        # Validate: each source must be uniformly fake or uniformly real.
        unique_labels = set(src_labels.tolist())
        if len(unique_labels) > 1:
            raise ValueError(
                f"Source '{src}' has mixed labels {unique_labels}. "
                f"Each source in the val manifest must be entirely fake (1) or real (0)."
            )

        n_src = int(mask.sum())
        vc = {v: sum(1 for x in src_verdicts if x == v)
              for v in ['auto_fake', 'likely_fake', 'to_review', 'auto_real']}
        if src_labels[0] == 1:  # fake source — catch rate (ensemble + deployed cascade)
            caught     = sum(1 for v in src_verdicts if v != 'auto_real')
            dep_caught = sum(1 for v in src_dep_verd if v != 'auto_real')
            per_source[src] = {
                'n': n_src,
                'catch_rate':          caught / n_src if n_src else 0.0,
                'deployed_catch_rate': dep_caught / n_src if n_src else 0.0,
                'verdict_counts': vc,
            }
        else:  # real source — false positive rate (ensemble + deployed cascade)
            fp     = sum(1 for v in src_verdicts if v != 'auto_real')
            dep_fp = sum(1 for v in src_dep_verd if v != 'auto_real')
            per_source[src] = {
                'n': n_src,
                'false_positive_rate':          fp / n_src if n_src else 0.0,
                'deployed_false_positive_rate': dep_fp / n_src if n_src else 0.0,
                'verdict_counts': vc,
            }

    # Per-component EERs (localises which sub-model is drifting; includes the
    # stage-1 LCNN screener).
    component_eers = {}
    for name, comp_scores in per_component.items():
        eer, _ = compute_eer(labels, np.array(comp_scores))
        component_eers[name] = eer

    ens_eer, ens_n_errors = compute_eer(labels, scores)
    dep_eer, _            = compute_eer(labels, deployed)
    stage1_resolution_pct = float((stages == 1).mean() * 100) if len(stages) else 0.0
    return {
        'n_samples':       len(scores),
        'n_skipped':       n_skipped,
        'n_errors_at_eer': ens_n_errors,
        'ensemble_eer':    ens_eer,
        'deployed_eer':    dep_eer,               # what the cascade actually delivers
        'stage1_resolution_pct': stage1_resolution_pct,
        'component_eer':   component_eers,        # includes 'lcnn'
        'per_source':      per_source,
    }

def evaluate_noizai(models, noizai_dir):
    """Run V8 on every Noiz.ai sample; return catch rate.
    Catch rate = fraction scored anything other than auto_real.
    This is an intentionally broad definition; to_review counts as caught.
    """
    if not os.path.isdir(noizai_dir):
        return {'available': False, 'reason': f'directory not found: {noizai_dir}'}

    audio_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(noizai_dir)
        for f in files
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        return {'available': False, 'reason': 'no audio files found'}

    results = []
    for fpath in audio_files:
        try:
            audio = load_audio(fpath)
            r = score_one(models, audio)
            results.append({
                'filename':        os.path.basename(fpath),
                'score':           r['ensemble'],
                'verdict':         verdict_from_score(r['ensemble']),
                'deployed_score':  r['deployed'],
                'deployed_verdict': verdict_from_score(r['deployed']),
                'lcnn':            r['lcnn'],
                'stage':           r['stage'],
            })
        except Exception as e:
            log.warning(f"Skipping {fpath}: {e}")

    n_total  = len(results)
    n_caught = sum(1 for r in results if r['verdict'] != 'auto_real')
    n_dep_caught = sum(1 for r in results if r['deployed_verdict'] != 'auto_real')
    n_stage1 = sum(1 for r in results if r['stage'] == 1)
    return {
        'available':  True,
        'n_samples':  n_total,
        'n_caught':   n_caught,
        'catch_rate': n_caught / n_total if n_total else 0.0,
        # What the deployed cascade actually catches (screener + ensemble):
        'deployed_n_caught':   n_dep_caught,
        'deployed_catch_rate': n_dep_caught / n_total if n_total else 0.0,
        'stage1_resolution_pct': (n_stage1 / n_total * 100) if n_total else 0.0,
        'per_file':   results,
    }

# ─── Main evaluation runner ────────────────────────────────────────────────

def run_evaluation(quick=False):
    """Run the full evaluation suite. Returns a metrics dict."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    log.info(f"Drift monitor run started: {timestamp}")
    log.info(f"Device: {DEVICE} | Mode: {'quick' if quick else 'full'}")

    models = load_models()

    with open(VAL_MANIFEST) as f:
        val_entries = json.load(f)
    val_df = pd.DataFrame(val_entries)
    _validate_manifest_schema(val_df)
    log.info(f"Val manifest: {len(val_df)} entries")

    n_samples = N_CLEAN_QUICK if quick else N_CLEAN_FULL
    log.info(f"Running clean evaluation on {n_samples} samples...")
    t0 = time.time()
    clean_results = evaluate_clean(models, val_df, n_samples)
    log.info(f"Clean eval done in {time.time()-t0:.0f}s. Ensemble EER: {clean_results['ensemble_eer']:.4f}")

    log.info("Running Noiz.ai evaluation...")
    t0 = time.time()
    noizai_results = evaluate_noizai(models, NOIZAI_DIR)
    if noizai_results.get('available'):
        log.info(f"Noiz.ai done in {time.time()-t0:.0f}s. "
                 f"Caught {noizai_results['n_caught']}/{noizai_results['n_samples']} "
                 f"({noizai_results['catch_rate']*100:.1f}%)")
    else:
        log.info(f"Noiz.ai skipped: {noizai_results.get('reason')}")

    manifest_hash = hashlib.sha256(
        json.dumps(val_entries, sort_keys=True).encode()
    ).hexdigest()[:16]

    metrics = {
        'timestamp':         timestamp,
        'mode':              'quick' if quick else 'full',
        'val_manifest_hash': manifest_hash,
        'val_manifest_n':    len(val_df),
        'clean':             clean_results,
        'noizai':            noizai_results,
    }

    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics

# ─── Drift detection ───────────────────────────────────────────────────────

def detect_drift(baseline, current):
    """
    Compare current metrics vs baseline.
    Returns list of alert dicts: {key, message, significant, p_value}.
    EER alerts include a significance test — threshold breach + p < alpha
    is required before the alert is promoted to 'confirmed'.
    """
    alerts = []

    # ── Clean ensemble EER ──────────────────────────────────────────────────
    if 'clean' in baseline and 'clean' in current:
        b_eer = baseline['clean'].get('ensemble_eer')
        c_eer = current['clean'].get('ensemble_eer')
        b_n   = baseline['clean'].get('n_samples', 0)
        c_n   = current['clean'].get('n_samples', 0)

        if b_eer is not None and c_eer is not None and not (np.isnan(b_eer) or np.isnan(c_eer)):
            delta     = (c_eer - b_eer) * 100
            threshold = ALERT_THRESHOLDS['clean_ensemble_eer_pp']
            # Alert on large increase OR suspiciously large decrease (possible eval bug)
            if abs(delta) > threshold:
                p_val = eer_significance_test(b_eer, c_eer, b_n, c_n)
                significant = p_val < EER_SIGNIFICANCE_ALPHA
                direction   = "increase" if delta > 0 else "decrease"
                alerts.append({
                    'key':         'clean_ensemble_eer',
                    'message':     (f"CLEAN ENSEMBLE EER {direction}: {b_eer:.4f} → {c_eer:.4f} "
                                   f"({delta:+.1f}pp, threshold ±{threshold}pp, "
                                   f"p={p_val:.3f}{'✓ significant' if significant else ' — may be noise'})"),
                    'significant': significant,
                    'p_value':     p_val,
                    'delta_pp':    delta,
                })

    # ── Deployed (cascade) EER — what production actually delivers ────────────
    if 'clean' in baseline and 'clean' in current:
        bd  = baseline['clean'].get('deployed_eer')
        cd  = current['clean'].get('deployed_eer')
        b_n = baseline['clean'].get('n_samples', 0)
        c_n = current['clean'].get('n_samples', 0)
        if bd is not None and cd is not None and not (np.isnan(bd) or np.isnan(cd)):
            delta     = (cd - bd) * 100
            threshold = ALERT_THRESHOLDS['clean_ensemble_eer_pp']
            if abs(delta) > threshold:
                p_val = eer_significance_test(bd, cd, b_n, c_n)
                significant = p_val < EER_SIGNIFICANCE_ALPHA
                direction   = "increase" if delta > 0 else "decrease"
                alerts.append({
                    'key':         'deployed_eer',
                    'message':     (f"DEPLOYED (cascade) EER {direction}: {bd:.4f} → {cd:.4f} "
                                   f"({delta:+.1f}pp, threshold ±{threshold}pp, "
                                   f"p={p_val:.3f}{'✓ significant' if significant else ' — may be noise'})"),
                    'significant': significant,
                    'p_value':     p_val,
                    'delta_pp':    delta,
                })

    # ── Per-source catch / FP rates ─────────────────────────────────────────
    for src, b_src in baseline.get('clean', {}).get('per_source', {}).items():
        c_src = current.get('clean', {}).get('per_source', {}).get(src)
        if c_src is None:
            continue
        if 'catch_rate' in b_src:
            delta_pp  = (c_src.get('catch_rate', 0) - b_src['catch_rate']) * 100
            threshold = (ALERT_THRESHOLDS['phone_catch_rate_pp']
                         if 'phone' in src.lower()
                         else ALERT_THRESHOLDS['per_source_catch_rate_pp'])
            if -delta_pp > threshold:
                alerts.append({
                    'key':         f'per_source_catch_{src}',
                    'message':     (f"FAKE SOURCE CATCH drift ({src}): "
                                   f"{b_src['catch_rate']*100:.1f}% → {c_src.get('catch_rate', 0)*100:.1f}% "
                                   f"({delta_pp:+.1f}pp, threshold -{threshold}pp)"),
                    'significant': True,  # catch rate drop doesn't need stat test — it's a count
                    'p_value':     None,
                    'delta_pp':    delta_pp,
                })

    # ── Noiz.ai catch rate ──────────────────────────────────────────────────
    b_noiz = baseline.get('noizai', {})
    c_noiz = current.get('noizai', {})
    if b_noiz.get('available') and c_noiz.get('available'):
        delta_pp  = (c_noiz['catch_rate'] - b_noiz['catch_rate']) * 100
        threshold = ALERT_THRESHOLDS['noizai_catch_rate_pp']
        if -delta_pp > threshold:
            alerts.append({
                'key':         'noizai_catch_rate',
                'message':     (f"NOIZ.AI CATCH RATE drift: "
                                f"{b_noiz['catch_rate']*100:.1f}% → {c_noiz['catch_rate']*100:.1f}% "
                                f"({delta_pp:+.1f}pp, threshold -{threshold}pp)"),
                'significant': True,
                'p_value':     None,
                'delta_pp':    delta_pp,
            })

    # ── Test set integrity ──────────────────────────────────────────────────
    if baseline.get('val_manifest_hash') != current.get('val_manifest_hash'):
        alerts.append({
            'key':         'manifest_changed',
            'message':     (f"TEST SET CHANGED: manifest hash differs from baseline "
                            f"(baseline: {baseline.get('val_manifest_hash')}, "
                            f"current: {current.get('val_manifest_hash')}). "
                            f"Re-run --init-baseline if this was intentional."),
            'significant': True,
            'p_value':     None,
            'delta_pp':    None,
        })

    return alerts

# ─── Confirmation state machine ────────────────────────────────────────────

def load_alert_state():
    """Load the consecutive-breach counter from disk. Returns dict keyed by alert key."""
    if ALERT_STATE_FILE.exists():
        try:
            with open(ALERT_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_alert_state(state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALERT_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

def update_confirmation_state(alerts, all_possible_keys):
    """
    Increment breach counters for triggered alerts; reset for clear ones.
    Returns (state, confirmed_alerts) where confirmed_alerts are those that
    have breached DRIFT_CONFIRM_RUNS consecutive times.
    """
    state = load_alert_state()
    triggered_keys = {a['key'] for a in alerts if a.get('significant', True)}

    for key in all_possible_keys:
        if key in triggered_keys:
            state[key] = state.get(key, 0) + 1
        else:
            if key in state:
                state[key] = 0  # reset on clear run

    confirmed = [a for a in alerts if state.get(a['key'], 0) >= DRIFT_CONFIRM_RUNS]
    save_alert_state(state)
    return state, confirmed

# ─── Email alerting ────────────────────────────────────────────────────────

def email_configured():
    return bool(EMAIL_CFG['smtp_host'] and EMAIL_CFG['smtp_user'] and EMAIL_CFG['alert_to'])

def send_alert_email(confirmed_alerts, metrics, state):
    """Send an email for confirmed drift alerts. Silently skips if email not configured."""
    if not confirmed_alerts:
        return
    if not email_configured():
        log.warning("Drift confirmed but email not configured — set DRIFT_SMTP_* env vars to enable alerts.")
        return

    subject = f"[VoiceGuard] Drift Alert — {len(confirmed_alerts)} confirmed issue(s)"

    body_lines = [
        "VoiceGuard Drift Monitor (active bundle)",
        "=" * 60,
        f"Timestamp: {metrics['timestamp']}",
        f"Mode:      {metrics['mode']}",
        f"Device:    {DEVICE}",
        "",
        f"CONFIRMED DRIFT DETECTED ({len(confirmed_alerts)} alert(s))",
        f"Each alert has breached its threshold for {DRIFT_CONFIRM_RUNS} consecutive run(s).",
        "-" * 60,
    ]
    for a in confirmed_alerts:
        body_lines.append(f"  ⚠  {a['message']}")
    body_lines += [
        "",
        "CURRENT METRICS",
        "-" * 60,
        f"  Ensemble EER:  {metrics['clean']['ensemble_eer']:.4f}",
        f"  N samples:     {metrics['clean']['n_samples']}",
    ]
    if metrics.get('noizai', {}).get('available'):
        body_lines.append(
            f"  Noiz.ai catch: {metrics['noizai']['catch_rate']*100:.1f}% "
            f"({metrics['noizai']['n_caught']}/{metrics['noizai']['n_samples']})"
        )
    body_lines += [
        "",
        "ACTION REQUIRED",
        "-" * 60,
        "  Review the detailed drift report on the server.",
        "  If drift is confirmed, consider retraining and promoting a new bundle.",
        "  To acknowledge and reset counters: delete drift_alert_state.json",
        "  To update baseline after retrain: run --init-baseline",
        "",
        "This is an automated alert from drift_monitor.py.",
    ]

    msg = MIMEMultipart()
    msg['From']    = EMAIL_CFG['smtp_user']
    msg['To']      = ', '.join(EMAIL_CFG['alert_to'])
    msg['Subject'] = subject
    msg.attach(MIMEText('\n'.join(body_lines), 'plain'))

    try:
        with smtplib.SMTP(EMAIL_CFG['smtp_host'], EMAIL_CFG['smtp_port']) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_CFG['smtp_user'], EMAIL_CFG['smtp_pass'])
            smtp.sendmail(EMAIL_CFG['smtp_user'], EMAIL_CFG['alert_to'], msg.as_string())
        log.info(f"Alert email sent to {EMAIL_CFG['alert_to']}")
    except Exception as e:
        log.error(f"Failed to send alert email: {e}")

# ─── Trend analysis ────────────────────────────────────────────────────────

def _active_bundle_version():
    try:
        import bundle_registry
        return bundle_registry.Registry().get_active()
    except Exception:
        return None


def fire_retrain_trigger(confirmed_alerts, metrics):
    """On CONFIRMED drift, raise a machine-readable retrain signal (retrain_trigger.json +
    an append-only log) and run $VOICEGUARD_RETRAIN_CMD if set. Actual retraining happens
    externally (GPU/Kaggle); this is the automation trigger a pipeline/operator acts on."""
    if not confirmed_alerts:
        return
    payload = {
        'retrain_needed': True,
        'triggered_at':   metrics.get('timestamp'),
        'active_bundle':  _active_bundle_version(),
        'confirm_runs':   DRIFT_CONFIRM_RUNS,
        'reasons':        [a['message'] for a in confirmed_alerts],
        'alert_keys':     [a['key'] for a in confirmed_alerts],
        'metrics_snapshot': {
            'clean_ensemble_eer': metrics.get('clean', {}).get('ensemble_eer'),
            'deployed_eer':       metrics.get('clean', {}).get('deployed_eer'),
            'noizai_catch_rate':  metrics.get('noizai', {}).get('catch_rate'),
            'per_source_catch':   metrics.get('clean', {}).get('per_source'),
        },
    }
    try:
        with open(RETRAIN_TRIGGER_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=_json_default)
        with open(RETRAIN_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, default=_json_default) + '\n')
        log.warning(f"RETRAIN TRIGGER fired ({len(confirmed_alerts)} confirmed alert(s)) "
                    f"→ {RETRAIN_TRIGGER_FILE}")
    except Exception as e:
        log.error(f"failed to write retrain trigger: {e}")
    if RETRAIN_CMD:
        try:
            log.warning(f"running VOICEGUARD_RETRAIN_CMD: {RETRAIN_CMD}")
            os.system(RETRAIN_CMD)
        except Exception as e:
            log.error(f"retrain command failed: {e}")


def retrain_status():
    """Current retrain-trigger state (for --retrain-status and the /drift endpoint)."""
    if RETRAIN_TRIGGER_FILE.exists():
        try:
            with open(RETRAIN_TRIGGER_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'retrain_needed': False}


def clear_retrain_trigger():
    """Clear the trigger once a retrain has been completed + promoted."""
    if RETRAIN_TRIGGER_FILE.exists():
        RETRAIN_TRIGGER_FILE.unlink()
        log.info("retrain trigger cleared")
    else:
        log.info("no retrain trigger to clear")


def load_trend(n=TREND_WINDOW):
    """Read the last N entries from the rolling log. Returns list of dicts."""
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries[-n:]

def format_trend(entries):
    if not entries:
        return "No run history found. Run the monitor at least once."
    lines = ["=" * 72, f"EER TREND (last {len(entries)} runs)", "=" * 72]
    eers = [e.get('clean_ens_eer') for e in entries if e.get('clean_ens_eer') is not None]
    for e in entries:
        ts  = e.get('timestamp', '')[:19].replace('T', ' ')
        eer = e.get('clean_ens_eer')
        n_a = e.get('n_alerts', 0)
        eer_str = f"{eer:.4f}" if eer is not None else "  n/a "
        alert_flag = f"  ⚠ {n_a} alert(s)" if n_a else ""
        lines.append(f"  {ts}   EER {eer_str}{alert_flag}")
    if len(eers) >= 3:
        slope, intercept, r, p, se = scipy_stats.linregress(range(len(eers)), eers)
        trend_dir = "↑ rising" if slope > 0 else "↓ falling"
        sig = "significant" if p < 0.05 else "not significant"
        lines += [
            "",
            f"  Trend (linear):  {trend_dir}  slope={slope*100:+.2f}pp/run  "
            f"p={p:.3f} ({sig})",
            f"  EER range:       {min(eers):.4f} – {max(eers):.4f}",
        ]
    lines.append("")
    return "\n".join(lines)

# ─── Reporting ─────────────────────────────────────────────────────────────

def format_report(metrics, baseline=None, alerts=None, confirmed_alerts=None, state=None):
    alerts           = alerts or []
    confirmed_alerts = confirmed_alerts or []
    alert_messages   = [a['message'] for a in alerts]
    confirmed_msgs   = {a['key'] for a in confirmed_alerts}

    lines = []
    lines.append("=" * 72)
    lines.append("VoiceGuard DRIFT MONITOR REPORT")
    lines.append("=" * 72)
    lines.append(f"Timestamp:   {metrics['timestamp']}")
    lines.append(f"Mode:        {metrics['mode']}")
    lines.append(f"Manifest:    {metrics['val_manifest_n']} entries (hash {metrics['val_manifest_hash']})")
    lines.append("")

    c = metrics['clean']
    lines.append("─" * 72)
    lines.append("CLEAN EVALUATION (val_v8_fresh)")
    lines.append("─" * 72)
    lines.append(f"  N samples:        {c['n_samples']}  (skipped: {c.get('n_skipped', 0)})")
    lines.append(f"  Ensemble EER:     {c['ensemble_eer']:.4f}")
    if 'deployed_eer' in c:
        lines.append(f"  Deployed EER:     {c['deployed_eer']:.4f}   (cascade — what production returns)")
        if baseline and baseline.get('clean', {}).get('deployed_eer') is not None:
            bd = baseline['clean']['deployed_eer']
            lines.append(f"    vs baseline:    {bd:.4f}  (Δ {(c['deployed_eer']-bd)*100:+.2f}pp)")
    if 'stage1_resolution_pct' in c:
        lines.append(f"  Stage-1 resolved: {c['stage1_resolution_pct']:.1f}%  "
                     f"(LCNN screener; rest escalate to the ensemble)")
    if baseline:
        b_eer = baseline.get('clean', {}).get('ensemble_eer')
        if b_eer is not None:
            delta = (c['ensemble_eer'] - b_eer) * 100
            b_n   = baseline.get('clean', {}).get('n_samples', 0)
            p_val = eer_significance_test(b_eer, c['ensemble_eer'], b_n, c['n_samples'])
            lines.append(f"    vs baseline:    {b_eer:.4f}  (Δ {delta:+.2f}pp,  p={p_val:.3f})")
    lines.append(f"  Component EERs:")
    for name, eer in c['component_eer'].items():
        lines.append(f"    {name:<10s} {eer:.4f}")
    lines.append("")
    lines.append("  Per-source breakdown:")
    for src, src_data in c['per_source'].items():
        if 'catch_rate' in src_data:
            lines.append(f"    {src:<25s} n={src_data['n']:>3d}  catch_rate={src_data['catch_rate']*100:>5.1f}%")
        else:
            lines.append(f"    {src:<25s} n={src_data['n']:>3d}  false_positive={src_data['false_positive_rate']*100:>5.1f}%")
    lines.append("")

    n = metrics.get('noizai', {})
    lines.append("─" * 72)
    lines.append("NOIZ.AI EVALUATION")
    lines.append("─" * 72)
    if n.get('available'):
        lines.append(f"  N samples:        {n['n_samples']}")
        lines.append(f"  Caught (ensemble):{n['n_caught']} ({n['catch_rate']*100:.1f}%)")
        if 'deployed_catch_rate' in n:
            lines.append(f"  Caught (deployed):{n.get('deployed_n_caught', 0)} "
                         f"({n['deployed_catch_rate']*100:.1f}%)   ← cascade, what production catches")
        if 'stage1_resolution_pct' in n:
            lines.append(f"  Stage-1 resolved: {n['stage1_resolution_pct']:.1f}%")
        if baseline:
            b_n = baseline.get('noizai', {})
            if b_n.get('available'):
                delta = (n['catch_rate'] - b_n['catch_rate']) * 100
                lines.append(f"    vs baseline:    {b_n['catch_rate']*100:.1f}% (Δ {delta:+.2f}pp)")
    else:
        lines.append(f"  Skipped: {n.get('reason', 'unknown')}")
    lines.append("")

    lines.append("─" * 72)
    lines.append("DRIFT ALERTS")
    lines.append("─" * 72)
    if alerts:
        for a in alerts:
            prefix = "  ⚠⚠ CONFIRMED" if a['key'] in confirmed_msgs else "  ⚠  PENDING  "
            lines.append(f"{prefix}  {a['message']}")
        lines.append("")
        lines.append(f"  Confirmation threshold: {DRIFT_CONFIRM_RUNS} consecutive run(s).")
        if state:
            lines.append("  Breach counters:")
            for k, v in state.items():
                if v > 0:
                    lines.append(f"    {k}: {v}/{DRIFT_CONFIRM_RUNS}")
    else:
        if baseline:
            lines.append("  None — V8 performance is within tolerance vs baseline.")
        else:
            lines.append("  No baseline to compare against. Run with --init-baseline first.")
    lines.append("")

    return "\n".join(lines)

# ─── Living test set: sample ingestion ─────────────────────────────────────

def cmd_add_samples(src_dir, label, source_name):
    """
    Ingest new audio files into the living val manifest.
    Skips duplicates via SHA256 content hash. Updates VAL_MANIFEST in place.
    """
    label = int(label)
    if label not in (0, 1):
        print("label must be 0 (real) or 1 (fake)")
        sys.exit(1)

    # Load existing manifest
    existing = []
    existing_hashes = set()
    if os.path.exists(VAL_MANIFEST):
        with open(VAL_MANIFEST) as f:
            existing = json.load(f)
        for e in existing:
            h = e.get('content_hash')
            if h:
                existing_hashes.add(h)

    audio_files = sorted(
        os.path.join(root, fname)
        for root, _, files in os.walk(src_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS
    )

    added = 0
    skipped_dup = 0
    for fpath in audio_files:
        with open(fpath, 'rb') as fh:
            content_hash = hashlib.sha256(fh.read()).hexdigest()
        if content_hash in existing_hashes:
            skipped_dup += 1
            continue
        existing.append({
            'path':         str(Path(fpath).resolve()),
            'label':        label,
            'source':       source_name,
            'content_hash': content_hash,
            'added':        datetime.now(timezone.utc).isoformat(),
        })
        existing_hashes.add(content_hash)
        added += 1

    with open(VAL_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)

    print(f"\n✓ Living test set updated: {VAL_MANIFEST}")
    print(f"  Added:           {added} new samples (source={source_name}, label={label})")
    print(f"  Skipped (dup):   {skipped_dup}")
    print(f"  Total manifest:  {len(existing)} entries")
    print(f"\n  Re-run --init-baseline after adding substantial new samples.")

# ─── Commands ──────────────────────────────────────────────────────────────

def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)

def cmd_init_baseline(quick=False):
    metrics = run_evaluation(quick=quick)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    # Reset alert state on re-baseline
    if ALERT_STATE_FILE.exists():
        ALERT_STATE_FILE.unlink()
    print(format_report(metrics))
    print(f"\n✓ Baseline saved → {BASELINE_FILE}")
    print(f"  Alert state reset.")
    print(f"  Future runs will compare against this baseline.")

def cmd_run(quick=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = None
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)
    else:
        log.warning(f"No baseline at {BASELINE_FILE} — drift detection disabled. Run --init-baseline.")

    metrics = run_evaluation(quick=quick)
    alerts  = detect_drift(baseline, metrics) if baseline else []

    # All possible alert keys (for resetting cleared ones in state)
    all_keys = (
        ['clean_ensemble_eer', 'deployed_eer', 'noizai_catch_rate', 'manifest_changed'] +
        [f'per_source_catch_{src}' for src in metrics['clean']['per_source'].keys()]
    )
    state, confirmed = update_confirmation_state(alerts, all_keys) if baseline else ({}, [])

    report = format_report(metrics, baseline=baseline, alerts=alerts,
                           confirmed_alerts=confirmed, state=state)
    print(report)

    # Save per-run files
    safe_ts   = metrics['timestamp'].replace(':', '').replace('-', '').split('.')[0]
    json_path = OUTPUT_DIR / f'drift_report_{safe_ts}.json'
    txt_path  = OUTPUT_DIR / f'drift_report_{safe_ts}.txt'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'alerts': alerts, 'confirmed': [a['key'] for a in confirmed]},
                  f, indent=2, default=_json_default)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log.info(f"Report saved → {json_path}")

    # Append to rolling log
    log_entry = {
        'timestamp':         metrics['timestamp'],
        'mode':              metrics['mode'],
        'clean_ens_eer':     metrics['clean']['ensemble_eer'],
        'deployed_eer':      metrics['clean'].get('deployed_eer'),
        'stage1_resolution_pct': metrics['clean'].get('stage1_resolution_pct'),
        'noizai_catch_rate': metrics.get('noizai', {}).get('catch_rate'),
        'noizai_deployed_catch_rate': metrics.get('noizai', {}).get('deployed_catch_rate'),
        'n_alerts':          len(alerts),
        'n_confirmed':       len(confirmed),
        'alerts':            [a['message'] for a in alerts],
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, default=_json_default) + '\n')
    log.info(f"Rolling log → {LOG_FILE}")

    # Fire email + retrain trigger if confirmed drift
    send_alert_email(confirmed, metrics, state)
    fire_retrain_trigger(confirmed, metrics)

    if confirmed:
        log.warning(f"{len(confirmed)} CONFIRMED drift alert(s). Email sent if configured.")
    elif alerts:
        log.info(f"{len(alerts)} pending alert(s) — not yet confirmed ({DRIFT_CONFIRM_RUNS} runs required).")

def cmd_diff():
    if not BASELINE_FILE.exists():
        print(f"No baseline at {BASELINE_FILE}. Run --init-baseline first.")
        return
    json_files = sorted(OUTPUT_DIR.glob('drift_report_*.json'))
    if not json_files:
        print(f"No run JSON files in {OUTPUT_DIR}.")
        return
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    with open(json_files[-1]) as f:
        latest = json.load(f)
    metrics   = latest['metrics']
    alerts    = [{'key': k, 'message': m, 'significant': True, 'p_value': None, 'delta_pp': None}
                 for k, m in zip(latest.get('confirmed', []), latest.get('alerts', []))]
    print(format_report(metrics, baseline=baseline, alerts=alerts))

def cmd_trend():
    entries = load_trend()
    print(format_trend(entries))

def cmd_schedule():
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    print()
    print("Add ONE or BOTH of these to your crontab (crontab -e):")
    print()
    print("  # Daily quick check at 3am UTC")
    print(f"  0 3 * * * {python_path} {script_path} --quick >> /var/log/v8_drift.log 2>&1")
    print()
    print("  # Full weekly run Sundays at 3am UTC")
    print(f"  0 3 * * 0 {python_path} {script_path} >> /var/log/v8_drift.log 2>&1")
    print()
    print("Email alerts require DRIFT_SMTP_* environment variables to be set.")
    print("See the module docstring for the full list.")
    print()

# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='V8 drift monitor — living test set + automated alerting (Phase 5 Task 5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (set before running for email alerts):
  DRIFT_SMTP_HOST, DRIFT_SMTP_PORT, DRIFT_SMTP_USER, DRIFT_SMTP_PASS
  DRIFT_ALERT_TO       (comma-separated recipients)
  DRIFT_CONFIRM_RUNS   (consecutive breaches before email fires, default 2)
  DRIFT_OUTPUT_DIR     (default: ./output)
  DRIFT_NOIZAI_DIR     (default: ./data/noizai)
        """
    )
    parser.add_argument('--init-baseline', action='store_true',
                        help='Compute and save baseline metrics. Run once initially.')
    parser.add_argument('--quick', action='store_true',
                        help='Use smaller sample size for faster runs.')
    parser.add_argument('--diff', action='store_true',
                        help='Show latest run vs baseline without re-running.')
    parser.add_argument('--trend', action='store_true',
                        help=f'Show EER trend across last {TREND_WINDOW} runs.')
    parser.add_argument('--schedule', action='store_true',
                        help='Print crontab lines for automation.')
    parser.add_argument('--add-samples', nargs=3,
                        metavar=('DIR', 'LABEL', 'SOURCE'),
                        help='Ingest new audio into the living test set. '
                             'LABEL=0 (real) or 1 (fake). SOURCE=a short name string.')
    parser.add_argument('--retrain-status', action='store_true',
                        help='Print the current retrain-trigger state (retrain_needed + reasons).')
    parser.add_argument('--clear-trigger', action='store_true',
                        help='Clear the retrain trigger after a retrain has been promoted.')
    args = parser.parse_args()

    if args.retrain_status:
        print(json.dumps(retrain_status(), indent=2, default=_json_default))
    elif args.clear_trigger:
        clear_retrain_trigger()
    elif args.add_samples:
        cmd_add_samples(*args.add_samples)
    elif args.schedule:
        cmd_schedule()
    elif args.trend:
        cmd_trend()
    elif args.init_baseline:
        cmd_init_baseline(quick=args.quick)
    elif args.diff:
        cmd_diff()
    else:
        cmd_run(quick=args.quick)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
cascade_and_distillation.py — Phase 7, Tasks 1-3: Cascade Screener,
Cascade Router, and Knowledge Distillation

Builds two separate compact models and the routing logic between them:

  1. STAGE-1 SCREENER (LCNN): a small, fast model trained with plain
     supervised learning (cross-entropy on real/fake labels) on cheap
     spectrogram features. Its only job is to resolve EASY cases quickly —
     clearly real or clearly fake audio — without invoking the full,
     expensive 3-model V8 ensemble.

  2. CASCADE ROUTER: decision logic. If the screener's output is confident
     (close to 0 or close to 1), trust it and return immediately. If it's
     uncertain (middle range), forward the sample to the full V8 ensemble
     and use ITS score as the final answer. Most traffic should resolve at
     stage 1; only ambiguous cases pay the cost of the full ensemble.

  3. DISTILLED STUDENT MODEL: a separate compact model trained via
     KNOWLEDGE DISTILLATION — not just on hard labels, but on the full
     ensemble's SOFT scores as a training target. The goal is a single
     small model (<200MB) whose standalone EER is within 3% of the full
     ensemble's EER — a compact stand-in for situations where running the
     full 3-model ensemble isn't practical (e.g. resource-constrained
     deployment).

These are two DIFFERENT models with two DIFFERENT training objectives,
even though they may share the same underlying architecture (LCNN):
  - Screener: trained for fast triage, optimized for resolving easy cases
    confidently. It is allowed to be uncertain on hard cases — uncertainty
    is the SIGNAL that routes to the full ensemble.
  - Student: trained to be the closest possible single-model approximation
    of the full ensemble's behavior, including on hard cases. It does not
    get to defer to anything else.

EXIT GATES this script is built to help verify:
  - Cascade resolves >=80% of cases at stage 1, each in <50ms
  - Student model: EER within 3% (absolute) of the full ensemble, file
    size <200MB

HONEST SCOPE NOTE
------------------------------------------------------------------------
<50ms latency cannot be honestly verified here — it depends on the actual
deployment hardware (CPU/GPU, batch size 1 vs batched). This script
measures WALL-CLOCK inference time on whatever machine runs it and reports
it plainly, but you should re-measure on the actual production hardware
before treating the latency exit gate as satisfied.

USAGE
-----
    python cascade_and_distillation.py --train-screener
    python cascade_and_distillation.py --validate-cascade
    python cascade_and_distillation.py --train-student
    python cascade_and_distillation.py --evaluate-student
    python cascade_and_distillation.py --full          # all of the above, in order
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy.io import wavfile
from sklearn.metrics import roc_curve

# ─── Configuration ──────────────────────────────────────────────────────────

DATASET_ROOT = 'C:\\Users\\Michael Ologungbara\\Downloads\\voice_guard 0ffline'
TRAIN_MANIFEST = f'{DATASET_ROOT}/models/train_v8_fresh.json'
VAL_MANIFEST   = f'{DATASET_ROOT}/models/val_v8_fresh.json'

OUTPUT_DIR = Path('C:\\Users\\Michael Ologungbara\\Downloads\\voice_guard 0ffline')
SCREENER_CKPT = OUTPUT_DIR / 'lcnn_screener.pt'
STUDENT_CKPT  = OUTPUT_DIR / 'lcnn_student.pt'

SR = 16000
SAMPLE_LEN = 4 * SR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# Cascade routing thresholds — a screener score outside this band is
# trusted directly; inside the band, defer to the full ensemble.
# Starting point: symmetric band around 0.5; tune based on validation.
CASCADE_CONFIDENT_LOW  = 0.15   # below this -> trust screener's "real" call
CASCADE_CONFIDENT_HIGH = 0.85   # above this -> trust screener's "fake" call

# Spectrogram feature config (cheap, fixed-size — NOT the expensive raw
# SincConv pipeline AASIST/RawNet3 use. Speed is the entire point of stage 1.)
N_MELS = 64
N_FRAMES = 128  # fixed width via crop/pad

SCREENER_EPOCHS = 15
STUDENT_EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 32
DISTILL_TEMP = 3.0
DISTILL_WEIGHT = 0.7  # weight on the distillation (soft-target) loss vs hard-label CE

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Audio loading + cheap spectrogram features ────────────────────────────

def load_audio(path, max_samples=SAMPLE_LEN):
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t:
        wav_path = t.name
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(path), '-ar', str(SR), '-ac', '1',
             '-acodec', 'pcm_s16le', wav_path],
            capture_output=True, timeout=15)
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

def compute_logmel(audio_np, sr=SR, n_mels=N_MELS, n_frames=N_FRAMES):
    """Cheap, fixed-size log-mel spectrogram — this is the whole reason
    stage 1 can be fast: no SincConv, no learned front-end, just a standard
    STFT-based feature that's quick to compute and quick to run a small
    CNN over."""
    import librosa
    mel = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512)
    logmel = librosa.power_to_db(mel, ref=np.max)
    # Fix width to n_frames via center-crop or pad
    if logmel.shape[1] > n_frames:
        start = (logmel.shape[1] - n_frames) // 2
        logmel = logmel[:, start:start + n_frames]
    elif logmel.shape[1] < n_frames:
        pad = n_frames - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)))
    # Normalize to roughly [-1, 1] range for stable training
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)

# ─── LCNN architecture (Light CNN, Wu et al. 2018 — standard anti-spoofing
#     baseline architecture; Max-Feature-Map activation) ──────────────────

class MFM(nn.Module):
    """Max-Feature-Map: splits channels into two halves, takes the
    element-wise max. Halves the channel count and acts as both a
    nonlinearity and an implicit feature-competition mechanism — the
    standard activation used in LCNN-family anti-spoofing models."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)

class LCNN(nn.Module):
    """Compact Light-CNN for spectrogram-based real/fake classification.
    Deliberately small: this is meant to be fast (<50ms) and small (<200MB
    as a standalone student), not state-of-the-art.
    Input: (B, 1, N_MELS, N_FRAMES) log-mel spectrogram.
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.mfm1 = MFM()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2a = nn.Conv2d(16, 32, kernel_size=1)
        self.mfm2a = MFM()
        self.conv2b = nn.Conv2d(16, 48, kernel_size=3, padding=1)
        self.mfm2b = MFM()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3a = nn.Conv2d(24, 48, kernel_size=1)
        self.mfm3a = MFM()
        self.conv3b = nn.Conv2d(24, 64, kernel_size=3, padding=1)
        self.mfm3b = MFM()
        self.pool3 = nn.MaxPool2d(2)

        self.bn = nn.BatchNorm2d(24)

        # Flattened size depends on N_MELS/N_FRAMES after 3x MaxPool2d(2):
        # N_MELS=64 -> 8, N_FRAMES=128 -> 16, channels after pool3 = 32
        flat_size = 32 * (N_MELS // 8) * (N_FRAMES // 8)
        self.fc1 = nn.Linear(flat_size, 160)
        self.mfm_fc = MFM()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(80, num_classes)

    def forward(self, x):
        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.mfm2a(self.conv2a(x))
        x = self.pool2(self.mfm2b(self.conv2b(x)))
        x = self.bn(x)
        x = self.mfm3a(self.conv3a(x))
        x = self.pool3(self.mfm3b(self.conv3b(x)))
        x = x.flatten(1)
        x = self.mfm_fc(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def model_size_mb(model):
    n_params = sum(p.numel() for p in model.parameters())
    # float32 = 4 bytes/param; this is the in-memory/serialized estimate
    return n_params * 4 / (1024 ** 2)

# ─── Dataset preparation ─────────────────────────────────────────────────────

def load_manifest_df(path):
    with open(path) as f:
        return pd.DataFrame(json.load(f))

def build_feature_cache(df, cache_path, n_samples=None, seed=SEED):
    """Precompute log-mel features for a manifest subset and cache to disk
    as a single .npz — avoids recomputing spectrograms every epoch."""
    if n_samples:
        n_per_class = n_samples // 2
        real = df[df['label'] == 0].sample(min(n_per_class, (df['label']==0).sum()), random_state=seed)
        fake = df[df['label'] == 1].sample(min(n_per_class, (df['label']==1).sum()), random_state=seed)
        df = pd.concat([real, fake]).reset_index(drop=True)

    X, y, paths = [], [], []
    for i, row in df.iterrows():
        try:
            audio = load_audio(row['path'])
            feat = compute_logmel(audio)
            X.append(feat); y.append(int(row['label'])); paths.append(row['path'])
        except Exception as e:
            print(f"  skipping {row['path']}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"  features computed: {i+1}/{len(df)}", flush=True)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(cache_path, X=X, y=y, paths=json.dumps(paths))
    print(f"Feature cache saved: {cache_path} ({len(X)} samples)", flush=True)
    return X, y, paths

def load_feature_cache(cache_path):
    data = np.load(cache_path, allow_pickle=False)
    return data['X'], data['y'], json.loads(str(data['paths']))

# ─── EER ─────────────────────────────────────────────────────────────────────

def compute_eer(labels, scores):
    if len(set(labels)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)

# ─── Task 1: train the stage-1 screener (plain supervised CE) ──────────────

def train_lcnn(X, y, epochs, lr=LR, batch_size=BATCH_SIZE, val_frac=0.2, seed=SEED,
                teacher_scores=None, distill_weight=0.0, distill_temp=DISTILL_TEMP):
    """Shared training loop for both the screener (distill_weight=0, plain
    CE) and the student (distill_weight>0, CE + KD against teacher_scores).
    teacher_scores, if given, must be aligned 1:1 with X/y (the full
    ensemble's calibrated score for each sample)."""
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    t_train = teacher_scores[train_idx] if teacher_scores is not None else None
    t_val = teacher_scores[val_idx] if teacher_scores is not None else None

    model = LCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_eer = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(X_train))
        epoch_loss = epoch_ce = epoch_kd = 0.0
        n_batches = 0
        for start in range(0, len(X_train), batch_size):
            bidx = perm[start:start + batch_size]
            xb = torch.from_numpy(X_train[bidx]).float().unsqueeze(1).to(DEVICE)
            yb = torch.from_numpy(y_train[bidx]).long().to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            ce_loss = F.cross_entropy(logits, yb)

            if distill_weight > 0 and t_train is not None:
                teacher_p = torch.from_numpy(t_train[bidx]).float().to(DEVICE)
                # Build a 2-class soft target [1-p, p] from the teacher's
                # single fake-probability score, temperature-scaled.
                teacher_soft = torch.stack([1 - teacher_p, teacher_p], dim=1)
                teacher_soft = torch.clamp(teacher_soft, 1e-6, 1 - 1e-6)
                student_log_soft = F.log_softmax(logits / distill_temp, dim=1)
                kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (distill_temp ** 2)
                loss = (1 - distill_weight) * ce_loss + distill_weight * kd_loss
                epoch_kd += kd_loss.item()
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); epoch_ce += ce_loss.item(); n_batches += 1

        model.eval()
        val_scores = []
        with torch.no_grad():
            for start in range(0, len(X_val), batch_size):
                xv = torch.from_numpy(X_val[start:start+batch_size]).float().unsqueeze(1).to(DEVICE)
                val_scores.append(F.softmax(model(xv), dim=1)[:, 1].cpu().numpy())
        val_scores = np.concatenate(val_scores)
        val_eer = compute_eer(y_val, val_scores)

        history.append({'epoch': epoch+1, 'loss': epoch_loss/max(n_batches,1),
                        'ce': epoch_ce/max(n_batches,1), 'kd': epoch_kd/max(n_batches,1),
                        'val_eer': val_eer})
        print(f"  epoch {epoch+1}/{epochs}: loss={history[-1]['loss']:.4f} val_EER={val_eer:.4f}", flush=True)

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_eer, history, (X_val, y_val, t_val)

# ─── Task 2: cascade router + validation ────────────────────────────────────

def cascade_decide(screener_score, low=CASCADE_CONFIDENT_LOW, high=CASCADE_CONFIDENT_HIGH):
    """Returns (resolved_at_stage1: bool, verdict_if_resolved)."""
    if screener_score <= low:
        return True, 'auto_real_via_screener'
    if screener_score >= high:
        return True, 'auto_fake_via_screener'
    return False, None  # ambiguous -> defer to full ensemble

def validate_cascade(screener, X_val, y_val, full_ensemble_score_fn=None,
                     low=CASCADE_CONFIDENT_LOW, high=CASCADE_CONFIDENT_HIGH):
    """Measures: % resolved at stage 1, per-sample latency, and — for cases
    that DO get deferred — confirms they're actually the harder cases
    (i.e. screener's uncertainty correlates with the full ensemble disagreeing
    or being less confident, not just routing arbitrarily).
    full_ensemble_score_fn: optional callable(index) -> full ensemble score,
    used only to sanity-check deferred cases if available.
    """
    screener.eval()
    n = len(X_val)
    resolved_flags = []
    latencies_ms = []
    screener_scores = []

    for i in range(n):
        x = torch.from_numpy(X_val[i:i+1]).float().unsqueeze(1).to(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            score = F.softmax(screener(x), dim=1)[0, 1].item()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        screener_scores.append(score)
        resolved, _ = cascade_decide(score, low, high)
        resolved_flags.append(resolved)

    resolved_flags = np.array(resolved_flags)
    latencies_ms = np.array(latencies_ms)
    pct_resolved = resolved_flags.mean()

    result = {
        'n_samples': n,
        'pct_resolved_at_stage1': float(pct_resolved),
        'mean_latency_ms': float(latencies_ms.mean()),
        'p95_latency_ms': float(np.percentile(latencies_ms, 95)),
        'p99_latency_ms': float(np.percentile(latencies_ms, 99)),
        'low_threshold': low, 'high_threshold': high,
    }

    # Accuracy check ONLY on stage-1-resolved cases — these should be
    # reliably correct, since this is where the cascade commits without
    # the full ensemble's input.
    resolved_idx = np.where(resolved_flags)[0]
    if len(resolved_idx) > 0:
        resolved_scores = np.array(screener_scores)[resolved_idx]
        resolved_labels = y_val[resolved_idx]
        resolved_preds = (resolved_scores >= 0.5).astype(int)
        stage1_accuracy = float((resolved_preds == resolved_labels).mean())
        result['stage1_resolved_accuracy'] = stage1_accuracy
    else:
        result['stage1_resolved_accuracy'] = None

    return result

# ─── Task 3: distillation evaluation ────────────────────────────────────────

def evaluate_student_vs_ensemble(student, X_val, y_val, ensemble_eer, batch_size=BATCH_SIZE):
    student.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(X_val), batch_size):
            xv = torch.from_numpy(X_val[start:start+batch_size]).float().unsqueeze(1).to(DEVICE)
            scores.append(F.softmax(student(xv), dim=1)[:, 1].cpu().numpy())
    scores = np.concatenate(scores)
    student_eer = compute_eer(y_val, scores)
    delta = abs(student_eer - ensemble_eer)
    size_mb = model_size_mb(student)
    return {
        'student_eer': student_eer,
        'ensemble_eer': ensemble_eer,
        'eer_delta_absolute': delta,
        'within_3pp_target': delta <= 0.03,
        'model_size_mb': size_mb,
        'under_200mb_target': size_mb < 200,
    }

# ─── CLI ────────────────────────────────────────────────────────────────────

def cmd_train_screener(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / 'screener_features.npz'
    if cache_path.exists() and not args.regenerate_features:
        print(f"Loading cached features: {cache_path}")
        X, y, paths = load_feature_cache(cache_path)
    else:
        df = load_manifest_df(TRAIN_MANIFEST)
        print(f"Computing log-mel features for {args.n_samples} training samples...")
        X, y, paths = build_feature_cache(df, cache_path, n_samples=args.n_samples)

    print(f"\nTraining stage-1 LCNN screener on {len(X)} samples...")
    model, best_eer, history, _ = train_lcnn(X, y, epochs=args.epochs)
    print(f"\nScreener best val EER: {best_eer:.4f}")
    print(f"Screener model size: {model_size_mb(model):.2f} MB")

    torch.save({'model_state': model.state_dict()}, SCREENER_CKPT)
    print(f"Saved: {SCREENER_CKPT}")

def cmd_validate_cascade(args):
    if not SCREENER_CKPT.exists():
        print("No trained screener found. Run --train-screener first.")
        sys.exit(1)
    screener = LCNN().to(DEVICE)
    screener.load_state_dict(torch.load(SCREENER_CKPT, map_location=DEVICE, weights_only=False)['model_state'])

    df = load_manifest_df(VAL_MANIFEST)
    cache_path = OUTPUT_DIR / 'cascade_val_features.npz'
    if cache_path.exists() and not args.regenerate_features:
        X_val, y_val, _ = load_feature_cache(cache_path)
    else:
        X_val, y_val, _ = build_feature_cache(df, cache_path, n_samples=args.n_samples)

    print(f"\nValidating cascade routing on {len(X_val)} held-out samples...")
    result = validate_cascade(screener, X_val, y_val, low=args.low, high=args.high)

    print(f"\n{'='*72}\nCASCADE VALIDATION RESULT\n{'='*72}")
    print(json.dumps(result, indent=2))
    print()
    print(f"EXIT GATE (>=80% resolved at stage 1): "
          f"{'PASSED' if result['pct_resolved_at_stage1'] >= 0.80 else 'FAILED'} "
          f"({result['pct_resolved_at_stage1']*100:.1f}%)")
    print(f"Latency (THIS MACHINE, not necessarily production hardware): "
          f"mean={result['mean_latency_ms']:.2f}ms, p95={result['p95_latency_ms']:.2f}ms")
    print(f"  Re-measure on actual production hardware before treating the")
    print(f"  <50ms exit gate as confirmed.")
    if result['stage1_resolved_accuracy'] is not None:
        print(f"Stage-1-resolved accuracy: {result['stage1_resolved_accuracy']*100:.1f}% "
              f"(should be high — these are the 'easy' cases by construction)")

def cmd_train_student(args):
    """Trains the distilled student. Requires teacher (full ensemble) scores
    for the same training samples — these must be precomputed and supplied,
    since this script doesn't load the full V8 ensemble itself (keeps this
    script's dependencies light; reuse drift_monitor.py's load_v8/score_one
    to generate a teacher_scores.npz file, see --teacher-scores-path)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / 'student_features.npz'
    if cache_path.exists() and not args.regenerate_features:
        X, y, paths = load_feature_cache(cache_path)
    else:
        df = load_manifest_df(TRAIN_MANIFEST)
        X, y, paths = build_feature_cache(df, cache_path, n_samples=args.n_samples)

    if not args.teacher_scores_path or not os.path.exists(args.teacher_scores_path):
        print(f"ERROR: --teacher-scores-path not found: {args.teacher_scores_path}")
        print(f"Knowledge distillation requires the full V8 ensemble's score for each")
        print(f"training sample. Generate this with a small companion script using")
        print(f"drift_monitor.py's load_v8()/score_one() over the same `paths` saved")
        print(f"in {cache_path}, then save as a .npy array aligned to those paths.")
        sys.exit(1)

    teacher_scores = np.load(args.teacher_scores_path)
    if len(teacher_scores) != len(X):
        print(f"ERROR: teacher_scores length ({len(teacher_scores)}) != features length ({len(X)}). "
              f"They must be aligned 1:1 with the same sample order.")
        sys.exit(1)

    print(f"\nTraining distilled student on {len(X)} samples (CE + KD vs teacher)...")
    model, best_eer, history, (X_val, y_val, t_val) = train_lcnn(
        X, y, epochs=args.epochs, teacher_scores=teacher_scores,
        distill_weight=DISTILL_WEIGHT)
    print(f"\nStudent best val EER (on hard labels): {best_eer:.4f}")
    print(f"Student model size: {model_size_mb(model):.2f} MB")

    torch.save({'model_state': model.state_dict()}, STUDENT_CKPT)
    print(f"Saved: {STUDENT_CKPT}")

    # Save the val split's indices/scores too, so --evaluate-student can
    # compare against the SAME held-out teacher scores later.
    np.savez(OUTPUT_DIR / 'student_val_split.npz', X_val=X_val, y_val=y_val, t_val=t_val)

def cmd_evaluate_student(args):
    if not STUDENT_CKPT.exists():
        print("No trained student found. Run --train-student first.")
        sys.exit(1)
    student = LCNN().to(DEVICE)
    student.load_state_dict(torch.load(STUDENT_CKPT, map_location=DEVICE, weights_only=False)['model_state'])

    split_path = OUTPUT_DIR / 'student_val_split.npz'
    if not split_path.exists():
        print("No saved validation split found. Run --train-student first (it saves this).")
        sys.exit(1)
    data = np.load(split_path)
    X_val, y_val, t_val = data['X_val'], data['y_val'], data['t_val']

    ensemble_eer = compute_eer(y_val, t_val)
    result = evaluate_student_vs_ensemble(student, X_val, y_val, ensemble_eer)

    print(f"\n{'='*72}\nSTUDENT VS ENSEMBLE EVALUATION\n{'='*72}")
    print(json.dumps(result, indent=2))
    print()
    print(f"EXIT GATE (EER within 3pp of ensemble): "
          f"{'PASSED' if result['within_3pp_target'] else 'FAILED'}")
    print(f"EXIT GATE (size < 200MB): "
          f"{'PASSED' if result['under_200mb_target'] else 'FAILED'} "
          f"({result['model_size_mb']:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description='Phase 7 Tasks 1-3: cascade screener, router, distillation')
    parser.add_argument('--train-screener', action='store_true')
    parser.add_argument('--validate-cascade', action='store_true')
    parser.add_argument('--train-student', action='store_true')
    parser.add_argument('--evaluate-student', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=SCREENER_EPOCHS)
    parser.add_argument('--regenerate-features', action='store_true')
    parser.add_argument('--low', type=float, default=CASCADE_CONFIDENT_LOW)
    parser.add_argument('--high', type=float, default=CASCADE_CONFIDENT_HIGH)
    parser.add_argument('--teacher-scores-path', type=str, default=None)
    args = parser.parse_args()

    if args.full:
        cmd_train_screener(args)
        cmd_validate_cascade(args)
        if args.teacher_scores_path:
            cmd_train_student(args)
            cmd_evaluate_student(args)
        else:
            print("\n--full stopped before student training: --teacher-scores-path not given.")
        return

    if args.train_screener: cmd_train_screener(args)
    elif args.validate_cascade: cmd_validate_cascade(args)
    elif args.train_student: cmd_train_student(args)
    elif args.evaluate_student: cmd_evaluate_student(args)
    else:
        parser.error('Specify --train-screener, --validate-cascade, --train-student, --evaluate-student, or --full')

if __name__ == '__main__':
    main()

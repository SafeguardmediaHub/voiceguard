#!/usr/bin/env python3
"""
fewshot_adapt.py — V8 Few-Shot Adaptation Pipeline (Phase 5 Task 6)

General-purpose tool to rapidly adapt V8 when a NEW, UNKNOWN attack type
appears (novel TTS model, new voice conversion method, unforeseen effect
chain, etc). You don't need to know in advance what the next attack looks
like — you need 10-20 labeled examples of it and this pipeline does the
rest, with an automatic safety gate that refuses to deploy an adaptation
that regresses existing performance.

DESIGN (see Phase5_handoff_summary.md for the four decisions this resolves)
-----------------------------------------------------------------------
1. Component fine-tuned: AASIST ONLY. Highest XGB ensemble weight (0.501),
   smallest model, fastest to adapt, and the known weak point. Wav2Vec and
   RawNet3 are left untouched (RawNet3 in particular contributes little —
   component EER 0.355 on val_v8_fresh — so adapting it on 10-20 samples
   is pure overfitting risk for ~no leverage).

2. Loss: cross-entropy on new samples + KL-divergence distillation against
   a FROZEN COPY of the original AASIST ("teacher"), computed on a replay
   buffer resampled from the original training manifest. This anchors the
   student to old behavior while nudging it toward the new attack type.

3. Forgetting prevention: (a) low LR (default 2e-6), (b) only the last 2
   ResBlocks + both GAT layers + classifier are unfrozen — SincConv and
   the first 4 ResBlocks (the low-level spectral front-end) stay frozen,
   (c) replay buffer mixed into every batch, weighted so new samples don't
   dominate the gradient.

4. Validation gate: re-runs a clean-set eval (reusing drift_monitor.py's
   eval logic/thresholds) before and after adaptation. If clean ensemble
   EER regresses beyond ALERT_THRESHOLDS['clean_ensemble_eer_pp'], or any
   per-source catch rate drops beyond 'per_source_catch_rate_pp', the new
   checkpoint is AUTOMATICALLY DISCARDED and the original is kept. Mirrors
   the honest-failure discipline from the Phase 5 effects-augmentation
   attempt (50% regression, discarded, V8 kept as production).

USAGE
-----
    # New samples as a folder: new_attack/fake/*.wav (+ optional real/*.wav)
    python fewshot_adapt.py --new-samples-dir /kaggle/working/new_attack \
                             --attack-name "elevenlabs_v3"

    # Or as an explicit manifest: [{"path": ..., "label": 0|1}, ...]
    python fewshot_adapt.py --new-samples-manifest new_attack.json \
                             --attack-name "elevenlabs_v3"

    python fewshot_adapt.py ... --dry-run     # train + validate, never save
    python fewshot_adapt.py ... --quick       # smaller replay buffer / eval set

OUTPUTS
-------
    adapted_checkpoints/aasist_v8_adapted_<attack_name>_<utc>.pt   (if passed)
    adaptation_log.jsonl    — one line per attempt, append-only, honest record
                              of passes AND failures
    adaptation_report_<utc>.txt

This file is intended to live next to drift_monitor.py on Kaggle / wherever
V8 is hosted. It reuses the exact AASIST/RawNet3/Wav2Vec architectures and
scoring conventions from drift_monitor.py so checkpoints stay compatible.
"""

import os, sys, json, time, copy, argparse, hashlib, gc
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import pandas as pd
from scipy.io import wavfile
from sklearn.metrics import roc_curve
from transformers import Wav2Vec2Model

# ─── Configuration (top of file for easy operator tuning) ──────────────────

DATASET_ROOT = '/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts'
OUTPUT_DIR   = Path('C:/Users/Michael Ologungbara/Downloads/voice_guard 0ffline')
CKPT_DIR     = OUTPUT_DIR / 'adapted_checkpoints'

AASIST_CKPT  = 'models/aasist_v8.pt'
WAV2VEC_CKPT = 'models/wav2vec_v8.pt'
RAWNET_CKPT  = 'models/rawnet3.pt'
XGB_PATH     = 'models/xgb_v8.json'
CAL_PATH     = 'models/cal_v8_params.json'
VAL_MANIFEST = 'models/val_v8_fresh.json'
TRAIN_MANIFEST = 'models/train_v8_fresh.json'

LOG_FILE = OUTPUT_DIR / 'adaptation_log.jsonl'

# Production thresholds (match deployed server.py / drift_monitor.py)
T_AUTO_FAKE = 0.85
T_LIKELY    = 0.55
T_REVIEW    = 0.30

# Validation gate thresholds — same philosophy as drift_monitor.py.
# An adaptation that regresses beyond these on EXISTING data is rejected,
# regardless of how well it learns the new attack.
GATE_THRESHOLDS = {
    'clean_ensemble_eer_pp':    3.0,   # +3pp ensemble EER on val set → reject
    'per_source_catch_rate_pp': 10.0,  # any existing fake source drops 10pp → reject
    'real_fp_rate_pp':          10.0,  # any existing real source FP rises 10pp → reject
}

# Training hyperparameters
LR              = 2e-6      # deliberately conservative — see Phase 5 Task 2 findings
EPOCHS          = 8
REPLAY_RATIO    = 4         # replay samples per new sample, per batch
DISTILL_TEMP    = 3.0       # softmax temperature for KD
DISTILL_WEIGHT  = 0.6       # weight on KD loss vs CE loss (CE gets 1 - this... see below)
BATCH_SIZE      = 4

N_CLEAN_FULL  = 200
N_CLEAN_QUICK = 60
N_REPLAY_FULL  = 200   # replay buffer pool size to draw from each epoch
N_REPLAY_QUICK = 60

SR = 16000
SAMPLE_LEN = 4 * SR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Model class definitions (identical to drift_monitor.py — must match for
#     checkpoint compatibility) ─────────────────────────────────────────────

class SincConv(nn.Module):
    def __init__(self, out_channels=70, kernel_size=128, sample_rate=16000,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size - (kernel_size % 2 == 0)
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz; self.min_band_hz = min_band_hz
        low_hz = 30; high_hz = sample_rate/2 - (min_low_hz + min_band_hz)
        mel_pts = np.linspace(2595*np.log10(1+low_hz/700), 2595*np.log10(1+high_hz/700), out_channels+1)
        hz_pts = 700*(10**(mel_pts/2595) - 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz_pts[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_pts)).view(-1, 1))
        n_lin = torch.linspace(0, (self.kernel_size/2) - 1, steps=int(self.kernel_size/2))
        self.register_buffer("window_", 0.54 - 0.46*torch.cos(2*np.pi*n_lin/self.kernel_size))
        n = (self.kernel_size - 1)/2.0
        self.register_buffer("n_", 2*np.pi*torch.arange(-n, 0).view(1, -1)/sample_rate)
    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high - low)[:, 0]
        f_low = torch.matmul(low, self.n_); f_high = torch.matmul(high, self.n_)
        bp_left = ((torch.sin(f_high) - torch.sin(f_low))/(self.n_/2))*self.window_
        bp = torch.cat([bp_left, 2*band.view(-1, 1), torch.flip(bp_left, dims=[1])], dim=1)/(2*band[:, None])
        return F.conv1d(x, bp.view(self.out_channels, 1, self.kernel_size), stride=1, padding=self.kernel_size//2)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.selu = nn.SELU()
        self.downsample = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm1d(out_c))
    def forward(self, x):
        identity = self.downsample(x)
        out = self.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.selu(out + identity)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = dropout; self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, x):
        h = self.W(x); N = h.size(1)
        a_input = torch.cat([h.repeat(1, 1, N).view(h.size(0), N * N, -1), h.repeat(1, N, 1)], dim=2).view(h.size(0), N, N, -1)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))
        attention = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return F.elu(torch.matmul(attention, h))

class AASIST(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.sinc = SincConv(out_channels=70, kernel_size=128)
        self.sinc_bn = nn.BatchNorm1d(70); self.sinc_selu = nn.SELU(); self.sinc_pool = nn.MaxPool1d(3)
        layers = []; in_c = 70
        for out_c, stride in zip([32, 32, 64, 64, 128, 128], [1, 2, 1, 2, 1, 2]):
            layers.append(ResBlock(in_c, out_c, stride=stride)); in_c = out_c
        self.res_blocks = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        self.gat1 = GraphAttentionLayer(128, 64)
        self.gat2 = GraphAttentionLayer(64, 64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.SELU(), nn.Dropout(dropout * 0.5), nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.sinc_selu(self.sinc_bn(self.sinc(x))); x = self.sinc_pool(x)
        x = self.res_blocks(x); x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1); x = self.gat1(x); x = self.gat2(x)
        return self.classifier(self.dropout(x.mean(dim=1)))

class SincConvRaw(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 128; kernel_size = 513; sample_rate = 16000
        min_low_hz = 50; min_band_hz = 50
        self.out_channels = out_channels; self.kernel_size = kernel_size
        self.sample_rate = sample_rate; self.min_low_hz = min_low_hz; self.min_band_hz = min_band_hz
        mel = np.linspace(2595*np.log10(1+30/700), 2595*np.log10(1+(sample_rate/2-(min_low_hz+min_band_hz))/700), out_channels+1)
        hz = 700*(10**(mel/2595)-1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, (kernel_size/2)-1, steps=int(kernel_size/2))
        self.register_buffer("window_", 0.54 - 0.46*torch.cos(2*np.pi*n_lin/kernel_size))
        self.register_buffer("n_", 2*np.pi*torch.arange(-(kernel_size-1)/2.0, 0).view(1, -1)/sample_rate)
    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low+self.min_band_hz+torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high-low)[:, 0]
        f_low = torch.matmul(low, self.n_); f_high = torch.matmul(high, self.n_)
        bp_left = ((torch.sin(f_high)-torch.sin(f_low))/(self.n_/2))*self.window_
        bp = torch.cat([bp_left, 2*band.view(-1, 1), torch.flip(bp_left, dims=[1])], dim=1)/(2*band[:, None])
        return F.conv1d(x, bp.view(self.out_channels, 1, self.kernel_size), stride=1, padding=self.kernel_size//2)

class RawResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.selu = nn.SELU()
        self.fms = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(out_c, out_c), nn.Sigmoid())
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm1d(out_c))
    def forward(self, x):
        identity = x
        out = self.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        s = self.fms(out).unsqueeze(-1)
        return self.selu(out * s + identity * (1 - s))

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.Tanh(), nn.Linear(in_dim // 2, 1))
    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        return (x_t * torch.softmax(self.attention(x_t), dim=1)).sum(dim=1)

class RawNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = SincConvRaw(); self.sinc_bn = nn.BatchNorm1d(128); self.sinc_act = nn.SELU(); self.sinc_pool = nn.MaxPool1d(3)
        layers = []; in_ch = 128
        for out_ch, stride in zip([128, 128, 256, 256], [1, 2, 1, 2]):
            layers.append(RawResBlock(in_ch, out_ch, stride=stride)); in_ch = out_ch
        self.res_blocks = nn.Sequential(*layers)
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
        self.attention_pool = AttentionPooling(256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.SELU(), nn.Dropout(0.25), nn.Linear(128, 2))
    def forward(self, x):
        x = self.sinc_act(self.sinc_bn(self.sinc(x))); x = self.sinc_pool(x); x = self.res_blocks(x)
        x = x.permute(0, 2, 1); x, _ = self.gru(x); x = x.permute(0, 2, 1); x = self.attention_pool(x)
        return self.classifier(self.dropout(x))

class Wav2VecHead(nn.Module):
    def __init__(self, backbone_name='facebook/wav2vec2-base', num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes))
    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        feats = self.backbone(x).last_hidden_state
        return self.classifier(feats.mean(dim=1))

# ─── Audio loading (identical convention to drift_monitor.py) ──────────────

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

# ─── V8 loading & scoring ───────────────────────────────────────────────────

def unwrap_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ('model', 'model_state'):
            if key in ckpt: return ckpt[key]
    return ckpt

def load_full_ensemble():
    """Load all V8 components for end-to-end scoring (used in validation gate)."""
    aasist = AASIST().to(DEVICE)
    aasist.load_state_dict(unwrap_state(torch.load(AASIST_CKPT, map_location=DEVICE, weights_only=False)))
    aasist.eval()

    wav2vec = Wav2VecHead().to(DEVICE)
    wav2vec.load_state_dict(unwrap_state(torch.load(WAV2VEC_CKPT, map_location=DEVICE, weights_only=False)))
    wav2vec.eval()

    rawnet = RawNet3().to(DEVICE)
    rawnet.load_state_dict(unwrap_state(torch.load(RAWNET_CKPT, map_location=DEVICE, weights_only=False)))
    rawnet.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_PATH)
    with open(CAL_PATH) as f:
        cal = json.load(f)
    return {
        'aasist': aasist, 'wav2vec': wav2vec, 'rawnet': rawnet,
        'xgb': xgb_model,
        'cal_coef': float(cal['coef']),
        'cal_intercept': float(cal['intercept']),
    }

def calibrate(p_raw, coef, intercept, eps=1e-6):
    p_raw = np.clip(p_raw, eps, 1 - eps)
    logit = np.log(p_raw / (1 - p_raw))
    return 1.0 / (1.0 + np.exp(-(coef * logit + intercept)))

def verdict_from_score(score):
    if score >= T_AUTO_FAKE: return 'auto_fake'
    if score >= T_LIKELY:    return 'likely_fake'
    if score >= T_REVIEW:    return 'to_review'
    return 'auto_real'

def score_one(models, audio_np):
    x_2d = torch.from_numpy(audio_np).float().unsqueeze(0).to(DEVICE)
    x_3d = x_2d.unsqueeze(1)
    with torch.no_grad():
        s_a = F.softmax(models['aasist'](x_3d), dim=1)[0, 1].item()
        s_w = F.softmax(models['wav2vec'](x_2d), dim=1)[0, 1].item()
        s_r = F.softmax(models['rawnet'](x_3d), dim=1)[0, 1].item()
    raw_p = float(models['xgb'].predict_proba(np.array([[s_a, s_w, s_r]]))[0, 1])
    ensemble = float(calibrate(np.array([raw_p]), models['cal_coef'], models['cal_intercept'])[0])
    return ensemble, s_a, s_w, s_r

def compute_eer(labels, scores):
    if len(set(labels)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)

# ─── New-sample / replay-buffer loading ────────────────────────────────────

def load_new_samples(samples_dir=None, manifest_path=None):
    """Load the small new-attack sample set. Returns list of {'path','label'}.

    Supports either:
      - samples_dir/fake/*.{wav,mp3,...} and samples_dir/real/*.{wav,mp3,...}
      - an explicit JSON manifest: [{"path": ..., "label": 0|1}, ...]
    """
    entries = []
    exts = {'.wav', '.mp3', '.flac', '.m4a', '.opus', '.ogg'}

    if manifest_path:
        with open(manifest_path) as f:
            entries = json.load(f)
    elif samples_dir:
        samples_dir = Path(samples_dir)
        for label_name, label in [('fake', 1), ('real', 0)]:
            sub = samples_dir / label_name
            if not sub.is_dir():
                continue
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in exts:
                    entries.append({'path': str(f), 'label': label})
    else:
        raise ValueError("Provide either samples_dir or manifest_path")

    if len(entries) == 0:
        raise ValueError(f"No samples found (dir={samples_dir}, manifest={manifest_path})")
    n_fake = sum(1 for e in entries if e['label'] == 1)
    n_real = sum(1 for e in entries if e['label'] == 0)
    print(f"  Loaded {len(entries)} new samples: {n_fake} fake, {n_real} real", flush=True)
    if n_fake == 0:
        print("  ⚠ WARNING: no fake-labeled new samples — adaptation has nothing to learn.", flush=True)
    if len(entries) < 6:
        print(f"  ⚠ WARNING: only {len(entries)} samples. Few-shot adaptation is designed for "
              f"10-20; fewer than ~6 risks unstable gradients even with a frozen front-end.", flush=True)
    return entries

def build_replay_buffer(train_manifest_path, n_samples, seed=SEED):
    """Resample a balanced subset of the ORIGINAL training data to anchor
    the student against forgetting during adaptation."""
    with open(train_manifest_path) as f:
        train_entries = json.load(f)
    df = pd.DataFrame(train_entries)
    n_per_class = n_samples // 2
    real_sub = df[df['label'] == 0].sample(min(n_per_class, (df['label'] == 0).sum()), random_state=seed)
    fake_sub = df[df['label'] == 1].sample(min(n_per_class, (df['label'] == 1).sum()), random_state=seed)
    sub = pd.concat([real_sub, fake_sub]).reset_index(drop=True)
    print(f"  Replay buffer: {len(sub)} samples drawn from {train_manifest_path}", flush=True)
    return sub.to_dict('records')

# ─── Quick clean-set evaluator (reused for before/after validation gate) ───

def evaluate_clean_quick(models, val_df, n_samples, seed=SEED):
    """Same logic as drift_monitor.evaluate_clean — duplicated here (not
    imported) so this file has no hard dependency on drift_monitor.py being
    present in the same directory."""
    n_per_class = n_samples // 2
    real_sub = val_df[val_df['label'] == 0].sample(min(n_per_class, (val_df['label'] == 0).sum()), random_state=seed)
    fake_sub = val_df[val_df['label'] == 1].sample(min(n_per_class, (val_df['label'] == 1).sum()), random_state=seed)
    sub = pd.concat([real_sub, fake_sub]).reset_index(drop=True)

    scores, labels, sources = [], [], []
    for _, row in sub.iterrows():
        try:
            audio = load_audio(row['path'])
            s_ens, _, _, _ = score_one(models, audio)
            scores.append(s_ens); labels.append(int(row['label'])); sources.append(row['source'])
        except Exception as e:
            print(f"  skipping {row['path']}: {e}", flush=True)

    scores = np.array(scores); labels = np.array(labels); sources = np.array(sources)

    per_source = {}
    for src in sorted(set(sources)):
        mask = sources == src
        src_labels = labels[mask]
        src_verdicts = [verdict_from_score(s) for s in scores[mask]]
        if src_labels[0] == 1:
            caught = sum(1 for v in src_verdicts if v != 'auto_real')
            per_source[src] = {'n': int(mask.sum()), 'catch_rate': caught / mask.sum(), 'kind': 'fake'}
        else:
            fp = sum(1 for v in src_verdicts if v != 'auto_real')
            per_source[src] = {'n': int(mask.sum()), 'false_positive_rate': fp / mask.sum(), 'kind': 'real'}

    return {
        'n_samples':    len(scores),
        'ensemble_eer': compute_eer(labels, scores),
        'per_source':   per_source,
    }

def evaluate_new_samples(models, new_entries):
    """Score the held-out portion of new samples through the FULL ensemble."""
    scores, labels = [], []
    for e in new_entries:
        try:
            audio = load_audio(e['path'])
            s_ens, _, _, _ = score_one(models, audio)
            scores.append(s_ens); labels.append(int(e['label']))
        except Exception as ex:
            print(f"  skipping {e['path']}: {ex}", flush=True)
    scores = np.array(scores); labels = np.array(labels)
    verdicts = [verdict_from_score(s) for s in scores]
    if len(labels) and (labels == 1).any():
        fake_mask = labels == 1
        catch_rate = sum(1 for v, l in zip(verdicts, labels) if l == 1 and v != 'auto_real') / fake_mask.sum()
    else:
        catch_rate = None
    return {'n_samples': len(scores), 'catch_rate': catch_rate, 'scores': scores.tolist(), 'verdicts': verdicts}

# ─── Validation gate ────────────────────────────────────────────────────────

def run_validation_gate(before_clean, after_clean):
    """Compare before/after clean-eval metrics. Returns (passed: bool, reasons: list[str])."""
    reasons = []
    passed = True

    b_eer = before_clean['ensemble_eer']; a_eer = after_clean['ensemble_eer']
    delta_eer_pp = (a_eer - b_eer) * 100
    if delta_eer_pp > GATE_THRESHOLDS['clean_ensemble_eer_pp']:
        passed = False
        reasons.append(f"REJECTED: clean ensemble EER regressed {b_eer:.4f} → {a_eer:.4f} "
                        f"(+{delta_eer_pp:.1f}pp, limit +{GATE_THRESHOLDS['clean_ensemble_eer_pp']}pp)")
    else:
        reasons.append(f"OK: clean ensemble EER {b_eer:.4f} → {a_eer:.4f} ({delta_eer_pp:+.1f}pp)")

    for src, b_data in before_clean['per_source'].items():
        a_data = after_clean['per_source'].get(src)
        if a_data is None:
            continue
        if b_data.get('kind') == 'fake':
            delta_pp = (a_data['catch_rate'] - b_data['catch_rate']) * 100
            if -delta_pp > GATE_THRESHOLDS['per_source_catch_rate_pp']:
                passed = False
                reasons.append(f"REJECTED: {src} catch rate dropped "
                                f"{b_data['catch_rate']*100:.1f}% → {a_data['catch_rate']*100:.1f}% "
                                f"({delta_pp:+.1f}pp, limit -{GATE_THRESHOLDS['per_source_catch_rate_pp']}pp)")
            else:
                reasons.append(f"OK: {src} catch rate {b_data['catch_rate']*100:.1f}% → "
                                f"{a_data['catch_rate']*100:.1f}% ({delta_pp:+.1f}pp)")
        else:
            delta_pp = (a_data['false_positive_rate'] - b_data['false_positive_rate']) * 100
            if delta_pp > GATE_THRESHOLDS['real_fp_rate_pp']:
                passed = False
                reasons.append(f"REJECTED: {src} false-positive rate rose "
                                f"{b_data['false_positive_rate']*100:.1f}% → {a_data['false_positive_rate']*100:.1f}% "
                                f"(+{delta_pp:.1f}pp, limit +{GATE_THRESHOLDS['real_fp_rate_pp']}pp)")
            else:
                reasons.append(f"OK: {src} FP rate {b_data['false_positive_rate']*100:.1f}% → "
                                f"{a_data['false_positive_rate']*100:.1f}% ({delta_pp:+.1f}pp)")

    return passed, reasons

# ─── Training loop ──────────────────────────────────────────────────────────

def freeze_for_adaptation(student):
    """Freeze SincConv + first 4 ResBlocks. Unfreeze last 2 ResBlocks + GAT + classifier."""
    for p in student.parameters():
        p.requires_grad = False
    for p in student.sinc.parameters():
        p.requires_grad = False  # stays frozen regardless

    # res_blocks is an nn.Sequential of 6 ResBlocks: unfreeze indices 4, 5 (last two)
    for idx in (4, 5):
        for p in student.res_blocks[idx].parameters():
            p.requires_grad = True
    for p in student.gat1.parameters():
        p.requires_grad = True
    for p in student.gat2.parameters():
        p.requires_grad = True
    for p in student.classifier.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)", flush=True)

def adapt(new_entries, replay_entries, epochs=EPOCHS, lr=LR,
          distill_temp=DISTILL_TEMP, distill_weight=DISTILL_WEIGHT,
          batch_size=BATCH_SIZE, replay_ratio=REPLAY_RATIO):
    """Fine-tune AASIST student with CE (new samples) + KD (replay buffer vs teacher).
    Returns the adapted student model (still needs validation gate before saving)."""

    # Teacher: frozen original AASIST
    teacher = AASIST().to(DEVICE)
    teacher.load_state_dict(unwrap_state(torch.load(AASIST_CKPT, map_location=DEVICE, weights_only=False)))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student: copy of original, partially unfrozen
    student = AASIST().to(DEVICE)
    student.load_state_dict(unwrap_state(torch.load(AASIST_CKPT, map_location=DEVICE, weights_only=False)))
    freeze_for_adaptation(student)

    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=lr)

    # Pre-load all audio once (small N, fits in memory easily)
    print("  Loading new-sample audio...", flush=True)
    new_audio, new_labels = [], []
    for e in new_entries:
        try:
            new_audio.append(load_audio(e['path']))
            new_labels.append(int(e['label']))
        except Exception as ex:
            print(f"  skipping {e['path']}: {ex}", flush=True)

    print("  Loading replay-buffer audio...", flush=True)
    replay_audio, replay_labels = [], []
    for e in replay_entries:
        try:
            replay_audio.append(load_audio(e['path']))
            replay_labels.append(int(e['label']))
        except Exception as ex:
            print(f"  skipping {e['path']}: {ex}", flush=True)

    new_audio = np.stack(new_audio); new_labels = np.array(new_labels)
    replay_audio = np.stack(replay_audio); replay_labels = np.array(replay_labels)

    n_new = len(new_audio)
    n_replay_per_step = min(len(replay_audio), batch_size * replay_ratio)

    student.train()
    # Keep BatchNorm layers in frozen blocks in eval mode even though .train()
    # was called — only unfrozen blocks should update BN running stats.
    for idx in (0, 1, 2, 3):
        student.res_blocks[idx].eval()

    history = []
    for epoch in range(epochs):
        perm = np.random.permutation(n_new)
        epoch_loss, epoch_ce, epoch_kd = 0.0, 0.0, 0.0
        n_batches = 0

        for start in range(0, n_new, batch_size):
            idx_new = perm[start:start + batch_size]
            x_new = new_audio[idx_new]; y_new = new_labels[idx_new]

            idx_replay = np.random.choice(len(replay_audio), n_replay_per_step, replace=False)
            x_replay = replay_audio[idx_replay]

            x_new_t = torch.from_numpy(x_new).float().unsqueeze(1).to(DEVICE)
            y_new_t = torch.from_numpy(y_new).long().to(DEVICE)
            x_replay_t = torch.from_numpy(x_replay).float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            # CE loss on new samples
            logits_new = student(x_new_t)
            ce_loss = F.cross_entropy(logits_new, y_new_t)

            # KD loss on replay buffer (anchor against teacher, no labels needed)
            student_logits_replay = student(x_replay_t)
            with torch.no_grad():
                teacher_logits_replay = teacher(x_replay_t)
            kd_loss = F.kl_div(
                F.log_softmax(student_logits_replay / distill_temp, dim=1),
                F.softmax(teacher_logits_replay / distill_temp, dim=1),
                reduction='batchmean'
            ) * (distill_temp ** 2)

            loss = (1 - distill_weight) * ce_loss + distill_weight * kd_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item(); epoch_ce += ce_loss.item(); epoch_kd += kd_loss.item()
            n_batches += 1

        history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss / max(n_batches, 1),
            'ce_loss': epoch_ce / max(n_batches, 1),
            'kd_loss': epoch_kd / max(n_batches, 1),
        })
        print(f"  Epoch {epoch+1}/{epochs}: loss={history[-1]['loss']:.4f} "
              f"(CE={history[-1]['ce_loss']:.4f}, KD={history[-1]['kd_loss']:.4f})", flush=True)

    student.eval()
    return student, history

# ─── Report ─────────────────────────────────────────────────────────────────

def format_report(attack_name, new_eval_before, new_eval_after,
                   gate_passed, gate_reasons, history, dry_run):
    lines = []
    lines.append("=" * 72)
    lines.append("V8 FEW-SHOT ADAPTATION REPORT")
    lines.append("=" * 72)
    lines.append(f"Attack name:   {attack_name}")
    lines.append(f"Timestamp:     {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Mode:          {'DRY RUN (no checkpoint will be saved)' if dry_run else 'LIVE'}")
    lines.append("")
    lines.append("─" * 72)
    lines.append("NEW-ATTACK CATCH RATE")
    lines.append("─" * 72)
    bc = new_eval_before.get('catch_rate')
    ac = new_eval_after.get('catch_rate')
    lines.append(f"  Before adaptation:  {f'{bc*100:.1f}%' if bc is not None else 'N/A'} "
                 f"(n={new_eval_before['n_samples']})")
    lines.append(f"  After adaptation:   {f'{ac*100:.1f}%' if ac is not None else 'N/A'} "
                 f"(n={new_eval_after['n_samples']})")
    lines.append("")
    lines.append("─" * 72)
    lines.append("TRAINING HISTORY")
    lines.append("─" * 72)
    for h in history:
        lines.append(f"  epoch {h['epoch']}: loss={h['loss']:.4f} (CE={h['ce_loss']:.4f}, KD={h['kd_loss']:.4f})")
    lines.append("")
    lines.append("─" * 72)
    lines.append("VALIDATION GATE (existing performance, before vs after)")
    lines.append("─" * 72)
    for r in gate_reasons:
        lines.append(f"  {r}")
    lines.append("")
    lines.append("─" * 72)
    lines.append("DECISION")
    lines.append("─" * 72)
    if dry_run:
        lines.append("  DRY RUN — no checkpoint saved regardless of gate result.")
    elif gate_passed:
        lines.append("  ✓ GATE PASSED — adapted checkpoint saved.")
    else:
        lines.append("  ✗ GATE FAILED — adapted checkpoint DISCARDED. Original V8 AASIST retained.")
    lines.append("")
    return "\n".join(lines)

# ─── Main command ───────────────────────────────────────────────────────────

def cmd_adapt(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    n_replay = N_REPLAY_QUICK if args.quick else N_REPLAY_FULL
    n_clean  = N_CLEAN_QUICK if args.quick else N_CLEAN_FULL

    print(f"\nFew-shot adaptation run: attack_name='{args.attack_name}'", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Mode:   {'quick' if args.quick else 'full'}{'  [DRY RUN]' if args.dry_run else ''}\n", flush=True)

    print("Loading new attack samples...", flush=True)
    new_entries = load_new_samples(args.new_samples_dir, args.new_samples_manifest)

    print("\nBuilding replay buffer from original training manifest...", flush=True)
    replay_entries = build_replay_buffer(TRAIN_MANIFEST, n_replay)

    print("\nLoading val manifest for validation gate...", flush=True)
    with open(VAL_MANIFEST) as f:
        val_df = pd.DataFrame(json.load(f))

    # ─── BEFORE: load full ensemble, measure baseline on new samples + clean val ───
    print("\nLoading current production ensemble for BEFORE measurement...", flush=True)
    models_before = load_full_ensemble()
    print("  Evaluating new samples (before)...", flush=True)
    new_eval_before = evaluate_new_samples(models_before, new_entries)
    print(f"    Catch rate before: {new_eval_before.get('catch_rate')}", flush=True)
    print("  Evaluating clean val set (before)...", flush=True)
    t0 = time.time()
    clean_before = evaluate_clean_quick(models_before, val_df, n_clean)
    print(f"    Done in {time.time()-t0:.0f}s. Ensemble EER: {clean_before['ensemble_eer']:.4f}", flush=True)
    del models_before; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ─── ADAPT: fine-tune AASIST student ───
    print("\nAdapting AASIST (frozen front-end, CE + KD loss)...", flush=True)
    student, history = adapt(new_entries, replay_entries,
                              epochs=args.epochs, lr=args.lr)

    # ─── AFTER: swap in adapted AASIST, re-measure everything ───
    print("\nLoading ensemble with ADAPTED AASIST for AFTER measurement...", flush=True)
    models_after = load_full_ensemble()
    models_after['aasist'] = student  # swap in adapted student, keep wav2vec/rawnet/xgb/cal
    print("  Evaluating new samples (after)...", flush=True)
    new_eval_after = evaluate_new_samples(models_after, new_entries)
    print(f"    Catch rate after: {new_eval_after.get('catch_rate')}", flush=True)
    print("  Evaluating clean val set (after)...", flush=True)
    t0 = time.time()
    clean_after = evaluate_clean_quick(models_after, val_df, n_clean)
    print(f"    Done in {time.time()-t0:.0f}s. Ensemble EER: {clean_after['ensemble_eer']:.4f}", flush=True)

    # ─── GATE ───
    gate_passed, gate_reasons = run_validation_gate(clean_before, clean_after)

    report = format_report(args.attack_name, new_eval_before, new_eval_after,
                            gate_passed, gate_reasons, history, args.dry_run)
    print("\n" + report)

    safe_ts = datetime.now(timezone.utc).isoformat().replace(':', '').replace('-', '').split('.')[0]
    report_path = OUTPUT_DIR / f'adaptation_report_{args.attack_name}_{safe_ts}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    ckpt_path = None
    if not args.dry_run and gate_passed:
        ckpt_path = CKPT_DIR / f'aasist_v8_adapted_{args.attack_name}_{safe_ts}.pt'
        torch.save({'model_state': student.state_dict()}, ckpt_path)
        print(f"✓ Adapted checkpoint saved: {ckpt_path}")
        print(f"  To deploy: update AASIST_CKPT in drift_monitor.py / server.py to point here,")
        print(f"  then re-run drift_monitor.py to confirm before flipping production traffic.")
    elif args.dry_run:
        print("  Dry run — checkpoint not saved regardless of gate outcome.")
    else:
        print("✗ Gate failed — checkpoint discarded. Original V8 AASIST remains production model.")

    log_entry = {
        'timestamp':        datetime.now(timezone.utc).isoformat(),
        'attack_name':      args.attack_name,
        'n_new_samples':    len(new_entries),
        'dry_run':          args.dry_run,
        'gate_passed':      gate_passed,
        'catch_rate_before': new_eval_before.get('catch_rate'),
        'catch_rate_after':  new_eval_after.get('catch_rate'),
        'clean_eer_before':  clean_before['ensemble_eer'],
        'clean_eer_after':   clean_after['ensemble_eer'],
        'checkpoint_saved':  str(ckpt_path) if ckpt_path else None,
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
        f.write('\n')
    print(f"Logged to {LOG_FILE}")

# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V8 few-shot adaptation pipeline (Phase 5 Task 6)')
    parser.add_argument('--new-samples-dir', type=str, default=None,
                        help='Folder with fake/ and (optionally) real/ subfolders of new attack samples.')
    parser.add_argument('--new-samples-manifest', type=str, default=None,
                        help='JSON manifest [{"path":..., "label":0|1}, ...] as an alternative to --new-samples-dir.')
    parser.add_argument('--attack-name', type=str, required=True,
                        help='Short identifier for this attack type, used in filenames (e.g. "elevenlabs_v3").')
    parser.add_argument('--quick', action='store_true', help='Smaller replay buffer / eval set, faster run.')
    parser.add_argument('--dry-run', action='store_true', help='Train + validate, never save a checkpoint.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    args = parser.parse_args()

    if not args.new_samples_dir and not args.new_samples_manifest:
        parser.error('Provide --new-samples-dir or --new-samples-manifest')

    cmd_adapt(args)

if __name__ == '__main__':
    main()
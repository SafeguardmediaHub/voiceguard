"""
refit_ensemble_v9.py — Re-fit XGBoost fusion + logistic calibration after AASIST retrain.

Pipeline:
  1. Load all 3 models (AASIST V9, Wav2Vec2 V8, RawNet3 V8)
  2. Extract P(fake) scores from each model on train + val + held-out sets
  3. Re-fit XGBoost on training scores → fused score
  4. Re-fit logistic calibration on val scores → calibrated probability
  5. Evaluate full ensemble on val + held-out with per-source breakdown
  6. Save xgb_v9.json, cal_v9_params.json, thresholds_v9.json

Usage (Kaggle):
    !python refit_ensemble_v9.py --dry-run     # quick sanity check
    !python refit_ensemble_v9.py               # full run
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchaudio
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "torchaudio", "--quiet"])
    import torchaudio

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "xgboost", "--quiet"])
    import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

try:
    from transformers import Wav2Vec2Model
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "transformers", "--quiet"])
    from transformers import Wav2Vec2Model


# ===================================================================
# PATHS (Kaggle defaults)
# ===================================================================
ARTEFACTS = "/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts"
OUTPUT_DIR = "/kaggle/working"

PATHS = {
    "aasist_ckpt":    os.path.join(OUTPUT_DIR, "aasist_v9_best.pt"),
    "wav2vec_ckpt":   os.path.join(ARTEFACTS, "wav2vec_v8.pt"),
    "rawnet_ckpt":    os.path.join(ARTEFACTS, "rawnet3.pt"),
    "train_manifest": os.path.join(OUTPUT_DIR, "train_v9.json"),
    "val_manifest":   os.path.join(ARTEFACTS, "val_v8_fresh.json"),
    "eval_manifest":  os.path.join(OUTPUT_DIR, "eval_v9_heldout.json"),
}

SAMPLE_RATE = 16000
MAX_SECONDS = 4.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECONDS)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# V8 verdict thresholds (starting point for V9)
V8_THRESHOLDS = {
    "auto_fake": 0.85,
    "likely_fake": 0.55,
    "to_review": 0.30,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ===================================================================
# MODEL DEFINITIONS
# ===================================================================

# ---- SincConv (shared by AASIST and RawNet3) ----
class SincConv(nn.Module):
    """Learnable sinc-function bandpass filters."""

    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 min_low_hz=50.0, min_band_hz=50.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = min_low_hz
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel_low = 2595.0 * math.log10(1.0 + low_hz / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + high_hz / 700.0)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).unsqueeze(1))

        self.register_buffer(
            "hamming",
            0.54 - 0.46 * torch.cos(
                2 * math.pi * torch.arange(0, kernel_size).float() / kernel_size))
        self.register_buffer(
            "n_",
            2 * math.pi * torch.arange(
                -(kernel_size - 1) / 2.0,
                (kernel_size - 1) / 2.0 + 1).float() / sample_rate)

    def sinc(self, x):
        x_safe = torch.where(x == 0, torch.ones_like(x), x)
        return torch.where(x == 0, torch.ones_like(x), torch.sin(x_safe) / x_safe)

    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            min=self.min_low_hz, max=self.sample_rate / 2.0)
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate
        bp_low = 2.0 * f_low * self.sinc(self.n_ * f_low * self.sample_rate)
        bp_high = 2.0 * f_high * self.sinc(self.n_ * f_high * self.sample_rate)
        filters = (bp_high - bp_low) * self.hamming
        filters = filters / (filters.abs().sum(dim=1, keepdim=True) + 1e-8)
        return F.conv1d(x, filters.unsqueeze(1), padding=self.kernel_size // 2)


# ---- AASIST V8/V9 ----
class ResBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.selu = nn.SELU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch))

    def forward(self, x):
        identity = self.downsample(x)
        out = self.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.selu(out + identity)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.W(x)
        h_i = h.unsqueeze(2).expand(-1, -1, T, -1)
        h_j = h.unsqueeze(1).expand(-1, T, -1, -1)
        e = self.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        alpha = F.softmax(e, dim=-1)
        return F.elu(torch.bmm(alpha, h))


class AASIST_V8(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = SincConv(70, 127, SAMPLE_RATE)
        self.bn0 = nn.BatchNorm1d(70)
        self.selu = nn.SELU(inplace=True)
        self.pool0 = nn.MaxPool1d(3)

        channels = [32, 32, 64, 64, 128, 128]
        strides = [1, 2, 1, 2, 1, 2]
        in_ch = 70
        self.res_blocks = nn.ModuleList()
        for out_ch, s in zip(channels, strides):
            self.res_blocks.append(ResBlock1d(in_ch, out_ch, s))
            in_ch = out_ch

        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        self.gat1 = GraphAttentionLayer(128, 64)
        self.gat2 = GraphAttentionLayer(64, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64), nn.SELU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        x = self.pool0(self.selu(self.bn0(self.sinc(x))))
        for block in self.res_blocks:
            x = block(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        x = self.gat1(x)
        x = self.gat2(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ---- Wav2Vec2 Classifier (V8) ----
class Wav2VecClassifier(nn.Module):
    """Wav2Vec2-base frozen + trainable classifier head. Takes 2D input (B, T)."""

    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for p in self.wav2vec.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 2))

    def forward(self, x):
        # x: (B, T) raw waveform, 2D
        outputs = self.wav2vec(x)
        hidden = outputs.last_hidden_state  # (B, T', 768)
        pooled = hidden.mean(dim=1)         # (B, 768)
        return self.classifier(pooled)


# ---- RawNet3 (V8) — exact copy from training notebook ----

class SincConvRaw(nn.Module):
    """Exact copy from Cell 60 of training notebook."""
    def __init__(self, out_channels=128, kernel_size=512,
                 sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size + (kernel_size % 2 == 0)  # force odd: 512->513
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30.0
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = np.linspace(
            self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 1)
        hz = self._mel2hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1,
            steps=int(self.kernel_size / 2))
        self.register_buffer(
            "window_",
            0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / self.kernel_size))

        n = (self.kernel_size - 1) / 2.0
        self.register_buffer(
            "n_",
            2 * np.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    @staticmethod
    def _hz2mel(hz): return 2595 * np.log10(1 + hz / 700)
    @staticmethod
    def _mel2hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_low = torch.matmul(low, self.n_)
        f_high = torch.matmul(high, self.n_)

        bp_left = ((torch.sin(f_high) - torch.sin(f_low))
                   / (self.n_ / 2)) * self.window_
        bp_center = 2 * band.view(-1, 1)
        bp_right = torch.flip(bp_left, dims=[1])

        bp = torch.cat([bp_left, bp_center, bp_right], dim=1)
        bp = bp / (2 * band[:, None])

        return F.conv1d(
            x, bp.view(self.out_channels, 1, self.kernel_size),
            stride=1, padding=self.kernel_size // 2)


class RawResBlock(nn.Module):
    """Exact copy — FMS as Sequential matching checkpoint keys."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.selu = nn.SELU()

        # FMS as Sequential: checkpoint keys fms.2.weight, fms.2.bias
        self.fms = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),      # 0
            nn.Flatten(start_dim=1),      # 1
            nn.Linear(out_channels, out_channels),  # 2
            nn.Sigmoid(),                 # 3
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = x
        out = self.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out = self.selu(out + identity)
        scale = self.fms(out)
        out = out * scale.unsqueeze(2)
        return out


class AttentionPooling(nn.Module):
    """Takes (B, C, T). Matches checkpoint keys."""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),   # 0
            nn.Tanh(),             # 1
            nn.Linear(128, 1),     # 2
        )

    def forward(self, x):
        # x: (B, C, T)
        x = x.permute(0, 2, 1)          # (B, T, C)
        w = self.attention(x)            # (B, T, 1)
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)        # (B, C)


class RawNet3(nn.Module):
    """Exact match to Cell 61 of training notebook."""
    def __init__(self):
        super().__init__()
        sinc_channels = 128
        sinc_kernel = 512
        res_channels = [128, 128, 256, 256]
        gru_hidden = 256
        gru_layers = 2
        dropout = 0.5

        self.sinc = SincConvRaw(sinc_channels, sinc_kernel, SAMPLE_RATE)
        self.sinc_bn = nn.BatchNorm1d(sinc_channels)
        self.sinc_act = nn.SELU()
        self.sinc_pool = nn.MaxPool1d(3)

        layers = []
        in_ch = sinc_channels
        strides = [1, 2, 1, 2]
        for out_ch, stride in zip(res_channels, strides):
            layers.append(RawResBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.res_blocks = nn.Sequential(*layers)

        self.gru = nn.GRU(
            input_size=res_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False)

        self.attention_pool = AttentionPooling(gru_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.SELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2))

    def forward(self, x):
        x = self.sinc(x)
        x = self.sinc_act(self.sinc_bn(x))
        x = self.sinc_pool(x)
        x = self.res_blocks(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)
        x = self.attention_pool(x)
        x = self.dropout(x)
        return self.classifier(x)



# ===================================================================
# DATASET
# ===================================================================
class AudioManifestDataset(Dataset):
    def __init__(self, manifest_path, max_samples=MAX_SAMPLES):
        with open(manifest_path) as f:
            self.entries = json.load(f)
        self.max_samples = max_samples

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        path = entry["path"]
        label = int(entry["label"])
        source = entry.get("source", "unknown")

        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            wav = torch.zeros(1, self.max_samples)
            sr = SAMPLE_RATE

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
        if wav.shape[1] > self.max_samples:
            wav = wav[:, :self.max_samples]
        elif wav.shape[1] < self.max_samples:
            wav = F.pad(wav, (0, self.max_samples - wav.shape[1]))

        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak

        return wav, label, source


# ===================================================================
# SCORE EXTRACTION
# ===================================================================
@torch.no_grad()
def extract_scores(model, loader, device, input_mode="3d"):
    """
    Run model on all samples, return P(fake) scores, labels, sources.
    input_mode: "3d" for AASIST/RawNet3 (B,1,T), "2d" for Wav2Vec2 (B,T)
    """
    model.eval()
    all_scores, all_labels, all_sources = [], [], []

    for wav, label, source in loader:
        wav = wav.to(device)
        if input_mode == "2d":
            wav = wav.squeeze(1)  # (B, 1, T) → (B, T)
        logits = model(wav)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(fake)
        all_scores.extend(probs.cpu().numpy().tolist())
        all_labels.extend(label.numpy().tolist())
        all_sources.extend(source)

    return np.array(all_scores), np.array(all_labels), all_sources


# ===================================================================
# METRICS
# ===================================================================
def compute_eer(scores, labels):
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    if np.all(np.isnan(diff)):
        return float("nan"), float("nan")
    idx = np.nanargmin(diff)
    return (fpr[idx] + fnr[idx]) / 2.0, thresholds[idx]


def compute_auc(scores, labels):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.0


def evaluate_ensemble(fused_scores, labels, sources, thresholds, title="Eval"):
    """Full evaluation with per-source breakdown and verdict distribution."""
    eer, eer_thresh = compute_eer(fused_scores, labels)
    auc = compute_auc(fused_scores, labels)

    eer_str = f"{eer*100:.2f}%" if not math.isnan(eer) else "N/A"
    print(f"\n  === {title} ===")
    print(f"  EER: {eer_str}  |  AUC: {auc:.4f}")

    # Per-source breakdown at EER threshold
    source_arr = np.array(sources)
    print(f"\n  {'Source':<30} {'N':>5}  {'Metric':>12}")
    print(f"  {'-'*30} {'-'*5}  {'-'*12}")
    for src in sorted(set(sources)):
        mask = source_arr == src
        s = fused_scores[mask]
        l = labels[mask]
        n = mask.sum()
        if len(set(l)) < 2:
            if l[0] == 1:
                if not math.isnan(eer_thresh):
                    cr = (s >= eer_thresh).mean()
                    print(f"  {src:<30} {n:>5}  catch {cr*100:.1f}%")
                else:
                    print(f"  {src:<30} {n:>5}  {'N/A':>12}")
            else:
                if not math.isnan(eer_thresh):
                    fp = (s >= eer_thresh).mean()
                    print(f"  {src:<30} {n:>5}  FP {fp*100:.1f}%")
                else:
                    print(f"  {src:<30} {n:>5}  {'N/A':>12}")
        else:
            src_eer, _ = compute_eer(s, l)
            print(f"  {src:<30} {n:>5}  EER {src_eer*100:.1f}%")

    # Verdict distribution using deployed thresholds
    print(f"\n  Verdict distribution (thresholds: "
          f"auto_fake≥{thresholds['auto_fake']}, "
          f"likely_fake≥{thresholds['likely_fake']}, "
          f"to_review≥{thresholds['to_review']}):")

    for cls_name, cls_val in [("FAKE", 1), ("REAL", 0)]:
        mask = labels == cls_val
        if mask.sum() == 0:
            continue
        s = fused_scores[mask]
        n = mask.sum()
        auto_fake = (s >= thresholds["auto_fake"]).sum()
        likely_fake = ((s >= thresholds["likely_fake"]) & (s < thresholds["auto_fake"])).sum()
        to_review = ((s >= thresholds["to_review"]) & (s < thresholds["likely_fake"])).sum()
        auto_real = (s < thresholds["to_review"]).sum()
        print(f"  {cls_name} (n={n}): auto_fake={auto_fake} ({auto_fake/n*100:.1f}%), "
              f"likely_fake={likely_fake} ({likely_fake/n*100:.1f}%), "
              f"to_review={to_review} ({to_review/n*100:.1f}%), "
              f"auto_real={auto_real} ({auto_real/n*100:.1f}%)")

    return {"eer": eer, "auc": auc, "eer_threshold": eer_thresh}


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Re-fit XGBoost + calibration")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for score extraction (lower = less VRAM)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--aasist-ckpt", type=str, default=PATHS["aasist_ckpt"])
    parser.add_argument("--wav2vec-ckpt", type=str, default=PATHS["wav2vec_ckpt"])
    parser.add_argument("--rawnet-ckpt", type=str, default=PATHS["rawnet_ckpt"])
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Disable cuDNN for GRU backward (RawNet3 requirement)
    torch.backends.cudnn.enabled = False

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    print("\n[1/6] Loading datasets...")
    train_ds = AudioManifestDataset(PATHS["train_manifest"])
    val_ds = AudioManifestDataset(PATHS["val_manifest"])
    eval_ds = None
    if os.path.isfile(PATHS["eval_manifest"]):
        eval_ds = AudioManifestDataset(PATHS["eval_manifest"])

    if args.dry_run:
        train_ds.entries = train_ds.entries[:64]
        val_ds.entries = val_ds.entries[:64]
        if eval_ds:
            reals = [e for e in eval_ds.entries if int(e["label"]) == 0]
            fakes = [e for e in eval_ds.entries if int(e["label"]) == 1]
            eval_ds.entries = reals[:16] + fakes[:16]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    eval_loader = None
    if eval_ds:
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} "
          f"| Held-out: {len(eval_ds) if eval_ds else 0}")

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    print("\n[2/6] Loading models...")

    # AASIST V9
    aasist = AASIST_V8().to(device)
    ckpt = torch.load(args.aasist_ckpt, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        aasist.load_state_dict(ckpt["model_state_dict"])
    else:
        aasist.load_state_dict(ckpt)
    aasist.eval()
    print(f"  AASIST V9: loaded from {args.aasist_ckpt}")

    # Wav2Vec2 V8
    wav2vec = Wav2VecClassifier().to(device)
    w2v_ckpt = torch.load(args.wav2vec_ckpt, map_location=device, weights_only=False)
    # Checkpoint wrapper key is "model" (contains both wav2vec + classifier)
    if "model" in w2v_ckpt and isinstance(w2v_ckpt["model"], dict):
        w2v_state = w2v_ckpt["model"]
    elif "model_state_dict" in w2v_ckpt:
        w2v_state = w2v_ckpt["model_state_dict"]
    elif "model_state" in w2v_ckpt:
        w2v_state = w2v_ckpt["model_state"]
    else:
        w2v_state = w2v_ckpt

    # Debug: show what keys the checkpoint actually has
    ckpt_cls_keys = [k for k in w2v_state.keys() if "classifier" in k]
    print(f"  W2V checkpoint classifier keys: {ckpt_cls_keys[:10]}")

    # Try full load first; if keys don't match, try classifier-only
    try:
        wav2vec.load_state_dict(w2v_state, strict=True)
        print(f"  Wav2Vec2 V8: loaded full state from {args.wav2vec_ckpt}")
    except RuntimeError as e:
        print(f"  W2V strict load failed, trying alternatives...")
        # Maybe only classifier weights were saved, or key prefix differs
        cls_state = {k.replace("classifier.", ""): v for k, v in w2v_state.items()
                     if k.startswith("classifier.")}
        if cls_state:
            wav2vec.classifier.load_state_dict(cls_state)
            print(f"  Wav2Vec2 V8: loaded classifier-only from {args.wav2vec_ckpt}")
        else:
            # Try with strict=False and report what matched
            result = wav2vec.load_state_dict(w2v_state, strict=False)
            print(f"  Wav2Vec2 V8: loaded partial — "
                  f"missing: {len(result.missing_keys)}, "
                  f"unexpected: {len(result.unexpected_keys)}")
            if result.unexpected_keys[:5]:
                print(f"    Sample unexpected keys: {result.unexpected_keys[:5]}")
    wav2vec.eval()

    # RawNet3 V8
    rawnet = RawNet3().to(device)
    rn_ckpt = torch.load(args.rawnet_ckpt, map_location=device, weights_only=False)
    if "model_state" in rn_ckpt:
        rn_state = rn_ckpt["model_state"]
    elif "model_state_dict" in rn_ckpt:
        rn_state = rn_ckpt["model_state_dict"]
    else:
        rn_state = rn_ckpt

    try:
        rawnet.load_state_dict(rn_state, strict=True)
        print(f"  RawNet3 V8: loaded from {args.rawnet_ckpt}")
    except RuntimeError as e:
        print(f"  ⚠ RawNet3 strict load failed: {e}")
        rawnet.load_state_dict(rn_state, strict=False)
        print(f"  RawNet3 V8: loaded with strict=False from {args.rawnet_ckpt}")
    rawnet.eval()

    # ------------------------------------------------------------------
    # 3. Extract scores from all 3 models
    # ------------------------------------------------------------------
    print("\n[3/6] Extracting scores (this may take a while)...")

    datasets_info = [
        ("train", train_loader, train_ds),
        ("val", val_loader, val_ds),
    ]
    if eval_loader:
        datasets_info.append(("eval", eval_loader, eval_ds))

    all_data = {}
    for name, loader, ds in datasets_info:
        t0 = time.time()
        print(f"\n  --- {name} ({len(ds)} samples) ---")

        print(f"    Extracting AASIST scores...", end=" ", flush=True)
        aasist_scores, labels, sources = extract_scores(
            aasist, loader, device, input_mode="3d")
        print(f"done ({time.time()-t0:.0f}s)")

        t1 = time.time()
        print(f"    Extracting Wav2Vec2 scores...", end=" ", flush=True)
        w2v_scores, _, _ = extract_scores(
            wav2vec, loader, device, input_mode="2d")
        print(f"done ({time.time()-t1:.0f}s)")

        t2 = time.time()
        print(f"    Extracting RawNet3 scores...", end=" ", flush=True)
        rn_scores, _, _ = extract_scores(
            rawnet, loader, device, input_mode="3d")
        print(f"done ({time.time()-t2:.0f}s)")

        # Stack into feature matrix
        features = np.column_stack([aasist_scores, w2v_scores, rn_scores])
        all_data[name] = {
            "features": features,
            "labels": labels,
            "sources": sources,
            "aasist_scores": aasist_scores,
            "w2v_scores": w2v_scores,
            "rn_scores": rn_scores,
        }
        print(f"    Feature matrix: {features.shape}")
        print(f"    Score ranges — AASIST: [{aasist_scores.min():.3f}, {aasist_scores.max():.3f}]  "
              f"W2V: [{w2v_scores.min():.3f}, {w2v_scores.max():.3f}]  "
              f"RN3: [{rn_scores.min():.3f}, {rn_scores.max():.3f}]")

    # Free GPU memory
    del aasist, wav2vec, rawnet
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Fit XGBoost
    # ------------------------------------------------------------------
    print("\n[4/6] Fitting XGBoost...")

    train_X = all_data["train"]["features"]
    train_y = all_data["train"]["labels"]
    val_X = all_data["val"]["features"]
    val_y = all_data["val"]["labels"]

    dtrain = xgb.DMatrix(train_X, label=train_y,
                         feature_names=["aasist", "wav2vec2", "rawnet3"])
    dval = xgb.DMatrix(val_X, label=val_y,
                       feature_names=["aasist", "wav2vec2", "rawnet3"])

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "min_child_weight": 5,
        "seed": 42,
        "verbosity": 1,
    }

    print(f"  XGBoost params: {xgb_params}")
    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=10,
    )

    print(f"\n  Best iteration: {bst.best_iteration}")
    print(f"  Best val score: {bst.best_score:.4f}")

    # Feature importance
    importance = bst.get_score(importance_type="weight")
    total_imp = sum(importance.values()) or 1
    print(f"\n  Feature importance (weight):")
    for feat in ["aasist", "wav2vec2", "rawnet3"]:
        w = importance.get(feat, 0) / total_imp
        print(f"    {feat:<12} {w:.3f}")

    # Save XGBoost model
    xgb_path = os.path.join(args.output_dir, "xgb_v9.json")
    bst.save_model(xgb_path)
    print(f"\n  Saved XGBoost model: {xgb_path}")

    # Raw XGBoost scores on val
    xgb_val_scores = bst.predict(dval)

    # ------------------------------------------------------------------
    # 5. Fit logistic calibration
    # ------------------------------------------------------------------
    print("\n[5/6] Fitting logistic calibration...")

    # Calibrate XGBoost output → well-calibrated probability
    cal_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal_model.fit(xgb_val_scores.reshape(-1, 1), val_y)

    cal_a = float(cal_model.coef_[0][0])
    cal_b = float(cal_model.intercept_[0])
    print(f"  Calibration: P = sigmoid({cal_a:.4f} * xgb_score + {cal_b:.4f})")

    cal_params = {"coef": cal_a, "intercept": cal_b}
    cal_path = os.path.join(args.output_dir, "cal_v9_params.json")
    with open(cal_path, "w") as f:
        json.dump(cal_params, f, indent=2)
    print(f"  Saved calibration params: {cal_path}")

    # Calibrated val scores
    def calibrate(raw_scores):
        return 1.0 / (1.0 + np.exp(-(cal_a * raw_scores + cal_b)))

    cal_val_scores = calibrate(xgb_val_scores)

    # ------------------------------------------------------------------
    # 6. Evaluate full ensemble
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating full ensemble...")

    # Try existing V8 thresholds first
    thresholds = V8_THRESHOLDS.copy()

    # Val set
    val_results = evaluate_ensemble(
        cal_val_scores, val_y, all_data["val"]["sources"],
        thresholds, "Val Set (val_v8_fresh.json) — Full Ensemble")

    # Held-out set
    if "eval" in all_data:
        eval_X = all_data["eval"]["features"]
        deval = xgb.DMatrix(eval_X, feature_names=["aasist", "wav2vec2", "rawnet3"])
        xgb_eval_scores = bst.predict(deval)
        cal_eval_scores = calibrate(xgb_eval_scores)

        eval_results = evaluate_ensemble(
            cal_eval_scores, all_data["eval"]["labels"],
            all_data["eval"]["sources"],
            thresholds, "Held-Out (studio + noiz.ai) — Full Ensemble")

    # Save thresholds (using V8 as starting point)
    thresh_path = os.path.join(args.output_dir, "thresholds_v9.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n  Saved thresholds: {thresh_path}")

    # Save raw scores for analysis
    scores_path = os.path.join(args.output_dir, "ensemble_scores_v9.json")
    scores_export = {
        "val": {
            "aasist": all_data["val"]["aasist_scores"].tolist(),
            "wav2vec2": all_data["val"]["w2v_scores"].tolist(),
            "rawnet3": all_data["val"]["rn_scores"].tolist(),
            "xgb_raw": xgb_val_scores.tolist(),
            "calibrated": cal_val_scores.tolist(),
            "labels": all_data["val"]["labels"].tolist(),
            "sources": all_data["val"]["sources"],
        },
    }
    if "eval" in all_data:
        scores_export["eval"] = {
            "aasist": all_data["eval"]["aasist_scores"].tolist(),
            "wav2vec2": all_data["eval"]["w2v_scores"].tolist(),
            "rawnet3": all_data["eval"]["rn_scores"].tolist(),
            "xgb_raw": xgb_eval_scores.tolist(),
            "calibrated": cal_eval_scores.tolist(),
            "labels": all_data["eval"]["labels"].tolist(),
            "sources": all_data["eval"]["sources"],
        }
    with open(scores_path, "w") as f:
        json.dump(scores_export, f)
    print(f"  Saved raw scores: {scores_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ENSEMBLE RE-FIT COMPLETE")
    print("=" * 65)
    print(f"  XGBoost:     {xgb_path}")
    print(f"  Calibration: {cal_path}")
    print(f"  Thresholds:  {thresh_path}")
    print(f"  Raw scores:  {scores_path}")
    print()
    print(f"  Val EER:  {val_results['eer']*100:.2f}%  (V8 baseline: 7.41%)")
    print(f"  Val AUC:  {val_results['auc']:.4f}  (V8 baseline: 0.979)")
    if "eval" in all_data:
        print(f"\n  Held-out EER: {eval_results['eer']*100:.2f}%")
    print()
    print("  Compare against V8 baselines above. If val metrics are")
    print("  comparable or better AND held-out studio FP / noiz.ai catch")
    print("  improved, the retrain is a success.")
    print("=" * 65)


if __name__ == "__main__":
    main()

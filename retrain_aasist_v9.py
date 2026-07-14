"""
retrain_aasist_v9.py — Full AASIST retrain from scratch on expanded V9 manifest.

Architecture matches V8 exactly:
  SincConv(70, kernel=127) → BN → SELU → MaxPool(3)
  → 6 ResBlocks [32,32,64,64,128,128] strides [1,2,1,2,1,2] (all with downsample)
  → AdaptiveAvgPool1d(64)
  → GAT(128→64) → GAT(64→64)
  → Linear(64→64) → SELU → Dropout → Linear(64→2)

Usage (Kaggle):
    !python retrain_aasist_v9.py                          # full run
    !python retrain_aasist_v9.py --epochs 5 --dry-run     # quick sanity check
    !python retrain_aasist_v9.py --resume /path/to/ckpt   # resume from checkpoint
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchaudio
    AUDIO_BACKEND = "torchaudio"
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "torchaudio", "--quiet"])
    import torchaudio
    AUDIO_BACKEND = "torchaudio"

# ===================================================================
# CONFIG
# ===================================================================
DEFAULT_CFG = {
    # Paths (Kaggle)
    "train_manifest": "/kaggle/working/train_v9.json",
    "val_manifest":   "/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts/val_v8_fresh.json",
    "eval_manifest":  "/kaggle/working/eval_v9_heldout.json",
    "output_dir":     "/kaggle/working",

    # Audio
    "sample_rate": 16000,
    "max_seconds": 4.0,        # 4s clips → 64,000 samples

    # AASIST architecture (V8-exact — do not change)
    "sinc_channels": 70,
    "sinc_kernel":   127,
    "res_channels":  [32, 32, 64, 64, 128, 128],
    "res_strides":   [1,  2,  1,  2,  1,   2],
    "pool_output":   64,       # AdaptiveAvgPool1d output
    "gat_dims":      [128, 64, 64],  # input, hidden, output
    "gat_heads":     1,
    "cls_hidden":    64,
    "dropout":       0.3,

    # Training
    "epochs":        100,
    "batch_size":    32,
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "lr_scheduler":  "cosine",    # "cosine" or "step"
    "lr_step_size":  30,
    "lr_gamma":      0.5,
    "warmup_epochs": 5,
    "patience":      15,         # early stopping on val EER
    "seed":          42,
    "num_workers":   2,
    "label_smoothing": 0.0,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ===================================================================
# SINC CONVOLUTION
# ===================================================================
class SincConv(nn.Module):
    """Learnable sinc-function bandpass filters (SincNet-style)."""

    def __init__(self, out_channels: int, kernel_size: int,
                 sample_rate: int = 16000, min_low_hz: float = 50.0,
                 min_band_hz: float = 50.0):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize filters: mel-spaced bandpass
        low_hz = min_low_hz
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)

        # Mel scale init
        mel_low = 2595.0 * math.log10(1.0 + low_hz / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + high_hz / 700.0)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))  # (C, 1)
        self.band_hz_ = nn.Parameter(
            (hz_points[1:] - hz_points[:-1]).unsqueeze(1))         # (C, 1)

        # Hamming window (fixed)
        n = (kernel_size - 1) / 2.0
        self.register_buffer(
            "hamming",
            0.54 - 0.46 * torch.cos(
                2.0 * math.pi * torch.arange(0, kernel_size).float()
                / kernel_size)
        )
        self.register_buffer(
            "n_", 2.0 * math.pi * torch.arange(
                -(kernel_size - 1) / 2.0,
                (kernel_size - 1) / 2.0 + 1
            ).float() / sample_rate
        )

    def sinc(self, x):
        """Normalized sinc: sin(x) / x, with sinc(0) = 1."""
        x_safe = torch.where(x == 0, torch.ones_like(x), x)
        return torch.where(x == 0, torch.ones_like(x),
                           torch.sin(x_safe) / x_safe)

    def forward(self, x):
        """x: (B, 1, T) → (B, out_channels, T')"""
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            min=self.min_low_hz,
            max=self.sample_rate / 2.0
        )

        # Compute bandpass filters: sinc(high) - sinc(low)
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate

        # (C, kernel_size)
        band_pass_low = 2.0 * f_low * self.sinc(
            self.n_ * f_low * self.sample_rate)
        band_pass_high = 2.0 * f_high * self.sinc(
            self.n_ * f_high * self.sample_rate)

        filters = (band_pass_high - band_pass_low) * self.hamming
        # Normalize
        filters = filters / (filters.abs().sum(dim=1, keepdim=True) + 1e-8)

        return F.conv1d(
            x, filters.unsqueeze(1),
            stride=1,
            padding=self.kernel_size // 2
        )


# ===================================================================
# RESBLOCK (1D, with downsample on ALL blocks per V8 spec)
# ===================================================================
class ResBlock1d(nn.Module):
    """1D Residual block. V8 spec: ALL blocks have downsample path."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.selu = nn.SELU(inplace=True)

        # Downsample on ALL blocks (V8 spec: even when in_ch == out_ch
        # and stride == 1, include the projection for architecture parity)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.selu(out + identity)
        return out


# ===================================================================
# GRAPH ATTENTION LAYER
# ===================================================================
class GraphAttentionLayer(nn.Module):
    """Single-head graph attention operating on temporal node sequence."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """x: (B, T, C_in) → (B, T, C_out)"""
        B, T, _ = x.shape
        h = self.W(x)  # (B, T, C_out)

        # Pairwise attention: for each pair of nodes (i, j)
        # Efficient: broadcast h_i || h_j
        h_i = h.unsqueeze(2).expand(-1, -1, T, -1)  # (B, T, T, C_out)
        h_j = h.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, T, C_out)
        e = self.leaky_relu(
            self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        )  # (B, T, T)

        alpha = F.softmax(e, dim=-1)  # (B, T, T)
        out = torch.bmm(alpha, h)     # (B, T, C_out)
        return F.elu(out)


# ===================================================================
# AASIST V8 MODEL
# ===================================================================
class AASIST_V8(nn.Module):
    """
    Exact V8 AASIST architecture:
      SincConv(70, k=127) → BN → SELU → MaxPool(3)
      → 6 ResBlocks [32,32,64,64,128,128] s=[1,2,1,2,1,2]
      → AdaptiveAvgPool1d(64)
      → GAT(128→64) → GAT(64→64)
      → Linear(64→64) → SELU → Dropout → Linear(64→2)
    Input: (B, 1, T)
    Output: (B, 2) logits
    """

    def __init__(self, cfg: dict):
        super().__init__()

        # --- Front-end: SincConv ---
        self.sinc = SincConv(
            out_channels=cfg["sinc_channels"],
            kernel_size=cfg["sinc_kernel"],
            sample_rate=cfg["sample_rate"],
        )
        self.bn0 = nn.BatchNorm1d(cfg["sinc_channels"])
        self.selu = nn.SELU(inplace=True)
        self.pool0 = nn.MaxPool1d(3)

        # --- ResBlocks ---
        channels = cfg["res_channels"]   # [32, 32, 64, 64, 128, 128]
        strides = cfg["res_strides"]     # [1, 2, 1, 2, 1, 2]
        in_ch = cfg["sinc_channels"]     # 70
        self.res_blocks = nn.ModuleList()
        for out_ch, s in zip(channels, strides):
            self.res_blocks.append(ResBlock1d(in_ch, out_ch, stride=s))
            in_ch = out_ch

        # --- Adaptive pooling ---
        self.adaptive_pool = nn.AdaptiveAvgPool1d(cfg["pool_output"])

        # --- GAT ---
        gat_dims = cfg["gat_dims"]  # [128, 64, 64]
        self.gat1 = GraphAttentionLayer(gat_dims[0], gat_dims[1])
        self.gat2 = GraphAttentionLayer(gat_dims[1], gat_dims[2])

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(gat_dims[2], cfg["cls_hidden"]),
            nn.SELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["cls_hidden"], 2),
        )

    def forward(self, x):
        """x: (B, 1, T) raw waveform"""
        # Front-end
        x = self.sinc(x)                # (B, 70, T)
        x = self.pool0(self.selu(self.bn0(x)))  # (B, 70, T//3)

        # ResBlocks
        for block in self.res_blocks:
            x = block(x)                # (B, 128, T')

        # Pool to fixed length
        x = self.adaptive_pool(x)       # (B, 128, 64)

        # GAT expects (B, T, C)
        x = x.permute(0, 2, 1)          # (B, 64, 128)
        x = self.gat1(x)                # (B, 64, 64)
        x = self.gat2(x)                # (B, 64, 64)

        # Aggregate nodes → single vector
        x = x.mean(dim=1)               # (B, 64)

        # Classify
        return self.classifier(x)       # (B, 2)


# ===================================================================
# DATASET
# ===================================================================
class AudioManifestDataset(Dataset):
    """Loads audio from a JSON manifest with path/label/source/weight."""

    def __init__(self, manifest_path: str, sample_rate: int = 16000,
                 max_samples: int = 64000, augment: bool = False):
        with open(manifest_path) as f:
            self.entries = json.load(f)
        self.sr = sample_rate
        self.max_samples = max_samples
        self.augment = augment

        # Verify a few paths exist
        missing = sum(1 for e in self.entries[:50]
                      if not os.path.isfile(e["path"]))
        if missing > 0:
            print(f"  ⚠ {missing}/{min(50, len(self.entries))} sampled paths "
                  f"missing in {manifest_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        path = entry["path"]
        label = int(entry["label"])
        source = entry.get("source", "unknown")

        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            # Fallback: return silence + label (don't crash training)
            print(f"  ⚠ Failed to load {path}: {e}")
            wav = torch.zeros(1, self.max_samples)
            sr = self.sr

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)

        # Trim or pad to max_samples
        if wav.shape[1] > self.max_samples:
            if self.augment:
                # Random crop during training
                start = random.randint(0, wav.shape[1] - self.max_samples)
                wav = wav[:, start:start + self.max_samples]
            else:
                wav = wav[:, :self.max_samples]
        elif wav.shape[1] < self.max_samples:
            pad = self.max_samples - wav.shape[1]
            wav = F.pad(wav, (0, pad))

        # Normalize waveform
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak

        return wav, label, source


# ===================================================================
# METRICS
# ===================================================================
def compute_eer(scores, labels):
    """Compute Equal Error Rate from score/label arrays."""
    from sklearn.metrics import roc_curve
    if len(set(labels)) < 2:
        # Single-class: EER is undefined
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    if np.all(np.isnan(diff)):
        return float("nan"), float("nan")
    idx = np.nanargmin(diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def compute_auc(scores, labels):
    """Compute AUC-ROC."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.0


# ===================================================================
# TRAINING
# ===================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    warmup_epochs, base_lr):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (wav, label, _) in enumerate(loader):
        # Warmup LR
        if epoch < warmup_epochs:
            warmup_frac = (epoch * len(loader) + batch_idx) / \
                          (warmup_epochs * len(loader))
            lr = base_lr * warmup_frac
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        wav = wav.to(device)      # (B, 1, T)
        label = label.to(device)  # (B,)

        optimizer.zero_grad()
        logits = model(wav)       # (B, 2)
        loss = criterion(logits, label)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * wav.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += wav.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    """Validate and return per-source breakdown."""
    model.eval()
    all_scores = []
    all_labels = []
    all_sources = []

    for wav, label, source in loader:
        wav = wav.to(device)
        logits = model(wav)
        # Score = softmax probability of class 1 (fake)
        probs = F.softmax(logits, dim=1)[:, 1]
        all_scores.extend(probs.cpu().numpy().tolist())
        all_labels.extend(label.numpy().tolist())
        all_sources.extend(source)

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # Overall metrics
    eer, eer_thresh = compute_eer(scores, labels)
    auc = compute_auc(scores, labels)

    # Per-source breakdown
    source_arr = np.array(all_sources)
    per_source = {}
    for src in sorted(set(all_sources)):
        mask = source_arr == src
        src_scores = scores[mask]
        src_labels = labels[mask]
        n = mask.sum()

        if len(set(src_labels)) < 2:
            # Single-class bucket: compute accuracy at EER threshold
            if math.isnan(eer_thresh):
                per_source[src] = {"n": int(n)}
            elif src_labels[0] == 1:  # all fake
                catch_rate = (src_scores >= eer_thresh).mean()
                per_source[src] = {"n": int(n), "catch_rate": float(catch_rate)}
            else:  # all real
                fp_rate = (src_scores >= eer_thresh).mean()
                per_source[src] = {"n": int(n), "fp_rate": float(fp_rate)}
        else:
            src_eer, _ = compute_eer(src_scores, src_labels)
            per_source[src] = {"n": int(n), "eer": float(src_eer)}

    return {
        "eer": float(eer),
        "eer_threshold": float(eer_thresh),
        "auc": float(auc),
        "per_source": per_source,
        "scores": scores,
        "labels": labels,
        "sources": all_sources,
    }


def print_val_results(results: dict, title: str = "Validation"):
    print(f"\n  --- {title} ---")
    eer_str = f"{results['eer']*100:.2f}%" if not math.isnan(results['eer']) else "N/A"
    thr_str = f"{results['eer_threshold']:.4f}" if not math.isnan(results['eer_threshold']) else "N/A"
    print(f"  EER: {eer_str}  |  AUC: {results['auc']:.4f}  "
          f"|  EER threshold: {thr_str}")
    print(f"  {'Source':<30} {'N':>5}  {'Metric':>12}")
    print(f"  {'-'*30} {'-'*5}  {'-'*12}")
    for src, info in sorted(results["per_source"].items()):
        n = info["n"]
        if "eer" in info:
            metric = f"EER {info['eer']*100:.1f}%"
        elif "catch_rate" in info:
            metric = f"catch {info['catch_rate']*100:.1f}%"
        elif "fp_rate" in info:
            metric = f"FP {info['fp_rate']*100:.1f}%"
        else:
            metric = "—"
        print(f"  {src:<30} {n:>5}  {metric:>12}")
    print()


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="AASIST V9 retrain")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 epoch with small subset for sanity check")
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--val-manifest", type=str, default=None)
    parser.add_argument("--eval-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    args = parser.parse_args()

    # Merge CLI into config
    cfg = DEFAULT_CFG.copy()
    for k in ["epochs", "batch_size", "lr", "patience", "seed",
              "train_manifest", "val_manifest", "eval_manifest",
              "output_dir", "label_smoothing"]:
        v = getattr(args, k.replace("-", "_"), None) if hasattr(
            args, k.replace("-", "_")) else getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if args.dry_run:
        cfg["epochs"] = 1

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    max_samples = int(cfg["sample_rate"] * cfg["max_seconds"])

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("\n[1/4] Loading datasets...")
    train_ds = AudioManifestDataset(
        cfg["train_manifest"], cfg["sample_rate"], max_samples, augment=True)
    val_ds = AudioManifestDataset(
        cfg["val_manifest"], cfg["sample_rate"], max_samples, augment=False)

    eval_ds = None
    if os.path.isfile(cfg["eval_manifest"]):
        eval_ds = AudioManifestDataset(
            cfg["eval_manifest"], cfg["sample_rate"], max_samples, augment=False)
        print(f"  Held-out eval set: {len(eval_ds)} entries")

    if args.dry_run:
        # Subset for quick check — keep balanced so metrics don't break
        train_ds.entries = train_ds.entries[:64]
        val_ds.entries = val_ds.entries[:64]
        if eval_ds:
            reals = [e for e in eval_ds.entries if int(e["label"]) == 0]
            fakes = [e for e in eval_ds.entries if int(e["label"]) == 1]
            eval_ds.entries = reals[:16] + fakes[:16]
        print("  [DRY RUN] Using small subsets")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True)
    eval_loader = None
    if eval_ds:
        eval_loader = DataLoader(
            eval_ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True)

    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} samples, {len(val_loader)} batches")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n[2/4] Building AASIST V8 architecture...")
    model = AASIST_V8(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_train:,} trainable")

    # Quick forward-pass sanity check
    dummy = torch.randn(2, 1, max_samples, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 2), f"Output shape mismatch: {out.shape}"
    print(f"  Forward pass OK: input {dummy.shape} → output {out.shape}")

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if cfg["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"] - cfg["warmup_epochs"],
            eta_min=cfg["lr"] * 0.01)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"])

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["label_smoothing"])

    start_epoch = 0
    best_eer = 1.0
    best_epoch = 0
    history = []

    # Resume if requested
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("best_eer", 1.0)
        print(f"  Resumed from epoch {start_epoch}, best EER {best_eer*100:.2f}%")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n[3/4] Training for {cfg['epochs']} epochs "
          f"(patience={cfg['patience']})...\n")
    print(f"  {'Epoch':>5} {'Loss':>8} {'Acc':>7} {'Val EER':>8} "
          f"{'Val AUC':>8} {'LR':>10} {'Time':>6}  {'Note'}")
    print(f"  {'-'*5} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*6}  {'-'*10}")

    no_improve = 0

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, cfg["warmup_epochs"], cfg["lr"])

        # Validate
        val_results = validate(model, val_loader, device)
        val_eer = val_results["eer"]
        val_auc = val_results["auc"]

        # LR step (skip during warmup)
        if epoch >= cfg["warmup_epochs"]:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Check improvement
        note = ""
        if val_eer < best_eer:
            best_eer = val_eer
            best_epoch = epoch
            no_improve = 0
            note = "★ best"

            # Save best checkpoint
            ckpt_path = os.path.join(cfg["output_dir"], "aasist_v9_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eer": best_eer,
                "val_auc": val_auc,
                "cfg": cfg,
            }, ckpt_path)
        else:
            no_improve += 1

        print(f"  {epoch:>5} {train_loss:>8.4f} {train_acc:>6.1%} "
              f"{val_eer:>7.2%} {val_auc:>8.4f} {current_lr:>10.2e} "
              f"{elapsed:>5.0f}s  {note}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_eer": val_eer,
            "val_auc": val_auc,
            "lr": current_lr,
        })

        # Print per-source every 10 epochs
        if (epoch + 1) % 10 == 0 or note == "★ best":
            print_val_results(val_results, f"Epoch {epoch} breakdown")

        # Early stopping
        if no_improve >= cfg["patience"]:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg['patience']} epochs)")
            break

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print(f"\n[4/4] Final evaluation (best model from epoch {best_epoch}, "
          f"EER {best_eer*100:.2f}%)...")

    # Reload best checkpoint
    ckpt_path = os.path.join(cfg["output_dir"], "aasist_v9_best.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # Val set final
    val_final = validate(model, val_loader, device)
    print_val_results(val_final, "FINAL — Val Set (val_v8_fresh.json)")

    # Held-out eval
    if eval_loader:
        eval_final = validate(model, eval_loader, device)
        print_val_results(eval_final, "FINAL — Held-Out (studio + noiz.ai)")

    # Save last checkpoint too
    last_path = os.path.join(cfg["output_dir"], "aasist_v9_last.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eer": best_eer,
        "cfg": cfg,
    }, last_path)

    # Save training history
    hist_path = os.path.join(cfg["output_dir"], "train_history_v9.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Saved:")
    print(f"    Best model:  {ckpt_path}")
    print(f"    Last model:  {last_path}")
    print(f"    History:     {hist_path}")
    print(f"\n  Best EER: {best_eer*100:.2f}% at epoch {best_epoch}")
    print(f"\n  Next step: re-fit XGBoost + calibration with the new AASIST scores.")


if __name__ == "__main__":
    main()

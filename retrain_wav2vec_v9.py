"""
retrain_wav2vec_v9.py — Retrain Wav2Vec2 classifier head on V9 manifest.

Architecture: Wav2Vec2-base (frozen) + trainable classifier:
  Linear(768→256) → ReLU → Dropout(0.3)
  → Linear(256→64) → ReLU → Dropout(0.15)
  → Linear(64→2)

The base model stays frozen — only the classifier head trains.
This eliminates the "clean = fake" bias from the W2V component.

Usage (Kaggle):
    !python retrain_wav2vec_v9.py --dry-run     # sanity check
    !python retrain_wav2vec_v9.py               # full run
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

from transformers import Wav2Vec2Model
from sklearn.metrics import roc_curve, roc_auc_score

# ===================================================================
# CONFIG
# ===================================================================
ARTEFACTS = "/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts"
OUTPUT_DIR = "/kaggle/working"

SAMPLE_RATE = 16000
MAX_SECONDS = 4.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECONDS)

DEFAULT_CFG = {
    "train_manifest": os.path.join(OUTPUT_DIR, "train_v9.json"),
    "val_manifest":   os.path.join(ARTEFACTS, "val_v8_fresh.json"),
    "eval_manifest":  os.path.join(OUTPUT_DIR, "eval_v9_heldout.json"),
    "output_dir":     OUTPUT_DIR,

    "epochs":         50,
    "batch_size":     16,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "warmup_epochs":  3,
    "patience":       10,
    "seed":           42,
    "num_workers":    2,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================================================================
# MODEL
# ===================================================================
class Wav2VecClassifier(nn.Module):
    """Wav2Vec2-base frozen + trainable classifier head.
    Classifier matches V8 checkpoint keys: classifier.{0,3,6}.*
    Input: (B, T) raw waveform — 2D, NOT 3D."""

    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        # Freeze entire base model
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        # Classifier head — indices match checkpoint: 0, 3, 6 have weights
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),     # 0
            nn.ReLU(),               # 1
            nn.Dropout(0.3),         # 2
            nn.Linear(256, 64),      # 3
            nn.ReLU(),               # 4
            nn.Dropout(0.15),        # 5
            nn.Linear(64, 2),        # 6
        )

    def forward(self, x):
        # x: (B, T) — 2D raw waveform
        with torch.no_grad():
            outputs = self.wav2vec(x)
        hidden = outputs.last_hidden_state  # (B, T', 768)
        pooled = hidden.mean(dim=1)         # (B, 768)
        return self.classifier(pooled)      # (B, 2)


# ===================================================================
# DATASET
# ===================================================================
class AudioManifestDataset(Dataset):
    def __init__(self, manifest_path, max_samples=MAX_SAMPLES, augment=False):
        with open(manifest_path) as f:
            self.entries = json.load(f)
        self.max_samples = max_samples
        self.augment = augment

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

        # Squeeze to 1D for Wav2Vec2
        wav = wav.squeeze(0)  # (T,)

        if wav.shape[0] > self.max_samples:
            if self.augment:
                start = random.randint(0, wav.shape[0] - self.max_samples)
                wav = wav[start:start + self.max_samples]
            else:
                wav = wav[:self.max_samples]
        elif wav.shape[0] < self.max_samples:
            wav = F.pad(wav, (0, self.max_samples - wav.shape[0]))

        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak

        return wav, label, source


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


# ===================================================================
# TRAINING
# ===================================================================
def train_one_epoch(model, loader, optimizer, criterion, device,
                    epoch, warmup_epochs, base_lr):
    model.train()
    # Keep wav2vec base in eval mode (frozen BN + dropout)
    model.wav2vec.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (wav, label, _) in enumerate(loader):
        # Warmup LR
        if epoch < warmup_epochs:
            warmup_frac = (epoch * len(loader) + batch_idx) / \
                          (warmup_epochs * len(loader))
            lr = base_lr * max(warmup_frac, 0.01)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        wav = wav.to(device)      # (B, T) — 2D
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(wav)
        loss = criterion(logits, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * wav.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += wav.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_scores, all_labels, all_sources = [], [], []

    for wav, label, source in loader:
        wav = wav.to(device)
        logits = model(wav)
        probs = F.softmax(logits, dim=1)[:, 1]
        all_scores.extend(probs.cpu().numpy().tolist())
        all_labels.extend(label.numpy().tolist())
        all_sources.extend(source)

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    eer, eer_thresh = compute_eer(scores, labels)
    auc = compute_auc(scores, labels)

    # Per-source
    source_arr = np.array(all_sources)
    per_source = {}
    for src in sorted(set(all_sources)):
        mask = source_arr == src
        s = scores[mask]
        l = labels[mask]
        n = mask.sum()
        if len(set(l)) < 2:
            if math.isnan(eer_thresh):
                per_source[src] = {"n": int(n)}
            elif l[0] == 1:
                per_source[src] = {"n": int(n),
                                   "catch_rate": float((s >= eer_thresh).mean())}
            else:
                per_source[src] = {"n": int(n),
                                   "fp_rate": float((s >= eer_thresh).mean())}
        else:
            src_eer, _ = compute_eer(s, l)
            per_source[src] = {"n": int(n), "eer": float(src_eer)}

    return {"eer": float(eer), "auc": float(auc),
            "eer_threshold": float(eer_thresh), "per_source": per_source}


def print_val(results, title="Validation"):
    eer_s = f"{results['eer']*100:.2f}%" if not math.isnan(results['eer']) else "N/A"
    thr_s = f"{results['eer_threshold']:.4f}" if not math.isnan(results['eer_threshold']) else "N/A"
    print(f"\n  --- {title} ---")
    print(f"  EER: {eer_s}  |  AUC: {results['auc']:.4f}  |  threshold: {thr_s}")
    print(f"  {'Source':<30} {'N':>5}  {'Metric':>12}")
    print(f"  {'-'*30} {'-'*5}  {'-'*12}")
    for src, info in sorted(results["per_source"].items()):
        n = info["n"]
        if "eer" in info:
            m = f"EER {info['eer']*100:.1f}%"
        elif "catch_rate" in info:
            m = f"catch {info['catch_rate']*100:.1f}%"
        elif "fp_rate" in info:
            m = f"FP {info['fp_rate']*100:.1f}%"
        else:
            m = "—"
        print(f"  {src:<30} {n:>5}  {m:>12}")
    print()


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Retrain Wav2Vec2 V9")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    for k in ["epochs", "batch_size", "lr", "patience", "output_dir"]:
        v = getattr(args, k, None)
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

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("\n[1/4] Loading datasets...")
    train_ds = AudioManifestDataset(cfg["train_manifest"], augment=True)
    val_ds = AudioManifestDataset(cfg["val_manifest"], augment=False)
    eval_ds = None
    if os.path.isfile(cfg["eval_manifest"]):
        eval_ds = AudioManifestDataset(cfg["eval_manifest"], augment=False)

    if args.dry_run:
        train_ds.entries = train_ds.entries[:64]
        val_ds.entries = val_ds.entries[:64]
        if eval_ds:
            reals = [e for e in eval_ds.entries if int(e["label"]) == 0]
            fakes = [e for e in eval_ds.entries if int(e["label"]) == 1]
            eval_ds.entries = reals[:16] + fakes[:16]

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

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} "
          f"| Held-out: {len(eval_ds) if eval_ds else 0}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n[2/4] Building Wav2Vec2 classifier...")
    model = Wav2VecClassifier().to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_total:,} total, {n_train:,} trainable "
          f"({n_train/n_total*100:.1f}% — classifier head only)")

    # Quick forward-pass check
    dummy = torch.randn(2, MAX_SAMPLES, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 2), f"Shape mismatch: {out.shape}"
    print(f"  Forward pass OK: ({2}, {MAX_SAMPLES}) → {out.shape}")

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss
    # ------------------------------------------------------------------
    # Only optimize classifier parameters (base is frozen)
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - cfg["warmup_epochs"],
        eta_min=cfg["lr"] * 0.01)

    criterion = nn.CrossEntropyLoss()

    best_eer = 1.0
    best_epoch = 0
    no_improve = 0
    history = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n[3/4] Training for {cfg['epochs']} epochs "
          f"(patience={cfg['patience']})...\n")
    print(f"  {'Epoch':>5} {'Loss':>8} {'Acc':>7} {'Val EER':>8} "
          f"{'Val AUC':>8} {'LR':>10} {'Time':>6}  {'Note'}")
    print(f"  {'-'*5} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*6}  {'-'*10}")

    for epoch in range(cfg["epochs"]):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, cfg["warmup_epochs"], cfg["lr"])

        val_results = validate(model, val_loader, device)
        val_eer = val_results["eer"]
        val_auc = val_results["auc"]

        if epoch >= cfg["warmup_epochs"]:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        note = ""
        if val_eer < best_eer:
            best_eer = val_eer
            best_epoch = epoch
            no_improve = 0
            note = "★ best"

            ckpt_path = os.path.join(cfg["output_dir"], "wav2vec_v9_best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "best_eer": best_eer,
                "val_auc": val_auc,
            }, ckpt_path)
        else:
            no_improve += 1

        print(f"  {epoch:>5} {train_loss:>8.4f} {train_acc:>6.1%} "
              f"{val_eer:>7.2%} {val_auc:>8.4f} {current_lr:>10.2e} "
              f"{elapsed:>5.0f}s  {note}")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "val_eer": val_eer,
            "val_auc": val_auc, "lr": current_lr,
        })

        if (epoch + 1) % 10 == 0 or note == "★ best":
            print_val(val_results, f"Epoch {epoch} breakdown")

        if no_improve >= cfg["patience"]:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg['patience']} epochs)")
            break

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print(f"\n[4/4] Final evaluation (best model from epoch {best_epoch})...")

    ckpt_path = os.path.join(cfg["output_dir"], "wav2vec_v9_best.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    val_final = validate(model, val_loader, device)
    print_val(val_final, "FINAL — Val Set")

    if eval_loader:
        eval_final = validate(model, eval_loader, device)
        print_val(eval_final, "FINAL — Held-Out (studio + noiz.ai)")

    # Save history
    hist_path = os.path.join(cfg["output_dir"], "w2v_train_history_v9.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Saved:")
    print(f"    Best model: {ckpt_path}")
    print(f"    History:    {hist_path}")
    print(f"    Best EER:   {best_eer*100:.2f}% at epoch {best_epoch}")
    print(f"\n  Next: re-run refit_ensemble_v9.py with --wav2vec-ckpt {ckpt_path}")


if __name__ == "__main__":
    main()

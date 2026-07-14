# %%  CELL: Train LCNN Screener + Distilled Student (Phase 7 Tasks 1-3)
# Run AFTER the teacher scores cell.
# Outputs: lcnn_screener_v9.pt, lcnn_student_v9.pt, lcnn_v9_results.json

import os, json, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ─── Config ──────────────────────────────────────────────────────────────────
TEACHER_TRAIN  = "/kaggle/working/teacher_scores_v9_train.json"
TEACHER_VAL    = "/kaggle/working/teacher_scores_v9_val.json"
OUT_SCREENER   = "/kaggle/working/lcnn_screener_v9.pt"
OUT_STUDENT    = "/kaggle/working/lcnn_student_v9.pt"
OUT_RESULTS    = "/kaggle/working/lcnn_v9_results.json"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR         = 16000
MAX_LEN    = 4 * SR
N_MELS     = 80
HOP_LENGTH = 160
WIN_LENGTH = 400

BATCH_SIZE   = 64
EPOCHS       = 50       # more room with stable training
LR           = 1e-3     # higher peak LR with warmup
WARMUP       = 5        # 5 epoch linear warmup
WEIGHT_DECAY = 1e-4
TEMPERATURE  = 4.0
ALPHA        = 0.7

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─── Mel transform (on GPU for speed during training) ────────────────────────
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=512, hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH, n_mels=N_MELS, f_min=20, f_max=8000
).to(DEVICE)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(DEVICE)

def wav_to_mel_gpu(wav_batch):
    """wav_batch: (B, T) on GPU → mel: (B, 1, N_MELS, T') normalised"""
    mel = mel_transform(wav_batch)
    mel = amplitude_to_db(mel)
    # Per-sample normalisation
    B = mel.shape[0]
    mel = mel.reshape(B, -1)
    mean = mel.mean(dim=1, keepdim=True)
    std  = mel.std(dim=1, keepdim=True) + 1e-8
    mel  = (mel - mean) / std
    mel  = mel.reshape(B, 1, N_MELS, -1)
    return mel

# ─── Audio loading ───────────────────────────────────────────────────────────
def load_audio(path):
    try:
        wav, sr = torchaudio.load(path)
        if sr != SR: wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(dim=0)
        if wav.shape[0] < SR // 4: return None
        if wav.shape[0] > MAX_LEN: wav = wav[:MAX_LEN]
        else: wav = F.pad(wav, (0, MAX_LEN - wav.shape[0]))
        return wav
    except:
        return None

# ─── Dataset ─────────────────────────────────────────────────────────────────
class DistillDataset(Dataset):
    def __init__(self, entries, augment=False):
        self.entries = entries
        self.augment = augment
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        e   = self.entries[idx]
        wav = load_audio(e["path"])
        if wav is None:
            wav = torch.zeros(MAX_LEN)
        if self.augment:
            if random.random() < 0.3:
                wav = wav + random.uniform(0.001, 0.005) * torch.randn_like(wav)
            if random.random() < 0.3:
                mask_len = random.randint(SR // 10, SR // 2)
                start    = random.randint(0, max(0, wav.shape[0] - mask_len))
                wav = wav.clone()
                wav[start:start + mask_len] = 0.0
        label        = torch.tensor(e["label"], dtype=torch.long)
        teacher_prob = torch.tensor(e.get("teacher_prob", float(e["label"])), dtype=torch.float32)
        weight       = torch.tensor(e.get("weight", 1.0), dtype=torch.float32)
        return wav, label, teacher_prob, weight

# ─── LCNN Architecture ───────────────────────────────────────────────────────
class MaxFeatureMap(nn.Module):
    def forward(self, x):
        half = x.shape[1] // 2
        return torch.max(x[:, :half], x[:, half:])

class LCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MaxFeatureMap()
    def forward(self, x):
        return self.mfm(F.relu(self.bn(self.conv(x))))

class LightCNN(nn.Module):
    """
    LCNN cascade screener / distilled student.
    Input: mel spectrogram (B, 1, N_MELS, T')
    Output: logits (B, 2)
    ~1.8M params, <200MB, CPU inference ~15-25ms for 4s audio
    """
    def __init__(self, n_mels=N_MELS):
        super().__init__()
        self.block1 = LCNNBlock(1, 16, kernel=(5,5), stride=(2,2), padding=(2,2))
        self.pool1  = nn.MaxPool2d((2,2), stride=(2,1))
        self.block2 = LCNNBlock(16, 32, kernel=(3,3), padding=(1,1))
        self.pool2  = nn.MaxPool2d((2,2), stride=(2,2))
        self.block3 = LCNNBlock(32, 64, kernel=(3,3), padding=(1,1))
        self.pool3  = nn.MaxPool2d((2,2), stride=(2,2))
        self.block4 = LCNNBlock(64, 64, kernel=(3,3), padding=(1,1))
        self.pool4  = nn.AdaptiveAvgPool2d((1, None))
        self.gru    = nn.GRU(64, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop   = nn.Dropout(0.3)
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.drop(x.mean(dim=1))    # mean-pool over all timesteps, not last
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ─── Distillation loss ───────────────────────────────────────────────────────
def distillation_loss(student_logits, teacher_probs, hard_labels, sample_weights,
                      T=TEMPERATURE, alpha=ALPHA):
    teacher_dist = torch.stack([1 - teacher_probs, teacher_probs], dim=1)
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_dist / T, dim=-1)
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction="none").sum(dim=-1) * T * T
    ce_loss = F.cross_entropy(student_logits, hard_labels, reduction="none")
    return ((alpha * kl_loss + (1 - alpha) * ce_loss) * sample_weights).mean()

# ─── EER ─────────────────────────────────────────────────────────────────────
def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return float(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)) * 100

# ─── Training loop ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    for wav_batch, labels, teacher_probs, weights in loader:
        wav_batch     = wav_batch.to(DEVICE)
        labels        = labels.to(DEVICE)
        teacher_probs = teacher_probs.to(DEVICE)
        weights       = weights.to(DEVICE)
        mel = wav_to_mel_gpu(wav_batch)
        optimizer.zero_grad()
        logits = model(mel)
        loss   = distillation_loss(logits, teacher_probs, labels, weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * wav_batch.size(0)
        n += wav_batch.size(0)
    if scheduler: scheduler.step()
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_scores, all_labels = [], []
    for wav_batch, labels, _, _ in loader:
        wav_batch = wav_batch.to(DEVICE)
        mel = wav_to_mel_gpu(wav_batch)
        probs = torch.softmax(model(mel), dim=-1)[:, 1].cpu().numpy()
        all_scores.extend(probs)
        all_labels.extend(labels.numpy())
    eer = compute_eer(np.array(all_scores), np.array(all_labels))
    return eer, np.array(all_scores), np.array(all_labels)

# ─── Cascade threshold calibration ───────────────────────────────────────────
def calibrate_cascade(scores, labels, target_res=0.80):
    results = []
    for low in np.arange(0.05, 0.45, 0.025):
        high = 1 - low
        resolved = (scores <= low) | (scores >= high)
        if resolved.sum() == 0: continue
        res_rate = resolved.mean()
        res_scores, res_labels = scores[resolved], labels[resolved]
        res_preds = (res_scores >= 0.5).astype(int)
        res_acc   = (res_preds == res_labels).mean()
        real_mask = res_labels == 0; fake_mask = res_labels == 1
        fpr = (res_preds[real_mask] == 1).mean() if real_mask.sum() > 0 else 0.0
        fnr = (res_preds[fake_mask] == 0).mean() if fake_mask.sum() > 0 else 0.0
        results.append({"low": float(low), "high": float(high),
                        "resolution_rate": float(res_rate), "resolved_accuracy": float(res_acc),
                        "resolved_fpr": float(fpr), "resolved_fnr": float(fnr)})
    good = [r for r in results if r["resolved_accuracy"] >= 0.95]
    if not good: good = results
    return max(good, key=lambda r: r["resolution_rate"])

# ─── Latency benchmark ───────────────────────────────────────────────────────
def benchmark_latency(model, n_runs=50):
    model.eval()
    cpu_model = model.cpu()
    mel_t_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=512, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS, f_min=20, f_max=8000)
    db_fn = torchaudio.transforms.AmplitudeToDB(top_db=80)
    wav = torch.zeros(MAX_LEN)
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        mel = db_fn(mel_t_fn(wav.unsqueeze(0)))
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel = mel.unsqueeze(0)
        with torch.no_grad(): _ = cpu_model(mel)
        times.append((time.perf_counter() - t) * 1000)
    model.to(DEVICE)
    p50, p95 = np.percentile(times, 50), np.percentile(times, 95)
    print(f"  Latency (CPU, {MAX_LEN/SR:.0f}s audio): p50={p50:.1f}ms  p95={p95:.1f}ms")
    return p50, p95

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading teacher-scored manifests...")
with open(TEACHER_TRAIN) as f: train_entries = json.load(f)
with open(TEACHER_VAL)   as f: val_entries   = json.load(f)
print(f"  Train: {len(train_entries)} | Val: {len(val_entries)}")

train_loader = DataLoader(DistillDataset(train_entries, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                          pin_memory=True, drop_last=True)
val_loader   = DataLoader(DistillDataset(val_entries, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                          pin_memory=True)

model     = LightCNN().to(DEVICE)
n_params  = model.count_params()
print(f"\nLightCNN params: {n_params:,} (~{n_params * 4 / 1e6:.1f}MB fp32)")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP, eta_min=1e-5)
scheduler    = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP])

best_eer, best_state, history = 100.0, None, []
print(f"\nTraining for {EPOCHS} epochs on {DEVICE}...\n")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_eer, val_scores, val_labels = evaluate(model, val_loader)
    elapsed = time.time() - t0
    print(f"Epoch {epoch:3d}/{EPOCHS} | loss={train_loss:.4f} | val_EER={val_eer:.2f}% | {elapsed:.0f}s")
    history.append({"epoch": epoch, "loss": train_loss, "val_eer": val_eer})
    if val_eer < best_eer:
        best_eer   = val_eer
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ✓ New best EER: {best_eer:.2f}%")
    if len(history) >= 15:
        recent = [h["val_eer"] for h in history[-15:]]
        if max(recent) - min(recent) < 0.3:
            print(f"\nPlateau detected at epoch {epoch}. Stopping early.")
            break

# Restore best
model.load_state_dict(best_state); model.to(DEVICE)

print("\n─── Final Evaluation ──────────────────────────────────────────────")
final_eer, final_scores, final_labels = evaluate(model, val_loader)
print(f"Screener EER on val set: {final_eer:.2f}%")

print("\n─── Cascade Threshold Calibration ─────────────────────────────────")

# Platt scaling: fit sigmoid to map raw LCNN scores → calibrated probabilities
from sklearn.linear_model import LogisticRegression
platt = LogisticRegression(solver="lbfgs", max_iter=1000)
platt.fit(final_scores.reshape(-1, 1), final_labels)
cal_scores = platt.predict_proba(final_scores.reshape(-1, 1))[:, 1]
platt_coef = float(platt.coef_[0, 0])
platt_intercept = float(platt.intercept_[0])
print(f"  Platt calibration: P = sigmoid({platt_coef:.4f} * score + {platt_intercept:.4f})")

# Calibrate cascade on Platt-scaled scores
cascade = calibrate_cascade(cal_scores, final_labels)
print(f"  Low threshold:     {cascade['low']:.3f}")
print(f"  High threshold:    {cascade['high']:.3f}")
print(f"  Resolution rate:   {cascade['resolution_rate']*100:.1f}%")
print(f"  Resolved accuracy: {cascade['resolved_accuracy']*100:.1f}%")
print(f"  Resolved FPR:      {cascade['resolved_fpr']*100:.1f}%")
print(f"  Resolved FNR:      {cascade['resolved_fnr']*100:.1f}%")

print("\n─── Latency Benchmark (CPU) ────────────────────────────────────────")
p50, p95 = benchmark_latency(model)

# Save
ckpt = {
    "model_state_dict": model.state_dict(),
    "n_params": n_params,
    "final_eer": final_eer, "best_eer": best_eer,
    "cascade_thresholds": {
        "low_thresh": cascade["low"], "high_thresh": cascade["high"],
        "resolution_rate": cascade["resolution_rate"],
        "resolved_accuracy": cascade["resolved_accuracy"]},
    "platt_calibration": {"coef": platt_coef, "intercept": platt_intercept},
    "latency_ms": {"p50": p50, "p95": p95},
    "architecture": "LightCNN (MFM conv + bi-GRU mean-pool)",
    "teacher": "AASIST V9 + Wav2Vec2 V9 + RawNet3 V8 + XGBoost",
    "distillation": {"temperature": TEMPERATURE, "alpha": ALPHA},
    "training_history": history}

torch.save(ckpt, OUT_SCREENER)
torch.save(ckpt, OUT_STUDENT)
print(f"\n✓ Saved: {OUT_SCREENER}")
print(f"✓ Saved: {OUT_STUDENT}")

results = {
    "screener_eer_percent": final_eer,
    "cascade_thresholds": cascade,
    "latency_cpu_ms": {"p50": p50, "p95": p95},
    "param_count": n_params,
    "model_size_mb_fp32": n_params * 4 / 1e6,
    "training_epochs": len(history),
    "training_history": history}
with open(OUT_RESULTS, "w") as f:
    json.dump(results, f, indent=2)
print(f"✓ Results: {OUT_RESULTS}")

print(f"\n─── Summary ───────────────────────────────────────────────────────")
print(f"  EER:          {final_eer:.2f}%")
print(f"  Resolution:   {cascade['resolution_rate']*100:.1f}%")
print(f"  Latency p50:  {p50:.1f}ms")
print(f"  Model size:   {n_params * 4 / 1e6:.1f}MB (fp32)")
print(f"\n✓ Phase 7 Tasks 1-3 complete. Download lcnn_screener_v9.pt + lcnn_student_v9.pt.")

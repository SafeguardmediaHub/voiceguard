"""
Phase 7 — Task 1a: Generate V9 Ensemble Teacher Scores
=======================================================
Runs the full V9 ensemble over the training manifest and saves soft labels
(calibrated probabilities) for use in LCNN distillation training.

Output: teacher_scores_v9.json
  [{"path": ..., "label": 0|1, "source": ..., "weight": ...,
    "teacher_prob": 0.0-1.0,   # P(fake) from calibrated ensemble
    "teacher_logit": float},   # raw XGB score before sigmoid calibration
   ...]

Run BEFORE train_lcnn_distill_v9.py
"""

import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import xgboost as xgb
from scipy.special import expit as sigmoid

# ─── Paths ───────────────────────────────────────────────────────────────────
MANIFEST      = "/kaggle/working/train_v9.json"
VAL_MANIFEST  = "/kaggle/working/val_v8_fresh.json"
AASIST_CKPT   = "/kaggle/working/aasist_v9_best.pt"
WAV2VEC_CKPT  = "/kaggle/working/wav2vec_v9_best.pt"
RAWNET3_CKPT  = "/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts/rawnet3.pt"
XGB_MODEL     = "/kaggle/working/xgb_v9.json"
CAL_PARAMS    = "/kaggle/working/cal_v9_params.json"
OUT_TRAIN     = "/kaggle/working/teacher_scores_v9_train.json"
OUT_VAL       = "/kaggle/working/teacher_scores_v9_val.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR     = 16000
MAX_LEN = 4 * SR   # 4 seconds

# ─── AASIST V9 Architecture (verified against checkpoint) ─────────────────────
class SincConvAASIST(nn.Module):
    """Learned sinc filterbank matching checkpoint keys: low_hz_, band_hz_, hamming, n_"""
    def __init__(self, out_channels=70, kernel_size=127, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.sample_rate  = sample_rate
        low_hz, high_hz = 30.0, sample_rate / 2 - 50
        mel_low  = 2595 * np.log10(1 + low_hz  / 700)
        mel_high = 2595 * np.log10(1 + high_hz / 700)
        mel_pts  = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
        self.low_hz_  = nn.Parameter(torch.Tensor(hz_pts[:-1]).unsqueeze(1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_pts)).unsqueeze(1))
        half = (kernel_size - 1) // 2
        n_   = torch.arange(-half, half + 1).float()
        self.register_buffer("n_", n_)          # shape [127]
        hamming = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size).float() / (kernel_size - 1))
        self.register_buffer("hamming", hamming) # shape [127]

    def forward(self, x):
        low  = torch.clamp(self.low_hz_,  min=50) / (self.sample_rate / 2)
        high = torch.clamp(self.low_hz_ + torch.abs(self.band_hz_),
                           min=50, max=self.sample_rate / 2 - 50) / (self.sample_rate / 2)
        n_ = self.n_.unsqueeze(0)        # [1, 127]
        hw = self.hamming.unsqueeze(0)   # [1, 127]
        sinc_low  = torch.sinc(n_ * low)  * 2 * low
        sinc_high = torch.sinc(n_ * high) * 2 * high
        filters   = (sinc_high - sinc_low) * hw
        filters   = filters / (filters.norm(dim=1, keepdim=True) + 1e-8)
        filters   = filters.unsqueeze(1)
        return F.conv1d(x, filters, padding=self.kernel_size // 2)

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act   = nn.SELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)
        )
    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + self.downsample(x))

class GATLayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W = nn.Linear(in_d, out_d, bias=False)
        self.a = nn.Linear(2 * out_d, 1, bias=False)   # checkpoint key: .a not .att
    def forward(self, x):
        h = self.W(x)
        B, T, D = h.shape
        src = h.unsqueeze(2).expand(-1, -1, T, -1)
        dst = h.unsqueeze(1).expand(-1, T, -1, -1)
        e   = torch.tanh(self.a(torch.cat([src, dst], dim=-1))).squeeze(-1)
        a   = torch.softmax(e, dim=-1)
        return torch.bmm(a, h)

class AASIST_V9(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc  = SincConvAASIST(70, 127)
        self.bn0   = nn.BatchNorm1d(70)
        self.act   = nn.SELU()
        self.mpool = nn.MaxPool1d(3)
        channels = [32, 32, 64, 64, 128, 128]
        strides  = [1,  2,  1,  2,  1,   2]
        in_ch    = 70
        blocks   = []
        for out_ch, s in zip(channels, strides):
            blocks.append(ResBlock1D(in_ch, out_ch, stride=s))
            in_ch = out_ch
        self.res_blocks = nn.ModuleList(blocks)   # checkpoint key: res_blocks
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.gat1 = GATLayer(128, 64)
        self.gat2 = GATLayer(64, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64), nn.SELU(), nn.Dropout(0.3), nn.Linear(64, 2)
        )
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.mpool(self.act(self.bn0(self.sinc(x))))
        for blk in self.res_blocks:
            x = blk(x)
        x = self.pool(x)              # (B, 128, 64)
        x = x.permute(0, 2, 1)       # (B, 64, 128)
        x = self.gat1(x)
        x = self.gat2(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ─── Wav2Vec2 V9 ─────────────────────────────────────────────────────────────
from transformers import Wav2Vec2Model

class Wav2Vec2Classifier_V9(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec    = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for p in self.wav2vec.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        out = self.wav2vec(x).last_hidden_state.mean(dim=1)
        return self.classifier(out)

# ─── RawNet3 V8 ──────────────────────────────────────────────────────────────
class SincConvRaw(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.sample_rate = sample_rate
        low_hz, high_hz = 30.0, sample_rate / 2 - 50
        n_filters = out_channels          # NOT out_channels // 2
        mel_low  = 2595 * np.log10(1 + low_hz  / 700)
        mel_high = 2595 * np.log10(1 + high_hz / 700)
        mel_pts  = np.linspace(mel_low, mel_high, n_filters + 1)
        hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
        self.low_hz_  = nn.Parameter(torch.Tensor(hz_pts[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_pts)).view(-1, 1))
        half = (kernel_size - 1) // 2
        n_ = torch.arange(-half, 0).view(1, -1).float()
        self.register_buffer("n_", n_)
        window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(half) / (kernel_size - 1))
        self.register_buffer("window_", window_)

    def forward(self, x):
        low  = torch.clamp(self.low_hz_,  min=50)
        high = torch.clamp(low + torch.clamp(self.band_hz_, min=50), max=self.sample_rate / 2 - 50)
        f_low  = 2 * low  / self.sample_rate
        f_high = 2 * high / self.sample_rate
        half_k = (2 * f_high * torch.sinc(f_high * self.n_ * 2) -
                  2 * f_low  * torch.sinc(f_low  * self.n_ * 2)) * self.window_
        filters = torch.cat([half_k.flip(dims=[1]),
                              (f_high - f_low),
                              half_k], dim=1)
        filters = filters.unsqueeze(1)   # (128, 1, K) — no neg mirror
        return F.conv1d(x, filters, padding=self.kernel_size // 2)

class RawResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        # FMS as flat Sequential — checkpoint keys: fms.0 (pool), fms.1 (flatten), fms.2 (linear), fms.3 (sigmoid)
        self.fms   = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_ch, out_ch),
            nn.Sigmoid()
        )
        self.act   = nn.SELU()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.downsample = None
    def forward(self, x):
        r = self.act(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        scale = self.fms(r).unsqueeze(-1)
        r = r * scale
        if self.downsample: x = self.downsample(x)
        return self.act(r + x)

class RawNet3_V8(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc    = SincConvRaw(128, 512)
        self.sinc_bn = nn.BatchNorm1d(128)
        self.act     = nn.SELU()
        self.pool    = nn.MaxPool1d(3)
        self.res_blocks = nn.ModuleList([       # checkpoint key: res_blocks
            RawResBlock(128, 128, 1),
            RawResBlock(128, 128, 2),
            RawResBlock(128, 256, 1),
            RawResBlock(256, 256, 2),
        ])
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True,
                          dropout=0.5, bidirectional=False)
        # checkpoint key: attention_pool.attention
        self.attention_pool = nn.Module()
        self.attention_pool.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.drop       = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.SELU(), nn.Dropout(0.25), nn.Linear(128, 2)
        )
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pool(self.act(self.sinc_bn(self.sinc(x))))
        for blk in self.res_blocks:
            x = blk(x)
        x = x.permute(0, 2, 1)     # (B, T', 256) — batch_first for GRU
        x, _ = self.gru(x)         # (B, T', 256)
        # Attention pooling
        w = torch.softmax(self.attention_pool.attention(x), dim=1)
        x = (x * w).sum(dim=1)
        x = self.drop(x)
        return self.classifier(x)

# ─── Audio loading ────────────────────────────────────────────────────────────
def load_audio(path):
    try:
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(dim=0)
        if wav.shape[0] < SR // 4:
            return None
        if wav.shape[0] > MAX_LEN:
            wav = wav[:MAX_LEN]
        else:
            wav = torch.nn.functional.pad(wav, (0, MAX_LEN - wav.shape[0]))
        return wav
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None

# ─── Load models ─────────────────────────────────────────────────────────────
def load_models():
    print("Loading AASIST V9...")
    aasist = AASIST_V9().to(DEVICE)
    ck = torch.load(AASIST_CKPT, map_location=DEVICE)
    aasist.load_state_dict(ck.get("model_state_dict", ck))
    aasist.eval()

    print("Loading Wav2Vec2 V9...")
    w2v = Wav2Vec2Classifier_V9().to(DEVICE)
    ck  = torch.load(WAV2VEC_CKPT, map_location=DEVICE)
    w2v.load_state_dict(ck.get("model", ck))
    w2v.eval()

    print("Loading RawNet3 V8...")
    rn3 = RawNet3_V8().to(DEVICE)
    torch.backends.cudnn.enabled = False
    ck  = torch.load(RAWNET3_CKPT, map_location=DEVICE)
    rn3.load_state_dict(ck.get("model_state", ck))
    rn3.eval()

    print("Loading XGBoost + calibration...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL)
    with open(CAL_PARAMS) as f:
        cal = json.load(f)
    cal_a, cal_b = cal["a"], cal["b"]

    return aasist, w2v, rn3, xgb_model, cal_a, cal_b

# ─── Score one sample ─────────────────────────────────────────────────────────
@torch.no_grad()
def score_sample(wav, aasist, w2v, rn3, xgb_model, cal_a, cal_b):
    wav_t = wav.unsqueeze(0).to(DEVICE)

    # AASIST
    logit_a = aasist(wav_t.unsqueeze(1))
    prob_a  = torch.softmax(logit_a, dim=-1)[0, 1].item()

    # Wav2Vec2
    logit_w = w2v(wav_t)
    prob_w  = torch.softmax(logit_w, dim=-1)[0, 1].item()

    # RawNet3
    logit_r = rn3(wav_t)
    prob_r  = torch.softmax(logit_r, dim=-1)[0, 1].item()

    # XGBoost fusion
    feats = np.array([[prob_a, prob_w, prob_r]])
    xgb_score  = xgb_model.predict_proba(feats)[0, 1]
    teacher_prob = float(sigmoid(cal_a * xgb_score + cal_b))

    return teacher_prob, float(xgb_score)

# ─── Process manifest ─────────────────────────────────────────────────────────
def process_manifest(manifest_path, out_path, aasist, w2v, rn3, xgb_model, cal_a, cal_b):
    with open(manifest_path) as f:
        data = json.load(f)

    print(f"\nProcessing {len(data)} entries from {manifest_path}")
    results, skipped = [], 0
    t0 = time.time()

    for i, entry in enumerate(data):
        wav = load_audio(entry["path"])
        if wav is None:
            skipped += 1
            continue

        teacher_prob, xgb_score = score_sample(wav, aasist, w2v, rn3, xgb_model, cal_a, cal_b)
        results.append({
            **entry,
            "teacher_prob":  teacher_prob,
            "teacher_logit": xgb_score
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(data) - i - 1) / rate
            print(f"  [{i+1}/{len(data)}] {rate:.1f} samples/s | ETA {eta:.0f}s")

    print(f"\nDone. {len(results)} scored, {skipped} skipped.")

    # Sanity check: teacher agreement with hard labels
    correct = sum(
        1 for r in results
        if (r["teacher_prob"] >= 0.5) == (r["label"] == 1)
    )
    print(f"Teacher accuracy on manifest: {correct/len(results)*100:.1f}%")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    aasist, w2v, rn3, xgb_model, cal_a, cal_b = load_models()

    process_manifest(MANIFEST, OUT_TRAIN, aasist, w2v, rn3, xgb_model, cal_a, cal_b)

    if os.path.exists(VAL_MANIFEST):
        process_manifest(VAL_MANIFEST, OUT_VAL, aasist, w2v, rn3, xgb_model, cal_a, cal_b)

    print("\n✓ Teacher scores ready. Run train_lcnn_distill_v9.py next.")

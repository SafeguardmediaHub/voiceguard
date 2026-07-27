#!/usr/bin/env python3
"""
VoiceGuard V9 — Detection Core (framework-agnostic), Cascade Edition
This module holds the model + scoring core (no web framework). The FastAPI
web layer lives in api.py; run the server with `uvicorn api:app --port 7860`.

V9 Hybrid Ensemble (post-AASIST fix):
  - AASIST V8: ORIGINAL architecture from training notebook. Uses SincConv
    with sin/matmul bandpass (NOT torch.sinc), AdaptiveAvgPool1d(32),
    GAT with LeakyReLU+elu+dropout(0.6). Checkpoint wrapper key: 'model'.
    Class 1 = fake (standard).
    *** The prior SincConvAASIST was WRONG — it used torch.sinc() instead of
    torch.sin() with pre-scaled radians, causing saturated output (all ≈1.0).
    This was the root cause of AASIST contributing nothing to the ensemble. ***
  - Wav2Vec2 V9: classifier head retrained (768→256→64→2), base frozen.
    Backbone attribute named `wav2vec` (checkpoint wrapper key `model`).
  - RawNet3 V8: kept from V8; XGBoost downweights it to ~0.02.
    Checkpoint wrapper key `model_state`.
  - XGBoost hybrid + logistic calibration: refit with V8-AASIST + V9-W2V + V8-RN3.
    P = sigmoid(coef * xgb_prob + intercept).

Cascade router (Phase 7 Tasks 1-3), applied PER CHUNK:
  - Stage 1: LightCNN (LCNN) screener on a mel-spectrogram → Platt-calibrated
    probability. If confident (≤ low_thresh or ≥ high_thresh) the chunk is
    resolved at stage 1 (fast, CPU). cascade_thresholds + platt_calibration
    are read from the LCNN checkpoint.
  - Stage 2: uncertain chunks escalate to the full hybrid ensemble
    (AASIST V8 + Wav2Vec2 V9 + RawNet3 V8 → XGBoost → calibration).
  Long-audio chunking (4s windows, 50% overlap) and segment-level flagging are
  preserved, so the cascade runs independently on every chunk.

Mel spectrogram is computed in pure torch (no torchaudio dependency) faithful
to torchaudio.transforms.MelSpectrogram(n_fft=512, hop=160, win=400, n_mels=80,
f_min=20, f_max=8000, mel_scale='htk') + AmplitudeToDB(top_db=80). Verified to
reproduce the validated cascade routing on held-out clips.
"""
import os, json, time, hashlib, subprocess, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
import xgboost as xgb_lib

from input_randomization import randomize_chunk
import explain_signals
import gradcam

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
SR     = 16000
CHUNK  = 64000          # 4s @ 16kHz — the fixed window every model expects
DEVICE = torch.device("cpu")
# ffmpeg binary — 'ffmpeg' on PATH (Linux/Docker); a full path on Windows. Override with VOICEGUARD_FFMPEG.
FFMPEG = os.environ.get("VOICEGUARD_FFMPEG",
                        "C:/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe" if os.name == "nt" else "ffmpeg")

# Mel config (must match the LCNN's training/teacher pipeline exactly)
N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH = 80, 512, 160, 400

# Input randomization config (Phase 5 Task 3) — applies to the stage-2 ensemble.
# When OFF (default): deterministic, identical to a single clean pass.
INPUT_RANDOMIZATION_ENABLED = False
INPUT_RANDOMIZATION_MODE    = 'gaussian'  # 'gaussian' | 'crop_pad' | 'multi_sample'
INPUT_RANDOMIZATION_SIGMA   = 0.002
INPUT_RANDOMIZATION_N_SAMPLES = 5

# Request protection config (Phase 5 Task 4) — rate limiting + anomaly detection
REQUEST_PROTECTION_ENABLED = os.environ.get("VOICEGUARD_REQUEST_PROTECTION", "1") != "0"

print("VoiceGuard V9 Offline Server (cascade) starting...")
print(f"Base directory:   {BASE}")
print(f"Models directory: {MODELS}")
print(f"NumPy version:    {np.__version__}")
print(f"PyTorch version:  {torch.__version__}")
print()


# ══════════════════════════════════════════════════════════════════════════
#  Model definitions — AASIST uses original training notebook architecture
#  (SincConv with sin/matmul, pool=32, GAT with LeakyReLU+elu)
#  W2V + RN3 verified against V9/V8 checkpoints
# ══════════════════════════════════════════════════════════════════════════

# ── LCNN stage-1 screener ───────────────────────────────────────────────────
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
    def __init__(self, n_mels=N_MELS):
        super().__init__()
        self.block1 = LCNNBlock(1, 16, kernel=(5, 5), stride=(2, 2), padding=(2, 2))
        self.pool1  = nn.MaxPool2d((2, 2), stride=(2, 1))
        self.block2 = LCNNBlock(16, 32, kernel=(3, 3), padding=(1, 1))
        self.pool2  = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.block3 = LCNNBlock(32, 64, kernel=(3, 3), padding=(1, 1))
        self.pool3  = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.block4 = LCNNBlock(64, 64, kernel=(3, 3), padding=(1, 1))
        self.pool4  = nn.AdaptiveAvgPool2d((1, None))
        self.gru    = nn.GRU(64, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop   = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = x.squeeze(2).permute(0, 2, 1)     # (B, T', 64)
        x, _ = self.gru(x)
        x = self.drop(x.mean(dim=1))          # mean-pool over timesteps
        return self.classifier(x)


# ── AASIST — ORIGINAL architecture from training notebook ─────────────────────
# CRITICAL FIX: The prior SincConvAASIST used torch.sinc() which is sin(πx)/(πx).
# The original training code uses torch.sin() with pre-scaled n_ = 2π*n/sr and
# matmul-based bandpass filter construction. This difference caused AASIST to
# produce saturated output (~1.0 for all inputs), rendering it useless.
# The classes below are copy-pasted from the verified training notebook (cell 46).

class SincConv(nn.Module):
    """Original SincConv from training notebook. Uses sin/matmul bandpass,
    NOT torch.sinc(). Checkpoint keys: low_hz_ [70,1], band_hz_ [70,1],
    window_ [63], n_ [1,63]."""
    def __init__(self, out_channels=70, kernel_size=128,
                 sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels  = out_channels
        self.kernel_size   = kernel_size - (kernel_size % 2 == 0)  # force odd: 128→127
        self.sample_rate   = sample_rate
        self.min_low_hz    = min_low_hz
        self.min_band_hz   = min_band_hz

        low_hz  = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel_pts = np.linspace(
            self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 1)
        hz_pts  = self._mel2hz(mel_pts)

        self.low_hz_  = nn.Parameter(torch.Tensor(hz_pts[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_pts)).view(-1, 1))

        # Half-kernel window: 63 points
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int(self.kernel_size / 2))
        self.register_buffer("window_",
            0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / self.kernel_size))

        # n_ is PRE-SCALED by 2π/sample_rate — THIS is the critical difference
        # from the broken version which used raw integer arange
        n = (self.kernel_size - 1) / 2.0
        self.register_buffer("n_",
            2 * np.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    @staticmethod
    def _hz2mel(hz):  return 2595 * np.log10(1 + hz / 700)
    @staticmethod
    def _mel2hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x):
        low  = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        # Matmul: low [70,1] × n_ [1,63] → [70,63]  (freq × time products)
        f_times_t_low  = torch.matmul(low,  self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Bandpass via torch.sin(), NOT torch.sinc()
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
            / (self.n_ / 2)
        ) * self.window_

        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right  = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        filters   = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    """Original ResBlock from training notebook."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.selu  = nn.SELU()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = x
        out = self.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.selu(out + identity)


class GraphAttentionLayer(nn.Module):
    """Original GAT from training notebook. Uses LeakyReLU + elu (NOT tanh),
    with dropout=0.6."""
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W       = nn.Linear(in_features, out_features, bias=False)
        self.a       = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky   = nn.LeakyReLU(alpha)

    def forward(self, x):
        h       = self.W(x)
        B, N, Fout = h.shape
        h_i     = h.unsqueeze(2).expand(B, N, N, Fout)
        h_j     = h.unsqueeze(1).expand(B, N, N, Fout)
        e       = self.leaky(
            self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        attn    = self.dropout(torch.softmax(e, dim=-1))
        out     = torch.bmm(attn, h)
        return F.elu(out)


class AASIST(nn.Module):
    """Original AASIST from training notebook.
    Key differences from the broken AASIST_V9:
      - SincConv: sin/matmul bandpass, not torch.sinc()
      - AdaptiveAvgPool1d(32), not 64
      - GAT: LeakyReLU+elu+dropout(0.6), not tanh
      - res_blocks: nn.Sequential (matches checkpoint numbering)
      - Dropout: 0.5 main, 0.25 classifier (not 0.3)
    Checkpoint wrapper key: 'model' (V8) or 'model_state_dict' (V9).
    Class 1 = fake (standard, same as W2V and RN3)."""
    def __init__(self, sinc_out=70, sinc_kernel=128,
                 res_channels=[32, 32, 64, 64, 128, 128],
                 gat_features=64, num_classes=2,
                 dropout=0.5, sample_rate=16000):
        super().__init__()
        self.sinc      = SincConv(sinc_out, sinc_kernel, sample_rate)
        self.sinc_bn   = nn.BatchNorm1d(sinc_out)
        self.sinc_selu = nn.SELU()
        self.sinc_pool = nn.MaxPool1d(3)

        layers = []
        in_ch  = sinc_out
        strides = [1, 2, 1, 2, 1, 2]
        for out_ch, stride in zip(res_channels, strides):
            layers.append(ResBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.res_blocks = nn.Sequential(*layers)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)  # 32, NOT 64

        self.gat1 = GraphAttentionLayer(res_channels[-1], gat_features, dropout)
        self.gat2 = GraphAttentionLayer(gat_features, gat_features, dropout)

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gat_features, 64),
            nn.SELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes))

    def _encode(self, x):
        x = self.sinc_selu(self.sinc_bn(self.sinc(x)))
        x = self.sinc_pool(x)
        x = self.res_blocks(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        x = self.gat1(x)
        x = self.gat2(x)
        return x.mean(dim=1)

    def forward(self, x):
        return self.classifier(self.dropout(self._encode(x)))


# ── Wav2Vec2 V9 ──────────────────────────────────────────────────────────────
class Wav2Vec2Classifier_V9(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for p in self.wav2vec.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 2))

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        with torch.no_grad():
            feats = self.wav2vec(x).last_hidden_state
        return self.classifier(feats.mean(dim=1))


# ── RawNet3 V8 (unchanged model, kept in the V9 ensemble) ───────────────────
class SincConvRaw(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size, self.sample_rate = kernel_size, sample_rate
        low_hz, high_hz = 30.0, sample_rate / 2 - 50
        mel_pts = np.linspace(2595 * np.log10(1 + low_hz / 700),
                              2595 * np.log10(1 + high_hz / 700), out_channels + 1)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        self.low_hz_  = nn.Parameter(torch.Tensor(hz_pts[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_pts)).view(-1, 1))
        half = (kernel_size - 1) // 2
        self.register_buffer("n_", torch.arange(-half, 0).view(1, -1).float())
        self.register_buffer("window_",
            0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(half) / (kernel_size - 1)))

    def forward(self, x):
        low  = torch.clamp(self.low_hz_, min=50)
        high = torch.clamp(low + torch.clamp(self.band_hz_, min=50),
                           max=self.sample_rate / 2 - 50)
        f_low, f_high = 2 * low / self.sample_rate, 2 * high / self.sample_rate
        half_k = (2 * f_high * torch.sinc(f_high * self.n_ * 2)
                  - 2 * f_low * torch.sinc(f_low * self.n_ * 2)) * self.window_
        filters = torch.cat([half_k.flip(dims=[1]), (f_high - f_low), half_k], dim=1)
        return F.conv1d(x, filters.unsqueeze(1), padding=self.kernel_size // 2)


class RawResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.fms   = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                   nn.Linear(out_ch, out_ch), nn.Sigmoid())
        self.act   = nn.SELU()
        self.downsample = (nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)) if stride != 1 or in_ch != out_ch else None)

    def forward(self, x):
        r = self.act(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r)); r = r * self.fms(r).unsqueeze(-1)
        if self.downsample:
            x = self.downsample(x)
        return self.act(r + x)


class RawNet3_V8(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = SincConvRaw(128, 512); self.sinc_bn = nn.BatchNorm1d(128)
        self.act = nn.SELU(); self.pool = nn.MaxPool1d(3)
        self.res_blocks = nn.ModuleList([
            RawResBlock(128, 128, 1), RawResBlock(128, 128, 2),
            RawResBlock(128, 256, 1), RawResBlock(256, 256, 2)])
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, dropout=0.5)
        self.attention_pool = nn.Module()
        self.attention_pool.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.drop = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.SELU(), nn.Dropout(0.25), nn.Linear(128, 2))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.act(self.sinc_bn(self.sinc(x))))
        for b in self.res_blocks:
            x = b(x)
        x = x.permute(0, 2, 1); x, _ = self.gru(x)
        w = torch.softmax(self.attention_pool.attention(x), dim=1)
        return self.classifier(self.drop((x * w).sum(dim=1)))


# ══════════════════════════════════════════════════════════════════════════
#  Pure-torch mel spectrogram (faithful to torchaudio MelSpectrogram defaults)
# ══════════════════════════════════════════════════════════════════════════
def _hz_to_mel_htk(f):  return 2595.0 * math.log10(1.0 + f / 700.0)
def _mel_to_hz_htk(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _melscale_fbanks(n_freqs, f_min, f_max, n_mels, sr):
    all_freqs = torch.linspace(0, sr // 2, n_freqs)
    m_pts = torch.linspace(_hz_to_mel_htk(f_min), _hz_to_mel_htk(f_max), n_mels + 2)
    f_pts = torch.tensor([_mel_to_hz_htk(m.item()) for m in m_pts])
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)   # (n_freqs, n_mels+2)
    down = -slopes[:, :-2] / f_diff[:-1]
    up   = slopes[:, 2:]  / f_diff[1:]
    return torch.clamp(torch.min(down, up), min=0.0)       # (n_freqs, n_mels)


_MEL_WINDOW = torch.hann_window(WIN_LENGTH)
_MEL_FB     = _melscale_fbanks(N_FFT // 2 + 1, 20, 8000, N_MELS, SR)

# Mel band-center frequencies (Hz) for the Grad-CAM heatmap's frequency axis — the same
# htk mel points as _melscale_fbanks (f_min=20, f_max=8000), inner 80 centers.
_MEL_M_PTS   = torch.linspace(_hz_to_mel_htk(20), _hz_to_mel_htk(8000), N_MELS + 2)
MEL_FREQS_HZ = [round(_mel_to_hz_htk(m.item()), 1) for m in _MEL_M_PTS[1:-1]]


def wav_to_mel(wav_1d):
    """wav_1d: 1-D float tensor (CHUNK samples). Returns (1,1,80,T') mel, per-
    sample standardized — matches the LCNN training/teacher transform."""
    spec = torch.stft(wav_1d, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      win_length=WIN_LENGTH, window=_MEL_WINDOW,
                      center=True, pad_mode="reflect", return_complex=True)
    power = spec.abs() ** 2                                       # (freq, time)
    mel = torch.matmul(power.transpose(0, 1), _MEL_FB).transpose(0, 1)  # (80, T')
    mel_db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))      # AmplitudeToDB(power)
    mel_db = torch.clamp(mel_db, min=float(mel_db.max()) - 80.0)  # top_db=80 floor
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)     # per-sample norm
    return mel_db.unsqueeze(0).unsqueeze(0)


# ── Resolve the active model bundle (falls back to hardcoded V9 during rollout) ──
from bundle_registry import Registry as _Registry

def _resolve_bundle_paths():
    """Return (paths_dict, version, manifest). Uses the active registry bundle
    if present; otherwise falls back to the legacy models/*_v9* layout.

    $VOICEGUARD_FORCE_BUNDLE overrides the active pointer with a specific
    registered version, WITHOUT touching ACTIVE.json. This lets the promotion
    health gate (scripts/submodel_health.py via bundle_registry) load and probe
    a *candidate* bundle before it is promoted — a pre-promotion check with no
    live-pointer flip. Never set it on a serving process."""
    forced = os.environ.get("VOICEGUARD_FORCE_BUNDLE")
    try:
        reg = _Registry()
        version = forced or reg.get_active()
        if forced:
            print(f"  [FORCE_BUNDLE] overriding active pointer -> {forced}")
    except Exception as e:
        print(f"  registry unavailable ({e}); using legacy model paths"); version = None
    if version:
        entry = reg.get_bundle(version)
        if entry is not None:
            d = entry["dir"]
            paths = {
                "aasist": f"{d}/aasist.pt", "wav2vec": f"{d}/wav2vec.pt",
                "rawnet": f"{d}/rawnet.pt", "xgb": f"{d}/xgb.json",
                "cal": f"{d}/cal.json", "thresholds": f"{d}/thresholds.json",
                "lcnn": f"{d}/lcnn.pt",
            }
            return paths, version, entry["manifest"]
        else:
            print(f"  active version {version!r} missing from registry; using legacy model paths")
    # legacy fallback — AASIST uses V8 checkpoint (original architecture)
    return ({
        "aasist": f"{MODELS}/aasist_v8.pt", "wav2vec": f"{MODELS}/wav2vec_v9_best.pt",
        "rawnet": f"{MODELS}/rawnet3.pt", "xgb": f"{MODELS}/xgb_v9_hybrid.json",
        "cal": f"{MODELS}/cal_v9_hybrid.json", "thresholds": f"{MODELS}/thresholds_v9.json",
        "lcnn": f"{MODELS}/lcnn_screener_v9.pt",
    }, "v9-hybrid-legacy", None)

_BUNDLE_PATHS, ACTIVE_VERSION, _ACTIVE_MANIFEST = _resolve_bundle_paths()
print(f"  Active bundle: {ACTIVE_VERSION}")


# ══════════════════════════════════════════════════════════════════════════
#  Load V9 models + LCNN cascade
# ══════════════════════════════════════════════════════════════════════════
print("Loading V9 models...")
t0 = time.time()
torch.backends.cudnn.enabled = False   # RawNet3 GRU backward compat (harmless on CPU)


def _load_ckpt(path):
    return torch.load(path, map_location=DEVICE, weights_only=False)


def _state(ck, *keys):
    """Return the state dict stored under the first matching wrapper key."""
    if isinstance(ck, dict):
        for k in keys:
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck


# CHANGED: AASIST now uses original architecture (AASIST class, not AASIST_V9).
# Checkpoint wrapper key: 'model' (V8) or 'model_state_dict' (V9).
aasist = AASIST().to(DEVICE).eval()
aasist.load_state_dict(_state(_load_ckpt(_BUNDLE_PATHS["aasist"]),
                              "model", "model_state_dict"))
print(f"  AASIST       {time.time()-t0:.1f}s")

rawnet = RawNet3_V8().to(DEVICE).eval()
rawnet.load_state_dict(_state(_load_ckpt(_BUNDLE_PATHS["rawnet"]), "model_state", "model"))
print(f"  RawNet3 (V8) {time.time()-t0:.1f}s")

wav2vec = Wav2Vec2Classifier_V9().to(DEVICE).eval()
wav2vec.load_state_dict(_state(_load_ckpt(_BUNDLE_PATHS["wav2vec"]), "model", "model_state_dict"))
print(f"  Wav2Vec2 V9  {time.time()-t0:.1f}s")

# XGBoost V9 fusion (XGBClassifier → predict_proba) + logistic calibration
xgb_model = xgb_lib.XGBClassifier()
xgb_model.load_model(_BUNDLE_PATHS["xgb"])
with open(_BUNDLE_PATHS["cal"]) as f:
    cal_params = json.load(f)
ENS_CAL_COEF, ENS_CAL_INT = float(cal_params["coef"]), float(cal_params["intercept"])
with open(_BUNDLE_PATHS["thresholds"]) as f:
    thresholds = json.load(f)
print(f"  Ensemble V9  {time.time()-t0:.1f}s")

# LCNN stage-1 screener + cascade band + Platt calibration (all from checkpoint)
lcnn = LightCNN().to(DEVICE).eval()
_lcnn_ck = _load_ckpt(_BUNDLE_PATHS["lcnn"])
lcnn.load_state_dict(_state(_lcnn_ck, "model_state_dict"))
CASCADE_LOW   = float(_lcnn_ck["cascade_thresholds"]["low_thresh"])
CASCADE_HIGH  = float(_lcnn_ck["cascade_thresholds"]["high_thresh"])
LCNN_PLATT_C  = float(_lcnn_ck["platt_calibration"]["coef"])
LCNN_PLATT_I  = float(_lcnn_ck["platt_calibration"]["intercept"])
print(f"  LCNN screener {time.time()-t0:.1f}s")
print(f"  Cascade band: stage-1 resolves p<={CASCADE_LOW:.2f} or p>={CASCADE_HIGH:.2f}")
print(f"  Thresholds: to_review>={thresholds['to_review']:.2f}  "
      f"likely_fake>={thresholds['likely_fake']:.2f}  "
      f"auto_fake>={thresholds['auto_fake']:.2f}")
print()


# ══════════════════════════════════════════════════════════════════════════
#  Ensemble + cascade + helpers
# ══════════════════════════════════════════════════════════════════════════
def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def predict_ensemble(s_a, s_w, s_r):
    """V9 feature order [aasist, wav2vec, rawnet] → calibrated p(fake)."""
    xgb_prob = float(xgb_model.predict_proba(np.array([[s_a, s_w, s_r]]))[0, 1])
    return _sigmoid(ENS_CAL_COEF * xgb_prob + ENS_CAL_INT)


def lcnn_score(wav_1d):
    """Stage-1 screener → Platt-calibrated p(fake) for one CHUNK-length window."""
    with torch.no_grad():
        raw = torch.softmax(lcnn(wav_to_mel(wav_1d)), dim=-1)[0, 1].item()
    return _sigmoid(LCNN_PLATT_C * raw + LCNN_PLATT_I)


def ensemble_score_variants(chunk_3d):
    """Stage-2: full ensemble on a (1,1,CHUNK) chunk, averaging over any
    randomization variants (single clean pass when randomization is OFF)."""
    variants = randomize_chunk(
        chunk_3d,
        enabled=INPUT_RANDOMIZATION_ENABLED,
        mode=INPUT_RANDOMIZATION_MODE,
        sigma=INPUT_RANDOMIZATION_SIGMA,
        n_samples=INPUT_RANDOMIZATION_N_SAMPLES,
    )
    s_a_l, s_w_l, s_r_l = [], [], []
    for v in variants:
        with torch.no_grad():
            # All three models: class 1 = fake (standard orientation)
            # AASIST uses the original SincConv — no longer saturated
            s_a_l.append(torch.softmax(aasist(v),               dim=1)[0, 1].item())
            s_r_l.append(torch.softmax(rawnet(v),               dim=1)[0, 1].item())
            s_w_l.append(torch.softmax(wav2vec(v.squeeze(1)),   dim=1)[0, 1].item())
    s_a = sum(s_a_l) / len(s_a_l)
    s_w = sum(s_w_l) / len(s_w_l)
    s_r = sum(s_r_l) / len(s_r_l)
    return s_a, s_w, s_r, predict_ensemble(s_a, s_w, s_r)


def cascade_score_chunk(wav_1d):
    """Run the two-stage cascade on one CHUNK-length window (1-D tensor).
    Returns a dict with the final prob, which stage resolved it, the LCNN prob,
    and per-model scores (present only when stage 2 ran).

    Stage 1 (LCNN) uses the raw waveform: its mel is per-sample standardized,
    which is scale-invariant, so peak-normalization has no effect there.
    Stage 2 (ensemble) MUST peak-normalize: AASIST/Wav2Vec/RawNet weights and
    the XGBoost fusion + calibration (xgb_v9/cal_v9) were all fit on
    peak-normalized audio (retrain_*_v9.py, refit_ensemble_v9.py)."""
    lcnn_p = lcnn_score(wav_1d)
    if lcnn_p <= CASCADE_LOW or lcnn_p >= CASCADE_HIGH:
        return {'score': lcnn_p, 'stage': 1, 'lcnn': lcnn_p,
                'aasist': None, 'wav2vec': None, 'rawnet': None}
    # Peak-normalize for the ensemble (matches its training/calibration).
    peak = wav_1d.abs().max()
    ens_wav = wav_1d / peak if peak > 1e-8 else wav_1d
    chunk_3d = ens_wav.unsqueeze(0).unsqueeze(0)   # (1,1,CHUNK)
    s_a, s_w, s_r, ens = ensemble_score_variants(chunk_3d)
    return {'score': ens, 'stage': 2, 'lcnn': lcnn_p,
            'aasist': s_a, 'wav2vec': s_w, 'rawnet': s_r}


def cw_score(scores):
    """Confidence-weighted aggregation across chunks (unchanged from V7/V8)."""
    s = np.array(scores)
    w = np.abs(s - 0.5) * 2 + 0.1
    return float(np.average(s, weights=w))


def get_codec(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            capture_output=True, text=True, timeout=5)
        data   = json.loads(r.stdout)
        stream = next((s for s in data.get("streams", [])
                       if s.get("codec_type") == "audio"), {})
        return stream.get("codec_name", "unknown").lower()
    except Exception:
        return "unknown"


def verdict_from_score(score):
    """Map calibrated probability to verdict using V9 thresholds
    {auto_fake, likely_fake, to_review}."""
    if score >= thresholds['auto_fake']:
        return "AUTO_FAKE"
    elif score >= thresholds['likely_fake']:
        return "LIKELY_FAKE"
    elif score >= thresholds['to_review']:
        return "REVIEW"
    else:
        return "AUTO_REAL"


def _active_model_versions():
    """Which model weights produced this score, for the audit record: the active
    bundle version plus each sub-model checkpoint's short SHA-256 from the active
    manifest. Content hashes ARE the version identity here — the registry tracks
    weights by hash, not an integer counter (torch.save is non-deterministic, so
    a byte hash is the only stable fingerprint)."""
    m = _ACTIVE_MANIFEST or {}
    files = m.get("files", {})
    versions = {"bundle": ACTIVE_VERSION}
    for fname in ("lcnn.pt", "aasist.pt", "wav2vec.pt", "rawnet.pt", "xgb.json"):
        sha = files.get(fname, {}).get("sha256")
        if sha:
            versions[fname.split(".")[0]] = sha[:12]
    return versions


def _write_audit_log(audit_id, intake_sha256, verdict, score, model_versions):
    """Append one tamper-evident, hash-chained entry to the governance audit log
    (governance/audit_log.jsonl) for this detection. Each entry embeds the prior
    entry's hash, so any later edit, deletion, or reorder is detectable with
    `python governance.py verify-chain` — a plain jsonl line (the earlier design)
    could be edited undetectably. Never fatal: an audit-write failure must not
    fail a detection the caller already paid for.

    Chain integrity assumes a single writer. Real detections run only in the
    sequential worker (worker.py), which processes one job at a time; the API
    workers enqueue but never call detect(), and startup_check() passes
    audit=False so concurrent boots don't interleave appends here."""
    try:
        import governance
        governance.AuditLog().append(
            audit_id=audit_id,
            intake_sha256=intake_sha256,
            model_versions=model_versions,
            score=round(float(score), 4),
            verdict=verdict,
        )
    except Exception as e:
        print(f"  audit log write failed: {e}")


def merge_chunks_to_segments(chunk_results):
    """Merge adjacent chunks with the same verdict into segments.
    chunk_results: list of dicts with 'start_sec','end_sec','score','verdict'."""
    if not chunk_results:
        return []
    segments = []
    cur = {'start_sec': chunk_results[0]['start_sec'],
           'end_sec':   chunk_results[0]['end_sec'],
           'verdict':   chunk_results[0]['verdict'],
           'scores':    [chunk_results[0]['score']]}
    for c in chunk_results[1:]:
        if c['verdict'] == cur['verdict']:
            cur['end_sec'] = c['end_sec']; cur['scores'].append(c['score'])
        else:
            segments.append({'start_sec': round(cur['start_sec'], 2),
                             'end_sec':   round(cur['end_sec'], 2),
                             'score':     round(float(np.mean(cur['scores'])), 4),
                             'verdict':   cur['verdict']})
            cur = {'start_sec': c['start_sec'], 'end_sec': c['end_sec'],
                   'verdict': c['verdict'], 'scores': [c['score']]}
    segments.append({'start_sec': round(cur['start_sec'], 2),
                     'end_sec':   round(cur['end_sec'], 2),
                     'score':     round(float(np.mean(cur['scores'])), 4),
                     'verdict':   cur['verdict']})
    return segments


# ── Phase 4 non-acoustic signals (independent; do not change the verdict) ────
try:
    from watermark_detection import detect_watermark as detect_audioseal
    AUDIOSEAL_AVAILABLE = True
    print("Watermark detection module loaded.")
except Exception as e:
    AUDIOSEAL_AVAILABLE = False; detect_audioseal = None
    print(f"Watermark detection unavailable: {e}")

try:
    from metadata_forensics import analyze_metadata
    METADATA_AVAILABLE = True
    print("Metadata forensics module loaded.")
except Exception as e:
    METADATA_AVAILABLE = False; analyze_metadata = None
    print(f"Metadata forensics unavailable: {e}")

try:
    from c2pa_validation import validate_c2pa
    C2PA_AVAILABLE = True
    print("C2PA validation module loaded.")
except Exception as e:
    C2PA_AVAILABLE = False; validate_c2pa = None
    print(f"C2PA validation unavailable: {e}")

try:
    from mic_signature import analyze_mic_signature
    MIC_SIG_AVAILABLE = True
    print("Microphone signature module loaded.")
except Exception as e:
    MIC_SIG_AVAILABLE = False; analyze_mic_signature = None
    print(f"Microphone signature unavailable: {e}")

# Adversarial-attack risk monitor (advisory; reuses the loaded ensemble models).
# On by default; set VOICEGUARD_ADVERSARIAL=0 to disable. It adds latency on
# 'real'-verdict audio (the perturbation path), so it runs in the worker only.
try:
    from adversarial_monitor_v2 import AdversarialMonitor
    ADVERSARIAL_ENABLED = os.environ.get("VOICEGUARD_ADVERSARIAL", "1") != "0"
    _adv_monitor = (AdversarialMonitor(aasist, wav2vec, rawnet, xgb_model,
                                       ENS_CAL_COEF, ENS_CAL_INT, device=DEVICE)
                    if ADVERSARIAL_ENABLED else None)
    ADVERSARIAL_AVAILABLE = _adv_monitor is not None
    print("Adversarial monitor " + ("loaded." if ADVERSARIAL_AVAILABLE
                                     else "disabled (VOICEGUARD_ADVERSARIAL=0)."))
except Exception as e:
    ADVERSARIAL_AVAILABLE = False; _adv_monitor = None
    print(f"Adversarial monitor unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  Detection
# ══════════════════════════════════════════════════════════════════════════
def detect(file_path, audit=True):
    """V9 cascade detection over a chunked long-audio pipeline.

    audit=True writes a tamper-evident audit-log entry for this detection (the
    live-path default). Internal calls that are not customer detections — e.g.
    the startup smoke-check — pass audit=False so they don't pollute the
    chain-of-custody log or interleave appends across concurrent boots."""
    t0 = time.time()
    codec = get_codec(file_path)

    # Metadata forensics (Phase 4)
    if METADATA_AVAILABLE:
        try:
            metadata_result = analyze_metadata(file_path)
        except Exception as e:
            metadata_result = {'available': False, 'findings': [], 'lean': 'neutral',
                               'summary': 'Metadata analysis failed', 'error': str(e)}
    else:
        metadata_result = {'available': False, 'findings': [], 'lean': 'neutral',
                           'summary': 'Metadata analysis unavailable', 'error': None}

    # C2PA content credentials (Phase 4)
    if C2PA_AVAILABLE:
        try:
            c2pa_result = validate_c2pa(file_path)
        except Exception as e:
            c2pa_result = {'available': False, 'has_credentials': False,
                           'format_supported': False, 'error': str(e),
                           'message': 'C2PA validation failed'}
    else:
        c2pa_result = {'available': False, 'has_credentials': False,
                       'format_supported': False, 'error': 'Module not loaded',
                       'message': 'C2PA validation unavailable'}

    # ── Load + decode audio (ffmpeg → 16k mono → float; NO peak normalization) ──
    wav_path = file_path + "_converted.wav"
    try:
        result = subprocess.run(
            [FFMPEG, "-y", "-i", file_path,
             "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", wav_path],
            capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg error: " + result.stderr.decode()[-200:])
        from scipy.io import wavfile as scipy_wav
        rate, data = scipy_wav.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"Audio loading failed ({os.path.basename(file_path)}): {e}")
    finally:
        try: os.unlink(wav_path)
        except Exception: pass

    y = wav.squeeze(0).numpy()

    # AudioSeal watermark (Phase 4)
    if AUDIOSEAL_AVAILABLE:
        try:
            watermark_result = detect_audioseal(y, SR)
        except Exception as e:
            watermark_result = {'available': False, 'detected': False,
                                'error': str(e), 'message': 'Watermark check failed'}
    else:
        watermark_result = {'available': False, 'detected': False,
                            'error': 'Module not loaded',
                            'message': 'Watermark detection unavailable'}

    # Microphone signature (Phase 4)
    if MIC_SIG_AVAILABLE:
        try:
            mic_sig_result = analyze_mic_signature(y, SR)
        except Exception as e:
            mic_sig_result = {'available': False, 'findings': [], 'lean': 'neutral',
                              'summary': '', 'error': str(e)}
    else:
        mic_sig_result = {'available': False, 'findings': [], 'lean': 'neutral',
                          'summary': '', 'error': 'Module not loaded'}

    silence_ratio = float((np.abs(y) < 0.005).mean())
    duration      = wav.shape[-1] / SR

    # ── Chunk into 4s windows with 50% overlap ──────────────────────────────
    chunk_samples = int(4.0 * SR)
    hop           = int(chunk_samples * 0.5)
    total         = wav.shape[-1]
    chunks, start = [], 0
    while start < total:
        end = min(start + chunk_samples, total)
        chunks.append(wav[:, start:end])
        if end == total:
            break
        start += hop
    if not chunks:
        chunks = [wav]

    # ── Cascade-score every chunk ────────────────────────────────────────────
    aa_s, w2v_s, rn_s, lcnn_s, ensemble_scores = [], [], [], [], []
    chunk_results = []          # per-chunk detail for segment-level reporting
    stage1_count = stage2_count = 0
    peak_score, peak_wav_1d, peak_range = -1.0, None, (0.0, 0.0)
    peak2_score, peak2_feats, peak2_range = -1.0, None, (0.0, 0.0)

    for chunk_idx, chunk in enumerate(chunks):
        chunk_start_sec = (chunk_idx * hop) / SR
        chunk_end_sec   = min(chunk_start_sec + 4.0, duration)

        n = chunk.shape[-1]
        chunk = F.pad(chunk, (0, CHUNK - n)) if n < CHUNK else chunk[:, :CHUNK]
        # Raw chunk (no peak-norm here); cascade_score_chunk peak-normalizes
        # internally for the stage-2 ensemble only (see its docstring).
        wav_1d = chunk.squeeze(0).to(DEVICE)     # (CHUNK,)

        r = cascade_score_chunk(wav_1d)
        if r['score'] > peak_score:
            peak_score, peak_wav_1d = r['score'], wav_1d
            peak_range = (chunk_start_sec, chunk_end_sec)
        if r['stage'] == 2 and r['score'] > peak2_score:
            peak2_score = r['score']
            peak2_feats = (r['aasist'], r['wav2vec'], r['rawnet'])
            peak2_range = (chunk_start_sec, chunk_end_sec)
        ensemble_scores.append(r['score'])
        lcnn_s.append(r['lcnn'])
        if r['stage'] == 1:
            stage1_count += 1
        else:
            stage2_count += 1
            aa_s.append(r['aasist']); w2v_s.append(r['wav2vec']); rn_s.append(r['rawnet'])

        chunk_results.append({
            'start_sec': chunk_start_sec, 'end_sec': chunk_end_sec,
            'score': r['score'], 'verdict': verdict_from_score(r['score']),
            'stage': r['stage'],
        })

    score = cw_score(ensemble_scores)

    policy = "standard"
    if silence_ratio > 0.55 and score >= thresholds['likely_fake']:
        score  = min(score, thresholds['likely_fake'] - 0.01)
        policy = "silence-cap"

    verdict = verdict_from_score(score)

    with open(file_path, "rb") as f:
        sha256_full = hashlib.sha256(f.read()).hexdigest()
    sha256 = sha256_full[:16] + "..."          # truncated for the response display

    audit_id   = explain_signals.new_audit_id()
    confidence = explain_signals.compute_confidence([c['score'] for c in chunk_results], score)

    heatmap = None
    if peak_wav_1d is not None:
        try:
            heatmap = gradcam.lcnn_gradcam(lcnn, wav_to_mel, peak_wav_1d, MEL_FREQS_HZ,
                                           HOP_LENGTH / SR, peak_range)
        except Exception as e:
            print(f"  gradcam failed: {e}")

    shap = None
    if peak2_feats is not None:
        try:
            booster = xgb_model.get_booster()
            # the booster was fit with named features; pred_contribs rejects an
            # unnamed DMatrix, so pass the booster's own feature_names to match.
            dm = xgb_lib.DMatrix(np.array([list(peak2_feats)], dtype=float),
                                 feature_names=booster.feature_names)
            contribs = booster.predict(dm, pred_contribs=True)[0]
            shap = explain_signals.shap_from_contribs(contribs)
            shap["chunk_range_sec"] = [round(peak2_range[0], 2), round(peak2_range[1], 2)]
            shap["note"] = "margin-space contributions from the XGBoost fusion (pre-calibration)"
        except Exception as e:
            print(f"  shap failed: {e}")

    # Adversarial-attack risk assessment on the peak chunk (advisory, non-fatal).
    # Peak-normalize to match the ensemble path the monitor scores against.
    adversarial = None
    if _adv_monitor is not None and peak_wav_1d is not None:
        try:
            _pk = peak_wav_1d.abs().max()
            _adv_wav = peak_wav_1d / _pk if _pk > 1e-8 else peak_wav_1d
            adversarial = _adv_monitor.assess(_adv_wav.to(DEVICE))
        except Exception as e:
            print(f"  adversarial monitor failed: {e}")

    if audit:
        # Full intake hash into the durable audit record (not the truncated display).
        _write_audit_log(audit_id, sha256_full, verdict, score, _active_model_versions())

    # Explainability (Phase 3)
    try:
        from explainability import explain_audio
        explanation = explain_audio(y, SR, verdict=verdict)
    except Exception as e:
        explanation = {'observations': [], 'measurements': {}, 'summary': '',
                       'error': str(e)}

    segments = merge_chunks_to_segments(chunk_results)
    flagged_segments = [s for s in segments
                        if s['verdict'] in ('REVIEW', 'LIKELY_FAKE', 'AUTO_FAKE')]

    mean_lcnn = float(np.mean(lcnn_s)) if lcnn_s else None
    mean_aa   = float(np.mean(aa_s))  if aa_s  else None
    mean_w2v  = float(np.mean(w2v_s)) if w2v_s else None
    mean_rn   = float(np.mean(rn_s))  if rn_s  else None
    n_chunks  = len(chunks)

    return {
        "verdict":       verdict,
        "score":         round(score, 4),
        "pct":           round(score * 100, 1),
        "duration":      round(duration, 1),
        "chunks":        n_chunks,
        "codec":         codec,
        "silence_ratio": round(silence_ratio, 3),
        # Per-model means are over stage-2 (escalated) chunks only; null if every
        # chunk resolved at stage 1. LCNN screener mean is always present.
        "lcnn":          round(mean_lcnn * 100, 1) if mean_lcnn is not None else None,
        "w2v":           round(mean_w2v * 100, 1) if mean_w2v is not None else None,
        "aasist":        round(mean_aa  * 100, 1) if mean_aa  is not None else None,
        "rawnet":        round(mean_rn  * 100, 1) if mean_rn  is not None else None,
        "ensemble":      round(score    * 100, 1),
        "xgb":           round(score    * 100, 1),   # backwards-compat for V7/V8 UI
        "cascade": {
            "stage1_chunks":  stage1_count,
            "stage2_chunks":  stage2_count,
            "resolution_pct": round(100.0 * stage1_count / max(n_chunks, 1), 1),
            "band":           [CASCADE_LOW, CASCADE_HIGH],
        },
        "policy":        policy,
        "model_version": "V9",
        "sha256":        sha256,
        "elapsed":       round(time.time() - t0, 2),
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "explanation":      explanation,
        "audit_id":     audit_id,
        "confidence":   confidence,
        "heatmap":      heatmap,
        "shap":         shap,
        "adversarial":  adversarial,
        "segments":         segments,
        "flagged_segments": flagged_segments,
        "watermark":        watermark_result,
        "metadata":         metadata_result,
        "c2pa":             c2pa_result,
        "mic_signature":    mic_sig_result,
    }


def startup_check():
    """Score the active bundle's smoke clip; on mismatch/failure, log CRITICAL.
    Returns True if OK or if no smoke_check is declared."""
    sc = (_ACTIVE_MANIFEST or {}).get("smoke_check")
    if not sc:
        return True
    clip = os.path.join(BASE, sc["clip"])
    if not os.path.exists(clip):
        print(f"  [smoke-check] clip not found ({clip}); skipping"); return True
    try:
        v = detect(clip, audit=False)["verdict"]
    except Exception as e:
        print(f"  [smoke-check] CRITICAL: scoring failed: {e}"); return False
    ok = v in sc["expected_verdicts"]
    print(f"  [smoke-check] {clip} -> {v} "
          f"({'OK' if ok else 'MISMATCH, expected ' + str(sc['expected_verdicts'])})")
    return ok
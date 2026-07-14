"""diag_cascade.py — for one file, run BOTH the LCNN screener AND the full ensemble
on EVERY chunk (even chunks the screener would fast-resolve), so we can see whether
the ensemble would have caught fakes the screener passed straight to "real".

Usage:  python diag_cascade.py "C:\\path\\to\\audio.mp3"
"""
import sys, os, subprocess
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # use the cached wav2vec2-base; no network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F
import detector as D

if len(sys.argv) < 2:
    print('usage: python diag_cascade.py "C:\\path\\to\\audio.mp3"')
    raise SystemExit(1)

path = sys.argv[1]
wav_path = path + "_diag.wav"
subprocess.run(["C:/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe", "-y", "-i", path,
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", wav_path],
               capture_output=True, timeout=30)
from scipy.io import wavfile as _wav
rate, data = _wav.read(wav_path)
try: os.unlink(wav_path)
except Exception: pass
if data.ndim > 1:
    data = data.mean(axis=1)
data = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
wav = torch.tensor(data, dtype=torch.float32)

total = wav.shape[-1]
cs, hop = int(4.0 * D.SR), int(4.0 * D.SR * 0.5)
print(f"\nfile: {os.path.basename(path)}   band=[{D.CASCADE_LOW:.2f},{D.CASCADE_HIGH:.2f}]")
print(f"{'chunk':<5} {'t(s)':<6} {'LCNN%':<7} {'cascade-did':<12} {'AASIST%':<8} {'Wav2Vec%':<9} {'RawNet%':<8} {'ENSEMBLE%':<9}")
i, start = 0, 0
missed = 0
while start < total:
    chunk = wav[start:start + cs]
    n = chunk.shape[-1]
    chunk = F.pad(chunk, (0, D.CHUNK - n)) if n < D.CHUNK else chunk[:D.CHUNK]
    wav_1d = chunk.to(D.DEVICE)
    lcnn_p = D.lcnn_score(wav_1d)
    peak = wav_1d.abs().max()
    ew = wav_1d / peak if peak > 1e-8 else wav_1d
    s_a, s_w, s_r, ens = D.ensemble_score_variants(ew.unsqueeze(0).unsqueeze(0))
    did = "REAL(s1)" if lcnn_p <= D.CASCADE_LOW else ("FAKE(s1)" if lcnn_p >= D.CASCADE_HIGH else "escalate")
    flag = ""
    if did == "REAL(s1)" and ens >= 0.55:          # screener said real, ensemble would flag fake
        flag = "  <== ENSEMBLE WOULD FLAG"
        missed += 1
    print(f"{i:<5} {start/D.SR:<6.1f} {lcnn_p*100:<7.1f} {did:<12} {s_a*100:<8.1f} {s_w*100:<9.1f} {s_r*100:<8.1f} {ens*100:<9.1f}{flag}")
    i += 1
    start += hop
print(f"\nchunks the screener sent to REAL but the ensemble would flag fake: {missed}")
if missed:
    print("=> the screener's 'real' early-out is bypassing the ensemble on this file.")

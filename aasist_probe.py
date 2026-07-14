"""aasist_probe.py — is AASIST_V9 recoverable? Probe its raw logits on real vs fake clips.

The scoring code reads softmax(logits)[:,1] as p(fake). We check whether class1 (or the
logit margin) actually separates real from fake, or whether the model has collapsed to one
class regardless of input.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import random
import numpy as np
import torch
import torch.nn.functional as F
import detector as D
import sweep_cascade as SW          # reuse _expand + _load


def aasist_logits(p):
    w = SW._load(p)
    c = w[:D.CHUNK]
    c = F.pad(c, (0, D.CHUNK - c.shape[-1])) if c.shape[-1] < D.CHUNK else c
    peak = c.abs().max()
    ew = c / peak if peak > 1e-8 else c
    with torch.no_grad():
        lg = D.aasist(ew.unsqueeze(0).unsqueeze(0))[0]
    p_fake = torch.softmax(lg, dim=0)[1].item()
    return float(lg[0]), float(lg[1]), p_fake


def summarize(files, lbl):
    rows = []
    for f in files:
        try:
            rows.append(aasist_logits(f))
        except Exception:
            pass
    l0 = np.array([r[0] for r in rows]); l1 = np.array([r[1] for r in rows])
    margin = l1 - l0                    # class1 - class0 ; code treats class1 as "fake"
    pf = np.array([r[2] for r in rows])
    print(f"{lbl:5} n={len(rows):3}  logit0(mean±sd)={l0.mean():7.2f}±{l0.std():5.2f}  "
          f"logit1={l1.mean():7.2f}±{l1.std():5.2f}  margin(l1-l0)={margin.mean():7.2f}±{margin.std():5.2f}  "
          f"p_fake(mean)={pf.mean():.4f}")
    return margin


random.seed(0)
real = SW._expand(["studio_clips", "bias_audit/real"])
fake = SW._expand(["studio_fake_test", "bias_audit_fakes", "bias_audit/fake"])
real = random.sample(real, min(20, len(real)))
fake = random.sample(fake, min(20, len(fake)))

print("AASIST logit probe (code uses class1 = fake):")
mr = summarize(real, "REAL")
mf = summarize(fake, "FAKE")

# Would ANY margin threshold separate them? report best split accuracy on l1-l0.
allm = np.concatenate([mr, mf])
labels = np.concatenate([np.zeros(len(mr)), np.ones(len(mf))])
best_acc, best_thr = 0.0, 0.0
for thr in np.unique(allm):
    acc = max(((allm >= thr) == labels).mean(), ((allm < thr) == labels).mean())
    if acc > best_acc:
        best_acc, best_thr = acc, thr
print(f"\nbest single-threshold separation on margin(l1-l0): acc={best_acc:.2f} at thr={best_thr:.2f}")
print("acc~0.5 => collapsed (no signal, retrain). acc>>0.5 => signal exists (readout/calibration recoverable).")

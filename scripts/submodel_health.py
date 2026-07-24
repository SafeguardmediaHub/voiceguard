#!/usr/bin/env python3
"""submodel_health.py — per-sub-model health gate for the active bundle.

WHY THIS EXISTS
---------------
Every other VoiceGuard control validates either FILE IDENTITY (SHA-256 in
bundle_registry) or ENSEMBLE OUTPUT (golden regression, startup smoke check,
drift monitor). None validates that each *individual* sub-model still
contributes discriminative signal.

That is exactly the gap the AASIST V9 collapse fell through: the checkpoint
loaded cleanly, hashes matched, the golden regression passed, and the startup
smoke check passed — while AASIST had collapsed to AUC 0.5258 (coin-flip) and
was contributing nothing to the fusion (see docs/MODEL_DEVELOPMENT_HISTORY.md
§12 and docs/GRC_CONTROL_PACK.md G1/R4).

This script turns the manual probe (aasist_probe.py) into a gate that scores
ALL FOUR models — LCNN, AASIST, Wav2Vec2, RawNet3 — on a fixed real/fake probe
set and checks each for the specific failure that actually occurred.

WHAT IT CHECKS — and why the primary check is SPREAD, not AUC
------------------------------------------------------------
The AASIST V9 failure was SATURATION: the model emitted a near-constant softmax
(fake_mean 0.9997±0.0007, real_mean 0.9995±0.0033 — notebook [423]) regardless
of input, so the XGBoost fusion saw a dead, constant feature. The detectable
signature of that is near-ZERO output SPREAD across inputs — the model says the
same thing no matter what you feed it.

So the gate has two tiers:

  * COLLAPSE (hard FAIL): output spread < --min-spread. A model that barely
    varies across real vs fake inputs is contributing nothing to the fusion.
    This is the exact AASIST V9 signature and the thing that must never ship
    undetected again. Exit code 1.

  * WEAK (WARN, not fail): the model varies (not collapsed) but its AUC on this
    probe is below --min-auc. This is EXPECTED for deliberately down-weighted
    members — RawNet3 carries only 0.153 fusion weight and is individually weak
    by design. A warning surfaces it for human judgement without blocking a
    known-good bundle. Exit code still 0 if nothing collapsed.

This distinction matters: the deployed, accepted v9h bundle has individually
weak sub-models (AASIST ~0.54, RawNet3 ~0.49 AUC on the studio probe) that are
NOT collapsed (spread 0.2–0.4). A gate that hard-failed on AUC would reject the
very bundle in production. A gate that hard-fails on spread catches the real
failure and lets the accepted bundle through.

USAGE
-----
    python scripts/submodel_health.py                    # human-readable report
    python scripts/submodel_health.py --json             # machine-readable
    python scripts/submodel_health.py --min-spread 0.05  # collapse floor
    python scripts/submodel_health.py --min-auc 0.65     # weak-warning floor

Exit code 0 = no sub-model collapsed; 1 = at least one collapsed (or an error).
Intended to run in the model-promotion gate BEFORE bundle_registry.py promote.
"""
import argparse
import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import detector as D
import sweep_cascade as SW      # reuse _expand (folder->files) + _load (file->waveform)

# Default real/fake probe sources (local folders). Kept identical to
# aasist_probe.py so the gate and the manual probe agree.
REAL_SOURCES = ["studio_clips", "bias_audit/real"]
FAKE_SOURCES = ["studio_fake_test", "bias_audit_fakes", "bias_audit/fake"]

# Collapse floor: a saturated model outputs a near-constant value -> spread ~0.
# The collapsed AASIST V9 had softmax std ~0.003; every healthy v9h sub-model
# measured here has spread >= 0.22. 0.05 sits unambiguously between them.
DEFAULT_MIN_SPREAD = 0.05
# Weak-warning floor (does NOT fail the gate): flags individually-weak members
# for human judgement. A collapsed model sits at ~0.50.
DEFAULT_MIN_AUC = 0.65
DEFAULT_N_PER_CLASS = 20
PROBE_SEED = 0              # fixed set -> reproducible gate


def _auc(scores, labels):
    """AUC via the rank-sum (Mann-Whitney U) identity. No sklearn dependency.
    labels: 1 = fake (positive), 0 = real. Returns 0.5 for degenerate input."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties, so a saturated (all-equal) model scores exactly 0.5
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank_per_group = (start + cum + 1) / 2.0
    ranks = avg_rank_per_group[inv]
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _prep_chunk(path):
    """Load a file and return a peak-normalized (1,1,CHUNK) tensor for the ensemble
    models, plus the raw CHUNK-length 1-D waveform for the (scale-invariant) LCNN."""
    w = SW._load(path)                       # 1-D waveform at detector.SR
    c = w[:D.CHUNK]
    if c.shape[-1] < D.CHUNK:
        c = torch.nn.functional.pad(c, (0, D.CHUNK - c.shape[-1]))
    peak = c.abs().max()
    ens = c / peak if peak > 1e-8 else c     # peak-norm: matches ensemble training/calibration
    return ens.unsqueeze(0).unsqueeze(0), c  # (1,1,CHUNK) for ensemble, 1-D for LCNN


def _model_scores(files):
    """Return {model: [p_fake, ...]} for each of the four models over `files`.
    Any file that fails to load is skipped for all models consistently."""
    out = {"lcnn": [], "aasist": [], "wav2vec": [], "rawnet": []}
    used = 0
    for f in files:
        try:
            ens_3d, raw_1d = _prep_chunk(f)
        except Exception:
            continue
        with torch.no_grad():
            out["lcnn"].append(float(D.lcnn_score(raw_1d)))
            out["aasist"].append(torch.softmax(D.aasist(ens_3d), dim=1)[0, 1].item())
            out["rawnet"].append(torch.softmax(D.rawnet(ens_3d), dim=1)[0, 1].item())
            out["wav2vec"].append(torch.softmax(D.wav2vec(ens_3d.squeeze(1)), dim=1)[0, 1].item())
        used += 1
    return out, used


def run_health_check(min_spread=DEFAULT_MIN_SPREAD, min_auc=DEFAULT_MIN_AUC,
                     n_per_class=DEFAULT_N_PER_CLASS,
                     real_sources=None, fake_sources=None):
    """Score all four sub-models on a fixed real/fake probe set.

    A model is COLLAPSED (hard fail) if its output spread < min_spread — it emits
    a near-constant value regardless of input, feeding the fusion a dead feature.
    A model is WEAK (warning only) if it varies but its AUC < min_auc.

    Returns a dict:
      {
        "active_version": str, "min_spread": float, "min_auc": float,
        "n_real": int, "n_fake": int,
        "models": {name: {"auc", "fake_mean", "real_mean", "spread",
                          "collapsed": bool, "weak": bool}},
        "passed": bool,            # False iff any model collapsed
        "collapsed": [name, ...],  # hard failures
        "weak": [name, ...],       # warnings, do not fail the gate
      }
    """
    random.seed(PROBE_SEED)
    real = SW._expand(real_sources or REAL_SOURCES)
    fake = SW._expand(fake_sources or FAKE_SOURCES)
    if not real or not fake:
        raise RuntimeError(
            f"probe set empty (real={len(real)}, fake={len(fake)}); "
            "run from the repo root where the probe folders exist")
    real = random.sample(real, min(n_per_class, len(real)))
    fake = random.sample(fake, min(n_per_class, len(fake)))

    real_s, n_real = _model_scores(real)
    fake_s, n_fake = _model_scores(fake)

    models = {}
    collapsed = []
    weak = []
    for name in ("lcnn", "aasist", "wav2vec", "rawnet"):
        rs, fs = real_s[name], fake_s[name]
        scores = np.array(rs + fs)
        labels = np.array([0] * len(rs) + [1] * len(fs))
        auc = _auc(scores, labels)
        fake_mean = float(np.mean(fs)) if fs else float("nan")
        real_mean = float(np.mean(rs)) if rs else float("nan")
        # Spread across BOTH classes. A saturated/collapsed model outputs a
        # near-constant value regardless of input -> spread ~0. This is the
        # primary, unambiguous collapse signal.
        spread = float(np.std(scores)) if len(scores) else 0.0
        # bool() casts away numpy.bool_ (not JSON-serializable) from the comparisons.
        is_collapsed = bool(spread < min_spread)
        # "weak" only meaningful when not collapsed (a collapsed model is trivially
        # also below the AUC floor, but we report it as collapsed, not weak).
        is_weak = bool((not is_collapsed) and (auc < min_auc))
        models[name] = {"auc": round(float(auc), 4), "fake_mean": round(fake_mean, 4),
                        "real_mean": round(real_mean, 4), "spread": round(float(spread), 4),
                        "collapsed": is_collapsed, "weak": is_weak}
        if is_collapsed:
            collapsed.append(name)
        elif is_weak:
            weak.append(name)

    return {
        "active_version": D.ACTIVE_VERSION,
        "min_spread": min_spread, "min_auc": min_auc,
        "n_real": n_real, "n_fake": n_fake,
        "models": models,
        "passed": len(collapsed) == 0,
        "collapsed": collapsed,
        "weak": weak,
    }


def _print_report(result):
    print(f"Per-sub-model health check — active bundle: {result['active_version']}")
    print(f"Probe set: {result['n_real']} real / {result['n_fake']} fake   "
          f"collapse floor (spread): {result['min_spread']}   weak floor (AUC): {result['min_auc']}")
    print()
    print(f"  {'model':10s} {'AUC':>7s}  {'real_mean':>9s}  {'fake_mean':>9s}  {'spread':>7s}  verdict")
    for name, m in result["models"].items():
        if m["collapsed"]:
            verdict = "*** COLLAPSED ***"
        elif m["weak"]:
            verdict = "weak (warn)"
        else:
            verdict = "OK"
        print(f"  {name:10s} {m['auc']:7.4f}  {m['real_mean']:9.4f}  "
              f"{m['fake_mean']:9.4f}  {m['spread']:7.4f}  {verdict}")
    print()
    if result["weak"]:
        print(f"WARN: individually weak on this probe (not a gate failure): {', '.join(result['weak'])}")
        print("      Expected for down-weighted members (e.g. RawNet3, fusion weight 0.153).")
    if result["passed"]:
        print("RESULT: PASS — no sub-model has collapsed; every model varies with input.")
    else:
        print(f"RESULT: FAIL — COLLAPSED: {', '.join(result['collapsed'])}")
        print("      A near-constant output feeds the fusion a dead feature — the AASIST V9")
        print("      failure mode (see docs/MODEL_DEVELOPMENT_HISTORY.md §12). Do not promote.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-sub-model health gate for the active bundle")
    ap.add_argument("--min-spread", type=float, default=DEFAULT_MIN_SPREAD,
                    help=f"collapse floor: fail if a model's output spread is below this "
                         f"(default {DEFAULT_MIN_SPREAD})")
    ap.add_argument("--min-auc", type=float, default=DEFAULT_MIN_AUC,
                    help=f"weak-warning floor: warn (do not fail) if AUC is below this "
                         f"(default {DEFAULT_MIN_AUC})")
    ap.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS,
                    help=f"probe files per class (default {DEFAULT_N_PER_CLASS})")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    try:
        result = run_health_check(min_spread=args.min_spread, min_auc=args.min_auc,
                                  n_per_class=args.n_per_class)
    except Exception as e:
        print(f"health check error: {e}", file=sys.stderr)
        return 1

    if args.json:
        # Single-line, sentinel-prefixed: detector's import prints a lot of noise to
        # stdout, so a parent process greps for this exact prefix rather than trying
        # to parse the whole stream. (indent=2 would also break last-line parsing.)
        print("HEALTH_RESULT_JSON " + json.dumps(result))
    else:
        _print_report(result)
    return 0 if result["passed"] else 1


HEALTH_JSON_PREFIX = "HEALTH_RESULT_JSON "


if __name__ == "__main__":
    raise SystemExit(main())

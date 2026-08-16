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

WHICH PROBE SET IT USES
-----------------------
Auto-selected, richest first:

  1. The large local corpora (studio_clips, bias_audit_fakes, ...) when present.
     Labelled and plentiful, so the AUC tier is meaningful. This is a dev machine.
  2. Otherwise the small fixed set shipped inside the image
     (tests/probe_clips, built by scripts/build_probe_set.py). This is the
     container, where the corpora above are excluded from the build context.

Before the shipped set existed, case 2 always exited "cannot certify" and the only
way to promote in production was --skip-health — the exact habit this gate exists
to prevent (REMEDIATION_PLAN D9).

USAGE
-----
    python scripts/submodel_health.py                    # human-readable report
    python scripts/submodel_health.py --json             # machine-readable
    python scripts/submodel_health.py --min-spread 0.05  # collapse floor
    python scripts/submodel_health.py --min-auc 0.65     # weak-warning floor
    python scripts/submodel_health.py --probe-set DIR    # force a specific set

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
# aasist_probe.py so the gate and the manual probe agree. These are large local
# corpora, excluded from the Docker build context — present on a dev machine,
# absent inside the image.
REAL_SOURCES = ["studio_clips", "bias_audit/real"]
FAKE_SOURCES = ["studio_fake_test", "bias_audit_fakes", "bias_audit/fake"]

# The fallback that makes the gate enforceable in production: a small fixed probe
# set that ships inside the image (built by scripts/build_probe_set.py) — CC0
# Common Voice reals plus the two fake clips already shipped for the golden
# regression test. It drives the COLLAPSE gate at full strength; with only 2 fakes
# the AUC weak-warning tier is reported as n/a (see MIN_FAKE_FOR_AUC). See that
# script for the licensing rationale and REMEDIATION_PLAN D9 for why shipping one
# was necessary at all.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_PROBE_DIR = os.path.join(REPO_ROOT, "tests", "probe_clips")

# Collapse floor: a saturated model outputs a near-constant value -> spread ~0.
# The collapsed AASIST V9 had softmax std ~0.003; every healthy v9h sub-model
# measured here has spread >= 0.22. 0.05 sits unambiguously between them.
DEFAULT_MIN_SPREAD = 0.05
# Weak-warning floor (does NOT fail the gate): flags individually-weak members
# for human judgement. A collapsed model sits at ~0.50.
DEFAULT_MIN_AUC = 0.65
DEFAULT_N_PER_CLASS = 20
PROBE_SEED = 0              # fixed set -> reproducible gate

# Below this many decoded fake clips, AUC is reported as n/a rather than computed.
# The shipped probe set carries 2 fakes; on those the four models "score" 0.87-1.00
# versus 0.49-0.69 on the full labelled corpora. Printing that would suggest the
# ensemble is near-perfect and contradict REMEDIATION_PLAN F1. The collapse gate
# needs only a second class to exist, not a statistically meaningful one.
MIN_FAKE_FOR_AUC = 10

# Importing detector floods stdout/stderr with model-loading noise, so the parent
# process (bundle_registry._run_health_gate) greps for these markers rather than
# trying to parse the whole stream. Single line each; indent=2 would break that.
HEALTH_JSON_PREFIX = "HEALTH_RESULT_JSON "
HEALTH_ERROR_PREFIX = "HEALTH_ERROR "


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


def _load_probe_dir(probe_dir):
    """(real_files, fake_files, description) from a probe dir with a manifest.json."""
    mpath = os.path.join(probe_dir, "manifest.json")
    if not os.path.exists(mpath):
        raise RuntimeError(f"probe set has no manifest.json: {probe_dir}")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    real, fake = [], []
    for entry in manifest.get("clips", []):
        path = os.path.join(probe_dir, entry["file"])
        if not os.path.exists(path):
            raise RuntimeError(f"probe set manifest lists a missing clip: {entry['file']}")
        (fake if entry.get("label") == "fake" else real).append(path)
    desc = (f"{os.path.relpath(probe_dir, os.path.dirname(SHIPPED_PROBE_DIR))} "
            f"[{manifest.get('corpus', 'unknown corpus')}, {manifest.get('licence', '?')}]")
    return real, fake, desc


def resolve_probe_sources(probe_dir=None, real_sources=None, fake_sources=None):
    """Pick the probe set, preferring the richest one available.

    Order matters. A dev machine holds the large local corpora and should keep
    using them — they are labelled, so the AUC tier stays meaningful and the
    numbers stay comparable with every measurement recorded to date. Only when
    those are absent (which is exactly the situation inside the image) does the
    gate fall back to the small shipped CC0 set. So this changes nothing about
    how the gate behaves on a workstation; it only stops it being unrunnable in
    the place it actually gates.
    """
    if probe_dir:
        return _load_probe_dir(probe_dir)

    rs = real_sources or REAL_SOURCES
    fs = fake_sources or FAKE_SOURCES
    real = SW._expand(rs)
    fake = SW._expand(fs)
    if real and fake:
        return real, fake, f"local corpora ({', '.join(rs)} / {', '.join(fs)})"

    if os.path.isdir(SHIPPED_PROBE_DIR):
        return _load_probe_dir(SHIPPED_PROBE_DIR)

    raise RuntimeError(
        f"no probe set available: local corpora absent (real={len(real)}, "
        f"fake={len(fake)}) and no shipped set at {SHIPPED_PROBE_DIR}")


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
    """Return ({model: [p_fake, ...]}, n_used, first_error) over `files`.
    Any file that fails to load is skipped for all models consistently, but the
    first failure is returned so the caller can tell "the probe set would not
    decode" apart from "the models collapsed" — those look identical downstream
    (both yield zero spread) and have opposite remedies."""
    out = {"lcnn": [], "aasist": [], "wav2vec": [], "rawnet": []}
    used = 0
    first_error = None
    for f in files:
        try:
            ens_3d, raw_1d = _prep_chunk(f)
        except Exception as e:
            if first_error is None:
                first_error = f"{os.path.basename(f)}: {e}"
            continue
        with torch.no_grad():
            out["lcnn"].append(float(D.lcnn_score(raw_1d)))
            out["aasist"].append(torch.softmax(D.aasist(ens_3d), dim=1)[0, 1].item())
            out["rawnet"].append(torch.softmax(D.rawnet(ens_3d), dim=1)[0, 1].item())
            out["wav2vec"].append(torch.softmax(D.wav2vec(ens_3d.squeeze(1)), dim=1)[0, 1].item())
        used += 1
    return out, used, first_error


def run_health_check(min_spread=DEFAULT_MIN_SPREAD, min_auc=DEFAULT_MIN_AUC,
                     n_per_class=DEFAULT_N_PER_CLASS,
                     real_sources=None, fake_sources=None, probe_dir=None):
    """Score all four sub-models on a fixed probe set.

    A model is COLLAPSED (hard fail) if its output spread < min_spread — it emits
    a near-constant value regardless of input, feeding the fusion a dead feature.
    A model is WEAK (warning only) if it varies but its AUC < min_auc.

    The probe set may be UNLABELLED (real audio only — the shipped CC0 set). The
    collapse gate is unaffected: spread asks "does this model vary with input at
    all", which diverse real audio answers on its own. The AUC tier needs both
    classes, so on an unlabelled set it reports None rather than a number computed
    from labels that do not exist.

    Returns a dict:
      {
        "active_version": str, "min_spread": float, "min_auc": float,
        "n_real": int, "n_fake": int, "probe_source": str, "labelled": bool,
        "models": {name: {"auc"|None, "fake_mean"|None, "real_mean", "spread",
                          "collapsed": bool, "weak": bool}},
        "passed": bool,            # False iff any model collapsed
        "collapsed": [name, ...],  # hard failures
        "weak": [name, ...],       # warnings, do not fail the gate
      }
    """
    random.seed(PROBE_SEED)
    real, fake, probe_source = resolve_probe_sources(
        probe_dir=probe_dir, real_sources=real_sources, fake_sources=fake_sources)
    labelled = bool(fake)
    if not real and not fake:
        raise RuntimeError(f"probe set empty ({probe_source})")
    real = random.sample(real, min(n_per_class, len(real)))
    if fake:
        fake = random.sample(fake, min(n_per_class, len(fake)))

    real_s, n_real, real_err = _model_scores(real)
    fake_s, n_fake, fake_err = _model_scores(fake)

    # A probe set that would not decode produces zero spread for every model, which
    # is byte-identical to a total collapse. Reporting that as "all four sub-models
    # COLLAPSED" sends the operator hunting a model failure that did not happen —
    # and, finding the gate apparently broken, straight to --skip-health. Raise
    # instead: bundle_registry maps a missing result to CANNOT CERTIFY, which is
    # the honest outcome and still refuses the promotion.
    if n_real == 0 or (labelled and n_fake == 0):
        raise RuntimeError(
            f"probe set did not decode (usable: {n_real} real, {n_fake} fake; "
            f"source: {probe_source}). This is an environment fault, NOT a model "
            f"verdict. First error — real: {real_err or 'n/a'}; "
            f"fake: {fake_err or 'n/a'}. Check that ffmpeg is present and "
            f"$VOICEGUARD_FFMPEG (currently {D.FFMPEG!r}) points at it.")

    # AUC needs both classes AND enough of the rare one to mean anything.
    auc_meaningful = labelled and n_fake >= MIN_FAKE_FOR_AUC

    models = {}
    collapsed = []
    weak = []
    for name in ("lcnn", "aasist", "wav2vec", "rawnet"):
        rs, fs = real_s[name], fake_s[name]
        scores = np.array(rs + fs)
        labels = np.array([0] * len(rs) + [1] * len(fs))
        auc = _auc(scores, labels) if auc_meaningful else None
        fake_mean = float(np.mean(fs)) if fs else None
        real_mean = float(np.mean(rs)) if rs else float("nan")
        # Spread across BOTH classes. A saturated/collapsed model outputs a
        # near-constant value regardless of input -> spread ~0. This is the
        # primary, unambiguous collapse signal.
        spread = float(np.std(scores)) if len(scores) else 0.0
        # bool() casts away numpy.bool_ (not JSON-serializable) from the comparisons.
        is_collapsed = bool(spread < min_spread)
        # "weak" only meaningful when not collapsed (a collapsed model is trivially
        # also below the AUC floor, but we report it as collapsed, not weak).
        is_weak = bool((not is_collapsed) and auc_meaningful and (auc < min_auc))
        models[name] = {"auc": round(float(auc), 4) if auc is not None else None,
                        "fake_mean": round(fake_mean, 4) if fake_mean is not None else None,
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
        "probe_source": probe_source, "labelled": labelled,
        "auc_reported": auc_meaningful,
        "models": models,
        "passed": len(collapsed) == 0,
        "collapsed": collapsed,
        "weak": weak,
    }


def _print_report(result):
    print(f"Per-sub-model health check — active bundle: {result['active_version']}")
    print(f"Probe source: {result['probe_source']}")
    print(f"Probe set: {result['n_real']} real / {result['n_fake']} fake   "
          f"collapse floor (spread): {result['min_spread']}   weak floor (AUC): {result['min_auc']}")
    if not result.get("auc_reported", True):
        print(f"Only {result['n_fake']} fake clip(s) (< {MIN_FAKE_FOR_AUC}): COLLAPSE gate "
              "enforced; AUC weak-warning tier reported as n/a.")
    print()
    print(f"  {'model':10s} {'AUC':>7s}  {'real_mean':>9s}  {'fake_mean':>9s}  {'spread':>7s}  verdict")
    for name, m in result["models"].items():
        if m["collapsed"]:
            verdict = "*** COLLAPSED ***"
        elif m["weak"]:
            verdict = "weak (warn)"
        else:
            verdict = "OK"
        auc_s = f"{m['auc']:7.4f}" if m["auc"] is not None else f"{'n/a':>7s}"
        fake_s = f"{m['fake_mean']:9.4f}" if m["fake_mean"] is not None else f"{'n/a':>9s}"
        print(f"  {name:10s} {auc_s}  {m['real_mean']:9.4f}  "
              f"{fake_s}  {m['spread']:7.4f}  {verdict}")
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
    ap.add_argument("--probe-set", default=None,
                    help="directory holding a probe set with a manifest.json. Overrides "
                         "auto-selection, which prefers the large local corpora and falls "
                         f"back to the shipped CC0 set ({os.path.relpath(SHIPPED_PROBE_DIR, REPO_ROOT)})")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    try:
        result = run_health_check(min_spread=args.min_spread, min_auc=args.min_auc,
                                  n_per_class=args.n_per_class, probe_dir=args.probe_set)
    except Exception as e:
        print(HEALTH_ERROR_PREFIX + " ".join(str(e).split()))
        print(f"health check error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(HEALTH_JSON_PREFIX + json.dumps(result))
    else:
        _print_report(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

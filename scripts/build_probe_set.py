#!/usr/bin/env python3
"""build_probe_set.py — assemble the shipped health-gate probe set.

WHY THIS EXISTS
---------------
scripts/submodel_health.py gates model promotion on per-sub-model collapse. Its
default probe sources (studio_clips, bias_audit_fakes, ...) are excluded from the
Docker build context, so inside the deployed image the gate could never certify
anything and the only way to promote was --skip-health — the exact habit the gate
exists to prevent (REMEDIATION_PLAN D9).

This script builds a SMALL, FIXED probe set that ships with the image.

WHY CC0 ONLY
------------
The image is pushed to a container registry, so anything baked in is
redistributed. Common Voice v26 is CC0 (public domain dedication), so it carries
no redistribution restriction — unlike studio_clips (YouTube-sourced, see
REMEDIATION_PLAN C1) or the edge-tts/gtts bias fakes, whose upstream service
terms restrict redistribution. Keeping the shipped set CC0-only means enforcing
the gate in production adds no licensing exposure while C1/C2 are with counsel.

WHY THE SET STILL NEEDS A FAKE HALF
-----------------------------------
A real-audio-only probe set does NOT work, and it fails in a way that looks
convincing until you check it. Collapse is measured as near-zero output SPREAD,
and on a set of all-real clips a *correct* model produces a tight cluster by
design — it confidently says "real" to every input. Measured on this exact set:
LCNN spread 0.0076 and RawNet3 spread 0.0001, both healthy models, both reported
COLLAPSED. Spread only separates healthy from collapsed when the probe set spans
both classes.

So the fake half comes from tests/golden_clips/ — the two fake clips that ALREADY
ship inside the image for the golden regression test. Reusing them adds no new
redistribution, which keeps the CC0/no-new-exposure property intact. With them
included, every sub-model shows healthy spread (0.14–0.39).

AUC IS DELIBERATELY NOT REPORTED
--------------------------------
Two fake clips cannot support an AUC. On this set the four models "score" 0.87 to
1.00, versus 0.49–0.69 on the full labelled corpora — an artefact of the sample
size that would flatly contradict the F1 finding in REMEDIATION_PLAN. The gate
therefore reports AUC as n/a below MIN_FAKE_FOR_AUC rather than printing a
flattering number. The collapse gate — the control that matters, and the one that
would have caught AASIST V9 — is fully enforced.

Selection is deterministic (sorted, fixed seed) and speaker-diverse: one clip per
distinct client_id, spread across all three locales, preferring clips at least
CHUNK-length so the models see real audio rather than zero padding.

USAGE
    python scripts/build_probe_set.py                 # build tests/probe_clips/
    python scripts/build_probe_set.py --per-locale 8  # larger set
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_ROOT = os.path.join(REPO, "cv-corpus-26.0-2026-06-12")
LOCALES = ("ha", "ig", "yo")
OUT_DIR = os.path.join(REPO, "tests", "probe_clips")

CORPUS = "Common Voice Corpus 26.0 (2026-06-12)"
LICENCE = "CC0-1.0"
LICENCE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

MIN_MS = 4000       # CHUNK is 4s @ 16kHz; shorter clips get zero-padded
MAX_MS = 12000      # keep the shipped set small
SEED = 0

# The fake half, reused from the golden-regression clips already inside the image.
GOLDEN_DIR = os.path.join(REPO, "tests", "golden_clips")
GOLDEN_FAKES = ("fake_noizai_a4cd.mp3", "fake_concert_hall.mp3")


def _durations(locale):
    path = os.path.join(CV_ROOT, locale, "clip_durations.tsv")
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            try:
                out[row["clip"]] = int(row["duration[ms]"])
            except (KeyError, ValueError):
                continue
    return out


def _candidates(locale):
    """Validated clips of usable length, one per distinct speaker, sorted."""
    durations = _durations(locale)
    by_speaker = {}
    path = os.path.join(CV_ROOT, locale, "validated.tsv")
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            clip = row.get("path")
            ms = durations.get(clip)
            if not clip or ms is None or not (MIN_MS <= ms <= MAX_MS):
                continue
            if not os.path.exists(os.path.join(CV_ROOT, locale, "clips", clip)):
                continue
            speaker = row.get("client_id") or clip
            # One clip per speaker: deterministically the lexicographically first,
            # so the set does not change with tsv row order.
            prev = by_speaker.get(speaker)
            if prev is None or clip < prev["clip"]:
                by_speaker[speaker] = {"clip": clip, "ms": ms, "speaker": speaker}
    return sorted(by_speaker.values(), key=lambda r: r["clip"])


def build(per_locale, out_dir=OUT_DIR):
    if not os.path.isdir(CV_ROOT):
        raise SystemExit(
            f"Common Voice corpus not found at {CV_ROOT}.\n"
            "The probe set is built from it; this script is a build-time tool and "
            "is not needed to RUN the gate (tests/probe_clips/ is committed).")

    os.makedirs(out_dir, exist_ok=True)
    for stale in os.listdir(out_dir):
        if stale.endswith(".mp3"):
            os.remove(os.path.join(out_dir, stale))

    rng = random.Random(SEED)
    entries = []
    for locale in LOCALES:
        cands = _candidates(locale)
        if len(cands) < per_locale:
            raise SystemExit(f"{locale}: only {len(cands)} usable candidates, "
                             f"need {per_locale}")
        for row in rng.sample(cands, per_locale):
            src = os.path.join(CV_ROOT, locale, "clips", row["clip"])
            dst_name = f"cv_{locale}_{row['clip'].split('_')[-1]}"
            shutil.copy2(src, os.path.join(out_dir, dst_name))
            with open(src, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            entries.append({
                "file": dst_name,
                "label": "real",
                "locale": locale,
                "duration_ms": row["ms"],
                "sha256": sha,
                "source_clip": row["clip"],
            })

    # Fake half: referenced in place from tests/golden_clips (already in the image),
    # so the probe set adds no audio beyond the CC0 reals copied above.
    for fname in GOLDEN_FAKES:
        src = os.path.join(GOLDEN_DIR, fname)
        if not os.path.exists(src):
            raise SystemExit(f"golden fake clip missing: {src}")
        with open(src, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        entries.append({
            "file": os.path.join("..", "golden_clips", fname).replace("\\", "/"),
            "label": "fake",
            "locale": None,
            "sha256": sha,
            "source_clip": fname,
            "note": "already shipped for the golden regression test; referenced, not copied",
        })

    entries.sort(key=lambda e: e["file"])
    manifest = {
        "purpose": "Fixed probe set for the per-sub-model promotion health gate "
                   "(scripts/submodel_health.py). Ships inside the Docker image.",
        "labelled": True,
        "auc_note": "Only 2 fake clips — too few for a meaningful AUC, so the gate "
                    "reports it as n/a (see MIN_FAKE_FOR_AUC). The COLLAPSE gate "
                    "(output spread) is fully enforced.",
        "corpus": CORPUS,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "selection": {"seed": SEED, "per_locale": per_locale, "locales": list(LOCALES),
                      "min_ms": MIN_MS, "max_ms": MAX_MS,
                      "rule": "one clip per distinct client_id, deterministic"},
        "clips": entries,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    copied = [e for e in entries if e["label"] == "real"]
    total = sum(os.path.getsize(os.path.join(out_dir, e["file"])) for e in copied)
    print(f"Wrote {len(copied)} CC0 clips to {out_dir} ({total/1024:.0f} KB); "
          f"referenced {len(entries) - len(copied)} already-shipped fake(s)")
    for e in entries:
        if e["label"] == "real":
            print(f"  {e['file']:28s} {e['locale']}  {e['duration_ms']:>6d} ms  real")
        else:
            print(f"  {e['file']:28s} {'-':>2s}  {'(referenced)':>10s}  fake")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the shipped CC0 health-gate probe set")
    ap.add_argument("--per-locale", type=int, default=5,
                    help="clips per locale (default 5 -> 15 total)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)
    build(args.per_locale, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""retune_cascade.py — deterministically re-tune an LCNN screener's cascade band.

WHY THIS EXISTS
---------------
The deployed v9h screener (model_store/v9h/lcnn.pt, sha f0f92004…) matched NO
retained source artifact — it was produced by editing the cascade early-out
threshold inside a checkpoint at bundling time, but the transformation was never
scripted or committed. That left the model making ~86% of production decisions
with no reproducible provenance (see docs/MODEL_INVENTORY.md §9.1 / GRC R3-G2).

This script is that missing transformation, made explicit and reproducible:
it takes a source LCNN checkpoint plus new cascade thresholds and writes a new
checkpoint that differs ONLY in `cascade_thresholds`. detector.py reads
CASCADE_LOW/HIGH from inside the checkpoint (detector.py:528-529), so re-tuning
the routing band is exactly a `cascade_thresholds` edit — no retraining, no
weight change.

PROVENANCE PROOF
----------------
v9h's lcnn.pt is v9's lcnn.pt (== models/lcnn_screener_v9.pt, sha ea5db300…)
with low_thresh 0.20 -> 0.10 and nothing else changed (verified: all 40 weight
tensors identical, every other field equal). To re-establish provenance:

    python scripts/retune_cascade.py \
        --src models/lcnn_screener_v9.pt \
        --low 0.10 \
        --expect-sha f0f920043de1f4206c5192a4f4c06fe30908e38365409a71c6b21bec5104b3b8 \
        --out /tmp/lcnn_v9h_reproduced.pt

IMPORTANT — torch.save is NOT byte-deterministic. Re-saving the *same* content
twice produces different bytes (zip metadata / pickle memo ordering). Verified on
this system. Therefore NO checkpoint here is byte-reproducible from its inputs,
and --expect-sha (byte match) will essentially always fail. The correct provenance
check is CONTENT equality: --match-content compares the retuned checkpoint's
logical contents (every weight tensor + every metadata field) against a target.
That is how v9h's screener provenance was established: content-identical to
models/lcnn_screener_v9.pt retuned to low_thresh=0.10.

    python scripts/retune_cascade.py \
        --src models/lcnn_screener_v9.pt --low 0.10 \
        --match-content model_store/v9h/lcnn.pt --check-only

USAGE
-----
    python scripts/retune_cascade.py --src IN.pt --low 0.10 [--high 0.80] --out OUT.pt
    python scripts/retune_cascade.py --src IN.pt --low 0.10 --match-content TARGET.pt --check-only
    python scripts/retune_cascade.py --src IN.pt --low 0.10 --expect-sha <hex> --out OUT.pt
"""
import argparse
import hashlib
import json
import os
import sys

import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ckpt(path):
    # weights_only=False: VoiceGuard checkpoints store numpy scalars + metadata dicts
    # (PyTorch >=2.6 default changed). See Handoff_Summary_V9_Phase7.md.
    return torch.load(path, map_location="cpu", weights_only=False)


def retune(ckpt, low=None, high=None):
    """Return a shallow-copied checkpoint with cascade_thresholds updated.
    Only low_thresh / high_thresh are touched; every other field is preserved."""
    if "cascade_thresholds" not in ckpt:
        raise KeyError("checkpoint has no 'cascade_thresholds' — is this an LCNN screener?")
    out = dict(ckpt)
    ct = dict(ckpt["cascade_thresholds"])       # copy so the source dict is untouched
    if low is not None:
        ct["low_thresh"] = float(low)
    if high is not None:
        ct["high_thresh"] = float(high)
    out["cascade_thresholds"] = ct
    return out


def content_diff(a, b):
    """Compare two checkpoints logically (not byte-wise). Returns a list of human
    strings describing every difference — empty list means logically identical."""
    diffs = []
    keys = set(a) | set(b)
    for k in sorted(keys):
        if k not in a:
            diffs.append(f"{k}: only in B"); continue
        if k not in b:
            diffs.append(f"{k}: only in A"); continue
        if k == "model_state_dict":
            ta, tb = a[k], b[k]
            if set(ta) != set(tb):
                diffs.append(f"{k}: tensor name set differs")
            else:
                bad = [t for t in ta if not torch.equal(ta[t], tb[t])]
                if bad:
                    diffs.append(f"{k}: {len(bad)} tensor(s) differ: {bad[:3]}")
        else:
            if a[k] != b[k]:
                diffs.append(f"{k}: {json.dumps(a[k])[:80]} != {json.dumps(b[k])[:80]}")
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deterministically re-tune an LCNN cascade band")
    ap.add_argument("--src", required=True, help="source LCNN checkpoint")
    ap.add_argument("--low", type=float, help="new cascade low_thresh (real early-out)")
    ap.add_argument("--high", type=float, help="new cascade high_thresh (fake early-out)")
    ap.add_argument("--out", help="output checkpoint path (omit with --check-only)")
    ap.add_argument("--expect-sha", help="assert the output FILE matches this sha256 "
                    "(note: torch.save is non-deterministic; prefer --match-content)")
    ap.add_argument("--match-content", help="target checkpoint to compare CONTENT against "
                    "(the correct provenance check — weights + metadata, not bytes)")
    ap.add_argument("--check-only", action="store_true",
                    help="load, retune, verify, but do not write an output file")
    args = ap.parse_args(argv)

    if args.low is None and args.high is None:
        ap.error("nothing to do: pass --low and/or --high")
    if not args.check_only and not args.out:
        ap.error("--out is required unless --check-only")

    src = load_ckpt(args.src)
    print(f"source: {args.src}  (sha {sha256_file(args.src)[:16]}…)")
    print(f"source cascade_thresholds: "
          f"low={src['cascade_thresholds'].get('low_thresh')} "
          f"high={src['cascade_thresholds'].get('high_thresh')}")

    tuned = retune(src, low=args.low, high=args.high)
    print(f"retuned cascade_thresholds: "
          f"low={tuned['cascade_thresholds'].get('low_thresh')} "
          f"high={tuned['cascade_thresholds'].get('high_thresh')}")

    # Confirm the ONLY logical change is cascade_thresholds.
    diffs = content_diff(src, tuned)
    non_threshold = [d for d in diffs if not d.startswith("cascade_thresholds")]
    if non_threshold:
        print("WARNING: changes beyond cascade_thresholds:", non_threshold, file=sys.stderr)
        return 1
    print("verified: only cascade_thresholds changed (weights + all metadata preserved)")

    # Content-level provenance check — the correct one, given torch.save is not
    # byte-deterministic. Compares every weight tensor and metadata field.
    if args.match_content:
        target = load_ckpt(args.match_content)
        cdiffs = content_diff(tuned, target)
        if not cdiffs:
            print(f"PROVENANCE PROVEN (content): retuned checkpoint is logically identical "
                  f"to {args.match_content}.")
        else:
            print(f"CONTENT MISMATCH vs {args.match_content}:", file=sys.stderr)
            for d in cdiffs:
                print(f"   - {d}", file=sys.stderr)
            return 2

    if args.check_only:
        return 0

    torch.save(tuned, args.out)
    out_sha = sha256_file(args.out)
    print(f"wrote: {args.out}  (sha {out_sha[:16]}…)")

    if args.expect_sha:
        if out_sha == args.expect_sha:
            print(f"PROVENANCE PROVEN: output reproduces {args.expect_sha[:16]}… byte-for-byte.")
            return 0
        # Byte mismatch — check whether it is nonetheless logically the target.
        print(f"byte mismatch: got {out_sha[:16]}…, expected {args.expect_sha[:16]}…")
        print("This can happen because torch.save serialization is not guaranteed stable")
        print("across PyTorch versions. Re-load both and compare content to classify it:")
        print("  python scripts/retune_cascade.py --src <target.pt> --low <same> --check-only")
        print("A CONTENT match means provenance is established logically; the byte delta is")
        print("a serialization artifact, not a difference in the model.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

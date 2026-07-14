"""
build_v9_manifest.py — Construct expanded training manifest for AASIST V9 retrain.

Adds two new source buckets to the existing train_v8_fresh.json:
  • real_studio  (~226 professionally-produced real clips)
  • fake_noizai  (~70 noiz.ai TTS fakes)

Also builds a matching held-out evaluation manifest from the held_out/ split
so validation is self-contained.

Usage (Kaggle notebook cell):
    !python build_v9_manifest.py                       # defaults
    !python build_v9_manifest.py --cap-real 200 --cap-fake 70  # custom caps
    !python build_v9_manifest.py --dry-run              # preview only
"""

import argparse
import json
import os
import glob
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Kaggle paths (adjust if running locally)
# ---------------------------------------------------------------------------
ARTEFACTS_ROOT = "/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts"
V9_DATA_ROOT   = "/kaggle/input/datasets/michaelologungbara/v9-train-test"

TRAIN_MANIFEST_IN  = os.path.join(ARTEFACTS_ROOT, "train_v8_fresh.json")
VAL_MANIFEST_IN    = os.path.join(ARTEFACTS_ROOT, "val_v8_fresh.json")

# Output goes to /kaggle/working/ (writable)
OUTPUT_DIR = "/kaggle/working"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def scan_audio_files(directory: str) -> list[str]:
    """Recursively find audio files in a directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in AUDIO_EXTS:
                files.append(os.path.join(root, fn))
    return files


def load_manifest(path: str) -> list[dict]:
    """Load a JSON manifest (list of dicts)."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def bucket_summary(entries: list[dict], label_key: str = "label",
                   source_key: str = "source") -> dict:
    """Return {source: {label: count}} breakdown."""
    breakdown = {}
    for e in entries:
        src = e.get(source_key, "unknown")
        lab = e.get(label_key, "unknown")
        breakdown.setdefault(src, Counter())[lab] += 1
    return breakdown


def print_breakdown(title: str, breakdown: dict):
    """Pretty-print a source-bucket breakdown table."""
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    print(f"  {'Source':<30} {'real':>6} {'fake':>6} {'total':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*7}")
    grand_real, grand_fake, grand_total = 0, 0, 0
    for src in sorted(breakdown.keys()):
        r = breakdown[src].get("real", 0) + breakdown[src].get(0, 0)
        f_ = breakdown[src].get("fake", 0) + breakdown[src].get(1, 0)
        t = r + f_
        grand_real += r
        grand_fake += f_
        grand_total += t
        print(f"  {src:<30} {r:>6} {f_:>6} {t:>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*7}")
    print(f"  {'TOTAL':<30} {grand_real:>6} {grand_fake:>6} {grand_total:>7}")
    print()


def infer_manifest_schema(entries: list[dict]) -> dict:
    """Detect which keys the existing manifest uses for path/label/source."""
    sample = entries[0]
    schema = {}

    # Path key
    for k in ["path", "file", "filepath", "audio_path", "audio_file"]:
        if k in sample:
            schema["path_key"] = k
            break
    else:
        # fallback: first string-valued key that looks like a path
        for k, v in sample.items():
            if isinstance(v, str) and ("/" in v or "\\" in v):
                schema["path_key"] = k
                break

    # Label key
    for k in ["label", "class", "target", "is_fake"]:
        if k in sample:
            schema["label_key"] = k
            break

    # Source key
    for k in ["source", "source_bucket", "dataset", "origin"]:
        if k in sample:
            schema["source_key"] = k
            break

    return schema


def build_new_entries(audio_files: list[str], label: str, source: str,
                      path_key: str, label_key: str, source_key: str,
                      cap: int | None = None) -> list[dict]:
    """Build manifest entries for new audio files."""
    entries = []
    for fp in audio_files:
        entries.append({
            path_key: fp,
            label_key: label,
            source_key: source,
        })
    if cap is not None and len(entries) > cap:
        print(f"  [cap] {source}: {len(entries)} files found, capping to {cap}")
        entries = entries[:cap]
    else:
        print(f"  [all] {source}: {len(entries)} files (no cap applied)")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Build V9 training manifest with studio reals + noiz.ai fakes")
    parser.add_argument("--cap-real", type=int, default=None,
                        help="Max studio-real samples to add (default: all ~226)")
    parser.add_argument("--cap-fake", type=int, default=None,
                        help="Max noiz.ai-fake samples to add (default: all ~70)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing output files")
    parser.add_argument("--train-in", type=str, default=TRAIN_MANIFEST_IN,
                        help="Path to existing train manifest")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory for output manifests")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load existing manifest and inspect schema
    # ------------------------------------------------------------------
    print("[1/5] Loading existing training manifest...")
    existing = load_manifest(args.train_in)
    print(f"       Loaded {len(existing)} entries from {args.train_in}")

    schema = infer_manifest_schema(existing)
    path_key   = schema.get("path_key", "path")
    label_key  = schema.get("label_key", "label")
    source_key = schema.get("source_key", "source")
    print(f"       Schema detected: path='{path_key}', label='{label_key}', "
          f"source='{source_key}'")
    print(f"       Sample entry: {existing[0]}")

    # Show existing breakdown
    bd_before = bucket_summary(existing, label_key, source_key)
    print_breakdown("EXISTING MANIFEST (train_v8_fresh.json)", bd_before)

    # ------------------------------------------------------------------
    # 2. Scan new audio directories
    # ------------------------------------------------------------------
    print("[2/5] Scanning new audio directories...")

    # Double-nested structure: new_samples/new_samples/real/ etc.
    real_dir = os.path.join(V9_DATA_ROOT, "new_samples", "new_samples", "real")
    fake_dir = os.path.join(V9_DATA_ROOT, "new_samples", "new_samples", "fake")

    if not os.path.isdir(real_dir):
        # Try single-nested fallback
        real_dir = os.path.join(V9_DATA_ROOT, "new_samples", "real")
    if not os.path.isdir(fake_dir):
        fake_dir = os.path.join(V9_DATA_ROOT, "new_samples", "fake")

    real_files = scan_audio_files(real_dir)
    fake_files = scan_audio_files(fake_dir)
    print(f"       Studio reals found: {len(real_files)}  in {real_dir}")
    print(f"       Noiz.ai fakes found: {len(fake_files)}  in {fake_dir}")

    if len(real_files) == 0:
        print("  ⚠ WARNING: No studio real files found. Check directory path.")
    if len(fake_files) == 0:
        print("  ⚠ WARNING: No noiz.ai fake files found. Check directory path.")

    # ------------------------------------------------------------------
    # 3. Build new entries with optional caps
    # ------------------------------------------------------------------
    print("\n[3/5] Building new manifest entries...")

    # Determine label format from existing data (string vs int)
    existing_labels = {e[label_key] for e in existing}
    if existing_labels <= {0, 1}:
        real_label, fake_label = 0, 1
        print("       Label format: integer (0=real, 1=fake)")
    elif "bonafide" in existing_labels or "spoof" in existing_labels:
        real_label, fake_label = "bonafide", "spoof"
        print("       Label format: ASVspoof-style (bonafide/spoof)")
    else:
        real_label, fake_label = "real", "fake"
        print(f"       Label format: string (real/fake) "
              f"[existing labels: {existing_labels}]")

    new_reals = build_new_entries(
        real_files, label=real_label, source="real_studio",
        path_key=path_key, label_key=label_key, source_key=source_key,
        cap=args.cap_real)

    new_fakes = build_new_entries(
        fake_files, label=fake_label, source="fake_noizai",
        path_key=path_key, label_key=label_key, source_key=source_key,
        cap=args.cap_fake)

    # ------------------------------------------------------------------
    # 4. Merge and save training manifest
    # ------------------------------------------------------------------
    print("\n[4/5] Merging manifests...")
    combined = existing + new_reals + new_fakes
    print(f"       Combined: {len(existing)} + {len(new_reals)} + "
          f"{len(new_fakes)} = {len(combined)}")

    bd_after = bucket_summary(combined, label_key, source_key)
    print_breakdown("EXPANDED MANIFEST (train_v9.json)", bd_after)

    # Class balance check
    total_real = sum(1 for e in combined
                     if e[label_key] in ("real", "bonafide", 0))
    total_fake = sum(1 for e in combined
                     if e[label_key] in ("fake", "spoof", 1))
    ratio = total_real / total_fake if total_fake > 0 else float("inf")
    print(f"  Class balance — real: {total_real}, fake: {total_fake}, "
          f"ratio: {ratio:.2f}:1")
    if ratio > 2.0 or ratio < 0.5:
        print("  ⚠ WARNING: Class imbalance > 2:1. Consider adjusting caps "
              "or oversampling the minority class during training.")

    if args.dry_run:
        print("\n  [DRY RUN] No files written.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    train_out = os.path.join(args.output_dir, "train_v9.json")
    with open(train_out, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  ✓ Saved training manifest: {train_out}")

    # ------------------------------------------------------------------
    # 5. Build held-out evaluation manifest
    # ------------------------------------------------------------------
    print("\n[5/5] Building held-out evaluation manifest...")

    heldout_real_dir = os.path.join(V9_DATA_ROOT, "held_out", "held_out", "real")
    heldout_fake_dir = os.path.join(V9_DATA_ROOT, "held_out", "held_out", "fake")

    if not os.path.isdir(heldout_real_dir):
        heldout_real_dir = os.path.join(V9_DATA_ROOT, "held_out", "real")
    if not os.path.isdir(heldout_fake_dir):
        heldout_fake_dir = os.path.join(V9_DATA_ROOT, "held_out", "fake")

    ho_reals = scan_audio_files(heldout_real_dir)
    ho_fakes = scan_audio_files(heldout_fake_dir)
    print(f"       Held-out reals: {len(ho_reals)}  in {heldout_real_dir}")
    print(f"       Held-out fakes: {len(ho_fakes)}  in {heldout_fake_dir}")

    eval_entries = []
    for fp in ho_reals:
        eval_entries.append({
            path_key: fp,
            label_key: real_label,
            source_key: "heldout_studio_real",
        })
    for fp in ho_fakes:
        eval_entries.append({
            path_key: fp,
            label_key: fake_label,
            source_key: "heldout_noizai_fake",
        })

    eval_out = os.path.join(args.output_dir, "eval_v9_heldout.json")
    with open(eval_out, "w") as f:
        json.dump(eval_entries, f, indent=2)
    print(f"  ✓ Saved held-out eval manifest: {eval_out} ({len(eval_entries)} entries)")

    bd_eval = bucket_summary(eval_entries, label_key, source_key)
    print_breakdown("HELD-OUT EVAL MANIFEST", bd_eval)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  DONE. Next steps:")
    print(f"  1. Verify manifests:  cat {train_out} | python -m json.tool | head")
    print(f"  2. Spot-check paths:  head -5 {train_out}")
    print(f"  3. Proceed to AASIST retrain using {train_out}")
    print("=" * 65)


if __name__ == "__main__":
    main()

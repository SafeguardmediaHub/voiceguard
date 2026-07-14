#!/usr/bin/env python3
"""
Golden-file regression test for the VoiceGuard V9 cascade server.

Pins server.detect() output on a fixed set of local clips. Scoring is
deterministic (input randomization OFF, models in eval mode, CPU), so exact
scores are reproducible run-to-run; this test catches any *unintended* change
to the scoring core — model swaps, threshold-schema edits, preprocessing
changes, checkpoint drift, calibration edits, cascade-band changes.

USAGE
    python tests/test_golden.py            # assert against the golden baseline
    python tests/test_golden.py --update   # regenerate the baseline (do this
                                            # ONLY after an intentional, reviewed
                                            # change, and eyeball the diff)

    pytest tests/test_golden.py            # same assertions under pytest

The clips live in tests/golden_clips/ so the test is self-contained and does
NOT depend on data/held_out/ (which may be regenerated). Expected values live
in tests/golden_manifest.json.
"""
import os, sys, json, argparse
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

pytestmark = pytest.mark.weights

CLIPS_DIR = os.path.join(HERE, "golden_clips")
MANIFEST  = os.path.join(HERE, "golden_manifest.json")

# The fixed golden set. Each entry names a clip and the class we believe it is
# (documentation only — the test pins whatever the server actually produces).
CLIPS = [
    {"file": "real_studio_037.mp3", "truth": "real"},
    {"file": "real_studio_055.mp3", "truth": "real"},
    {"file": "fake_noizai_a4cd.mp3", "truth": "fake"},
    {"file": "fake_concert_hall.mp3", "truth": "fake"},
]

SCORE_TOL = 1e-3   # allow tiny float drift; verdict + cascade counts are exact


def _run(server, clip_path):
    """Run the server's detect() and pull the fields we pin."""
    r = server.detect(clip_path)
    return {
        "score":          round(float(r["score"]), 6),
        "verdict":        r["verdict"],
        "model_version":  r["model_version"],
        "chunks":         r["chunks"],
        "stage1_chunks":  r["cascade"]["stage1_chunks"],
        "stage2_chunks":  r["cascade"]["stage2_chunks"],
    }


def _load_server():
    # Importing server.py loads all V9 models (~5-8s).
    import importlib
    return importlib.import_module("detector")


def generate():
    server = _load_server()
    baseline = {"score_tol": SCORE_TOL, "clips": {}}
    for c in CLIPS:
        path = os.path.join(CLIPS_DIR, c["file"])
        assert os.path.exists(path), f"missing golden clip: {path}"
        out = _run(server, path)
        out["truth"] = c["truth"]
        baseline["clips"][c["file"]] = out
        print(f"  {c['file']:22s} -> {out['verdict']:12s} score={out['score']:.6f} "
              f"stage1/2={out['stage1_chunks']}/{out['stage2_chunks']}")
    with open(MANIFEST, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nWrote baseline: {MANIFEST}")


def check():
    """Return (n_pass, failures). Raises AssertionError-free; caller decides."""
    assert os.path.exists(MANIFEST), (
        f"no golden baseline at {MANIFEST}. Run: python tests/test_golden.py --update")
    baseline = json.load(open(MANIFEST))
    tol = baseline.get("score_tol", SCORE_TOL)
    server = _load_server()

    failures = []
    print(f"\n{'clip':24s} {'verdict':12s} {'score':>10s}  {'expected':>10s}  result")
    print("-" * 74)
    for fname, exp in baseline["clips"].items():
        path = os.path.join(CLIPS_DIR, fname)
        got = _run(server, path)
        problems = []
        if got["verdict"] != exp["verdict"]:
            problems.append(f"verdict {got['verdict']} != {exp['verdict']}")
        if abs(got["score"] - exp["score"]) > tol:
            problems.append(f"score {got['score']:.6f} != {exp['score']:.6f} (tol {tol})")
        if got["model_version"] != exp["model_version"]:
            problems.append(f"version {got['model_version']} != {exp['model_version']}")
        if got["stage1_chunks"] != exp["stage1_chunks"] or got["stage2_chunks"] != exp["stage2_chunks"]:
            problems.append(
                f"routing {got['stage1_chunks']}/{got['stage2_chunks']} != "
                f"{exp['stage1_chunks']}/{exp['stage2_chunks']}")
        status = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        print(f"{fname:24s} {got['verdict']:12s} {got['score']:10.6f}  "
              f"{exp['score']:10.6f}  {status}")
        if problems:
            failures.append((fname, problems))
    return len(baseline["clips"]) - len(failures), failures


def test_golden():
    """pytest entry point."""
    n_pass, failures = check()
    assert not failures, f"{len(failures)} golden clip(s) regressed: {failures}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="regenerate the golden baseline (after a reviewed change)")
    args = ap.parse_args()
    if args.update:
        print("Regenerating golden baseline...")
        generate()
        return 0
    n_pass, failures = check()
    print("-" * 74)
    if failures:
        print(f"RESULT: {len(failures)} FAILED, {n_pass} passed")
        return 1
    print(f"RESULT: all {n_pass} golden clips passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

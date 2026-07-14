"""explain_acceptance.py — run detect() over a folder of clips and dump the explainability
outputs (heatmap PNG + summary JSON) for the Phase 3 20-sample review.

Usage: python scripts/explain_acceptance.py --clips tests/golden_clips --out explain_out
"""
import argparse, base64, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder of audio files, or a glob")
    ap.add_argument("--out", default="explain_out")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    pattern = os.path.join(args.clips, "*") if os.path.isdir(args.clips) else args.clips
    files = [f for f in sorted(glob.glob(pattern)) if os.path.isfile(f)]

    n = 0
    for f in files:
        try:
            r = detector.detect(f)
        except Exception as e:
            print(f"  {os.path.basename(f):40s} ERROR {e}")
            continue
        aid = r["audit_id"]
        hm = r.get("heatmap")
        if hm and hm.get("png_base64"):
            with open(os.path.join(args.out, aid + ".png"), "wb") as fp:
                fp.write(base64.b64decode(hm["png_base64"]))
        with open(os.path.join(args.out, aid + ".json"), "w", encoding="utf-8") as fp:
            json.dump({"file": os.path.basename(f), "verdict": r["verdict"],
                       "score": r["score"], "confidence": r["confidence"],
                       "shap": r.get("shap"), "flagged_segments": r.get("flagged_segments")},
                      fp, indent=2)
        n += 1
        print(f"  {os.path.basename(f):40s} {r['verdict']:12s} "
              f"score={r['score']:.3f} conf={r['confidence']:.3f} -> {aid}")
    print(f"\n{n} clips -> {args.out}/")


if __name__ == "__main__":
    main()

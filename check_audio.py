"""check_audio.py — run VoiceGuard's detect() DIRECTLY on one file (no browser),
and print the verdict plus the cascade/screener breakdown.

Usage:  python check_audio.py "C:\\path\\to\\your-audio.mp3"
"""
import sys, json, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # use the cached wav2vec2-base; no network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import detector

if len(sys.argv) < 2:
    print('usage: python check_audio.py "C:\\path\\to\\audio.mp3"')
    raise SystemExit(1)

path = sys.argv[1]
r = detector.detect(path)

print("\n================ VoiceGuard direct check ================")
print("file           :", path)
print("sha256(bytes)  :", r["sha256"], " <- fingerprint of what was decoded")
print("VERDICT        :", r["verdict"], "   score =", r["score"], f"({r['pct']}% fake)")
print("confidence     :", r["confidence"])
print("--------------- cascade / screener --------------------")
c = r["cascade"]
print("total chunks   :", r["chunks"])
print("stage-1 (screener resolved) :", c["stage1_chunks"])
print("stage-2 (escalated to ensemble):", c["stage2_chunks"])
print("resolution_pct :", c["resolution_pct"], "% resolved by the screener alone")
print("screener band  :", c["band"], "(<=low => REAL, >=high => FAKE, else escalate)")
print("--------------- per-model means (% fake) --------------")
print("LCNN screener  :", r["lcnn"])
print("AASIST         :", r["aasist"])
print("Wav2Vec2       :", r["w2v"])
print("RawNet3        :", r["rawnet"])
print("========================================================\n")

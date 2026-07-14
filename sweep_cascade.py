"""sweep_cascade.py — sweep the screener's CASCADE_LOW ("real" early-out) and measure the
false-positive vs fake-recall trade-off, so we pick the threshold with data, not guesses.

For every chunk of every clip it computes BOTH the LCNN screener prob and the forced ensemble
prob once; then for each candidate CASCADE_LOW it re-decides each chunk (<=low or >=high -> use
the screener prob; otherwise -> ensemble prob), aggregates with the real cw_score, and applies
the real verdict thresholds.

Usage:
  python sweep_cascade.py --real studio_clips tests/golden_clips_real --fake studio_fake_test bias_audit_fakes
  (any number of --real / --fake dirs or files; label is by which flag you pass)
"""
import sys, os, glob, subprocess, argparse, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F
import detector as D

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac")


def _expand(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [f for f in sorted(glob.glob(os.path.join(p, "**", "*"), recursive=True))
                      if os.path.isfile(f) and f.lower().endswith(AUDIO_EXT)]
        elif os.path.isfile(p):
            files.append(p)
    return files


def _load(p):
    w = p + "_sw.wav"
    subprocess.run(["C:/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe", "-y", "-i", p,
                    "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", w],
                   capture_output=True, timeout=30)
    from scipy.io import wavfile
    _, d = wavfile.read(w)
    try: os.unlink(w)
    except Exception: pass
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d.astype(np.float32) / 32768.0 if d.dtype == np.int16 else d.astype(np.float32)
    return torch.tensor(d, dtype=torch.float32)


def chunk_signals(p):
    """[(lcnn_p, ensemble_p)] per chunk — runs the ensemble on EVERY chunk (the slow part)."""
    w = _load(p)
    total = w.shape[-1]
    cs, hop = int(4.0 * D.SR), int(4.0 * D.SR * 0.5)
    out, start = [], 0
    while start < total:
        c = w[start:start + cs]
        n = c.shape[-1]
        c = F.pad(c, (0, D.CHUNK - n)) if n < D.CHUNK else c[:D.CHUNK]
        wav_1d = c.to(D.DEVICE)
        lp = D.lcnn_score(wav_1d)
        peak = wav_1d.abs().max()
        ew = wav_1d / peak if peak > 1e-8 else wav_1d
        _, _, _, ens = D.ensemble_score_variants(ew.unsqueeze(0).unsqueeze(0))
        out.append((lp, ens))
        start += hop
    return out or [(0.0, 0.0)]


def file_score(sig, low, high):
    scores = [lp if (lp <= low or lp >= high) else ens for lp, ens in sig]
    return D.cw_score(scores)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--fake", nargs="*", default=[])
    ap.add_argument("--flag", type=float, default=0.55,
                    help="score >= this counts as 'called fake' (default 0.55 = likely_fake)")
    ap.add_argument("--sample", type=int, default=60, help="cap files per class (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    real_files, fake_files = _expand(args.real), _expand(args.fake)
    random.seed(args.seed)
    if args.sample and len(real_files) > args.sample:
        real_files = random.sample(real_files, args.sample)
    if args.sample and len(fake_files) > args.sample:
        fake_files = random.sample(fake_files, args.sample)
    print(f"real clips: {len(real_files)}   fake clips: {len(fake_files)}   "
          f"HIGH={D.CASCADE_HIGH:.2f}  current LOW={D.CASCADE_LOW:.2f}  flag>={args.flag}")
    print("computing per-chunk screener + forced-ensemble (once) ...")

    def collect(files):
        sigs, skipped = [], 0
        for f in files:
            try:
                sigs.append(chunk_signals(f))
            except Exception:
                skipped += 1          # unreadable / ffmpeg-failed file — skip, don't abort
        return sigs, skipped

    real_sig, r_skip = collect(real_files)
    fake_sig, f_skip = collect(fake_files)
    if r_skip or f_skip:
        print(f"(skipped unreadable: {r_skip} real, {f_skip} fake)")

    grid = [0.20, 0.15, 0.10, 0.075, 0.05, 0.02, 0.0]
    nf, nr = len(fake_sig), len(real_sig)
    print(f"\n(recall = fakes flagged, FP = reals flagged; at REVIEW>=0.30 and FAKE>=0.55)")
    print(f"{'LOW':<7}{'rec@.30':<10}{'FP@.30':<9}{'rec@.55':<10}{'FP@.55':<9}")
    for low in grid:
        fs = [file_score(s, low, D.CASCADE_HIGH) for s in fake_sig]
        rs = [file_score(s, low, D.CASCADE_HIGH) for s in real_sig]
        r30 = sum(1 for x in fs if x >= 0.30); p30 = sum(1 for x in rs if x >= 0.30)
        r55 = sum(1 for x in fs if x >= 0.55); p55 = sum(1 for x in rs if x >= 0.55)
        star = "  <- current" if abs(low - D.CASCADE_LOW) < 1e-9 else ""
        print(f"{low:<7.3f}{f'{r30}/{nf}':<10}{f'{p30}/{nr}':<9}{f'{r55}/{nf}':<10}{f'{p55}/{nr}':<9}{star}")
    print("\nLook at where recall climbs as LOW drops, and how much FP rises alongside it — "
          "pick the LOW that adds fake catches (esp. at REVIEW) with minimal real FP.")


if __name__ == "__main__":
    main()

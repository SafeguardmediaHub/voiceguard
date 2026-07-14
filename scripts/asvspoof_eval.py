"""asvspoof_eval.py — benchmark the active bundle (v9h) on an ASVspoof-style eval set.

Reports EER + AUC for each sub-model (AASIST, Wav2Vec2, RawNet3), the LCNN screener,
the XGBoost ensemble, and the full cascade, against a protocol/keys file whose lines
contain a `bonafide`/`spoof` label. Works with:
  - ASVspoof 2019 LA: ASVspoof2019.LA.cm.eval.trl.txt   (line: spk utt - attack label)
  - ASVspoof 2021 LA: keys/LA/CM/trial_metadata.txt     (line: spk utt codec tx attack label trim split)
In both, the utterance id is column 2 and the label is the bonafide/spoof token.

Usage:
  python scripts/asvspoof_eval.py --protocol <keys.txt> --audio-dir <dir> --ext flac [--limit N]

Notes:
  - Sets VOICEGUARD_ADVERSARIAL=0 (advisory monitor is irrelevant + slow for a benchmark).
  - ASVspoof utterances are short; the first 4s window is scored (padded if shorter).
  - --limit N samples the first N valid entries so you can smoke-test before the full run.
"""
import os, sys, argparse, time, subprocess, glob, re
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VOICEGUARD_ADVERSARIAL", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
import detector as D

_FFMPEG = os.environ.get("VOICEGUARD_FFMPEG",
                         "C:/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe" if os.name == "nt" else "ffmpeg")
_ATTACK_RE = re.compile(r"^A\d{2}$")          # ASVspoof attack ids: A01..A19
_SPLITS = {"progress", "eval", "hidden"}      # ASVspoof 2021 trial_metadata split column


def eer_pct(scores, labels):
    """Equal Error Rate (%). labels: 1=spoof(fake), 0=bonafide(real); score = p_fake."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0 * 100.0)


def auc(scores, labels):
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def load_16k(path):
    wav_path = path + "_asv.wav"
    subprocess.run([_FFMPEG, "-y", "-i", path, "-ar", "16000", "-ac", "1",
                    "-acodec", "pcm_s16le", "-f", "wav", wav_path], capture_output=True, timeout=30)
    from scipy.io import wavfile
    _, d = wavfile.read(wav_path)
    try: os.unlink(wav_path)
    except Exception: pass
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d.astype(np.float32) / 32768.0 if d.dtype == np.int16 else d.astype(np.float32)
    return torch.tensor(d, dtype=torch.float32)


def parse_protocol(path, utt_col):
    """[(utt_id, label, attack, split)] — label 1=spoof/0=bonafide; attack e.g. 'A07'
    (None for bonafide); split e.g. 'eval'/'progress' (None if the file has no split column)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            label = 1 if "spoof" in parts else (0 if "bonafide" in parts else None)
            if label is None:
                continue
            utt = parts[utt_col] if len(parts) > utt_col else parts[-1]
            attack = next((p for p in parts if _ATTACK_RE.match(p)), None)
            split = next((p for p in parts if p in _SPLITS), None)
            rows.append((utt, label, attack, split))
    return rows


def index_audio(audio_dir, ext):
    """basename(no ext) -> full path, recursive."""
    idx = {}
    for p in glob.glob(os.path.join(audio_dir, "**", "*." + ext.lstrip(".")), recursive=True):
        idx[os.path.splitext(os.path.basename(p))[0]] = p
    return idx


def score_file(path):
    """Return (lcnn, aasist, wav2vec, rawnet, ensemble, cascade) p_fake for one file."""
    wav = load_16k(path)
    c = wav[:D.CHUNK]
    c = F.pad(c, (0, D.CHUNK - c.shape[-1])) if c.shape[-1] < D.CHUNK else c
    c = c.to(D.DEVICE)
    lcnn_p = D.lcnn_score(c)
    peak = c.abs().max()
    ew = c / peak if peak > 1e-8 else c
    s_a, s_w, s_r, ens = D.ensemble_score_variants(ew.unsqueeze(0).unsqueeze(0))
    casc = D.cascade_score_chunk(c)["score"]
    return lcnn_p, s_a, s_w, s_r, ens, casc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--ext", default="flac")
    ap.add_argument("--utt-col", type=int, default=1, help="0-based column of the utterance id")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N valid files (0=all)")
    ap.add_argument("--split", default=None,
                    help="keep only rows tagged with this split, e.g. 'eval' (ASVspoof 2021 only)")
    args = ap.parse_args(argv)

    entries = parse_protocol(args.protocol, args.utt_col)
    if args.split:
        tagged = [e for e in entries if e[3] == args.split]
        if tagged:
            print(f"--split '{args.split}': {len(tagged)}/{len(entries)} rows kept")
            entries = tagged
        else:
            print(f"--split '{args.split}': no rows carry a split column (e.g. 2019 LA) — ignoring")
    audio = index_audio(args.audio_dir, args.ext)
    print(f"protocol entries: {len(entries)}   audio files indexed: {len(audio)}   "
          f"active bundle: {D.ACTIVE_VERSION}")

    streams = {k: [] for k in ("lcnn", "aasist", "wav2vec", "rawnet", "ensemble", "cascade")}
    labels, attacks, missing, n = [], [], 0, 0
    t0 = time.time()
    for utt, lab, attack, _split in entries:
        path = audio.get(utt)
        if path is None:
            missing += 1
            continue
        try:
            lc, sa, sw, sr, en, ca = score_file(path)
        except Exception as e:
            print(f"  skip {utt}: {e}")
            continue
        streams["lcnn"].append(lc); streams["aasist"].append(sa); streams["wav2vec"].append(sw)
        streams["rawnet"].append(sr); streams["ensemble"].append(en); streams["cascade"].append(ca)
        labels.append(lab); attacks.append(attack)
        n += 1
        if n % 500 == 0:
            print(f"  scored {n} ({(time.time()-t0)/n:.2f}s/file)")
        if args.limit and n >= args.limit:
            break

    labels = np.asarray(labels)
    print(f"\nscored {n} files ({missing} protocol entries had no matching audio)")
    n_spoof = int(labels.sum()); n_bona = len(labels) - n_spoof
    print(f"labels: {n_bona} bonafide (real), {n_spoof} spoof (fake)\n")
    targets = {"aasist": "<5%", "wav2vec": "<8%", "rawnet": "<10%", "ensemble": "<5%", "cascade": "<5%"}
    print(f"{'stream':12}{'EER%':>8}{'AUC':>8}   {'target':>8}")
    print("-" * 40)
    for k in ("lcnn", "aasist", "wav2vec", "rawnet", "ensemble", "cascade"):
        print(f"{k:12}{eer_pct(streams[k], labels):8.2f}{auc(streams[k], labels):8.3f}   "
              f"{targets.get(k, '-'):>8}")

    # Per-attack breakdown for the deployed CASCADE stream. Each attack's EER pools all
    # bonafide scores (shared negatives) with that one attack's spoof scores — the standard
    # ASVspoof per-attack EER. Sorted worst-first so weak spots surface at the top.
    casc = np.asarray(streams["cascade"])
    attacks = np.asarray([a if a is not None else "-" for a in attacks])
    bona_scores = casc[labels == 0]
    attack_ids = sorted(set(attacks[labels == 1].tolist()))
    if len(bona_scores) and attack_ids:
        print(f"\nper-attack EER (cascade; each vs the shared bonafide set):")
        print(f"{'attack':10}{'n':>7}{'EER%':>9}")
        print("-" * 28)
        rows = []
        for aid in attack_ids:
            spoof_scores = casc[(labels == 1) & (attacks == aid)]
            s = np.concatenate([bona_scores, spoof_scores])
            y = np.concatenate([np.zeros(len(bona_scores)), np.ones(len(spoof_scores))])
            rows.append((aid, len(spoof_scores), eer_pct(s, y)))
        for aid, cnt, e in sorted(rows, key=lambda r: (-(r[2] if r[2] == r[2] else -1))):
            print(f"{aid:10}{cnt:>7}{e:>9.2f}")


if __name__ == "__main__":
    main()

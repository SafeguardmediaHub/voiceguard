# %%  CELL: Adversarial Monitor v2 — Deliberate Attack Detection
# Separate from the cascade detector. Advisory only — never overrides verdict.
#
# v2 fixes: v1 flagged 84% of normal fakes because ensemble disagreement
# and perturbation sensitivity naturally correlate with fakeness.
# v2 decouples the adversarial signal from the fakeness signal by asking:
#   "Is someone trying to make a FAKE look REAL?"
#
# Key principle: adversarial attacks TARGET the detector. So the signals are:
#   1. A sample the cascade calls "real" but individual models disagree about
#   2. A sample whose verdict FLIPS under tiny noise (fragile classification)
#   3. Spectral fingerprints of gradient-based perturbations
#
# Output: {"flag": "normal" | "attack_suspected", "confidence": 0.0-1.0, ...}

import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import xgboost as xgb
from scipy.special import expit as sigmoid_fn
from transformers import Wav2Vec2Model

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR      = 16000
MAX_LEN = 4 * SR

# ═══════════════════════════════════════════════════════════════════════════════
#  Signal 1: Ensemble Disagreement (conditioned on verdict)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Normal fake:  cascade says fake, models mostly agree → NOT suspicious
# Normal real:  cascade says real, models mostly agree → NOT suspicious
# Adversarial:  cascade says real, but one+ model says fake → SUSPICIOUS
#               (attack fooled the fusion but not every sub-model)
#
# Also suspicious: cascade says real but with low margin (barely real)

def compute_disagreement(prob_a, prob_w, prob_r, ensemble_prob):
    """
    Adversarial-specific disagreement: only fires when the ensemble leans
    "real" but individual models disagree.

    Args:
        prob_a, prob_w, prob_r: individual model P(fake)
        ensemble_prob: calibrated ensemble P(fake)

    Returns: (score [0,1], detail_dict)
    """
    probs = np.array([prob_a, prob_w, prob_r])
    preds = (probs >= 0.5).astype(int)
    ensemble_says_fake = ensemble_prob >= 0.5

    std = float(np.std(probs))
    n_say_fake = int(preds.sum())
    n_say_real = 3 - n_say_fake

    # Which model is the outlier?
    majority = int(np.median(preds))
    one_vs_two = (n_say_fake == 1 or n_say_real == 1)
    outlier_model = None
    if one_vs_two:
        outlier_idx = int(np.argmax(preds != majority))
        outlier_model = ["AASIST", "Wav2Vec2", "RawNet3"][outlier_idx]

    # ── Adversarial-specific scoring ──────────────────────────────────
    score = 0.0

    if not ensemble_says_fake:
        # CASCADE SAYS REAL — this is where adversarial attacks live.
        # How many individual models think it's fake despite the ensemble?
        if n_say_fake >= 2:
            # 2 of 3 models say fake but ensemble says real → very suspicious
            score = 0.9
        elif n_say_fake == 1:
            # 1 model says fake, ensemble says real → moderately suspicious
            # Weight by how confident the dissenting model is
            fake_probs = probs[preds == 1]
            max_dissent = float(np.max(fake_probs))
            score = 0.3 + 0.4 * min(1.0, (max_dissent - 0.5) / 0.4)
        else:
            # All models agree real — but is the ensemble margin thin?
            # ensemble_prob close to 0.5 from the real side is suspicious
            if ensemble_prob >= 0.35:
                score = 0.2 * ((ensemble_prob - 0.35) / 0.15)
    else:
        # CASCADE SAYS FAKE — disagreement here is normal (models
        # have different sensitivities to different fake types).
        # Only flag if ALL models say real but ensemble somehow says fake
        # (shouldn't happen but check anyway).
        if n_say_fake == 0:
            score = 0.5  # weird edge case

    return score, {
        "score": round(score, 4),
        "std": round(std, 4),
        "n_say_fake": n_say_fake,
        "one_vs_two": one_vs_two,
        "outlier_model": outlier_model,
        "ensemble_says_fake": bool(ensemble_says_fake),
        "ensemble_prob": round(float(ensemble_prob), 4),
        "model_probs": {"aasist": round(prob_a, 4),
                        "wav2vec2": round(prob_w, 4),
                        "rawnet3": round(prob_r, 4)}
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Signal 2: Perturbation Sensitivity (flip-based, not variance-based)
# ═══════════════════════════════════════════════════════════════════════════════
#
# v1 problem: measured score VARIANCE under noise. But normal fakes near the
# decision boundary naturally have high variance. That's not adversarial.
#
# v2 fix: only count DECISION FLIPS, and weight by how much noise was needed.
# Adversarial examples are fragile: even tiny noise (35-40dB SNR) flips them.
# Normal borderline cases might flip at 25dB but not at 35dB.
#
# We test at two noise levels:
#   - Light: 40dB SNR (~0.01% noise) — only adversarial examples flip here
#   - Medium: 30dB SNR (~0.1% noise) — borderline naturals may flip here

N_PERTURBATIONS  = 5
NOISE_LIGHT_DB   = 40.0
NOISE_MEDIUM_DB  = 30.0

def add_noise(wav, snr_db):
    signal_power = wav.pow(2).mean()
    snr_linear   = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_linear
    return wav + torch.randn_like(wav) * noise_power.sqrt()

def compute_perturbation_sensitivity(wav, score_fn, n_runs=N_PERTURBATIONS):
    """
    Flip-based perturbation sensitivity.

    Returns: (score [0,1], detail_dict)
    """
    original_score = score_fn(wav)
    original_pred  = int(original_score >= 0.5)

    # Light noise — adversarial-grade sensitivity
    light_scores = []
    light_flips  = 0
    for _ in range(n_runs):
        s = score_fn(add_noise(wav, NOISE_LIGHT_DB))
        light_scores.append(s)
        if int(s >= 0.5) != original_pred:
            light_flips += 1

    # Medium noise — natural borderline sensitivity
    medium_scores = []
    medium_flips  = 0
    for _ in range(n_runs):
        s = score_fn(add_noise(wav, NOISE_MEDIUM_DB))
        medium_scores.append(s)
        if int(s >= 0.5) != original_pred:
            medium_flips += 1

    light_scores  = np.array(light_scores)
    medium_scores = np.array(medium_scores)

    # ── Adversarial-specific scoring ──────────────────────────────────
    score = 0.0

    # Flips at light noise (40dB) are very suspicious — natural audio doesn't flip
    if light_flips > 0:
        score += 0.5 + 0.1 * min(light_flips, 5)

    # Flips at medium noise (30dB) are mildly suspicious if there were also light flips
    if medium_flips > 0 and light_flips > 0:
        score += 0.15

    # Even without flips, large score shifts at light noise are suspicious
    light_shift = float(np.max(np.abs(light_scores - original_score)))
    if light_shift > 0.15 and light_flips == 0:
        # Score moved a lot but didn't flip — still somewhat suspicious
        score += 0.2 * min(1.0, light_shift / 0.30)

    score = min(1.0, score)

    return score, {
        "score": round(score, 4),
        "original_score": round(original_score, 4),
        "light_flips": light_flips,
        "medium_flips": medium_flips,
        "light_std": round(float(np.std(light_scores)), 4),
        "light_max_shift": round(light_shift, 4),
        "medium_std": round(float(np.std(medium_scores)), 4),
        "n_perturbations": n_runs
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Signal 3: Spectral Anomaly (unchanged from v1 — it wasn't the problem)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spectral_anomaly(wav):
    """
    Returns anomaly score [0, 1] and detail dict.
    """
    wav_np = wav.numpy() if isinstance(wav, torch.Tensor) else wav

    n_fft = 2048
    spectrum = np.abs(np.fft.rfft(wav_np, n=n_fft))
    freqs    = np.fft.rfftfreq(n_fft, d=1.0 / SR)

    # Spectral flatness
    log_spectrum = np.log(spectrum + 1e-10)
    geo_mean     = np.exp(np.mean(log_spectrum))
    arith_mean   = np.mean(spectrum)
    spectral_flatness = float(geo_mean / (arith_mean + 1e-10))

    # High-frequency energy ratio (>6kHz)
    hf_mask  = freqs >= 6000
    hf_energy = float(np.sum(spectrum[hf_mask] ** 2))
    total_energy = float(np.sum(spectrum ** 2)) + 1e-10
    hf_ratio = hf_energy / total_energy

    # Crest factor
    rms = float(np.sqrt(np.mean(wav_np ** 2)))
    peak = float(np.max(np.abs(wav_np)))
    crest_factor = peak / (rms + 1e-10)

    # Noise floor (quietest 10% of frames)
    frame_size = 400
    n_frames   = len(wav_np) // frame_size
    if n_frames > 0:
        frames = wav_np[:n_frames * frame_size].reshape(n_frames, frame_size)
        frame_energy = np.mean(frames ** 2, axis=1)
        noise_floor  = float(np.sqrt(np.percentile(frame_energy, 10)))
    else:
        noise_floor = rms

    # Combine
    anomaly = 0.0
    if spectral_flatness > 0.15:
        anomaly += min(0.35, (spectral_flatness - 0.15) / 0.30 * 0.35)
    if hf_ratio > 0.15:
        anomaly += min(0.30, (hf_ratio - 0.15) / 0.20 * 0.30)
    if crest_factor < 4.0:
        anomaly += min(0.20, (4.0 - crest_factor) / 3.0 * 0.20)
    if noise_floor > 0.01:
        anomaly += min(0.15, (noise_floor - 0.01) / 0.05 * 0.15)
    anomaly = min(1.0, anomaly)

    return anomaly, {
        "score": round(anomaly, 4),
        "spectral_flatness": round(spectral_flatness, 4),
        "hf_energy_ratio": round(hf_ratio, 4),
        "crest_factor": round(crest_factor, 2),
        "noise_floor_rms": round(noise_floor, 6),
        "rms": round(rms, 6),
        "peak": round(peak, 6)
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Adversarial Monitor v2
# ═══════════════════════════════════════════════════════════════════════════════

WEIGHT_DISAGREEMENT  = 0.40
WEIGHT_PERTURBATION  = 0.40
WEIGHT_SPECTRAL      = 0.20

FLAG_THRESHOLD       = 0.45

class AdversarialMonitor:
    """
    Advisory module for detecting deliberate adversarial attacks.
    Completely separate from the primary cascade detector.

    v2: Decoupled from fakeness signal.
    - Disagreement only fires when cascade says "real" but models disagree
    - Perturbation sensitivity only counts decision flips, not score variance
    - Two noise levels distinguish adversarial fragility from natural borderline

    Usage:
        monitor = AdversarialMonitor(aasist, w2v, rn3, xgb_model, cal_a, cal_b)
        result  = monitor.assess(wav)
        # result = {"flag": "normal", "confidence": 0.12, "signals": {...}}
    """

    def __init__(self, aasist, w2v, rn3, xgb_model, cal_a, cal_b, device=DEVICE):
        self.aasist    = aasist
        self.w2v       = w2v
        self.rn3       = rn3
        self.xgb_model = xgb_model
        self.cal_a     = cal_a
        self.cal_b     = cal_b
        self.device    = device

    @torch.no_grad()
    def _get_model_probs(self, wav):
        wav_t = wav.unsqueeze(0).to(self.device)
        prob_a = torch.softmax(self.aasist(wav_t.unsqueeze(1)), dim=-1)[0, 1].item()
        prob_w = torch.softmax(self.w2v(wav_t),                 dim=-1)[0, 1].item()
        prob_r = torch.softmax(self.rn3(wav_t),                 dim=-1)[0, 1].item()
        return prob_a, prob_w, prob_r

    @torch.no_grad()
    def _get_ensemble_prob(self, wav):
        prob_a, prob_w, prob_r = self._get_model_probs(wav)
        feats     = np.array([[prob_a, prob_w, prob_r]])
        xgb_score = self.xgb_model.predict_proba(feats)[0, 1]
        return float(sigmoid_fn(self.cal_a * xgb_score + self.cal_b))

    @torch.no_grad()
    def _get_ensemble_score_only(self, wav):
        """Lighter version for perturbation loop — reuses cached model."""
        return self._get_ensemble_prob(wav)

    def assess(self, wav):
        """
        Run adversarial assessment on a single audio sample.

        Returns dict with flag, confidence, signals breakdown, latency.
        """
        t0 = time.perf_counter()

        # Get individual + ensemble scores
        prob_a, prob_w, prob_r = self._get_model_probs(wav)
        feats     = np.array([[prob_a, prob_w, prob_r]])
        xgb_score = self.xgb_model.predict_proba(feats)[0, 1]
        ensemble_prob = float(sigmoid_fn(self.cal_a * xgb_score + self.cal_b))

        # Signal 1: Ensemble disagreement (conditioned on verdict)
        disagree_score, disagree_detail = compute_disagreement(
            prob_a, prob_w, prob_r, ensemble_prob
        )

        # Signal 2: Perturbation sensitivity (flip-based)
        # ONLY run for samples the ensemble calls "real" — if the ensemble
        # already says fake, perturbation flips are irrelevant (it's caught).
        # Adversarial attacks make fakes look real, so fragility on "real"
        # verdicts is the signal.
        if ensemble_prob < 0.5:
            # Ensemble says real — test if that verdict is fragile
            perturb_score, perturb_detail = compute_perturbation_sensitivity(
                wav, self._get_ensemble_score_only, n_runs=N_PERTURBATIONS
            )
        else:
            # Ensemble says fake — skip perturbation (would waste time
            # and produce false positives on borderline fakes)
            perturb_score = 0.0
            perturb_detail = {
                "score": 0.0,
                "original_score": round(ensemble_prob, 4),
                "light_flips": 0, "medium_flips": 0,
                "light_std": 0.0, "light_max_shift": 0.0,
                "medium_std": 0.0, "n_perturbations": 0,
                "skipped": True,
                "reason": "ensemble_says_fake"
            }

        # Signal 3: Spectral anomaly
        spectral_score, spectral_detail = compute_spectral_anomaly(wav)

        # Weighted combination
        confidence = (
            WEIGHT_DISAGREEMENT * disagree_score +
            WEIGHT_PERTURBATION * perturb_score +
            WEIGHT_SPECTRAL     * spectral_score
        )
        confidence = min(1.0, confidence)

        flag = "attack_suspected" if confidence >= FLAG_THRESHOLD else "normal"

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "flag": flag,
            "confidence": round(confidence, 4),
            "threshold": FLAG_THRESHOLD,
            "signals": {
                "ensemble_disagreement": disagree_detail,
                "perturbation_sensitivity": perturb_detail,
                "spectral_anomaly": spectral_detail
            },
            "latency_ms": round(latency_ms, 1)
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline evaluation function
# ═══════════════════════════════════════════════════════════════════════════════

def run_adversarial_baseline(entries, aasist, w2v, rn3, xgb_model, cal_a, cal_b):
    monitor = AdversarialMonitor(aasist, w2v, rn3, xgb_model, cal_a, cal_b)

    results = []
    print(f"\nRunning adversarial assessment v2 on {len(entries)} samples...")
    t0 = time.time()

    for i, entry in enumerate(entries):
        wav = load_audio(entry["path"])
        if wav is None:
            continue

        result = monitor.assess(wav)
        result["label"]    = entry["label"]
        result["language"] = entry.get("language", "unknown")
        result["source"]   = entry.get("source", "unknown")
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(entries)}] {(i+1)/elapsed:.1f} samples/s")

    print(f"\nAssessed {len(results)} samples in {time.time()-t0:.0f}s")

    # ── Baseline statistics ───────────────────────────────────────────────
    confidences = np.array([r["confidence"] for r in results])
    flags       = [r["flag"] for r in results]
    n_flagged   = sum(1 for f in flags if f == "attack_suspected")

    disagree_scores = np.array([r["signals"]["ensemble_disagreement"]["score"] for r in results])
    perturb_scores  = np.array([r["signals"]["perturbation_sensitivity"]["score"] for r in results])
    spectral_scores = np.array([r["signals"]["spectral_anomaly"]["score"] for r in results])

    print(f"\n{'='*65}")
    print(f"  ADVERSARIAL MONITOR v2 BASELINE — {len(results)} samples")
    print(f"{'='*65}")
    print(f"\n  Flagged as attack_suspected: {n_flagged} / {len(results)} ({n_flagged/len(results)*100:.1f}%)")
    print(f"\n  Combined confidence:     p50={np.median(confidences):.3f}  p95={np.percentile(confidences,95):.3f}  max={np.max(confidences):.3f}")
    print(f"  Ensemble disagreement:   p50={np.median(disagree_scores):.3f}  p95={np.percentile(disagree_scores,95):.3f}  max={np.max(disagree_scores):.3f}")
    print(f"  Perturbation sensitivity:p50={np.median(perturb_scores):.3f}  p95={np.percentile(perturb_scores,95):.3f}  max={np.max(perturb_scores):.3f}")
    print(f"  Spectral anomaly:        p50={np.median(spectral_scores):.3f}  p95={np.percentile(spectral_scores,95):.3f}  max={np.max(spectral_scores):.3f}")

    for cls_name, cls_label in [("Real", 0), ("Fake", 1)]:
        mask = np.array([r["label"] == cls_label for r in results])
        if mask.sum() == 0: continue
        cls_conf = confidences[mask]
        cls_flag = sum(1 for r, m in zip(results, mask) if m and r["flag"] == "attack_suspected")
        cls_dis  = disagree_scores[mask]
        cls_per  = perturb_scores[mask]
        print(f"\n  {cls_name} samples ({mask.sum()}):")
        print(f"    Flagged: {cls_flag} ({cls_flag/mask.sum()*100:.1f}%)")
        print(f"    Confidence:   p50={np.median(cls_conf):.3f}  p95={np.percentile(cls_conf,95):.3f}")
        print(f"    Disagreement: p50={np.median(cls_dis):.3f}   p95={np.percentile(cls_dis,95):.3f}")
        print(f"    Perturbation: p50={np.median(cls_per):.3f}   p95={np.percentile(cls_per,95):.3f}")

    from collections import defaultdict
    lang_groups = defaultdict(list)
    for r in results:
        lang_groups[r["language"]].append(r)

    print(f"\n  {'Language':12s} {'N':>4s} {'Flagged':>8s} {'p50':>6s} {'p95':>6s} {'max':>6s}")
    print(f"  {'-'*46}")
    for lang in sorted(lang_groups):
        lr = lang_groups[lang]
        lc = np.array([r["confidence"] for r in lr])
        lf = sum(1 for r in lr if r["flag"] == "attack_suspected")
        print(f"  {lang:12s} {len(lr):4d} {lf:5d} ({lf/len(lr)*100:4.1f}%) {np.median(lc):.3f} {np.percentile(lc,95):.3f} {np.max(lc):.3f}")

    out_path = "/kaggle/working/adversarial_baseline_v2_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": {
            "n_samples": len(results), "n_flagged": n_flagged,
            "confidence_p50": round(float(np.median(confidences)), 4),
            "confidence_p95": round(float(np.percentile(confidences, 95)), 4),
            "flag_rate_real": round(sum(1 for r in results if r["label"]==0 and r["flag"]=="attack_suspected") / max(1, sum(1 for r in results if r["label"]==0)) * 100, 1),
            "flag_rate_fake": round(sum(1 for r in results if r["label"]==1 and r["flag"]=="attack_suspected") / max(1, sum(1 for r in results if r["label"]==1)) * 100, 1),
            "threshold": FLAG_THRESHOLD},
            "per_sample": results}, f, indent=2)
    print(f"\n✓ Saved to {out_path}")

    return results

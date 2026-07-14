"""
VoiceGuard V8 — AudioSeal Watermark Detection Module (Phase 4)
==============================================================
Wraps Meta's AudioSeal detector to provide a high-precision cryptographic
watermark detection signal alongside V8's deep-model ensemble.

Honest scope notes:
  - AudioSeal ONLY detects watermarks placed by AudioSeal-cooperating
    generators. Most current commercial AI voice systems (ElevenLabs, OpenAI,
    Microsoft, etc.) do NOT use AudioSeal watermarking, so the detector will
    correctly return "not detected" for those. This is expected, not a failure.
  - When AudioSeal does fire, it is near-100% reliable for the specific
    watermark family it was trained on. This is a high-precision signal.
  - Adoption of cooperative watermarking is expected to grow as regulatory
    pressure mounts (EU AI Act, US executive orders). Integrating now means
    V8 is ready for that shift.

Drop-in usage:
    from watermark_detection import detect_watermark
    result = detect_watermark(audio_array, sample_rate)
    # result['detected'] is True/False
    # result['message'] is human-readable text

Install requirement: `pip install audioseal`
"""

import numpy as np
from pathlib import Path

# Lazy-loaded singleton — only instantiate the detector when first called.
# This avoids loading the model at import time (which would slow server startup
# even if no audio is processed).
_detector = None
_load_attempted = False
_load_error = None


def _get_detector():
    """Lazy-load the AudioSeal detector. Returns None if the library is not
    installed or model loading fails — callers should handle that gracefully."""
    global _detector, _load_attempted, _load_error
    if _load_attempted:
        return _detector
    _load_attempted = True
    try:
        from audioseal import AudioSeal
        _detector = AudioSeal.load_detector("audioseal_detector_16bits")
        try:
            _detector.eval()
        except Exception:
            pass
        print("  AudioSeal detector loaded.")
    except ImportError:
        _load_error = (
            "audioseal package not installed. Run: pip install audioseal"
        )
        print(f"  AudioSeal not available: {_load_error}")
    except Exception as e:
        _load_error = f"Failed to load AudioSeal detector: {e}"
        print(f"  AudioSeal load failed: {_load_error}")
    return _detector


def _resample_if_needed(audio, sr, target_sr=16000):
    """AudioSeal expects 16 kHz audio. Resample if needed."""
    if sr == target_sr:
        return audio.astype(np.float32), target_sr
    try:
        from scipy.signal import resample_poly
        # use the closest rational ratio for scipy.signal.resample_poly
        from math import gcd
        g = gcd(int(target_sr), int(sr))
        up   = target_sr // g
        down = sr // g
        resampled = resample_poly(audio, up, down).astype(np.float32)
        return resampled, target_sr
    except Exception:
        # Last resort: simple decimation/interpolation via numpy
        if sr > target_sr:
            step = sr // target_sr
            return audio[::step].astype(np.float32), target_sr
        return audio.astype(np.float32), sr  # leave alone, AudioSeal may handle


def detect_watermark(audio, sample_rate, min_segment_ms=200, threshold=0.5):
    """
    Run AudioSeal's watermark detector on the given audio.

    Args:
        audio: 1D numpy array of audio samples in [-1, 1]
        sample_rate: original sample rate of the audio in Hz
        min_segment_ms: minimum duration for a localized watermark segment to
                        be reported (default 200ms — anything shorter is noise)
        threshold: probability threshold for "watermarked" classification
                   (default 0.5)

    Returns:
        dict with:
          - 'available':   True if AudioSeal is installed and loaded
          - 'detected':    True if any watermark was found
          - 'confidence':  Maximum per-sample watermark probability
          - 'mean_score':  Mean watermark probability across all samples
          - 'coverage':    Fraction of audio classified as watermarked
          - 'localized_segments': List of {start_sec, end_sec, confidence}
                                  for portions of audio with watermark
          - 'message':     Plain-language description of result
          - 'error':       Error string if detection failed (None otherwise)
    """
    # Default result for failure/unavailable cases
    blank = {
        'available':           False,
        'detected':            False,
        'confidence':          0.0,
        'mean_score':          0.0,
        'coverage':            0.0,
        'localized_segments':  [],
        'message': "AudioSeal watermark detection is not available in this "
                   "environment. Install with: pip install audioseal",
        'error':               None,
    }

    detector = _get_detector()
    if detector is None:
        blank['error'] = _load_error
        return blank

    blank['available'] = True

    # Audio prep: ensure mono, float32, in [-1, 1], at 16 kHz
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak > 1e-8 and peak > 1.0:
        audio = audio / peak

    if len(audio) < 1600:  # less than 0.1 seconds
        blank['message'] = ("Audio too short for AudioSeal watermark "
                            "detection (needs at least 100ms).")
        return blank

    audio, sr = _resample_if_needed(audio, sample_rate, target_sr=16000)

    # AudioSeal expects (batch=1, channels=1, samples)
    try:
        import torch
        audio_tensor = torch.from_numpy(audio).reshape(1, 1, -1).float()

        with torch.no_grad():
            # The detect_watermark API typically returns (per_sample_probs,
            # decoded_message) but some versions return just probabilities.
            # We handle both shapes defensively.
            try:
                result = detector.detect_watermark(audio_tensor, sample_rate=sr)
            except TypeError:
                # Older API without sample_rate kwarg
                result = detector.detect_watermark(audio_tensor)

            # result can be:
            #   (probs_tensor, message_tensor)  — newer API
            #   probs_tensor                    — simpler API
            if isinstance(result, tuple):
                probs_tensor = result[0]
            else:
                probs_tensor = result

            # probs_tensor shape can be:
            #   (1, 2, samples)  — class 0 = not-watermark, class 1 = watermark
            #   (1, samples)     — already the watermark probability
            #   (samples,)       — same as above, squeezed
            probs = probs_tensor.detach().cpu().numpy()
            if probs.ndim == 3:
                # take the watermark class probability
                if probs.shape[1] == 2:
                    watermark_probs = probs[0, 1, :]
                else:
                    watermark_probs = probs[0, 0, :]
            elif probs.ndim == 2:
                watermark_probs = probs[0, :]
            else:
                watermark_probs = probs

    except Exception as e:
        blank['error'] = f"Watermark detection raised an exception: {e}"
        blank['message'] = (
            "Could not analyze audio for AudioSeal watermark — the detector "
            "encountered an error. V8's main verdict is still valid; this "
            "supplementary signal is unavailable for this file."
        )
        return blank

    # Compute summary statistics
    max_conf = float(np.max(watermark_probs))
    mean_score = float(np.mean(watermark_probs))
    coverage = float(np.mean(watermark_probs > threshold))

    # Find localized watermark segments (where probability exceeds threshold)
    segments = []
    min_samples = int(min_segment_ms * sr / 1000)
    in_seg = False
    seg_start_idx = 0
    for i, p in enumerate(watermark_probs):
        if p > threshold and not in_seg:
            seg_start_idx = i
            in_seg = True
        elif p <= threshold and in_seg:
            seg_len = i - seg_start_idx
            if seg_len >= min_samples:
                segments.append({
                    'start_sec':  round(seg_start_idx / sr, 2),
                    'end_sec':    round(i / sr, 2),
                    'confidence': round(float(np.mean(
                        watermark_probs[seg_start_idx:i])), 4),
                })
            in_seg = False
    # close any open segment at end of audio
    if in_seg:
        seg_len = len(watermark_probs) - seg_start_idx
        if seg_len >= min_samples:
            segments.append({
                'start_sec':  round(seg_start_idx / sr, 2),
                'end_sec':    round(len(watermark_probs) / sr, 2),
                'confidence': round(float(np.mean(
                    watermark_probs[seg_start_idx:])), 4),
            })

    detected = (max_conf > threshold) and (len(segments) > 0 or coverage > 0.1)

    # Build the plain-language message
    duration = len(watermark_probs) / sr
    if detected:
        if len(segments) == 1 and segments[0]['end_sec'] - segments[0]['start_sec'] >= duration * 0.8:
            message = (f"AudioSeal watermark detected throughout this audio "
                       f"with {max_conf*100:.1f}% confidence. This audio "
                       f"carries a cryptographic watermark identifying it as "
                       f"AI-generated by an AudioSeal-cooperating system.")
        elif len(segments) >= 1:
            total_marked = sum(s['end_sec'] - s['start_sec'] for s in segments)
            message = (f"AudioSeal watermark detected in {len(segments)} "
                       f"segment(s) of the audio (total {total_marked:.1f}s "
                       f"of {duration:.1f}s). Maximum confidence: "
                       f"{max_conf*100:.1f}%. Portions of this audio carry "
                       f"a cryptographic watermark identifying them as "
                       f"AI-generated.")
        else:
            # detected but no localized segment crossed the duration threshold
            message = (f"AudioSeal watermark signal present "
                       f"(max confidence {max_conf*100:.1f}%) but no "
                       f"localized segment of sufficient duration. This may "
                       f"indicate partial or degraded watermarking.")
    else:
        message = (
            "No AudioSeal watermark detected in this audio. Important: "
            "AudioSeal only detects watermarks from generators that "
            "specifically use AudioSeal's watermarking system. Most "
            "commercial AI voice services (ElevenLabs, OpenAI, Microsoft, "
            "etc.) do not currently use AudioSeal. The absence of an "
            "AudioSeal watermark is NOT evidence that the audio is real — "
            "refer to V8's main verdict for the overall determination."
        )

    return {
        'available':           True,
        'detected':            detected,
        'confidence':          round(max_conf, 4),
        'mean_score':          round(mean_score, 4),
        'coverage':            round(coverage, 4),
        'localized_segments':  segments,
        'message':             message,
        'error':               None,
    }


# ────────────────────────────────────────────────────────────────────
# Standalone test: python watermark_detection.py path/to/audio.wav
# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, subprocess, tempfile, os
    if len(sys.argv) < 2:
        print("Usage: python watermark_detection.py <audio_file>")
        sys.exit(1)
    in_path = sys.argv[1]
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t:
        wav_path = t.name
    subprocess.run(
        ['ffmpeg', '-y', '-i', in_path, '-ar', '16000', '-ac', '1',
         '-acodec', 'pcm_s16le', wav_path],
        capture_output=True
    )
    from scipy.io import wavfile
    sr, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    os.unlink(wav_path)

    result = detect_watermark(data, sr)
    print()
    print("=" * 70)
    print("AUDIOSEAL WATERMARK DETECTION RESULT")
    print("=" * 70)
    print(f"  Detected:    {result['detected']}")
    print(f"  Confidence:  {result['confidence']}")
    print(f"  Mean score:  {result['mean_score']}")
    print(f"  Coverage:    {result['coverage']}")
    if result['localized_segments']:
        print(f"\n  Localized segments:")
        for seg in result['localized_segments']:
            print(f"    {seg['start_sec']}s — {seg['end_sec']}s  "
                  f"(confidence {seg['confidence']})")
    print(f"\n  Message: {result['message']}")
    if result['error']:
        print(f"\n  ERROR: {result['error']}")

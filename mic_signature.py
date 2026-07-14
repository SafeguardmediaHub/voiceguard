"""
VoiceGuard V8 — Microphone Signature Analyzer (Phase 4)
========================================================
Examines audio for acoustic signatures consistent with physical microphone
capture: ambient noise floor, low-frequency environmental presence,
spectral noise color, and dynamic-range characteristics.

Honest scope:
  - This module does NOT identify which specific microphone made a
    recording. It detects the PRESENCE vs ABSENCE of microphone-like
    acoustic signatures.
  - Real recordings that have been noise-suppressed, low-bitrate-encoded
    (Opus, AMR), or heavily denoised can look "AI-like" by these metrics
    even though they are genuine. The module flags this caveat explicitly.
  - This is a SUPPLEMENTARY signal, not a primary detector. V8's deep
    model remains the authoritative verdict.

Drop-in usage:
    from mic_signature import analyze_mic_signature
    result = analyze_mic_signature(audio_array, sample_rate)
    # result['findings']: list of observations with lean/confidence
    # result['lean']: overall direction (toward_real / toward_fake / neutral)
    # result['summary']: plain-language summary
"""

import numpy as np
from scipy import signal as sig_proc
from scipy.fft import rfft, rfftfreq


# ════════════════════════════════════════════════════════════════════
# Helpers — frame analysis, energy, silence detection
# ════════════════════════════════════════════════════════════════════

def _frame_energies(audio, sr, frame_ms=20, hop_ms=10):
    """RMS energy per frame. Returns dict with frames, energies (linear),
    energies_db, and time axis."""
    frame_len = int(frame_ms * sr / 1000)
    hop_len   = int(hop_ms   * sr / 1000)
    if len(audio) < frame_len:
        return None
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = audio[i*hop_len : i*hop_len + frame_len]
        energies[i] = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
    energies_db = 20.0 * np.log10(energies / max(np.max(energies), 1e-12))
    times = np.arange(n_frames) * hop_ms / 1000.0
    return {'frame_len': frame_len, 'hop_len': hop_len,
            'energies': energies, 'energies_db': energies_db, 'times': times}


def _identify_silent_frames(energies_db, silence_threshold_db=-35):
    """Boolean mask: True where the frame is below silence threshold."""
    return energies_db < silence_threshold_db


def _segment_silent_regions(audio, sr, silent_mask, frame_data,
                            min_duration_sec=0.15):
    """Return list of (start_sample, end_sample) for continuous silent
    regions of at least min_duration_sec."""
    hop_len = frame_data['hop_len']
    frame_len = frame_data['frame_len']
    min_frames = max(2, int(min_duration_sec * sr / hop_len))

    regions = []
    in_silent = False
    start_idx = 0
    for i, is_silent in enumerate(silent_mask):
        if is_silent and not in_silent:
            start_idx = i
            in_silent = True
        elif not is_silent and in_silent:
            length = i - start_idx
            if length >= min_frames:
                regions.append((start_idx * hop_len,
                                i * hop_len + frame_len))
            in_silent = False
    if in_silent:
        length = len(silent_mask) - start_idx
        if length >= min_frames:
            regions.append((start_idx * hop_len, len(audio)))
    return regions


def _noise_floor_stats(audio, sr, silent_regions):
    """Statistics over the silent regions: mean amplitude in dBFS,
    variance, fraction of samples at absolute digital silence."""
    if not silent_regions:
        return {'mean_db': None, 'std_db': None, 'absolute_silence_frac': None,
                'total_silent_sec': 0.0}

    silent_samples = []
    abs_silence_count = 0
    total_count = 0
    for s, e in silent_regions:
        seg = audio[s:e]
        silent_samples.append(seg)
        abs_silence_count += int(np.sum(np.abs(seg) < 1e-6))
        total_count += len(seg)

    if not silent_samples:
        return {'mean_db': None, 'std_db': None, 'absolute_silence_frac': None,
                'total_silent_sec': 0.0}

    combined = np.concatenate(silent_samples)
    if len(combined) < 10:
        return {'mean_db': None, 'std_db': None, 'absolute_silence_frac': None,
                'total_silent_sec': float(len(combined) / sr)}

    rms = float(np.sqrt(np.mean(combined ** 2)) + 1e-12)
    mean_db = 20.0 * np.log10(rms)

    # std of the dB-scale energy across short windows within silent regions
    win_len = max(160, int(0.01 * sr))  # 10ms windows
    short_rms = []
    for s, e in silent_regions:
        seg = audio[s:e]
        for i in range(0, len(seg) - win_len, win_len):
            short_rms.append(np.sqrt(np.mean(seg[i:i+win_len] ** 2)) + 1e-12)
    if len(short_rms) > 1:
        std_db = float(np.std(20.0 * np.log10(np.array(short_rms))))
    else:
        std_db = 0.0

    return {
        'mean_db':                mean_db,
        'std_db':                 std_db,
        'absolute_silence_frac':  abs_silence_count / max(total_count, 1),
        'total_silent_sec':       len(combined) / sr,
    }


def _low_frequency_presence(audio, sr, lf_cutoff_hz=80):
    """Energy below lf_cutoff_hz as a fraction of total energy. Real
    recordings usually have measurable low-frequency environmental content."""
    if len(audio) < 1024:
        return None
    n = min(len(audio), 32768)
    win = np.hanning(n)
    spec = np.abs(rfft(audio[:n] * win)) ** 2
    freqs = rfftfreq(n, 1.0 / sr)
    total = float(np.sum(spec) + 1e-12)
    low = float(np.sum(spec[freqs < lf_cutoff_hz]))
    return low / total


def _noise_color(audio, sr, silent_regions):
    """Estimate the spectral 'color' of noise in silent regions. Returns
    the spectral slope (in dB/decade) — pink noise is around -10, white
    noise is around 0, brown noise around -20. Real environments tend
    to be pink-to-brown; some AI 'silence' is white-noise-like or has
    no consistent color."""
    if not silent_regions:
        return None
    silent_audio = []
    for s, e in silent_regions:
        silent_audio.append(audio[s:e])
    if not silent_audio:
        return None
    combined = np.concatenate(silent_audio)
    if len(combined) < 4096:
        return None

    n = min(len(combined), 16384)
    win = np.hanning(n)
    spec = np.abs(rfft(combined[:n] * win)) ** 2
    freqs = rfftfreq(n, 1.0 / sr)

    # only fit over the useful band (avoid DC and very high frequencies)
    fmin, fmax = 50, min(sr // 2 - 100, 4000)
    mask = (freqs >= fmin) & (freqs <= fmax) & (spec > 0)
    if np.sum(mask) < 20:
        return None
    log_f = np.log10(freqs[mask])
    log_p = 10.0 * np.log10(spec[mask] + 1e-12)
    # linear fit: log_p = slope * log_f + intercept
    slope, _ = np.polyfit(log_f, log_p, 1)
    return float(slope)


# ════════════════════════════════════════════════════════════════════
# Interpretation — plain-language findings
# ════════════════════════════════════════════════════════════════════

def _interpret_noise_floor(stats):
    """Interpret noise-floor statistics."""
    findings = []

    if stats['absolute_silence_frac'] is None:
        return findings  # no silent regions found

    # Absolute digital silence — very rare in real recordings
    if stats['absolute_silence_frac'] > 0.5:
        findings.append({
            'title': 'Silent regions contain absolute digital silence',
            'detail': (
                f"{stats['absolute_silence_frac']*100:.0f}% of the 'silent' "
                f"portions of this audio are at true digital zero — meaning "
                f"the audio file contains exact 0.0 sample values. Real "
                f"microphone recordings essentially never produce true "
                f"digital silence; even quiet rooms have measurable noise "
                f"floor. This is a strong indicator of synthesized audio "
                f"or aggressive noise gating."
            ),
            'lean': 'toward_fake', 'confidence': 'high',
        })
    elif stats['absolute_silence_frac'] > 0.1:
        findings.append({
            'title': 'Some segments contain absolute digital silence',
            'detail': (
                f"{stats['absolute_silence_frac']*100:.0f}% of silent regions "
                f"contain true digital zero values. This is unusual for real "
                f"recordings and could indicate noise gating, AI synthesis, "
                f"or heavy post-processing."
            ),
            'lean': 'toward_fake', 'confidence': 'medium',
        })

    # Mean noise-floor level
    if stats['mean_db'] is not None:
        if stats['mean_db'] < -80:
            findings.append({
                'title': 'Noise floor is unusually quiet',
                'detail': (
                    f"The audio's silent regions measure at {stats['mean_db']:.1f} "
                    f"dBFS — significantly quieter than typical real "
                    f"recordings. Real microphone audio usually has a noise "
                    f"floor between -60 and -50 dBFS due to mic self-noise "
                    f"and ambient room noise. Very-quiet noise floors can "
                    f"indicate synthesized audio or heavy noise reduction."
                ),
                'lean': 'toward_fake', 'confidence': 'medium',
            })
        elif -65 <= stats['mean_db'] <= -40:
            findings.append({
                'title': 'Noise floor is in the typical microphone range',
                'detail': (
                    f"Silent regions measure at {stats['mean_db']:.1f} dBFS, "
                    f"consistent with a real microphone recording. Natural "
                    f"environmental and mic self-noise is present at the "
                    f"expected level."
                ),
                'lean': 'toward_real', 'confidence': 'medium',
            })

    # Variability of noise floor — real noise fluctuates; synthetic often doesn't
    if stats['std_db'] is not None and stats['mean_db'] is not None:
        if stats['std_db'] < 0.5 and stats['mean_db'] > -90:
            findings.append({
                'title': 'Noise floor is unnaturally constant',
                'detail': (
                    f"The noise level in silent regions is extremely stable "
                    f"(variation < 0.5 dB). Real environmental noise "
                    f"naturally fluctuates with people moving, air systems, "
                    f"and other ambient sources. Unnaturally constant "
                    f"noise can suggest synthesis or processed audio."
                ),
                'lean': 'toward_fake', 'confidence': 'low',
            })

    return findings


def _interpret_low_freq(low_freq_frac):
    """Interpret low-frequency content presence."""
    if low_freq_frac is None:
        return []
    # Real recordings: typically 5-25% of energy in <80Hz from environmental
    # rumble, HVAC, body movement. AI-clean audio: often <2%.
    if low_freq_frac < 0.01:
        return [{
            'title': 'Almost no low-frequency environmental content',
            'detail': (
                f"Only {low_freq_frac*100:.2f}% of the audio's energy is "
                f"below 80 Hz. Real microphone recordings typically contain "
                f"5-25% of energy in this range from environmental rumble, "
                f"HVAC systems, traffic, and physical movement. Near-absence "
                f"of low-frequency content suggests either AI synthesis "
                f"or aggressive high-pass filtering."
            ),
            'lean': 'toward_fake', 'confidence': 'medium',
        }]
    elif low_freq_frac < 0.04:
        return [{
            'title': 'Low-frequency content is reduced',
            'detail': (
                f"Energy below 80 Hz is {low_freq_frac*100:.2f}% — somewhat "
                f"low for a typical microphone recording. Could indicate "
                f"high-pass filtering, low-bitrate codec processing, or "
                f"AI synthesis."
            ),
            'lean': 'neutral', 'confidence': 'low',
        }]
    elif low_freq_frac > 0.05:
        return [{
            'title': 'Low-frequency environmental content is present',
            'detail': (
                f"Low-frequency energy ({low_freq_frac*100:.1f}%) is consistent "
                f"with environmental noise typical of microphone recordings — "
                f"HVAC, traffic rumble, body movement, etc."
            ),
            'lean': 'toward_real', 'confidence': 'medium',
        }]
    return []


def _interpret_noise_color(slope):
    """Interpret the spectral slope of silent-region noise."""
    if slope is None:
        return []
    # Real environments: pink to brown noise, slope ~ -10 to -25 dB/decade
    # White noise: slope ~ 0
    # AI 'clean' silence may have flat or no consistent slope
    if -5 < slope < 5:
        return [{
            'title': 'Noise color is unusually flat',
            'detail': (
                f"The noise in silent regions has a spectral slope of "
                f"{slope:+.1f} dB/decade — close to flat (white-noise-like). "
                f"Real environmental noise typically has a pink or brown "
                f"character (slope -10 to -25). Flat noise can suggest "
                f"synthesized audio or codec-induced artifacts."
            ),
            'lean': 'toward_fake', 'confidence': 'low',
        }]
    elif -25 < slope < -5:
        return [{
            'title': 'Noise color is consistent with environmental sound',
            'detail': (
                f"Spectral slope of silent regions ({slope:+.1f} dB/decade) "
                f"is in the pink-to-brown range typical of real "
                f"environmental noise."
            ),
            'lean': 'toward_real', 'confidence': 'low',
        }]
    return []


# ════════════════════════════════════════════════════════════════════
# Public interface
# ════════════════════════════════════════════════════════════════════

def analyze_mic_signature(audio, sample_rate):
    """
    Analyze audio for acoustic signatures consistent with physical
    microphone capture.

    Args:
        audio: 1D numpy array, samples in [-1, 1]
        sample_rate: sampling rate in Hz

    Returns:
        dict with:
          - 'available':  True if analysis ran successfully
          - 'findings':   list of finding dicts (title, detail, lean, confidence)
          - 'lean':       overall direction (toward_real/toward_fake/neutral)
          - 'summary':    plain-language summary
          - 'measurements': raw numerical features
          - 'caveat':     standard caveat about false positives on processed audio
          - 'error':      error string if analysis failed
    """
    blank = {
        'available':    False,
        'findings':     [],
        'lean':         'neutral',
        'summary':      '',
        'measurements': {},
        'caveat':       '',
        'error':        None,
    }

    if audio is None or len(audio) < sample_rate * 0.5:
        blank['error'] = 'Audio too short for microphone signature analysis (< 0.5s)'
        return blank

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio_norm = audio / peak
    else:
        blank['error'] = 'Audio appears silent (peak amplitude near zero)'
        return blank

    try:
        frame_data = _frame_energies(audio_norm, sample_rate)
        if frame_data is None:
            blank['error'] = 'Could not frame audio'
            return blank

        silent_mask = _identify_silent_frames(frame_data['energies_db'])
        silent_regions = _segment_silent_regions(
            audio_norm, sample_rate, silent_mask, frame_data
        )

        # If we have basically no silent regions, the audio is wall-to-wall
        # speech and the signal-based checks aren't applicable
        total_silent = sum(e - s for s, e in silent_regions) / sample_rate
        if total_silent < 0.1:
            blank['available'] = True
            blank['summary'] = (
                "This audio has no extended silent regions, so microphone "
                "signature analysis cannot examine the noise floor. The "
                "analysis is most useful on audio with natural pauses where "
                "ambient sound is audible."
            )
            blank['caveat'] = (
                "This module analyzes silent regions of audio to detect "
                "microphone-recording characteristics. Audio with no pauses "
                "cannot be analyzed this way."
            )
            blank['measurements']['total_silent_sec'] = round(total_silent, 2)
            return blank

        # Run the measurements
        noise_stats = _noise_floor_stats(audio_norm, sample_rate, silent_regions)
        low_freq = _low_frequency_presence(audio_norm, sample_rate)
        noise_slope = _noise_color(audio_norm, sample_rate, silent_regions)

        measurements = {
            'total_silent_sec':          round(noise_stats['total_silent_sec'], 2),
            'noise_floor_db':            (round(noise_stats['mean_db'], 1)
                                          if noise_stats['mean_db'] is not None
                                          else None),
            'noise_floor_std_db':        (round(noise_stats['std_db'], 2)
                                          if noise_stats['std_db'] is not None
                                          else None),
            'absolute_silence_fraction': (round(noise_stats['absolute_silence_frac'], 4)
                                          if noise_stats['absolute_silence_frac'] is not None
                                          else None),
            'low_freq_energy_fraction':  (round(low_freq, 4)
                                          if low_freq is not None else None),
            'noise_spectral_slope_db_per_decade': (round(noise_slope, 2)
                                                   if noise_slope is not None
                                                   else None),
        }

        # Generate findings
        findings = []
        findings += _interpret_noise_floor(noise_stats)
        findings += _interpret_low_freq(low_freq)
        findings += _interpret_noise_color(noise_slope)

        # Overall lean — weight by confidence
        weight = {'high': 3, 'medium': 2, 'low': 1}
        fake_w = sum(weight.get(f['confidence'], 1)
                     for f in findings if f['lean'] == 'toward_fake')
        real_w = sum(weight.get(f['confidence'], 1)
                     for f in findings if f['lean'] == 'toward_real')

        if fake_w - real_w >= 3:
            overall_lean = 'toward_fake'
        elif real_w - fake_w >= 3:
            overall_lean = 'toward_real'
        else:
            overall_lean = 'neutral'

        # Build summary
        fake_count = sum(1 for f in findings if f['lean'] == 'toward_fake')
        real_count = sum(1 for f in findings if f['lean'] == 'toward_real')

        if overall_lean == 'toward_fake':
            summary = (
                f"Microphone signature analysis found {fake_count} signal(s) "
                f"suggesting the audio lacks physical-recording characteristics. "
                f"The silent regions don't show the noise floor and environmental "
                f"content typical of real microphone capture."
            )
        elif overall_lean == 'toward_real':
            summary = (
                f"Microphone signature analysis found {real_count} signal(s) "
                f"consistent with physical microphone capture. The audio shows "
                f"the noise floor and environmental characteristics typical "
                f"of a real recording."
            )
        else:
            summary = (
                "Microphone signature analysis is mixed or inconclusive. "
                "Some characteristics suggest physical recording, others "
                "suggest processing or synthesis."
            )

        caveat = (
            "Note: real recordings that have been heavily noise-suppressed "
            "(modern smartphone audio is aggressively de-noised), "
            "low-bitrate-encoded (Opus, AMR), or post-produced may show "
            "'AI-like' microphone signatures even though they are genuine. "
            "Treat these findings as supplementary to V8's main verdict, "
            "not as definitive evidence."
        )

        blank['available']    = True
        blank['findings']     = findings
        blank['lean']         = overall_lean
        blank['summary']      = summary
        blank['measurements'] = measurements
        blank['caveat']       = caveat
        return blank

    except Exception as e:
        blank['error'] = f"Microphone signature analysis failed: {e}"
        return blank


# ════════════════════════════════════════════════════════════════════
# Standalone test: python mic_signature.py <audio_file>
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys, subprocess, tempfile, os
    if len(sys.argv) < 2:
        print("Usage: python mic_signature.py <audio_file>")
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

    result = analyze_mic_signature(data, sr)
    print()
    print("=" * 70)
    print("MICROPHONE SIGNATURE ANALYSIS")
    print("=" * 70)
    print(f"\n  Available:    {result['available']}")
    if result['error']:
        print(f"  ERROR:       {result['error']}")
    print(f"  Overall lean: {result['lean']}")
    print(f"\n  Summary: {result['summary']}\n")

    if result['findings']:
        print(f"  Findings ({len(result['findings'])}):")
        for f in result['findings']:
            marker = {'toward_fake': '⚠', 'toward_real': '✓',
                      'neutral': '○'}.get(f['lean'], '•')
            print(f"\n    {marker} [{f['confidence']}] {f['title']}")
            print(f"      {f['detail']}")

    if result['measurements']:
        print(f"\n  Measurements:")
        for k, v in result['measurements'].items():
            print(f"    {k:<40s} {v}")

    if result['caveat']:
        print(f"\n  Caveat: {result['caveat']}")

"""
VoiceGuard V8 — Explainability Module
=====================================
Extracts acoustic measurements that humans use (unconsciously) to distinguish
real speech from AI-generated audio, and translates each measurement into
plain English a non-technical user can understand.

Drop-in usage from server.py:
    from explainability import explain_audio
    explanation = explain_audio(audio_array, sample_rate)
    # explanation is a dict with 'observations' (list of plain-text bullets)
    # and 'measurements' (dict of raw numbers, for completeness)

Honest scope notes:
  - These are *signals consistent with* fake or real, not definitive proofs.
    The deep model (V8) makes the actual decision; these explanations
    describe *what V8 may be seeing* in terms a human can understand.
  - Thresholds are based on voice-quality research literature and are
    approximate. Real human speech varies a lot by speaker, language, and
    emotion. Treat the labels (low/typical/high) as rough guides.
  - No external libraries beyond numpy/scipy — works in any V8 environment.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq


# ────────────────────────────────────────────────────────────────────
# Low-level feature extractors (numpy + scipy only)
# ────────────────────────────────────────────────────────────────────

def _frame_audio(audio, sr, frame_ms=30, hop_ms=10):
    """Split audio into overlapping frames. Returns (frames, n_frames)."""
    frame_len = int(frame_ms * sr / 1000)
    hop_len   = int(hop_ms   * sr / 1000)
    if len(audio) < frame_len:
        return np.array([]), 0
    n = 1 + (len(audio) - frame_len) // hop_len
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n, frame_len),
        strides=(audio.strides[0] * hop_len, audio.strides[0]),
        writeable=False,
    ).copy()
    return frames, n


def _autocorr_f0(frame, sr, f_min=70, f_max=400):
    """Estimate F0 of one frame via autocorrelation.
    Returns 0 if unvoiced (low energy or no clear peak)."""
    if np.std(frame) < 1e-4:
        return 0.0
    frame = frame - np.mean(frame)
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr) // 2:]
    if corr[0] < 1e-6:
        return 0.0
    corr = corr / corr[0]
    min_lag = int(sr / f_max)
    max_lag = int(sr / f_min)
    if max_lag >= len(corr):
        return 0.0
    search = corr[min_lag:max_lag]
    if len(search) == 0:
        return 0.0
    peak = np.argmax(search) + min_lag
    if corr[peak] < 0.3:
        return 0.0
    return float(sr / peak)


def _extract_f0_contour(audio, sr):
    """Return array of F0 values per frame (0 where unvoiced)."""
    frames, n = _frame_audio(audio, sr)
    if n == 0:
        return np.array([])
    f0 = np.array([_autocorr_f0(f, sr) for f in frames])
    return f0


def _local_jitter(f0):
    """Local jitter — average abs difference between consecutive F0 periods,
    normalized by mean period. Returns percentage."""
    voiced = f0[f0 > 0]
    if len(voiced) < 3:
        return 0.0
    periods = 1.0 / voiced
    diffs = np.abs(np.diff(periods))
    return float(100 * np.mean(diffs) / np.mean(periods))


def _local_shimmer(audio, sr, f0):
    """Local shimmer — cycle-to-cycle amplitude variation in dB.
    Uses RMS amplitude per estimated glottal cycle."""
    voiced_frames = np.where(f0 > 0)[0]
    if len(voiced_frames) < 3:
        return 0.0
    frame_len = int(0.030 * sr)
    hop = int(0.010 * sr)
    amps = []
    for fi in voiced_frames:
        start = fi * hop
        end = min(start + frame_len, len(audio))
        if end - start < 100:
            continue
        amps.append(np.sqrt(np.mean(audio[start:end] ** 2)) + 1e-9)
    amps = np.array(amps)
    if len(amps) < 3:
        return 0.0
    # Convert to dB and take mean absolute difference between consecutive frames
    db = 20 * np.log10(amps)
    return float(np.mean(np.abs(np.diff(db))))


def _hnr_db(audio, sr):
    """Harmonics-to-noise ratio in dB via autocorrelation peak."""
    frames, n = _frame_audio(audio, sr)
    if n == 0:
        return 0.0
    vals = []
    for f in frames:
        if np.std(f) < 1e-4:
            continue
        f = f - np.mean(f)
        corr = np.correlate(f, f, mode='full')
        corr = corr[len(corr) // 2:]
        if corr[0] < 1e-6:
            continue
        corr = corr / corr[0]
        min_lag = int(sr / 400)
        max_lag = int(sr / 70)
        if max_lag >= len(corr):
            continue
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            continue
        peak_val = float(np.max(search))
        if peak_val <= 0 or peak_val >= 0.999:
            continue
        vals.append(10 * np.log10(peak_val / (1 - peak_val)))
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def _spectral_flatness(audio, sr):
    """Spectral flatness (Wiener entropy): geometric mean / arithmetic mean
    of the power spectrum. 0 = pure tone, 1 = white noise."""
    if len(audio) < 1024:
        return 0.0
    n = min(len(audio), 16384)
    win = np.hanning(n)
    spec = np.abs(rfft(audio[:n] * win)) ** 2
    spec = spec + 1e-12
    geom = np.exp(np.mean(np.log(spec)))
    arith = np.mean(spec)
    return float(geom / arith)


def _high_freq_ratio(audio, sr, cutoff_hz=4000):
    """Ratio of energy above cutoff_hz to total energy."""
    if len(audio) < 1024:
        return 0.0
    n = min(len(audio), 16384)
    win = np.hanning(n)
    spec = np.abs(rfft(audio[:n] * win)) ** 2
    freqs = rfftfreq(n, 1 / sr)
    total = np.sum(spec) + 1e-12
    high = np.sum(spec[freqs >= cutoff_hz])
    return float(high / total)


def _silence_pattern(audio, sr, threshold_db=-40):
    """Analyze silence/pause distribution. Returns:
      - silence_ratio: fraction of audio under threshold
      - pause_count_per_sec: how many silent intervals per second
      - pause_regularity: std-dev of pause durations (lower = more regular)
    """
    frame_len = int(0.020 * sr)
    hop = int(0.010 * sr)
    if len(audio) < frame_len:
        return {'silence_ratio': 0.0, 'pause_count_per_sec': 0.0,
                'pause_regularity': 0.0}
    n_frames = 1 + (len(audio) - frame_len) // hop
    energies = np.array([
        np.sqrt(np.mean(audio[i*hop:i*hop+frame_len] ** 2)) + 1e-9
        for i in range(n_frames)
    ])
    db = 20 * np.log10(energies / np.max(energies))
    is_silent = db < threshold_db
    silence_ratio = float(np.mean(is_silent))

    # find silent intervals
    in_pause = False
    pause_durs = []
    cur_len = 0
    for s in is_silent:
        if s:
            cur_len += 1
            in_pause = True
        else:
            if in_pause and cur_len > 3:  # minimum 30ms = real pause
                pause_durs.append(cur_len * 0.010)  # in seconds
            cur_len = 0
            in_pause = False
    if in_pause and cur_len > 3:
        pause_durs.append(cur_len * 0.010)

    duration = len(audio) / sr
    pause_count_per_sec = len(pause_durs) / max(duration, 0.1)
    pause_regularity = float(np.std(pause_durs)) if len(pause_durs) > 1 else 0.0
    return {
        'silence_ratio': silence_ratio,
        'pause_count_per_sec': pause_count_per_sec,
        'pause_regularity': pause_regularity,
    }


# ────────────────────────────────────────────────────────────────────
# Plain-language interpretation
# ────────────────────────────────────────────────────────────────────

def _interpret_f0_variation(f0_std):
    """F0 standard deviation in Hz. Real speech typically 15-40 Hz."""
    if f0_std < 5:
        return ('Pitch variation is very low',
                'Real speech naturally moves in pitch as people emphasize words. '
                'Audio with almost no pitch movement can suggest synthesis, '
                'though some speakers naturally speak in a monotone.')
    elif f0_std < 12:
        return ('Pitch variation is below typical',
                'Real speech usually has more natural pitch variation than this. '
                'Constrained pitch range is sometimes seen in AI voices, but '
                'also in calm or reading speech.')
    elif f0_std < 50:
        return ('Pitch variation is in the typical range for real speech',
                'Natural pitch movement consistent with normal human speech.')
    else:
        return ('Pitch variation is unusually wide',
                'Very wide pitch swings — could indicate emotional speech, '
                'singing, or unusual recording conditions.')


def _interpret_jitter(jitter_pct):
    """Local jitter. Real speech 0.5-2%."""
    if jitter_pct < 0.2:
        return ('Voice cycle irregularity (jitter) is very low',
                'Real vocal cords vibrate with small natural irregularities '
                'between cycles. Audio with extremely smooth, regular cycles '
                'often suggests synthesis.')
    elif jitter_pct < 0.5:
        return ('Voice cycle irregularity (jitter) is below typical',
                'Slightly smoother than typical human speech.')
    elif jitter_pct < 2.5:
        return ('Voice cycle irregularity is in the typical human range',
                'Natural micro-variation in vocal cord cycles — consistent '
                'with real speech.')
    else:
        return ('Voice cycle irregularity is high',
                'Very irregular cycles — could indicate a hoarse or '
                'pathological voice, or recording artifacts.')


def _interpret_shimmer(shimmer_db):
    """Local shimmer in dB. Real speech 0.5-3 dB."""
    if shimmer_db < 0.3:
        return ('Amplitude variation (shimmer) is very low',
                'Real speech has subtle variation in loudness from one '
                'syllable to the next. Audio with too-even amplitude can '
                'suggest synthesis.')
    elif shimmer_db < 0.8:
        return ('Amplitude variation is below typical',
                'Smoother amplitude than typical natural speech.')
    elif shimmer_db < 4:
        return ('Amplitude variation is in the typical human range',
                'Natural loudness variation consistent with real speech.')
    else:
        return ('Amplitude variation is high',
                'Strong amplitude variation — could indicate emotional '
                'speech, recording issues, or unusual content.')


def _interpret_hnr(hnr_db):
    """Harmonics-to-noise ratio. Real speech 15-25 dB."""
    if hnr_db < 8:
        return ('Voice signal is noisy or unclear',
                'Background noise or recording quality may be affecting the '
                'analysis. Hard to draw conclusions about real vs AI from '
                'noisy audio.')
    elif hnr_db < 18:
        return ('Voice clarity is in the typical range',
                'Normal voice-to-background ratio.')
    elif hnr_db < 28:
        return ('Voice signal is unusually clean',
                'Audio is cleaner than most real-world recordings. Some AI '
                'audio is unnaturally noise-free; though studio recordings '
                'can also look like this.')
    else:
        return ('Voice signal is extremely clean',
                'Very little background or vocal noise. This level of '
                'cleanliness is rarely seen in real-world phone or call '
                'recordings, and is more typical of synthesized audio.')


def _interpret_spectral_flatness(sf):
    """Spectral flatness. Real speech 0.1-0.3."""
    if sf < 0.05:
        return ('Frequency spectrum is unusually concentrated',
                'Very tonal spectrum — could indicate music, noise, or '
                'unusual synthesis. Atypical for normal speech.')
    elif sf < 0.4:
        return ('Frequency spectrum is in the typical speech range',
                'Spectral characteristics consistent with natural speech.')
    else:
        return ('Frequency spectrum is unusually flat',
                'Very flat or noisy spectrum — could indicate heavy '
                'compression, background noise, or specific synthesis methods.')


def _interpret_high_freq(hf_ratio):
    """High-frequency energy ratio (above 4kHz)."""
    if hf_ratio < 0.02:
        return ('Very little high-frequency content',
                'Audio has been heavily filtered, downsampled, or comes from '
                'a low-bandwidth source (e.g. telephone). Note: some AI '
                'voices also exhibit this pattern.')
    elif hf_ratio < 0.15:
        return ('High-frequency content is in the typical speech range',
                'Spectral balance consistent with natural speech.')
    else:
        return ('High-frequency content is elevated',
                'Higher than typical high-frequency energy. Could indicate '
                'sharp consonants, sibilance, or specific synthesis vocoder '
                'characteristics.')


def _interpret_pauses(pause_data):
    """Pause patterns. Real speech has irregular pauses."""
    p_count = pause_data['pause_count_per_sec']
    p_reg = pause_data['pause_regularity']

    msgs = []
    if p_count < 0.1:
        msgs.append(('Almost no pauses detected',
                     'The audio is continuous speech with no natural breaks. '
                     'Real speech usually has some micro-pauses for breath '
                     'or thought. Very fluent AI synthesis can also look '
                     'like this.'))
    elif p_count > 1.5:
        msgs.append(('Many short pauses',
                     'Frequent pauses — could indicate hesitant speech, '
                     'segmented synthesis, or specific speaking style.'))

    if 0.05 < p_reg < 0.15 and p_count > 0.3:
        msgs.append(('Pause patterns are unusually regular',
                     'When pauses occur at very consistent intervals, this '
                     'can sometimes suggest synthesized audio (real human '
                     'pauses tend to be more irregular).'))

    return msgs


# ────────────────────────────────────────────────────────────────────
# Public interface
# ────────────────────────────────────────────────────────────────────

def explain_audio(audio, sample_rate, verdict=None):
    """
    Extract acoustic features and produce plain-language explanations.

    Args:
        audio: 1D numpy array of audio samples (float in [-1, 1])
        sample_rate: sampling rate in Hz
        verdict: optional V8 verdict string (e.g. "AUTO_FAKE") — used to
                 frame the explanation appropriately

    Returns:
        dict with:
          - 'observations': list of dicts, each {'title': str, 'detail': str,
                            'lean': 'toward_fake'|'toward_real'|'neutral'}
          - 'measurements': dict of raw numerical features
          - 'summary': single-sentence plain-language summary
    """
    # ensure mono float32
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    # normalize
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak

    # extract features
    f0 = _extract_f0_contour(audio, sample_rate)
    voiced = f0[f0 > 0]
    f0_std = float(np.std(voiced)) if len(voiced) > 5 else 0.0
    f0_mean = float(np.mean(voiced)) if len(voiced) > 5 else 0.0

    jitter = _local_jitter(f0)
    shimmer = _local_shimmer(audio, sample_rate, f0)
    hnr = _hnr_db(audio, sample_rate)
    spectral_flat = _spectral_flatness(audio, sample_rate)
    high_freq = _high_freq_ratio(audio, sample_rate)
    pause_data = _silence_pattern(audio, sample_rate)

    measurements = {
        'f0_mean_hz':            round(f0_mean, 1),
        'f0_std_hz':             round(f0_std, 1),
        'jitter_pct':            round(jitter, 3),
        'shimmer_db':            round(shimmer, 3),
        'hnr_db':                round(hnr, 2),
        'spectral_flatness':     round(spectral_flat, 4),
        'high_freq_ratio':       round(high_freq, 4),
        'pause_count_per_sec':   round(pause_data['pause_count_per_sec'], 2),
        'pause_regularity_sec':  round(pause_data['pause_regularity'], 3),
        'silence_ratio':         round(pause_data['silence_ratio'], 3),
    }

    # build observations
    observations = []

    def add_obs(interpretation, lean):
        title, detail = interpretation
        observations.append({'title': title, 'detail': detail, 'lean': lean})

    # pitch variation
    pitch_interp = _interpret_f0_variation(f0_std)
    if f0_std > 0:  # only meaningful if we found voiced speech
        if f0_std < 12:
            add_obs(pitch_interp, 'toward_fake')
        elif f0_std < 50:
            add_obs(pitch_interp, 'toward_real')
        else:
            add_obs(pitch_interp, 'neutral')

    # jitter
    jitter_interp = _interpret_jitter(jitter)
    if jitter > 0:
        if jitter < 0.5:
            add_obs(jitter_interp, 'toward_fake')
        elif jitter < 2.5:
            add_obs(jitter_interp, 'toward_real')
        else:
            add_obs(jitter_interp, 'neutral')

    # shimmer
    shimmer_interp = _interpret_shimmer(shimmer)
    if shimmer > 0:
        if shimmer < 0.8:
            add_obs(shimmer_interp, 'toward_fake')
        elif shimmer < 4:
            add_obs(shimmer_interp, 'toward_real')
        else:
            add_obs(shimmer_interp, 'neutral')

    # HNR
    hnr_interp = _interpret_hnr(hnr)
    if hnr > 5:  # below 5 dB the analysis isn't reliable
        if hnr > 28:
            add_obs(hnr_interp, 'toward_fake')
        elif hnr < 18:
            add_obs(hnr_interp, 'toward_real')
        else:
            add_obs(hnr_interp, 'toward_real')

    # spectral flatness
    sf_interp = _interpret_spectral_flatness(spectral_flat)
    if spectral_flat < 0.05 or spectral_flat > 0.4:
        add_obs(sf_interp, 'neutral')
    else:
        add_obs(sf_interp, 'toward_real')

    # high frequency
    hf_interp = _interpret_high_freq(high_freq)
    add_obs(hf_interp, 'neutral')

    # pauses
    for pause_msg in _interpret_pauses(pause_data):
        add_obs(pause_msg, 'toward_fake' if 'regular' in pause_msg[0].lower()
                                          or 'no pauses' in pause_msg[0].lower()
                else 'neutral')

    # summary
    fake_signals = sum(1 for o in observations if o['lean'] == 'toward_fake')
    real_signals = sum(1 for o in observations if o['lean'] == 'toward_real')

    if verdict in ('AUTO_FAKE', 'LIKELY_FAKE'):
        if fake_signals > real_signals:
            summary = (f"The audio shows {fake_signals} measurable signs more "
                       f"consistent with AI-generated speech than with natural "
                       f"speech. The detector's verdict aligns with these "
                       f"signals.")
        else:
            summary = (f"V8 flagged this as suspicious based on patterns its "
                       f"trained model detected. The acoustic measurements "
                       f"alone are mixed — the detector likely identified "
                       f"more subtle synthesis fingerprints than these "
                       f"surface measurements capture.")
    elif verdict == 'AUTO_REAL':
        if real_signals > fake_signals:
            summary = (f"The audio shows {real_signals} measurable signs "
                       f"consistent with natural human speech. The detector's "
                       f"verdict aligns with these signals.")
        else:
            summary = (f"V8 cleared this as natural speech. While some "
                       f"individual measurements are atypical, the overall "
                       f"pattern is consistent with real speech in the "
                       f"detector's training experience.")
    else:  # REVIEW or unspecified
        summary = (f"The audio shows a mix of signals. The detector "
                   f"recommends human review because the evidence is "
                   f"not strongly one way or the other.")

    return {
        'observations': observations,
        'measurements': measurements,
        'summary': summary,
    }


# ────────────────────────────────────────────────────────────────────
# Standalone test (run `python explainability.py path/to/audio.wav`)
# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, subprocess, tempfile, os
    if len(sys.argv) < 2:
        print("Usage: python explainability.py <audio_file>")
        sys.exit(1)
    in_path = sys.argv[1]
    # convert to wav 16kHz mono via ffmpeg
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

    result = explain_audio(data, sr)
    print("\nSUMMARY:")
    print(f"  {result['summary']}\n")
    print("OBSERVATIONS:")
    for o in result['observations']:
        marker = {'toward_fake': '⚠', 'toward_real': '✓', 'neutral': '○'}[o['lean']]
        print(f"  {marker} {o['title']}")
        print(f"     {o['detail']}\n")
    print("RAW MEASUREMENTS:")
    for k, v in result['measurements'].items():
        print(f"  {k:<28s} {v}")

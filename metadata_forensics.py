"""
VoiceGuard V8 — Metadata Forensics Module (Phase 4)
====================================================
Examines audio file metadata for forensic evidence of synthetic origin:
  - Container metadata (ID3 tags, RIFF chunks, format headers)
  - Encoder signature strings (LAME, FFmpeg, vendor-specific)
  - Format consistency (declared vs actual stream characteristics)
  - Missing expected fields (real recordings have device info; AI doesn't)
  - Re-encoding fingerprints (multiple encoding layers visible)

Honest scope:
  - HIGH PRECISION, LOW RECALL signal. When it fires, it's reliable.
    When it doesn't fire, it tells you nothing (most AI audio is re-encoded
    and metadata-scrubbed).
  - Catches lazy/opportunistic fraud, not sophisticated attackers.
  - Complements (does not replace) V8's deep model verdict.

Drop-in usage:
    from metadata_forensics import analyze_metadata
    result = analyze_metadata('/path/to/audio.wav')
    # result['findings'] is list of plain-text observations
    # result['lean'] indicates overall direction: 'toward_fake', 'toward_real', 'neutral'

Requirement: ffprobe must be on PATH (same as V8 server already requires).
"""

import os
import re
import json
import subprocess
from pathlib import Path


# ════════════════════════════════════════════════════════════════════
# Signature database
# ════════════════════════════════════════════════════════════════════
# Each entry: substring (lowercased) to look for in metadata fields,
# mapped to {'lean': 'toward_fake' | 'toward_real' | 'neutral',
#            'confidence': 'high' | 'medium' | 'low',
#            'description': 'plain-language explanation'}
#
# Easy to extend: just add new entries as you encounter new vendor
# signatures. Match is case-insensitive substring.

ENCODER_SIGNATURES = {
    # FFmpeg / Lavf — generic re-encoding tool, ubiquitous but suspicious
    # when claimed source is a phone or natural recording
    'lavf': {
        'lean': 'neutral',
        'confidence': 'low',
        'description': (
            "File was encoded with FFmpeg (Lavf encoder). FFmpeg is the most "
            "common audio re-encoding tool — used legitimately for format "
            "conversion, but also a common step when laundering AI audio. "
            "Not by itself evidence of fakeness, but worth noting alongside "
            "other signals."
        ),
    },
    'lavc': {
        'lean': 'neutral',
        'confidence': 'low',
        'description': (
            "Audio stream was encoded with libavcodec (the FFmpeg codec "
            "library). Indicates re-encoding through FFmpeg, common in both "
            "legitimate processing and AI audio cleanup."
        ),
    },

    # LAME MP3 — a real-world signature for MP3 encoders
    'lame': {
        'lean': 'neutral',
        'confidence': 'low',
        'description': (
            "File was encoded with LAME (a standard MP3 encoder). LAME is "
            "used by many legitimate tools and many AI vendors. Not "
            "diagnostic by itself."
        ),
    },

    # Phone-recording markers — should be present on real phone recordings
    'apple inc.':       {'lean': 'toward_real', 'confidence': 'medium',
                         'description': "Contains Apple device metadata — "
                         "typical of iOS recordings; unusual in AI audio."},
    'samsung':          {'lean': 'toward_real', 'confidence': 'medium',
                         'description': "Contains Samsung device metadata — "
                         "typical of Android recordings."},
    'whatsapp':         {'lean': 'toward_real', 'confidence': 'medium',
                         'description': "WhatsApp identifier present in "
                         "metadata — consistent with a real WhatsApp voice "
                         "note rather than synthesized audio."},
    'voice memo':       {'lean': 'toward_real', 'confidence': 'medium',
                         'description': "Voice Memo metadata present — "
                         "typical of iOS Voice Memos app."},

    # Known AI/TTS vendor markers (when not stripped)
    'elevenlabs':       {'lean': 'toward_fake', 'confidence': 'high',
                         'description': "ElevenLabs identifier found in "
                         "metadata — directly indicates ElevenLabs voice "
                         "synthesis."},
    'openai':           {'lean': 'toward_fake', 'confidence': 'high',
                         'description': "OpenAI identifier found in "
                         "metadata — directly indicates OpenAI TTS."},
    'eleven_turbo':     {'lean': 'toward_fake', 'confidence': 'high',
                         'description': "ElevenLabs Turbo model identifier "
                         "found — directly indicates ElevenLabs synthesis."},
    'eleven_multilingual': {'lean': 'toward_fake', 'confidence': 'high',
                            'description': "ElevenLabs multilingual model "
                            "identifier — direct evidence of ElevenLabs "
                            "synthesis."},
    'azure-tts':        {'lean': 'toward_fake', 'confidence': 'high',
                         'description': "Microsoft Azure TTS marker found "
                         "in metadata."},
    'microsoft text':   {'lean': 'toward_fake', 'confidence': 'high',
                         'description': "Microsoft TTS identifier present."},
    'play.ht':          {'lean': 'toward_fake', 'confidence': 'high',
                         'description': "PlayHT identifier — direct evidence "
                         "of PlayHT voice synthesis."},
    'resemble':         {'lean': 'toward_fake', 'confidence': 'medium',
                         'description': "Resemble identifier — may indicate "
                         "Resemble AI voice synthesis."},
    'murf':             {'lean': 'toward_fake', 'confidence': 'medium',
                         'description': "Murf identifier — may indicate "
                         "Murf AI voice synthesis."},
    'descript':         {'lean': 'toward_fake', 'confidence': 'medium',
                         'description': "Descript identifier — may indicate "
                         "Descript Overdub voice synthesis (or legitimate "
                         "Descript editing of real audio)."},
    'speechify':        {'lean': 'toward_fake', 'confidence': 'medium',
                         'description': "Speechify identifier — likely TTS."},
    'uberduck':         {'lean': 'toward_fake', 'confidence': 'high',
                         'description': "Uberduck identifier — directly "
                         "indicates Uberduck voice synthesis."},
    'coqui':            {'lean': 'toward_fake', 'confidence': 'high',
                         'description': "Coqui TTS identifier present — "
                         "open-source TTS engine."},
    'bark':             {'lean': 'toward_fake', 'confidence': 'medium',
                         'description': "Bark identifier present — "
                         "Suno's Bark TTS model. Could be coincidental "
                         "(e.g. tag for dog audio)."},
    'tortoise':         {'lean': 'toward_fake', 'confidence': 'medium',
                         'description': "Tortoise TTS identifier — open "
                         "source voice cloning model."},

    # Common synthesis pipeline indicators
    'gradio':           {'lean': 'toward_fake', 'confidence': 'low',
                         'description': "Audio was processed through Gradio "
                         "(a tool commonly used for hosting AI model demos). "
                         "Not definitive — Gradio is also used for legitimate "
                         "audio processing apps."},
    'huggingface':      {'lean': 'toward_fake', 'confidence': 'low',
                         'description': "HuggingFace identifier found — "
                         "may indicate audio came from a HuggingFace Space "
                         "(many of which host AI voice generators)."},
}


# ════════════════════════════════════════════════════════════════════
# Core ffprobe wrapper
# ════════════════════════════════════════════════════════════════════

def _run_ffprobe(file_path, timeout=8):
    """Run ffprobe and return parsed JSON dict, or None on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", file_path],
            capture_output=True, text=True, timeout=timeout
        )
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None
    except Exception:
        return None


def _collect_metadata_strings(probe_data):
    """Extract all metadata string values from ffprobe output into a flat
    dict of {field_name: value}. Includes both format-level and stream-level
    tags."""
    if not probe_data:
        return {}
    collected = {}
    # format-level tags
    fmt = probe_data.get('format', {})
    for k, v in fmt.get('tags', {}).items():
        collected[f'format.tags.{k}'] = str(v)
    # also useful format fields themselves
    for k in ('format_name', 'format_long_name'):
        if k in fmt:
            collected[f'format.{k}'] = str(fmt[k])
    # stream-level tags
    for i, stream in enumerate(probe_data.get('streams', [])):
        for k, v in stream.get('tags', {}).items():
            collected[f'stream{i}.tags.{k}'] = str(v)
        for k in ('codec_name', 'codec_long_name', 'codec_tag_string',
                  'profile', 'sample_fmt', 'bits_per_sample'):
            if k in stream:
                collected[f'stream{i}.{k}'] = str(stream[k])
    return collected


# ════════════════════════════════════════════════════════════════════
# Finding generators
# ════════════════════════════════════════════════════════════════════

def _check_encoder_signatures(metadata_strings):
    """Look for known encoder/vendor signatures in metadata values.
    Returns list of findings."""
    findings = []
    seen_signatures = set()  # avoid duplicate findings for same signature

    for field, value in metadata_strings.items():
        value_lower = value.lower()
        for signature, info in ENCODER_SIGNATURES.items():
            if signature in value_lower and signature not in seen_signatures:
                seen_signatures.add(signature)
                findings.append({
                    'category':    'encoder_signature',
                    'signature':   signature,
                    'matched_in':  field,
                    'matched_value': value,
                    'lean':        info['lean'],
                    'confidence':  info['confidence'],
                    'title':       f"Encoder signature: {signature}",
                    'description': info['description'],
                })
    return findings


def _check_format_consistency(probe_data, metadata_strings):
    """Look for format inconsistencies that might indicate processing
    or synthesis. Returns list of findings."""
    findings = []
    if not probe_data:
        return findings

    fmt = probe_data.get('format', {})
    streams = probe_data.get('streams', [])
    audio_streams = [s for s in streams if s.get('codec_type') == 'audio']

    if not audio_streams:
        return findings

    audio = audio_streams[0]

    # Check 1: missing creation_time on a format that usually has it
    container_format = fmt.get('format_name', '').lower()
    has_creation = any('creation_time' in k.lower() for k in metadata_strings)
    if container_format in ('mp4', 'm4a', 'mov', '3gp') and not has_creation:
        findings.append({
            'category':    'missing_metadata',
            'title':       'No creation timestamp in container',
            'lean':        'toward_fake',
            'confidence':  'low',
            'description': (
                "This container format normally records a creation timestamp "
                "when the file is captured by a recording device. The "
                "timestamp is absent here, which can indicate the file was "
                "generated programmatically rather than recorded."
            ),
        })

    # Check 2: suspiciously round duration (AI often outputs exact-length clips)
    try:
        duration = float(fmt.get('duration', 0))
        if duration > 1.0:
            # check if duration is suspiciously close to a round number of seconds
            rounded = round(duration)
            if abs(duration - rounded) < 0.01 and rounded in (3, 5, 10, 15, 20, 30, 60):
                findings.append({
                    'category':    'format_anomaly',
                    'title':       f'Duration is exactly {rounded} seconds',
                    'lean':        'toward_fake',
                    'confidence':  'low',
                    'description': (
                        f"The audio is precisely {rounded:.0f} seconds long. "
                        f"Real-world recordings rarely have perfectly round "
                        f"durations. AI-generated audio is often produced "
                        f"with a fixed-length target."
                    ),
                })
    except (ValueError, TypeError):
        pass

    # Check 3: Phone-typical codecs without phone metadata
    codec = audio.get('codec_name', '').lower()
    sample_rate = int(audio.get('sample_rate', 0) or 0)
    has_phone_metadata = any(
        sig in value.lower()
        for value in metadata_strings.values()
        for sig in ('apple inc.', 'samsung', 'whatsapp', 'voice memo',
                    'android', 'ios', 'iphone', 'ipad')
    )

    # if it's a phone-typical codec at phone sample rate but no phone metadata
    # AND has FFmpeg/Lavf encoder, that's suspicious
    has_ffmpeg = any('lavf' in v.lower() or 'lavc' in v.lower()
                     for v in metadata_strings.values())
    is_phone_like = codec in ('opus', 'amr_nb', 'amr_wb', 'g722') or \
                    (codec == 'aac' and sample_rate in (8000, 16000))
    if is_phone_like and has_ffmpeg and not has_phone_metadata:
        findings.append({
            'category':    'format_anomaly',
            'title':       'Phone-like format but no recording device metadata',
            'lean':        'toward_fake',
            'confidence':  'medium',
            'description': (
                f"The audio is in {codec} format at {sample_rate} Hz, which "
                f"is typical for phone recordings — but the file has no "
                f"device or app metadata (no Apple, Samsung, WhatsApp, etc.) "
                f"and was processed through FFmpeg. This combination suggests "
                f"the file may have been generated synthetically and then "
                f"made to look like a phone recording via re-encoding."
            ),
        })

    # Check 4: number of metadata fields — real recordings usually have several
    n_meta_fields = len([k for k in metadata_strings.keys()
                         if k.endswith(f'tags.{k.split(".")[-1]}')])
    actual_tag_count = sum(1 for k in metadata_strings.keys() if '.tags.' in k)
    if actual_tag_count == 0 and container_format not in ('wav', 's16le', 'raw'):
        findings.append({
            'category':    'missing_metadata',
            'title':       'No metadata tags present',
            'lean':        'toward_fake',
            'confidence':  'low',
            'description': (
                "The container has no metadata tags at all. Real-world "
                "audio files usually carry some metadata (device info, "
                "creation date, software used). Complete absence of tags "
                "can indicate the file was generated and re-encoded to "
                "strip identifying information."
            ),
        })

    return findings


def _check_reencoding_chain(metadata_strings):
    """Look for evidence of multiple encoding passes — common in AI audio
    that's been generated then re-encoded for distribution."""
    findings = []

    # count distinct encoder identifiers
    encoder_strings = []
    for field, value in metadata_strings.items():
        if 'encoder' in field.lower() or 'encoded' in field.lower():
            encoder_strings.append(value)

    if len(set(encoder_strings)) >= 2:
        findings.append({
            'category':    'reencoding_chain',
            'title':       'Multiple encoding signatures detected',
            'lean':        'toward_fake',
            'confidence':  'low',
            'description': (
                f"Detected {len(set(encoder_strings))} distinct encoder "
                f"identifiers in this file's metadata, suggesting the audio "
                f"passed through multiple encoding stages. This is common "
                f"for AI audio that was generated by one tool and then "
                f"re-encoded for distribution. Legitimate single-recording "
                f"files usually show only one encoder."
            ),
        })

    return findings


# ════════════════════════════════════════════════════════════════════
# Public interface
# ════════════════════════════════════════════════════════════════════

def analyze_metadata(file_path):
    """
    Analyze file metadata for forensic evidence of synthetic origin.

    Args:
        file_path: path to audio file (any format ffprobe can read)

    Returns:
        dict with:
          - 'available':  True if analysis ran
          - 'findings':   list of finding dicts (each with title, description,
                          lean, confidence, category)
          - 'lean':       overall direction: 'toward_fake', 'toward_real',
                          or 'neutral' (based on weighted findings)
          - 'summary':    plain-language summary of the metadata analysis
          - 'raw_metadata': flat dict of all extracted metadata fields
          - 'error':      error string if analysis failed (None otherwise)
    """
    blank = {
        'available': False,
        'findings':  [],
        'lean':      'neutral',
        'summary':   "Metadata analysis was not available for this file.",
        'raw_metadata': {},
        'error':     None,
    }

    if not os.path.exists(file_path):
        blank['error'] = f"File not found: {file_path}"
        return blank

    probe_data = _run_ffprobe(file_path)
    if probe_data is None:
        blank['error'] = "ffprobe failed to read this file."
        blank['summary'] = (
            "Could not extract metadata from this file. ffprobe may not be "
            "installed, or the file format may not be recognized."
        )
        return blank

    metadata_strings = _collect_metadata_strings(probe_data)

    # Run each check
    findings = []
    findings += _check_encoder_signatures(metadata_strings)
    findings += _check_format_consistency(probe_data, metadata_strings)
    findings += _check_reencoding_chain(metadata_strings)

    # Determine overall lean by weighting high > medium > low
    weight = {'high': 3, 'medium': 2, 'low': 1}
    fake_weight = sum(weight.get(f['confidence'], 1)
                      for f in findings if f['lean'] == 'toward_fake')
    real_weight = sum(weight.get(f['confidence'], 1)
                      for f in findings if f['lean'] == 'toward_real')

    if fake_weight - real_weight >= 3:
        overall_lean = 'toward_fake'
    elif real_weight - fake_weight >= 3:
        overall_lean = 'toward_real'
    else:
        overall_lean = 'neutral'

    # Build summary
    fake_count = sum(1 for f in findings if f['lean'] == 'toward_fake')
    real_count = sum(1 for f in findings if f['lean'] == 'toward_real')
    high_confidence_fake = [f for f in findings
                            if f['lean'] == 'toward_fake'
                            and f['confidence'] == 'high']

    if high_confidence_fake:
        sigs = ', '.join(set(f.get('signature', f['title'])
                             for f in high_confidence_fake))
        summary = (f"Metadata analysis found {len(high_confidence_fake)} "
                   f"high-confidence indicator(s) of synthetic origin: {sigs}. "
                   f"This is forensic evidence consistent with AI-generated "
                   f"audio.")
    elif fake_count > 0 and real_count == 0:
        summary = (f"Metadata analysis found {fake_count} weak indicator(s) "
                   f"suggesting possible synthetic origin. None are "
                   f"definitive on their own — consider alongside V8's main "
                   f"verdict.")
    elif real_count > 0 and fake_count == 0:
        summary = (f"Metadata analysis found {real_count} indicator(s) "
                   f"consistent with a real recording (device or app "
                   f"signatures present).")
    elif fake_count > 0 and real_count > 0:
        summary = (f"Metadata analysis is mixed: {real_count} real-recording "
                   f"indicators and {fake_count} synthetic-origin indicators "
                   f"present.")
    else:
        summary = ("Metadata analysis found no notable signatures in this "
                   "file. Absence of evidence is not evidence of absence — "
                   "most AI audio in the wild has been metadata-scrubbed by "
                   "re-encoding.")

    return {
        'available':    True,
        'findings':     findings,
        'lean':         overall_lean,
        'summary':      summary,
        'raw_metadata': metadata_strings,
        'error':        None,
    }


# ════════════════════════════════════════════════════════════════════
# Standalone CLI: python metadata_forensics.py <audio_file>
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metadata_forensics.py <audio_file>")
        sys.exit(1)

    result = analyze_metadata(sys.argv[1])
    print()
    print("=" * 70)
    print("METADATA FORENSICS RESULT")
    print("=" * 70)
    if not result['available']:
        print(f"  ERROR: {result['error']}")
        sys.exit(1)

    print(f"\n  Overall lean: {result['lean']}")
    print(f"\n  Summary: {result['summary']}\n")

    if result['findings']:
        print(f"  Findings ({len(result['findings'])}):")
        for f in result['findings']:
            marker = {'toward_fake': '⚠', 'toward_real': '✓',
                      'neutral': '○'}.get(f['lean'], '•')
            print(f"\n    {marker} [{f['confidence']}] {f['title']}")
            print(f"      {f['description']}")
    else:
        print("  No findings to report.")

    print(f"\n  Raw metadata extracted ({len(result['raw_metadata'])} fields):")
    for k, v in sorted(result['raw_metadata'].items()):
        display_v = v if len(v) < 80 else v[:77] + '...'
        print(f"    {k}: {display_v}")

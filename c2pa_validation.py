"""c2pa_validation.py — Phase 4: C2PA Content Credentials validator.

Reads embedded C2PA provenance ("Content Credentials") from a media file and turns it into a
forensic signal for the detector:
  - a VALID manifest that DISCLOSES AI generation (digitalSourceType trainedAlgorithmicMedia /
    algorithmicMedia) -> strong 'toward_fake' signal (the file itself says it's synthetic);
  - a VALID manifest asserting a captured, non-synthetic source (digitalCapture) -> 'toward_real';
  - a manifest whose signature FAILS validation -> 'toward_fake' (tampered / forged provenance);
  - no manifest -> 'neutral' (most audio carries none).

Advisory / additive: it NEVER changes the primary verdict. Interface expected by detector.py:
    validate_c2pa(file_path) -> dict(available, format_supported, has_credentials, ...)
"""
import os
import json as _json

try:
    import c2pa
    _C2PA_LIB = True
except Exception:
    c2pa = None
    _C2PA_LIB = False

# Media extensions that can carry C2PA credentials (superset covers audio + the common image/video).
_SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".m4v", ".flac", ".mov", ".avi",
                  ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".avif", ".heic", ".pdf"}

# IPTC digitalSourceType values that indicate AI / synthetic generation, and authentic capture.
_AI_SOURCE_TYPES = {"trainedalgorithmicmedia", "algorithmicmedia",
                    "compositewithtrainedalgorithmicmedia", "trainedalgorithmicdata"}
_CAPTURE_SOURCE_TYPES = {"digitalcapture", "computationalcapture", "negativefilm", "positivefilm"}


def _empty(summary, format_supported=True, error=None):
    return {"available": _C2PA_LIB, "format_supported": format_supported,
            "has_credentials": False, "valid": None, "validation_state": None,
            "ai_generated": False, "digital_source_type": None,
            "claim_generator": None, "signer": None,
            "lean": "neutral", "findings": [], "summary": summary, "error": error}


def _scan_source_types(store):
    """Walk the manifest-store JSON for any digitalSourceType values (returned lowercased,
    URI tail only). Location-agnostic so it survives spec/nesting differences."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lower().replace("_", "") == "digitalsourcetype" and isinstance(v, str):
                    found.append(v.rstrip("/").split("/")[-1].lower())
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(store)
    return found


def validate_c2pa(file_path):
    if not _C2PA_LIB:
        return {"available": False, "format_supported": False, "has_credentials": False,
                "lean": "neutral", "findings": [], "summary": "c2pa library not installed",
                "error": "c2pa module unavailable"}

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _SUPPORTED_EXT:
        return _empty(f"format '{ext or '?'}' does not carry C2PA", format_supported=False)

    try:
        reader = c2pa.Reader.try_create(file_path)   # None if no manifest, no exception
    except Exception as e:
        return _empty("could not read file for C2PA", error=str(e))
    if reader is None:
        return _empty("no C2PA Content Credentials present")

    try:
        try:
            valid = bool(reader.is_valid())
        except Exception:
            valid = None
        try:
            state = str(reader.get_validation_state())
        except Exception:
            state = None
        try:
            store = _json.loads(reader.json())
        except Exception as e:
            return _empty("C2PA manifest present but unreadable", error=str(e))
    finally:
        try:
            reader.close()
        except Exception:
            pass

    # Active manifest details (defensive across schema shapes).
    active_label = store.get("active_manifest")
    manifests = store.get("manifests", {}) or {}
    active = manifests.get(active_label) or (next(iter(manifests.values()), {}) if manifests else {})
    claim_gen = active.get("claim_generator")
    info = active.get("claim_generator_info")
    if not claim_gen and isinstance(info, list) and info:
        claim_gen = info[0].get("name") if isinstance(info[0], dict) else str(info[0])
    sig = active.get("signature_info") or {}
    signer = sig.get("issuer") or sig.get("common_name") or sig.get("cert_serial_number")

    src_types = _scan_source_types(store)
    ai = any(s in _AI_SOURCE_TYPES for s in src_types)
    capture = any(s in _CAPTURE_SOURCE_TYPES for s in src_types)
    dst = src_types[0] if src_types else None

    findings = []
    if valid is False:
        lean = "toward_fake"
        summary = "Invalid Content Credentials — provenance signature could not be verified."
        findings.append("C2PA manifest signature FAILED validation (possible tampering/forgery)")
    elif ai:
        lean = "toward_fake"
        summary = "Content Credentials declare this is AI / synthetic media."
        findings.append(f"Content Credentials disclose an AI-generated source (digitalSourceType: {dst})")
    elif capture:
        lean = "toward_real"
        summary = "Content Credentials assert an authentic (captured) source."
        findings.append(f"Content Credentials assert a captured, non-synthetic source ({dst})")
    else:
        lean = "neutral"
        summary = "Valid Content Credentials present (no AI-generation disclosure)."
    if claim_gen:
        findings.append(f"Claim generator: {claim_gen}")
    if signer:
        findings.append(f"Signed by: {signer}")

    return {"available": True, "format_supported": True, "has_credentials": True,
            "valid": valid, "validation_state": state,
            "ai_generated": ai, "digital_source_type": dst,
            "claim_generator": claim_gen, "signer": signer,
            "lean": lean, "findings": findings, "summary": summary, "error": None}

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c2pa_validation as cv

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = os.path.join(HERE, "golden_clips", "fake_noizai_a4cd.mp3")


def test_no_manifest_is_neutral():
    r = cv.validate_c2pa(CLIP)
    assert r["available"] is True
    assert r["format_supported"] is True
    assert r["has_credentials"] is False
    assert r["lean"] == "neutral"


def test_unsupported_format():
    r = cv.validate_c2pa("clip.xyz")
    assert r["format_supported"] is False
    assert r["has_credentials"] is False
    assert r["lean"] == "neutral"


def test_scan_detects_ai_source_type():
    store = {"active_manifest": "m1", "manifests": {"m1": {"assertions": [
        {"label": "c2pa.actions", "data": {"actions": [
            {"action": "c2pa.created",
             "digitalSourceType":
                 "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}]}}]}}}
    types = cv._scan_source_types(store)
    assert "trainedalgorithmicmedia" in types
    assert any(t in cv._AI_SOURCE_TYPES for t in types)


def test_scan_detects_capture_source_type():
    store = {"m": [{"digital_source_type":
                    "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"}]}
    types = cv._scan_source_types(store)
    assert "digitalcapture" in types
    assert any(t in cv._CAPTURE_SOURCE_TYPES for t in types)

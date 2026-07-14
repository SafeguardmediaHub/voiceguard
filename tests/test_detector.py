# tests/test_detector.py
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector

pytestmark = pytest.mark.weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_core_exposes_expected_surface():
    for name in ["detect", "cascade_score_chunk", "lcnn_score",
                 "ensemble_score_variants", "verdict_from_score", "startup_check",
                 "CHUNK", "CASCADE_LOW", "CASCADE_HIGH", "thresholds", "ACTIVE_VERSION"]:
        assert hasattr(detector, name), f"detector missing {name}"

def test_detect_parity_on_golden_clips():
    cases = [
        ("tests/golden_clips/fake_noizai_a4cd.mp3",  "LIKELY_FAKE", 0.8167),
        ("tests/golden_clips/real_studio_037.mp3",   "AUTO_REAL",   0.1708),
    ]
    for rel, verdict, score in cases:
        r = detector.detect(os.path.join(REPO, rel))
        assert r["verdict"] == verdict, f"{rel}: {r['verdict']} != {verdict}"
        assert abs(r["score"] - score) < 1e-3, f"{rel}: {r['score']} != {score}"
        assert r["model_version"] == "V9"

def test_import_does_not_exit():
    # startup_check must be a function, not auto-run at import (importing above
    # already succeeded — a sys.exit at import would have aborted this module).
    assert callable(detector.startup_check)

def test_detect_has_explainability_fields():
    r = detector.detect("tests/golden_clips/fake_noizai_a4cd.mp3")
    # additive fields present
    assert r["audit_id"].startswith("aud_")
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["heatmap"]["target"] == "lcnn"
    assert len(r["heatmap"]["values"]) == detector.N_MELS
    assert len(r["heatmap"]["freq_hz"]) == detector.N_MELS
    # shap is present (this clip escalates) or explicitly null; if present, has the 4 keys
    if r["shap"] is not None:
        assert {"aasist", "wav2vec", "rawnet", "base"} <= set(r["shap"].keys())
    # whole response is JSON-serializable
    import json
    json.dumps(r)

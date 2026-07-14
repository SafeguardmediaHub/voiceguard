import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forensic_report as fr


def _res(score, verdict):
    return {"verdict": verdict, "score": score, "audit_id": "aud_x", "sha256": "abc123def456...",
            "model_version": "V9",
            "explanation": {"observations": [{"title": "Regular spectrum", "detail": "atypical",
                                              "lean": "toward_fake"}]},
            "flagged_segments": [{"start_sec": 1.0, "end_sec": 4.0, "verdict": verdict}]}


def test_band_mapping():
    assert fr._band(0.90)[0] == "auto_fake" and fr._band(0.90)[2] == "HIGH"
    assert fr._band(0.60)[0] == "likely_fake" and fr._band(0.60)[2] == "MODERATE"
    assert fr._band(0.40)[0] == "to_review" and fr._band(0.40)[2] == "LOW"
    assert fr._band(0.10)[0] == "auto_real"


def test_fake_report_content():
    h = fr.build_report_html(_res(0.70, "LIKELY_FAKE"), analyst="A. Analyst", exhibit="EX-1")
    assert "consistent with computer-generated" in h
    assert "Confidence level: MODERATE" in h
    assert 'class="cur"' in h and "likely_fake (0.55" in h
    assert "aud_x" in h and "abc123def456" in h
    assert "Regular spectrum" in h and "1.0" in h        # observation + flagged segment
    assert "A. Analyst" in h


def test_real_report_content():
    h = fr.build_report_html(_res(0.10, "AUTO_REAL"))
    assert "no significant indication" in h


def test_valid_html_and_legal_disclaimers():
    h = fr.build_report_html(_res(0.90, "AUTO_FAKE"))
    assert h.lstrip().startswith("<!doctype html>")
    assert "does not constitute legal proof" in h
    assert "certified forensic audio examiner" in h
    assert "Phone / call-quality audio" in h

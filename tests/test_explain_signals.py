import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import explain_signals as es


def test_new_audit_id_prefix_and_unique():
    a, b = es.new_audit_id(), es.new_audit_id()
    assert a.startswith("aud_") and len(a) > 4
    assert a != b


def test_confidence_decisive_and_agreeing():
    assert es.compute_confidence([0.95, 0.96, 0.94], 0.95) > 0.85


def test_confidence_fence_score_zero():
    assert es.compute_confidence([0.5, 0.5], 0.5) == 0.0


def test_confidence_disagreement_lowers():
    assert es.compute_confidence([0.1, 0.9], 0.9) < es.compute_confidence([0.9, 0.9], 0.9)


def test_confidence_single_chunk_is_decisiveness():
    assert es.compute_confidence([0.8], 0.8) == round(abs(0.8 - 0.5) * 2, 3)


def test_confidence_empty_zero():
    assert es.compute_confidence([], 0.9) == 0.0


def test_shap_from_contribs_mapping():
    assert es.shap_from_contribs([0.42, 1.13, -0.08, -0.30]) == {
        "aasist": 0.42, "wav2vec": 1.13, "rawnet": -0.08, "base": -0.30}

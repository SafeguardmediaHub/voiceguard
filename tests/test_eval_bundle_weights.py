# tests/test_eval_bundle_weights.py
"""End-to-end tier for the evaluation harness, against the real v9h bundle.

Importing eval_bundle is cheap, but these tests call through to detector, which
loads ~380 MB of weights -- hence the marker and the conftest collect_ignore entry.
The weights-free units live in tests/test_eval_bundle.py.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import detector as D                       # noqa: E402
import eval_bundle as E                    # noqa: E402
import governance                          # noqa: E402

pytestmark = pytest.mark.weights

GOLDEN = os.path.join(REPO, "tests", "golden_clips")


def _golden_entries():
    clips = [os.path.join(GOLDEN, f) for f in sorted(os.listdir(GOLDEN))
             if f.lower().endswith(E.AUDIO_EXTS)][:3]
    return [{"path": p, "label": 1 if "fake" in os.path.basename(p) else 0,
             "language": "english", "source": "golden"} for p in clips]


def test_device_defaults_to_cpu():
    """Production is a CPU droplet. $VOICEGUARD_DEVICE is unset in this process, so
    detector must have resolved to cpu -- the same behaviour as the hardcoded
    torch.device("cpu") it replaced. tests/test_docker_context.py fences the deploy
    config separately."""
    assert os.environ.get("VOICEGUARD_DEVICE") is None
    assert D.DEVICE.type == "cpu"


def test_gate_passes_on_the_bundle_this_process_actually_loaded():
    prov = E.assert_bundle_provenance(D.ACTIVE_VERSION)
    assert prov["bundle"] == D.ACTIVE_VERSION
    assert prov["artifacts"], "the artifact SHA map must not be empty"
    assert all(len(sha) == 64 for sha in prov["artifacts"].values())


def test_gate_refuses_a_version_that_is_not_the_loaded_one():
    with pytest.raises(E.ProvenanceError):
        E.assert_bundle_provenance("definitely-not-a-real-bundle")


def test_scoring_real_clips_produces_usable_rows(tmp_path):
    rows = E.score_manifest(_golden_entries(), str(tmp_path / "out"))
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok, [r.get("error") for r in rows]
    for r in ok:
        assert 0.0 <= r["score"] <= 1.0
        assert r["n_chunks"] >= 1
        assert r["stage1_chunks"] + r["stage2_chunks"] == r["n_chunks"]
        assert r["latency_ms"] > 0


def test_scoring_leaves_the_audit_chain_untouched(tmp_path):
    """detect(audit=False) should write nothing -- but 'should' is not verification.
    A regression here would put every evaluation detection into the chain of custody,
    indistinguishable from a real customer detection in a forensic export."""
    log_path = governance.AuditLog().path
    before = sum(1 for _ in open(log_path, encoding="utf-8")) if os.path.exists(log_path) else 0
    E.score_manifest(_golden_entries(), str(tmp_path / "out"))
    after = sum(1 for _ in open(log_path, encoding="utf-8")) if os.path.exists(log_path) else 0
    assert after == before, "evaluation detections leaked into the audit chain"

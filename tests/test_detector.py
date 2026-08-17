# tests/test_detector.py
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector

pytestmark = pytest.mark.weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _StubRegistry:
    """Stands in for bundle_registry.Registry so the load-time gate can be driven
    through each outcome without touching the real 386 MB bundle."""
    def __init__(self, problems):
        self._problems = problems

    def __call__(self):
        return self

    def integrity_problems(self, version):
        return self._problems


def test_verify_bundle_refuses_a_tampered_bundle(monkeypatch):
    # The gate must raise BEFORE torch.load unpickles a modified checkpoint.
    monkeypatch.setattr(detector, "_Registry",
                        _StubRegistry(["aasist.pt: sha256 deadbeef != registered cafe1234"]))
    with pytest.raises(RuntimeError, match="failed integrity verification"):
        detector._verify_bundle_before_load("v9h", {"files": {}})


def test_verify_bundle_refuses_an_unregistered_bundle(monkeypatch):
    monkeypatch.setattr(detector, "_Registry", _StubRegistry(None))
    with pytest.raises(RuntimeError, match="not in the registry"):
        detector._verify_bundle_before_load("v9h", {"files": {}})


def test_verify_bundle_accepts_an_intact_bundle(monkeypatch):
    monkeypatch.setattr(detector, "_Registry", _StubRegistry([]))
    detector._verify_bundle_before_load("v9h", {"files": {}})   # must not raise


def test_verify_bundle_warns_but_allows_the_legacy_fallback(capsys):
    # No manifest = the pre-registry models/ layout; nothing to verify against.
    detector._verify_bundle_before_load("v9-hybrid-legacy", None)
    assert "UNVERIFIED" in capsys.readouterr().out


def test_active_bundle_passes_integrity_verification():
    # The bundle this process actually loaded must be exactly what was registered.
    assert detector._Registry().integrity_problems(detector.ACTIVE_VERSION) == []


def test_detect_writes_verifiable_audit_entry(tmp_path, monkeypatch):
    # H6: a live detection must emit a tamper-evident, chain-verifiable audit entry.
    monkeypatch.setenv("VOICEGUARD_GOVERNANCE_DIR", str(tmp_path / "gov"))
    import importlib, governance
    governance = importlib.reload(governance)      # pick up the redirected dir

    r = detector.detect(os.path.join(REPO, "tests/golden_clips/fake_noizai_a4cd.mp3"))

    entries = governance.AuditLog().read_all()
    assert len(entries) == 1
    e = entries[0]
    assert e["audit_id"] == r["audit_id"]
    assert e["verdict"] == r["verdict"]
    assert len(e["intake_sha256"]) == 64           # full hash, not the truncated display
    assert e["model_versions"]["bundle"] == detector.ACTIVE_VERSION
    ok, details = governance.AuditLog().verify_chain()
    assert ok, details


def test_startup_check_does_not_write_audit_entry(tmp_path, monkeypatch):
    # The internal smoke-check must not pollute the chain-of-custody log.
    monkeypatch.setenv("VOICEGUARD_GOVERNANCE_DIR", str(tmp_path / "gov"))
    import importlib, governance
    governance = importlib.reload(governance)

    detector.startup_check()
    assert not governance.AUDIT_LOG_PATH.exists() or governance.AuditLog().read_all() == []

def test_core_exposes_expected_surface():
    for name in ["detect", "cascade_score_chunk", "lcnn_score",
                 "ensemble_score_variants", "verdict_from_score", "startup_check",
                 "CHUNK", "CASCADE_LOW", "CASCADE_HIGH", "thresholds", "ACTIVE_VERSION"]:
        assert hasattr(detector, name), f"detector missing {name}"

def test_detect_parity_on_golden_clips():
    """Read the expectation from the golden manifest, never hardcode it here.

    These values were hardcoded and went stale: they still held V9-era numbers
    (fake_noizai_a4cd LIKELY_FAKE/0.8167) after the deployed bundle became v9h
    with CASCADE_LOW retuned to 0.10, so this test failed against a bundle that
    was behaving exactly as its own baseline says it should. The manifest is the
    single source of truth, regenerated deliberately via
    `python tests/test_golden.py --update`."""
    import json
    with open(os.path.join(REPO, "tests/golden_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    tol = manifest.get("score_tol", 1e-3)

    for fname in ("fake_noizai_a4cd.mp3", "real_studio_037.mp3"):
        exp = manifest["clips"][fname]
        r = detector.detect(os.path.join(REPO, "tests/golden_clips", fname))
        assert r["verdict"] == exp["verdict"], f"{fname}: {r['verdict']} != {exp['verdict']}"
        assert abs(r["score"] - exp["score"]) < tol, f"{fname}: {r['score']} != {exp['score']}"
        assert r["model_version"] == exp["model_version"]

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

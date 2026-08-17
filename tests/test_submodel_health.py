# tests/test_submodel_health.py
"""The promotion health gate, run against the shipped probe set (REMEDIATION_PLAN D9).

Weights tier: importing submodel_health imports detector, which loads the bundle.
"""
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import detector as D                       # noqa: E402
import submodel_health as H                # noqa: E402

pytestmark = pytest.mark.weights

PROBE = os.path.join(REPO, "tests", "probe_clips")


class _Collapsed(torch.nn.Module):
    """A sub-model saturated to a near-constant output — the AASIST V9 signature
    (softmax ~0.9997 regardless of input, feeding the fusion a dead feature)."""
    def forward(self, x):
        return torch.tensor([[-4.0, 4.0]])


def test_shipped_probe_set_passes_a_healthy_bundle():
    r = H.run_health_check(probe_dir=PROBE)
    assert r["passed"], r["collapsed"]
    assert r["collapsed"] == []
    assert r["n_real"] > 0 and r["n_fake"] > 0


def test_shipped_probe_set_catches_a_collapsed_submodel(monkeypatch):
    """The whole point of the gate. If this cannot be demonstrated on the SHIPPED
    set, then enforcing the gate in production is theatre."""
    monkeypatch.setattr(D, "aasist", _Collapsed())
    r = H.run_health_check(probe_dir=PROBE)
    assert r["passed"] is False
    assert r["collapsed"] == ["aasist"]                 # and no false positives
    assert r["models"]["aasist"]["spread"] < r["min_spread"]


def test_auc_is_withheld_when_there_are_too_few_fakes():
    """2 fake clips cannot support an AUC. Reporting one would read as 0.87-1.00
    against 0.49-0.69 on the full corpora and contradict REMEDIATION_PLAN F1."""
    r = H.run_health_check(probe_dir=PROBE)
    assert r["n_fake"] < H.MIN_FAKE_FOR_AUC
    assert r["auc_reported"] is False
    for name, m in r["models"].items():
        assert m["auc"] is None, f"{name} reported an AUC from {r['n_fake']} fakes"
        assert m["weak"] is False                        # weak tier is inert without AUC
    # The collapse gate is unaffected — that is the tier that matters.
    assert all(m["spread"] > r["min_spread"] for m in r["models"].values())


def test_probe_source_is_recorded_for_the_promotion_record():
    """What certified a promotion has to be attributable, not implied."""
    r = H.run_health_check(probe_dir=PROBE)
    assert "probe_clips" in r["probe_source"]
    assert "CC0" in r["probe_source"]


def test_local_corpora_are_preferred_when_present(tmp_path):
    """A dev machine must keep using the large labelled corpora, so recorded
    numbers stay comparable; the shipped set is a fallback, not a replacement."""
    real, fake, desc = H.resolve_probe_sources()
    if os.path.isdir(os.path.join(REPO, "studio_clips")):
        assert "local corpora" in desc
        assert len(real) > 100 and len(fake) > 100
    else:                                    # inside the image: falls back
        assert "probe_clips" in desc


def test_undecodable_probe_set_is_not_reported_as_collapse(monkeypatch):
    """An environment fault and a total collapse both produce zero spread. They
    must not be reported the same way — the first sends an operator to
    --skip-health, which is the habit the gate exists to prevent."""
    monkeypatch.setattr(D, "FFMPEG", "/nonexistent/ffmpeg")
    with pytest.raises(RuntimeError, match="did not decode"):
        H.run_health_check(probe_dir=PROBE)

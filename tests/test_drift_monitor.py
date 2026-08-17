# tests/test_drift_monitor.py
"""Drift detection + the confirmation state machine.

These are the two pieces of drift monitoring that decide whether production is
still healthy, and neither was covered: tests/test_retrain_trigger.py starts one
step later, at fire_retrain_trigger(), and takes the confirmed-alert list as a
given. Everything here is pure metric arithmetic over dicts, so it needs no
weights and runs in the fast CI tier.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fresh_dm(tmp_path, monkeypatch):
    # OUTPUT_DIR-derived paths (incl. the alert-state file) are computed at import,
    # so point them at tmp_path and (re)load — same pattern as test_retrain_trigger.
    monkeypatch.setenv("DRIFT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("DRIFT_CONFIRM_RUNS", raising=False)
    import drift_monitor_3 as dm
    importlib.reload(dm)
    dm.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return dm


def _metrics(ensemble_eer=0.05, deployed_eer=0.06, n=1000,
             per_source=None, noizai_catch=0.90, manifest="hash_a"):
    """A metrics dict shaped like run_evaluation()'s output."""
    return {
        "timestamp": "2026-08-17T00:00:00Z",
        "val_manifest_hash": manifest,
        "clean": {
            "n_samples": n,
            "ensemble_eer": ensemble_eer,
            "deployed_eer": deployed_eer,
            "per_source": per_source if per_source is not None
                          else {"noizai_tts": {"n": 100, "catch_rate": 0.90}},
        },
        "noizai": {"available": True, "n_samples": 100, "catch_rate": noizai_catch},
    }


# ── detect_drift ────────────────────────────────────────────────────────────

def test_identical_metrics_produce_no_alerts(tmp_path, monkeypatch):
    """The load-bearing negative case: a healthy re-run must stay silent, or every
    alert downstream is noise and operators learn to ignore the monitor."""
    dm = _fresh_dm(tmp_path, monkeypatch)
    assert dm.detect_drift(_metrics(), _metrics()) == []


def test_eer_regression_past_the_threshold_alerts(tmp_path, monkeypatch):
    dm = _fresh_dm(tmp_path, monkeypatch)
    # +10pp on 1000 samples: past the 3pp threshold and statistically unambiguous.
    alerts = dm.detect_drift(_metrics(ensemble_eer=0.05), _metrics(ensemble_eer=0.15))
    by_key = {a["key"]: a for a in alerts}
    assert "clean_ensemble_eer" in by_key
    assert by_key["clean_ensemble_eer"]["significant"] is True
    assert by_key["clean_ensemble_eer"]["delta_pp"] == pytest.approx(10.0)


def test_eer_wobble_within_threshold_does_not_alert(tmp_path, monkeypatch):
    dm = _fresh_dm(tmp_path, monkeypatch)
    # +2pp < the 3pp threshold — sampling noise, not drift.
    alerts = dm.detect_drift(_metrics(ensemble_eer=0.05, deployed_eer=0.05),
                             _metrics(ensemble_eer=0.07, deployed_eer=0.07))
    assert [a["key"] for a in alerts] == []


def test_deployed_eer_is_tracked_separately_from_the_ensemble(tmp_path, monkeypatch):
    """deployed_eer is what the cascade actually ships; the ensemble can look fine
    while the deployed path regresses (e.g. a screener threshold change)."""
    dm = _fresh_dm(tmp_path, monkeypatch)
    alerts = dm.detect_drift(_metrics(ensemble_eer=0.05, deployed_eer=0.06),
                             _metrics(ensemble_eer=0.05, deployed_eer=0.16))
    keys = [a["key"] for a in alerts]
    assert "deployed_eer" in keys and "clean_ensemble_eer" not in keys


def test_per_source_catch_drop_alerts_but_an_improvement_does_not(tmp_path, monkeypatch):
    dm = _fresh_dm(tmp_path, monkeypatch)
    base = _metrics(per_source={"elevenlabs": {"n": 100, "catch_rate": 0.90}})

    dropped = _metrics(per_source={"elevenlabs": {"n": 100, "catch_rate": 0.70}})
    assert "per_source_catch_elevenlabs" in [a["key"] for a in dm.detect_drift(base, dropped)]

    improved = _metrics(per_source={"elevenlabs": {"n": 100, "catch_rate": 0.99}})
    assert [a["key"] for a in dm.detect_drift(base, improved)] == []


def test_noizai_catch_drop_alerts(tmp_path, monkeypatch):
    dm = _fresh_dm(tmp_path, monkeypatch)
    alerts = dm.detect_drift(_metrics(noizai_catch=0.90), _metrics(noizai_catch=0.60))
    assert "noizai_catch_rate" in [a["key"] for a in alerts]


def test_manifest_change_alerts_so_drift_is_not_measured_against_a_moved_target(
        tmp_path, monkeypatch):
    dm = _fresh_dm(tmp_path, monkeypatch)
    alerts = dm.detect_drift(_metrics(manifest="hash_a"), _metrics(manifest="hash_b"))
    assert "manifest_changed" in [a["key"] for a in alerts]


# ── update_confirmation_state ───────────────────────────────────────────────

ALL_KEYS = ["clean_ensemble_eer", "deployed_eer", "noizai_catch_rate",
            "manifest_changed", "per_source_catch_noizai_tts"]


def test_a_single_breach_is_not_confirmed(tmp_path, monkeypatch):
    """DRIFT_CONFIRM_RUNS defaults to 2 — one bad run must not page anyone."""
    dm = _fresh_dm(tmp_path, monkeypatch)
    alerts = dm.detect_drift(_metrics(noizai_catch=0.90), _metrics(noizai_catch=0.60))
    state, confirmed = dm.update_confirmation_state(alerts, ALL_KEYS)
    assert state["noizai_catch_rate"] == 1
    assert confirmed == []


def test_two_consecutive_breaches_confirm(tmp_path, monkeypatch):
    dm = _fresh_dm(tmp_path, monkeypatch)
    alerts = dm.detect_drift(_metrics(noizai_catch=0.90), _metrics(noizai_catch=0.60))
    dm.update_confirmation_state(alerts, ALL_KEYS)
    state, confirmed = dm.update_confirmation_state(alerts, ALL_KEYS)
    assert state["noizai_catch_rate"] == 2
    assert [a["key"] for a in confirmed] == ["noizai_catch_rate"]


def test_a_clear_run_resets_the_counter(tmp_path, monkeypatch):
    """Otherwise two unrelated bad runs a month apart would confirm as if they
    were a sustained regression."""
    dm = _fresh_dm(tmp_path, monkeypatch)
    bad = dm.detect_drift(_metrics(noizai_catch=0.90), _metrics(noizai_catch=0.60))
    dm.update_confirmation_state(bad, ALL_KEYS)

    state, confirmed = dm.update_confirmation_state([], ALL_KEYS)   # clean run
    assert state["noizai_catch_rate"] == 0
    assert confirmed == []

    state, confirmed = dm.update_confirmation_state(bad, ALL_KEYS)
    assert state["noizai_catch_rate"] == 1                          # counting from scratch
    assert confirmed == []


def test_confirmation_state_survives_process_restart(tmp_path, monkeypatch):
    """The monitor runs as a fresh cron container each night, so the counter only
    works if it round-trips through drift_alert_state.json on the volume."""
    dm = _fresh_dm(tmp_path, monkeypatch)
    alerts = dm.detect_drift(_metrics(noizai_catch=0.90), _metrics(noizai_catch=0.60))
    dm.update_confirmation_state(alerts, ALL_KEYS)
    assert dm.ALERT_STATE_FILE.exists()

    dm2 = _fresh_dm(tmp_path, monkeypatch)          # simulates the next night's run
    state, confirmed = dm2.update_confirmation_state(alerts, ALL_KEYS)
    assert state["noizai_catch_rate"] == 2
    assert [a["key"] for a in confirmed] == ["noizai_catch_rate"]


def test_an_insignificant_alert_does_not_advance_the_counter(tmp_path, monkeypatch):
    """update_confirmation_state only counts alerts marked significant, so a
    threshold breach that fails the z-test stays visible in the report without
    ever escalating to a retrain trigger."""
    dm = _fresh_dm(tmp_path, monkeypatch)
    noise = [{"key": "clean_ensemble_eer", "message": "borderline", "significant": False,
              "p_value": 0.4, "delta_pp": 3.5}]
    state, confirmed = dm.update_confirmation_state(noise, ALL_KEYS)
    assert state.get("clean_ensemble_eer", 0) == 0
    assert confirmed == []

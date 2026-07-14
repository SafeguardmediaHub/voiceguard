import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fresh_dm(tmp_path, monkeypatch):
    # OUTPUT_DIR-derived paths are computed at import, so set the env then (re)load.
    monkeypatch.setenv("DRIFT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("VOICEGUARD_RETRAIN_CMD", raising=False)
    import drift_monitor_3 as dm
    importlib.reload(dm)
    dm.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return dm


def test_empty_confirmed_does_not_fire(tmp_path, monkeypatch):
    dm = _fresh_dm(tmp_path, monkeypatch)
    dm.fire_retrain_trigger([], {"timestamp": "t"})
    assert not dm.RETRAIN_TRIGGER_FILE.exists()
    assert dm.retrain_status()["retrain_needed"] is False


def test_confirmed_fires_and_clears(tmp_path, monkeypatch):
    dm = _fresh_dm(tmp_path, monkeypatch)
    confirmed = [{"key": "clean_ensemble_eer", "message": "Ensemble EER +4pp"},
                 {"key": "noizai_catch_rate", "message": "Noiz.ai catch -18pp"}]
    metrics = {"timestamp": "2026-07-13T00:00:00Z",
               "clean": {"ensemble_eer": 17.5, "deployed_eer": 16.0, "per_source": {"yoruba": 0.7}},
               "noizai": {"catch_rate": 0.65}}
    dm.fire_retrain_trigger(confirmed, metrics)

    st = dm.retrain_status()
    assert st["retrain_needed"] is True
    assert st["reasons"] == ["Ensemble EER +4pp", "Noiz.ai catch -18pp"]
    assert st["metrics_snapshot"]["clean_ensemble_eer"] == 17.5
    assert dm.RETRAIN_TRIGGER_FILE.exists() and dm.RETRAIN_LOG_FILE.exists()

    dm.clear_retrain_trigger()
    assert dm.retrain_status()["retrain_needed"] is False
    assert not dm.RETRAIN_TRIGGER_FILE.exists()

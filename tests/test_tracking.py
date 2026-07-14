import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bundle_registry as br
import tracking


def _mini_bundle(tmp_path):
    d = str(tmp_path / "v1")
    os.makedirs(d, exist_ok=True)
    for name in br.BUNDLE_FILES:
        open(os.path.join(d, name), "wb").write(b"x")
    br.write_manifest(d, {"version": "v1", "metrics": {"val_eer": 0.13, "studio_fp": 0.12},
                          "preprocessing": {"ensemble_peak_norm": True, "lcnn_peak_norm": False}})
    return d


def test_log_bundle_never_raises_and_returns_id_or_none(tmp_path):
    d = _mini_bundle(tmp_path)
    uri = "file:" + str(tmp_path / "mlruns")
    run_id = tracking.log_bundle(d, tracking_uri=uri)
    try:
        import mlflow  # noqa
        assert isinstance(run_id, str) and run_id
    except ImportError:
        assert run_id is None

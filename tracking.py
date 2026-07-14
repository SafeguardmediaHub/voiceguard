"""Thin MLflow wrapper. Logs a bundle's metrics/params/tags + bundle.json —
NEVER the weight blobs. Degrades to a no-op if mlflow is unavailable so a
promote is never blocked by tracking being down."""
import os
from bundle_registry import read_manifest

# Enable file-based MLflow tracking by default (user can override via env var)
if "MLFLOW_ALLOW_FILE_STORE" not in os.environ:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

DEFAULT_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "voiceguard")


def log_bundle(bundle_dir, tracking_uri=None):
    try:
        import mlflow
    except ImportError:
        print("  [tracking] mlflow not installed; skipping (pip install mlflow to enable)")
        return None
    try:
        m = read_manifest(bundle_dir)
        mlflow.set_tracking_uri(tracking_uri or DEFAULT_URI)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=m["version"]) as run:
            pp = m.get("preprocessing", {})
            mlflow.log_params({
                "version": m["version"],
                "ensemble_peak_norm": pp.get("ensemble_peak_norm"),
                "lcnn_peak_norm": pp.get("lcnn_peak_norm"),
                **{f"threshold_{k}": v for k, v in m.get("verdict_thresholds", {}).items()},
            })
            metrics = m.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
            for lang, fpr in (metrics.get("per_language_fpr") or {}).items():
                mlflow.log_metric(f"fpr_{lang}", float(fpr))
            mlflow.set_tags({"git_sha": m.get("git_sha") or "",
                             "train_manifest_hash": m.get("train_manifest_hash") or ""})
            mlflow.log_artifact(os.path.join(bundle_dir, "bundle.json"))
            return run.info.run_id
    except Exception as e:
        print(f"  [tracking] MLflow logging failed ({e}); continuing")
        return None

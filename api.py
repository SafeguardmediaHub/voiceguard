# api.py
"""VoiceGuard V9 — FastAPI web layer over the detector core.
Run: uvicorn api:app --host 0.0.0.0 --port 7860
"""
import os, json, tempfile, traceback, math, csv, io, mimetypes, zipfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

import detector
import auth
import jobs
import evaluations
from request_protection import get_protection, hash_file_content

DRIFT_OUTPUT_DIR = os.environ.get("DRIFT_OUTPUT_DIR", os.path.join(detector.BASE, "output"))
JOBS_INPUT_DIR = os.environ.get("VOICEGUARD_JOBS_INPUT", os.path.join(detector.BASE, "jobs_input"))


@asynccontextmanager
async def lifespan(app):
    # Fail closed: refuse to start if the active bundle can't classify its fixture.
    if not detector.startup_check():
        raise RuntimeError("active bundle failed startup smoke-check; "
                           "roll back with `python bundle_registry.py rollback` and restart")
    yield


app = FastAPI(title="VoiceGuard V9", version="9", lifespan=lifespan)
# Server-to-server deployment: no browser origin needs access, so the default is
# an empty allowlist. Set VOICEGUARD_ALLOWED_ORIGINS (comma-separated) to serve the
# demo UI at / from a browser on another origin.
_ALLOWED_ORIGINS = [o.strip() for o in
                    os.environ.get("VOICEGUARD_ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

security = HTTPBearer(auto_error=False)


def require_api_key(creds: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """FastAPI dependency: require a valid Bearer API key; return the client record."""
    if creds is None or (creds.scheme or "").lower() != "bearer" or not creds.credentials:
        raise HTTPException(status_code=401,
                            detail="Missing or malformed API key (use 'Authorization: Bearer <key>')")
    rec = auth.verify_key(creds.credentials)
    if rec is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    return rec


def require_evaluation_admin(creds: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Restrict evaluation uploads/results to a separately-scoped admin key.

    Ordinary detection keys can submit customer files, but must never be able to
    inspect or alter the labelled evaluation corpus.
    """
    rec = require_api_key(creds)
    scopes = set(rec.get("scopes") or [])
    if not ({"admin", "evaluations:admin"} & scopes):
        raise HTTPException(status_code=403,
                            detail="This endpoint requires an admin evaluation API key")
    return rec


def _safe_unlink(path):
    try:
        os.unlink(path)
    except Exception:
        pass


def _json_safe(obj):
    """Recursively replace NaN/Inf (invalid JSON) with None, so drift responses
    built from persisted files serialize cleanly instead of raising a 500."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ── Drift read helpers (web-adjacent; read what the scheduled monitor writes) ──
def _drift_read_json(name):
    try:
        with open(os.path.join(DRIFT_OUTPUT_DIR, name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _drift_history(limit=50):
    path = os.path.join(DRIFT_OUTPUT_DIR, "drift_log.jsonl")
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        return []
    return rows[-limit:] if limit else rows


def _drift_latest_report():
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(DRIFT_OUTPUT_DIR, "drift_report_*.json")))
    if not files:
        return None
    try:
        with open(files[-1], encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.get("/")
def index():
    html_path = os.path.join(detector.BASE, "VoiceGuard_LiveDemo (2).html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return PlainTextResponse("VoiceGuard_LiveDemo (2).html not found in " + detector.BASE, status_code=404)


@app.get("/admin/evaluations")
def evaluation_dashboard():
    """Serve a harmless UI shell. Its data and mutations still require admin auth."""
    html_path = os.path.join(detector.BASE, "dashboard", "evaluations.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return PlainTextResponse("Evaluation dashboard asset not found", status_code=404)


@app.get("/ping")
def ping():
    m = detector._ACTIVE_MANIFEST or {}
    return {
        "status": "ready",
        "version": "V9",
        "cascade": True,
        "active_version": detector.ACTIVE_VERSION,
        "active_sha": m.get("files", {}).get("aasist.pt", {}).get("sha256", "")[:12],
        "modules": {
            "watermark": detector.AUDIOSEAL_AVAILABLE,
            "metadata":  detector.METADATA_AVAILABLE,
            "c2pa":      detector.C2PA_AVAILABLE,
            "mic_signature": detector.MIC_SIG_AVAILABLE,
        },
    }


# A sync `def` route: FastAPI runs it in its threadpool, so the blocking file
# write + sqlite enqueue don't serialize on the async event loop (which pushed
# submission p95 into the seconds under concurrency). Submission stays fast.
@app.post("/detect", status_code=202)
def detect_route(request: Request,
                 client: dict = Depends(require_api_key),
                 file: UploadFile | None = File(None)):
    if file is None:
        return JSONResponse(status_code=400, content={"error": "No file provided"})
    max_mb = int(os.environ.get("VOICEGUARD_MAX_UPLOAD_MB", 25))
    os.makedirs(JOBS_INPUT_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, dir=JOBS_INPUT_DIR, delete=False)
    try:
        tmp.write(file.file.read())          # sync read (we're in the threadpool)
        tmp.close()
    except Exception as e:
        _safe_unlink(tmp.name)
        return JSONResponse(status_code=400, content={"error": f"File handling failed: {e}"})

    if os.path.getsize(tmp.name) > max_mb * 1024 * 1024:
        _safe_unlink(tmp.name)
        return JSONResponse(status_code=413, content={"error": "File too large", "max_mb": max_mb})

    if detector.REQUEST_PROTECTION_ENABLED:
        file_hash = hash_file_content(tmp.name)
        allowed, retry_after, info = get_protection().check_request(client["key_id"], file_hash)
        if not allowed:
            _safe_unlink(tmp.name)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded",
                         "retry_after_sec": round(retry_after, 1),
                         "anomalies": info["anomalies"]},
                headers={"Retry-After": str(int(retry_after) + 1)})

    # Enqueue for the worker; the saved file persists until the worker consumes it.
    job_id = jobs.enqueue(client["client"], client["key_id"], tmp.name)
    return {"job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, client: dict = Depends(require_api_key)):
    job = jobs.get(job_id)
    if job is None or job.get("key_id") != client["key_id"]:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    status = job["status"]
    body = {"job_id": job["job_id"], "status": status,
            "created_at": job["created_at"], "status_url": f"/jobs/{job['job_id']}"}
    if status == "running":
        body["started_at"] = job["started_at"]
    elif status == "done":
        body["finished_at"] = job["finished_at"]
        body["result"] = job.get("result")
    elif status == "error":
        body["finished_at"] = job["finished_at"]
        body["error"] = job["error"]
    return _json_safe(body)


@app.get("/drift")
def drift_composite(client: dict = Depends(require_api_key)):
    baseline = _drift_read_json("drift_baseline.json")
    latest   = _drift_latest_report()
    return _json_safe({
        "available":   baseline is not None or latest is not None,
        "output_dir":  DRIFT_OUTPUT_DIR,
        "baseline":    baseline,
        "latest":      latest,
        "history":     _drift_history(),
        "alert_state": _drift_read_json("drift_alert_state.json"),
        "retrain":     _drift_read_json("retrain_trigger.json"),
    })


@app.get("/drift/latest")
def drift_latest(client: dict = Depends(require_api_key)):
    r = _drift_latest_report()
    if r is None:
        return JSONResponse(status_code=404, content={"available": False, "message": "no drift runs yet"})
    return _json_safe(r)


@app.get("/drift/history")
def drift_history_route(limit: int = 50, client: dict = Depends(require_api_key)):
    return _json_safe({"runs": _drift_history(limit)})


@app.get("/drift/baseline")
def drift_baseline_route(client: dict = Depends(require_api_key)):
    b = _drift_read_json("drift_baseline.json")
    if b is None:
        return JSONResponse(status_code=404, content={"available": False, "message": "no baseline set"})
    return _json_safe(b)


# ── Labelled evaluation dashboard API ───────────────────────────────────────
@app.get("/admin/evaluations/api/batches")
def evaluation_batches(limit: int = 50, client: dict = Depends(require_evaluation_admin)):
    return _json_safe({"batches": evaluations.list_batches(limit)})


@app.get("/admin/evaluations/api/batches/{batch_id}")
def evaluation_batch(batch_id: str, client: dict = Depends(require_evaluation_admin)):
    batch = evaluations.get_batch(batch_id, include_samples=True)
    if batch is None:
        return JSONResponse(status_code=404, content={"error": "batch not found"})
    return _json_safe(batch)


@app.get("/admin/evaluations/api/batches/{batch_id}/hard-cases.csv")
def evaluation_hard_cases_csv(batch_id: str, client: dict = Depends(require_evaluation_admin)):
    """Export mistakes as a review/training-selection manifest, never as labels.

    A human must still inspect provenance and consent before moving a clip into a
    training corpus.  The CSV makes that review repeatable without silently
    promoting candidate evaluation data into the frozen drift baseline.
    """
    batch = evaluations.get_batch(batch_id, include_samples=True)
    if batch is None:
        return JSONResponse(status_code=404, content={"error": "batch not found"})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=[
        "sample_id", "original_name", "ground_truth", "outcome", "verdict", "score",
        "lcnn_pct", "aasist_pct", "w2v_pct", "rawnet_pct", "ensemble_pct", "sha256",
    ])
    writer.writeheader()
    for sample in batch.get("samples", []):
        outcome = sample.get("outcome") or {}
        if outcome.get("correct") is not False:
            continue
        result = sample.get("result") or {}
        models = result.get("model_scores") or {}
        writer.writerow({
            "sample_id": sample["sample_id"], "original_name": sample["original_name"],
            "ground_truth": "fake" if batch["label"] else "real", "outcome": outcome.get("kind"),
            "verdict": result.get("verdict"), "score": result.get("score"),
            "lcnn_pct": models.get("lcnn"), "aasist_pct": models.get("aasist"),
            "w2v_pct": models.get("w2v"), "rawnet_pct": models.get("rawnet"),
            "ensemble_pct": models.get("ensemble"), "sha256": result.get("sha256"),
        })
    safe_name = "".join(c if c.isalnum() or c in "_.-" else "_" for c in batch["source"])
    return PlainTextResponse(output.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{safe_name}_hard_cases.csv"'})


@app.get("/admin/evaluations/api/samples/{sample_id}/audio")
def evaluation_sample_audio(sample_id: str, client: dict = Depends(require_evaluation_admin)):
    """Download original labelled evidence through the admin-only API."""
    sample = evaluations.get_sample(sample_id)
    if sample is None:
        return JSONResponse(status_code=404, content={"error": "sample not found"})
    path = sample["stored_path"]
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": "evaluation audio is no longer available"})
    media_type = mimetypes.guess_type(sample["original_name"])[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=sample["original_name"])


@app.post("/admin/evaluations/api/samples/{sample_id}/curation")
def evaluation_update_curation(sample_id: str, curation_status: str = Form(...),
                               dataset_split: str = Form(""), note: str = Form(""),
                               client: dict = Depends(require_evaluation_admin)):
    """Save an explicit reviewer decision; this never triggers model training."""
    try:
        sample = evaluations.update_curation(sample_id, curation_status, dataset_split, note)
    except KeyError:
        return JSONResponse(status_code=404, content={"error": "sample not found"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return _json_safe(sample)


@app.get("/admin/evaluations/api/curation-summary")
def evaluation_curation_summary(client: dict = Depends(require_evaluation_admin)):
    return _json_safe({"summary": evaluations.curation_summary()})


@app.get("/admin/evaluations/api/datasets")
def evaluation_datasets(limit: int = 50, client: dict = Depends(require_evaluation_admin)):
    return _json_safe({"datasets": evaluations.list_dataset_revisions(limit)})


@app.post("/admin/evaluations/api/datasets", status_code=201)
def evaluation_create_dataset(name: str = Form(...), notes: str = Form(""),
                              client: dict = Depends(require_evaluation_admin)):
    try:
        revision = evaluations.create_dataset_revision(client["key_id"], name, notes)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return _json_safe(revision)


def _delete_file(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@app.get("/admin/evaluations/api/datasets/{revision_id}/archive")
def evaluation_dataset_archive(revision_id: str, client: dict = Depends(require_evaluation_admin)):
    """Build a portable, immutable dataset snapshot for an offline GPU run.

    The archive is an export only: it does not train, promote, or alter the
    deployed bundle.  Every manifest row contains the frozen revision ID and
    SHA-256 so the GPU run can be reproduced and audited later.
    """
    revision, samples = evaluations.dataset_archive_entries(revision_id)
    if revision is None:
        return JSONResponse(status_code=404, content={"error": "dataset revision not found"})
    fd, archive_path = tempfile.mkstemp(prefix="voiceguard-dataset-", suffix=".zip")
    os.close(fd)
    try:
        manifests = {"train": [], "validation": [], "heldout": []}
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for sample in samples:
                split = sample["dataset_split"]
                manifest_split = "heldout" if split == "holdout" else split
                safe_name = "".join(c if c.isalnum() or c in "_.-" else "_" for c in sample["original_name"])
                archive_name = f"audio/{manifest_split}/{sample['sample_id']}_{safe_name}"
                if os.path.isfile(sample["stored_path"]):
                    archive.write(sample["stored_path"], archive_name)
                else:
                    # Do not silently manufacture a trainable archive with a missing
                    # file.  The manifest exposes the gap so it can be corrected.
                    archive_name = None
                manifests[manifest_split].append({
                    "path": archive_name,
                    "label": sample["label"],
                    "source": sample["source"],
                    "sample_id": sample["sample_id"],
                    "sha256": sample["sha256"],
                    "curation_note": sample["curation_note"],
                    "dataset_revision": revision_id,
                })
            for split, rows in manifests.items():
                archive.writestr(f"{split}.json", json.dumps(rows, indent=2) + "\n")
            archive.writestr("dataset_revision.json", json.dumps(revision, indent=2) + "\n")
            archive.writestr("README.txt", (
                "VoiceGuard curated dataset export\n\n"
                f"Revision: {revision_id}\n"
                "This export does NOT train or promote a model. Keep heldout.json out of training and "
                "use it only for final certification. Verify each file SHA-256 before a GPU run.\n"
            ))
    except Exception:
        _delete_file(archive_path)
        raise
    safe_name = revision["name"]
    return FileResponse(archive_path, media_type="application/zip",
                        filename=f"voiceguard_{safe_name}_{revision_id}.zip",
                        background=BackgroundTask(_delete_file, archive_path))


@app.post("/admin/evaluations/api/batches", status_code=201)
def evaluation_create_batch(name: str = Form(...), source: str = Form(...), label: int = Form(...),
                            notes: str = Form(""), client: dict = Depends(require_evaluation_admin)):
    try:
        batch = evaluations.create_batch(client["key_id"], name, source, label, notes)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return _json_safe(batch)


@app.post("/admin/evaluations/api/batches/{batch_id}/files")
def evaluation_upload_files(batch_id: str, files: list[UploadFile] = File(...),
                            client: dict = Depends(require_evaluation_admin)):
    max_mb = int(os.environ.get("VOICEGUARD_EVALUATION_MAX_UPLOAD_MB",
                                os.environ.get("VOICEGUARD_MAX_UPLOAD_MB", "25")))
    uploaded, duplicates, errors = [], [], []
    for file in files:
        try:
            result = evaluations.add_upload(batch_id, file.filename, file.file, max_mb * 1024 * 1024)
            (duplicates if result["status"] == "duplicate" else uploaded).append(result)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "batch not found"})
        except ValueError as e:
            errors.append({"filename": file.filename, "error": str(e)})
        finally:
            try:
                file.file.close()
            except Exception:
                pass
    code = 413 if any(e["error"] == "file too large" for e in errors) else 200
    return JSONResponse(status_code=code, content=_json_safe({
        "uploaded": uploaded, "duplicates": duplicates, "errors": errors, "max_mb": max_mb,
    }))


@app.post("/admin/evaluations/api/batches/{batch_id}/run", status_code=202)
def evaluation_run_batch(batch_id: str, client: dict = Depends(require_evaluation_admin)):
    try:
        batch = evaluations.start_batch(batch_id)
    except KeyError:
        return JSONResponse(status_code=404, content={"error": "batch not found"})
    except (PermissionError, ValueError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return _json_safe({"batch_id": batch_id, "status": batch["status"],
                       "message": "evaluation queued; refresh this batch for progress"})


@app.get("/admin/evaluations/api/metrics")
def evaluation_metrics(limit: int = 100, client: dict = Depends(require_evaluation_admin)):
    return _json_safe({"metrics": evaluations.metrics(limit)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

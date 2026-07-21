# api.py
"""VoiceGuard V9 — FastAPI web layer over the detector core.
Run: uvicorn api:app --host 0.0.0.0 --port 7860
"""
import os, json, tempfile, traceback, math
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import detector
import auth
import jobs
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

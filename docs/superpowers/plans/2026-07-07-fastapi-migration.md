# FastAPI Migration (REST Parity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move VoiceGuard's web layer from Flask to FastAPI at exact REST parity, by extracting the framework-agnostic scoring core into `detector.py` and building a FastAPI app (`api.py`) on top.

**Architecture:** `server.py` (1,023 lines) splits into `detector.py` (scoring core — no web imports) and `api.py` (FastAPI routes). Consumers (`drift_monitor_3.py`, `tests/test_golden.py`) repoint from `server` to `detector`. The move is behavior-preserving: the golden test's pinned scores and a new API-parity test are the correctness gate.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, python-multipart, starlette `TestClient`, PyTorch 2.12+cpu, pytest. Existing modules reused unchanged: `bundle_registry`, `request_protection`, `input_randomization`, Phase-4 signal modules.

## Global Constraints

- **Commits are the human's.** Implementers do NOT run `git add` or `git commit` — leave all changes in the working tree. Each task ends with verification, not a commit.
- **Move-only migration.** Do NOT change model behavior, thresholds, calibration, or scoring math. `detector.py`'s scoring code is copied verbatim from `server.py`.
- **`detector.py` must not import Flask or FastAPI** (no web framework in the core).
- **Parity is the gate.** The golden clips must keep their pinned verdicts/scores exactly: `real_studio_037`→AUTO_REAL 0.1708, `real_studio_055`→AUTO_REAL 0.1359, `fake_noizai_a4cd`→LIKELY_FAKE 0.8167, `fake_concert_hall`→REVIEW 0.3703. If any moves, the migration is not done.
- All `torch.load()` on VoiceGuard checkpoints use `weights_only=False` (inherited from the copied code).
- Platform: Windows, Git Bash; run tests with `python -m pytest`.
- Run command after migration: `uvicorn api:app --host 0.0.0.0 --port 7860` (replaces `python server.py`).

---

## File Structure

- Create `detector.py` — the scoring core: config constants, model classes, mel, bundle resolution + model loading, all scoring (`predict_ensemble`, `lcnn_score`, `ensemble_score_variants`, `cascade_score_chunk`, `verdict_from_score`, `merge_chunks_to_segments`, `get_codec`, `cw_score`, `detect`), Phase-4 module loading, and `startup_check()`. No web imports.
- Create `api.py` — the FastAPI app: lifespan (calls `detector.startup_check()`), CORS, routes (`/`, `/ping`, `/detect`, `/drift`, `/drift/latest`, `/drift/history`, `/drift/baseline`), and the drift file-reading helpers.
- Create `tests/test_detector.py` — core parity test.
- Create `tests/test_api.py` — API parity test via `TestClient`.
- Modify `drift_monitor_3.py` — `import server` → `import detector`.
- Modify `tests/test_golden.py` — `import server` → `import detector` (+ `server.detect` → `detector.detect`).
- Modify `requirements.txt` — add `fastapi`, `uvicorn[standard]`, `python-multipart`.
- Back up `server.py` → `server_flask_backup.py`; retire `server.py` from the run path.

---

## Task 1: Extract the scoring core into `detector.py`

**Files:**
- Create: `detector.py` (extracted from `server.py`)
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: existing `server.py` content; `bundle_registry`, `request_protection`, `input_randomization`, Phase-4 modules (imported inside the copied code).
- Produces: module `detector` exposing `detect(file_path) -> dict`, `cascade_score_chunk`, `lcnn_score`, `ensemble_score_variants`, `predict_ensemble`, `verdict_from_score`, `merge_chunks_to_segments`, `get_codec`, `startup_check() -> bool`, and globals `CHUNK`, `SR`, `DEVICE`, `BASE`, `CASCADE_LOW`, `CASCADE_HIGH`, `thresholds`, `ACTIVE_VERSION`, `_ACTIVE_MANIFEST`, `REQUEST_PROTECTION_ENABLED`, `AUDIOSEAL_AVAILABLE`, `METADATA_AVAILABLE`, `C2PA_AVAILABLE`, `MIC_SIG_AVAILABLE`.

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_detector.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_core_exposes_expected_surface():
    for name in ["detect", "cascade_score_chunk", "lcnn_score",
                 "ensemble_score_variants", "verdict_from_score", "startup_check",
                 "CHUNK", "CASCADE_LOW", "CASCADE_HIGH", "thresholds", "ACTIVE_VERSION"]:
        assert hasattr(detector, name), f"detector missing {name}"

def test_detect_parity_on_golden_clips():
    cases = [
        ("tests/golden_clips/fake_noizai_a4cd.mp3",  "LIKELY_FAKE", 0.8167),
        ("tests/golden_clips/real_studio_037.mp3",   "AUTO_REAL",   0.1708),
    ]
    for rel, verdict, score in cases:
        r = detector.detect(os.path.join(REPO, rel))
        assert r["verdict"] == verdict, f"{rel}: {r['verdict']} != {verdict}"
        assert abs(r["score"] - score) < 1e-3, f"{rel}: {r['score']} != {score}"
        assert r["model_version"] == "V9"

def test_import_does_not_exit():
    # startup_check must be a function, not auto-run at import (importing above
    # already succeeded — a sys.exit at import would have aborted this module).
    assert callable(detector.startup_check)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_detector.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'detector'`.

- [ ] **Step 3: Create `detector.py` by extracting the core from `server.py`**

Run this exact extraction script (it copies `server.py`, drops the Flask import, cuts the Flask-app section to EOF, removes the import-time smoke-check auto-run, and renames the smoke-check to a public `startup_check`):

```bash
python - <<'PY'
s = open("server.py", encoding="utf-8").read()

# 1) drop the Flask import
s = s.replace("from flask import Flask, request, jsonify, send_file\n", "")

# 2) cut everything from the Flask-app section header to EOF
idx        = s.index("#  Flask app")                    # unique marker
line_start = s.rindex("\n", 0, idx) + 1                 # start of "#  Flask app" line
div_start  = s.rindex("\n", 0, line_start - 1) + 1      # start of the "# ══" divider above it
s = s[:div_start].rstrip() + "\n"

# 3) remove the import-time smoke-check auto-run (api.py runs it via lifespan)
s = s.replace(
'''if not _startup_smoke_check():
    print("  [smoke-check] CRITICAL: active bundle failed its smoke-check. "
          "Roll back with `python bundle_registry.py rollback` and restart.")
    # Fail closed rather than serve a model that can't classify its own fixture.
    import sys as _sys; _sys.exit(1)
''', "")

# 4) make the smoke-check public
s = s.replace("def _startup_smoke_check():", "def startup_check():")

open("detector.py", "w", encoding="utf-8", newline="\n").write(s)
print("wrote detector.py")
PY
```

- [ ] **Step 4: Verify the extraction structurally**

Run:
```bash
python -c "import detector; print('import OK', detector.ACTIVE_VERSION)"
grep -c "Flask\|@app.route\|app.run" detector.py     # expect 0
grep -c "def startup_check" detector.py              # expect 1
grep -c "sys.exit\|_startup_smoke_check" detector.py # expect 0
```
Expected: `import OK v9`; then `0`, `1`, `0`. (If `import detector` prints a `[smoke-check]` line or exits, the auto-run block was not removed — re-check step 3.)

- [ ] **Step 5: Run the parity test to verify it passes**

Run: `python -m pytest tests/test_detector.py -q`
Expected: PASS (3 passed). Scores match the pinned values exactly.

---

## Task 2: Build the FastAPI app `api.py`

**Files:**
- Create: `api.py`
- Create: `tests/test_api.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `detector` (Task 1) — `detector.detect`, `detector.BASE`, `detector.ACTIVE_VERSION`, `detector._ACTIVE_MANIFEST`, `detector.REQUEST_PROTECTION_ENABLED`, `detector.AUDIOSEAL_AVAILABLE`/`METADATA_AVAILABLE`/`C2PA_AVAILABLE`/`MIC_SIG_AVAILABLE`, `detector.startup_check`; `request_protection.get_protection`, `request_protection.hash_file_content`.
- Produces: `api.app` (a FastAPI instance) with routes `GET /`, `GET /ping`, `POST /detect`, `GET /drift`, `GET /drift/latest`, `GET /drift/history`, `GET /drift/baseline`.

- [ ] **Step 1: Install FastAPI dependencies and record them**

Run:
```bash
python -m pip install fastapi "uvicorn[standard]" python-multipart
python - <<'PY'
p="requirements.txt"; s=open(p,encoding="utf-8").read()
for dep in ["fastapi","uvicorn[standard]","python-multipart"]:
    if dep.split("[")[0] not in s:
        s=s.rstrip("\n")+"\n"+dep+"\n"
open(p,"w",encoding="utf-8",newline="\n").write(s)
print("requirements updated")
PY
```
Expected: installs succeed; `requirements.txt` gains the three deps.

- [ ] **Step 2: Write the failing API-parity test**

```python
# tests/test_api.py
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
import api

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="module")
def client():
    # context manager triggers the lifespan startup (startup_check)
    with TestClient(api.app) as c:
        yield c

def test_ping(client):
    r = client.get("/ping")
    assert r.status_code == 200
    j = r.json()
    assert j["version"] == "V9"
    assert j["active_version"]
    assert set(["watermark","metadata","c2pa","mic_signature"]).issubset(j["modules"].keys())

def test_detect_fake(client):
    with open(os.path.join(REPO, "tests/golden_clips/fake_noizai_a4cd.mp3"), "rb") as f:
        r = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")})
    assert r.status_code == 200
    j = r.json()
    assert j["verdict"] == "LIKELY_FAKE"
    assert abs(j["score"] - 0.8167) < 1e-3
    assert j["model_version"] == "V9"
    for k in ["cascade","segments","flagged_segments","watermark","metadata",
              "c2pa","mic_signature","explanation","filename"]:
        assert k in j

def test_detect_missing_file(client):
    r = client.post("/detect")           # no file part
    assert r.status_code in (400, 422)   # FastAPI validation → 422; explicit → 400

def test_drift_composite(client):
    r = client.get("/drift")
    assert r.status_code == 200
    assert "available" in r.json()
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_api.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api'`.

- [ ] **Step 4: Implement `api.py`**

```python
# api.py
"""VoiceGuard V9 — FastAPI web layer over the detector core.
Run: uvicorn api:app --host 0.0.0.0 --port 7860
"""
import os, json, tempfile, traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import detector
from request_protection import get_protection, hash_file_content

DRIFT_OUTPUT_DIR = os.environ.get("DRIFT_OUTPUT_DIR", os.path.join(detector.BASE, "output"))


@asynccontextmanager
async def lifespan(app):
    # Fail closed: refuse to start if the active bundle can't classify its fixture.
    if not detector.startup_check():
        raise RuntimeError("active bundle failed startup smoke-check; "
                           "roll back with `python bundle_registry.py rollback` and restart")
    yield


app = FastAPI(title="VoiceGuard V9", version="9", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def _safe_unlink(path):
    try:
        os.unlink(path)
    except Exception:
        pass


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
    html_path = os.path.join(detector.BASE, "VoiceGuard_Demo.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return PlainTextResponse("VoiceGuard_Demo.html not found in " + detector.BASE, status_code=404)


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


@app.post("/detect")
async def detect_route(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        tmp.write(await file.read())
        tmp.close()
    except Exception as e:
        _safe_unlink(tmp.name)
        return JSONResponse(status_code=400, content={"error": f"File handling failed: {e}"})

    protection_info = {"flagged": False, "anomalies": []}
    if detector.REQUEST_PROTECTION_ENABLED:
        file_hash = hash_file_content(tmp.name)
        source = request.client.host if request.client else "unknown"
        allowed, retry_after, protection_info = get_protection().check_request(source, file_hash)
        if not allowed:
            _safe_unlink(tmp.name)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded",
                         "retry_after_sec": round(retry_after, 1),
                         "anomalies": protection_info["anomalies"]},
                headers={"Retry-After": str(int(retry_after) + 1)})

    try:
        result = detector.detect(tmp.name)
        result["filename"] = file.filename
        if detector.REQUEST_PROTECTION_ENABLED:
            result["request_protection"] = {"flagged": protection_info["flagged"],
                                            "anomalies": protection_info["anomalies"]}
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        _safe_unlink(tmp.name)


@app.get("/drift")
def drift_composite():
    baseline = _drift_read_json("drift_baseline.json")
    latest   = _drift_latest_report()
    return {
        "available":   baseline is not None or latest is not None,
        "output_dir":  DRIFT_OUTPUT_DIR,
        "baseline":    baseline,
        "latest":      latest,
        "history":     _drift_history(),
        "alert_state": _drift_read_json("drift_alert_state.json"),
    }


@app.get("/drift/latest")
def drift_latest():
    r = _drift_latest_report()
    if r is None:
        return JSONResponse(status_code=404, content={"available": False, "message": "no drift runs yet"})
    return r


@app.get("/drift/history")
def drift_history_route(limit: int = 50):
    return {"runs": _drift_history(limit)}


@app.get("/drift/baseline")
def drift_baseline_route():
    b = _drift_read_json("drift_baseline.json")
    if b is None:
        return JSONResponse(status_code=404, content={"available": False, "message": "no baseline set"})
    return b


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

- [ ] **Step 5: Run the API-parity test to verify it passes**

Run: `python -m pytest tests/test_api.py -q`
Expected: PASS (4 passed). `/ping` reports V9, `/detect` returns LIKELY_FAKE at 0.8167 with the full key set, `/drift` returns the composite.

---

## Task 3: Repoint the drift monitor and golden test at `detector`

**Files:**
- Modify: `tests/test_golden.py`
- Modify: `drift_monitor_3.py`

**Interfaces:**
- Consumes: `detector` (Task 1) in place of `server`.
- Produces: no new surface — behavior identical, dependency changed.

- [ ] **Step 1: Repoint the golden test**

In `tests/test_golden.py`, change the import and the call. The file currently does `import server` and calls `server.detect(...)` inside `_run`. Replace both:

```bash
python - <<'PY'
p="tests/test_golden.py"; s=open(p,encoding="utf-8").read()
s=s.replace("import server", "import detector as server")  # keep local alias; minimal churn
open(p,"w",encoding="utf-8",newline="\n").write(s)
print("golden test repointed" if "import detector as server" in s else "NO CHANGE")
PY
```
(Using `import detector as server` keeps every `server.detect` call working with a one-line change.)

- [ ] **Step 2: Run the golden test — pinned scores must be unchanged**

Run: `python tests/test_golden.py`
Expected: `RESULT: all 4 golden clips passed` — scores exactly 0.1708 / 0.1359 / 0.8167 / 0.3703.

- [ ] **Step 3: Repoint the drift monitor**

`drift_monitor_3.py` imports the scoring core as `server`. Repoint it to `detector` with the same alias so every `server.X` reference keeps working:

```bash
python - <<'PY'
p="drift_monitor_3.py"; s=open(p,encoding="utf-8").read()
assert "import server" in s, "expected 'import server' in drift_monitor_3.py"
s=s.replace("import server", "import detector as server")
open(p,"w",encoding="utf-8",newline="\n").write(s)
print("drift monitor repointed")
PY
python -m py_compile drift_monitor_3.py && echo "compile OK"
```
Expected: `drift monitor repointed`, `compile OK`.

- [ ] **Step 4: Smoke-test the repointed drift monitor**

Run (uses the small local fixtures; completes in ~1 min):
```bash
DRIFT_VAL_MANIFEST=tests/drift_fixtures/val_fast.json \
DRIFT_NOIZAI_DIR=tests/drift_fixtures/noiz_fast \
DRIFT_OUTPUT_DIR=output \
python drift_monitor_3.py --init-baseline --quick
```
Expected: completes with no traceback; logs `Active bundle: v9`; writes `output/drift_baseline.json`. (If the fast fixtures are absent, this is still a valid check that `import detector as server` resolves — the run will report a missing-manifest warning instead, which also confirms the import works.)

---

## Task 4: Retire the Flask server and update the run path

**Files:**
- Create: `server_flask_backup.py` (copy of the current Flask `server.py`)
- Modify: `docs/RUNBOOK-model-flow.md` (run-command reference)

- [ ] **Step 1: Back up the Flask server**

Run:
```bash
cp server.py server_flask_backup.py
echo "backed up Flask server -> server_flask_backup.py"
```
(The Flask `server.py` is retired from the run path but retained as `server_flask_backup.py` for rollback. Do NOT delete `server.py` in this task — leave it in the tree so the human can decide when to remove it; nothing runs it anymore.)

- [ ] **Step 2: Update the runbook's restart/run reference**

In `docs/RUNBOOK-model-flow.md`, the "Restart the server to apply" section references `python server.py`. Update the local-run line:

```bash
python - <<'PY'
p="docs/RUNBOOK-model-flow.md"; s=open(p,encoding="utf-8").read()
s=s.replace("stop and re-run `python server.py`",
            "stop and re-run `uvicorn api:app --host 0.0.0.0 --port 7860`")
open(p,"w",encoding="utf-8",newline="\n").write(s)
print("runbook updated" if "uvicorn api:app" in s else "NO CHANGE (check wording)")
PY
```
Expected: `runbook updated`. (If it prints `NO CHANGE`, open the file and update the local-run line by hand to `uvicorn api:app --host 0.0.0.0 --port 7860`.)

- [ ] **Step 3: Final end-to-end verification**

Run:
```bash
python -m pytest tests/test_detector.py tests/test_api.py -q
python tests/test_golden.py | grep RESULT
```
Expected: pytest all pass; `RESULT: all 4 golden clips passed`.

- [ ] **Step 4: Boot the real server once (manual smoke)**

Run (start, hit /ping, stop):
```bash
uvicorn api:app --host 0.0.0.0 --port 7860 &
sleep 20
curl -s http://localhost:7860/ping
# expect JSON: {"status":"ready","version":"V9","cascade":true,"active_version":"v9",...}
```
Then stop the server. Expected: `/ping` returns the V9 ready JSON and `/docs` is reachable in a browser.

---

## Self-Review

**Spec coverage:**
- §3 module split (`detector.py` no web imports; `api.py` web only) → Tasks 1, 2. ✓
- §4 core extraction (verbatim move) + `startup_check()` refactor (not auto-run; lifespan calls it) → Task 1 (extraction script removes auto-run + renames), Task 2 (lifespan). ✓
- §5 FastAPI app: routes at parity, CORS, `request_protection` reused, dict responses, drift helpers moved, `/docs` free, `uvicorn api:app` → Task 2. ✓
- §6 repoint drift + golden → Task 3. ✓
- §7 parity contract (golden pinned scores + API test) → Task 1 test, Task 2 test, Task 3 golden. ✓
- §10 testing (golden via core, TestClient API test, drift smoke) → Tasks 1-3. ✓
- §11 files/requirements/backup/run command → Task 2 (deps), Task 4 (backup + runbook). ✓

**Notes on intentional spec deviations:**
- Spec §5 mentioned a "small Pydantic `PingResponse` model"; the plan returns a plain dict from `/ping` (parity-exact with the Flask `jsonify(dict)`, YAGNI — typed models can come with the API-reference work). `/docs` still generates from the route.
- `test_detect_missing_file` accepts 400 **or** 422: FastAPI's `File(...)` returns 422 on a missing required file (framework default), which is acceptable parity for "no file provided."

**Placeholder scan:** no TBD/TODO; every code step has complete code. ✓

**Type consistency:** `detector` public names used in `api.py` and the tests (`detect`, `startup_check`, `BASE`, `ACTIVE_VERSION`, `_ACTIVE_MANIFEST`, `REQUEST_PROTECTION_ENABLED`, `*_AVAILABLE`, `CHUNK`, `CASCADE_LOW/HIGH`, `thresholds`) all exist in `server.py` today and are preserved by the verbatim copy. ✓

# FastAPI Migration (REST Parity) — Design Spec

- **Date:** 2026-07-07
- **Status:** Approved (design); pending implementation plan
- **Scope:** Sub-project A of the VoiceGuard API modernization (FastAPI move)
- **Follow-ups (separate specs):** B — real-time streaming (websocket); C — Phase 7 hardening (auth, rate-limit, TLS, encryption, load/pen test)

---

## 1. Context & Problem

`server.py` (~1,023 lines) is a Flask app that tangles three concerns in one file:

1. **Model layer** — the V9 model classes (LCNN / AASIST / Wav2Vec / RawNet), the
   pure-torch mel, bundle resolution, and model loading.
2. **Scoring core** — `predict_ensemble`, `lcnn_score`, `ensemble_score_variants`,
   `cascade_score_chunk`, `verdict_from_score`, `merge_chunks_to_segments`, `detect`.
3. **Web layer** — Flask routes (`/`, `/ping`, `/detect`, `/drift*`), CORS,
   file handling, request-protection.

`drift_monitor_3.py` currently does `import server` purely to reach the scoring
core — so a batch monitoring job depends on the whole Flask web app.

The project is moving to **FastAPI** because real-time / streaming inference is a
first-class enterprise use case (websocket streaming is a poor fit for Flask), and
the API surface is still small (5 routes) — the cheapest time to migrate. This
sub-project does the **framework move at REST parity**; streaming and hardening
are separate follow-ups that build on the result.

## 2. Goals / Non-Goals

**Goals:**

1. Extract the framework-agnostic scoring core into **`detector.py`**.
2. Build a **FastAPI** app (**`api.py`**) that reproduces today's endpoints and JSON
   **byte-for-byte** in behavior.
3. Repoint `drift_monitor_3.py` and `tests/test_golden.py` at **`detector`** (not the
   web app), so the monitor no longer depends on a web framework.
4. Prove parity: the golden test's pinned scores must not move, and an API-level
   parity test must pass.

**Non-Goals (explicitly out of scope; later sub-projects):**

- Websocket / streaming inference (sub-project B).
- Authentication, rate-limit hardening beyond today's `request_protection`, TLS,
  encryption at rest, load/pen testing (sub-project C).
- Full Pydantic response models for the large `/detect` response (return the dict as
  today; a `/ping` model is included). Full schemas belong with the API-reference work.
- A `Detector` **class** refactor. This migration keeps the existing
  **module-singleton** pattern (models load once at import; functions reference module
  globals) — a *move*, not a restructure. The class refactor is a possible later cleanup.
- Any change to model behavior, thresholds, or scoring math.

## 3. Module Split

```
detector.py   ← the scoring core (no web imports; no Flask/FastAPI)
api.py        ← the FastAPI app (imports detector; only web concerns)
```

- `detector.py` **must not** import Flask or FastAPI. Its only heavy imports are the
  ML stack + the existing helper modules (`bundle_registry`, `input_randomization`,
  and the Phase-4 signal modules), exactly as `server.py` uses them today.
- `api.py` imports `detector` and the web stack only.

## 4. The Core — `detector.py`

**What moves (verbatim from `server.py`, behavior-identical):**

- Config constants: `BASE`, `MODELS`, `SR`, `CHUNK`, `N_MELS/N_FFT/HOP_LENGTH/WIN_LENGTH`,
  `DEVICE`, `INPUT_RANDOMIZATION_*`, `REQUEST_PROTECTION_ENABLED`.
- Model classes: `MaxFeatureMap`, `LCNNBlock`, `LightCNN`, `SincConvAASIST`,
  `ResBlock1D`, `GATLayer`, `AASIST_V9`, `Wav2Vec2Classifier_V9`, `SincConvRaw`,
  `RawResBlock`, `RawNet3_V8`.
- Mel helpers: `_hz_to_mel_htk`, `_mel_to_hz_htk`, `_melscale_fbanks`, `_MEL_WINDOW`,
  `_MEL_FB`, `wav_to_mel`.
- Bundle resolution + model loading: `_resolve_bundle_paths`, `_BUNDLE_PATHS`,
  `ACTIVE_VERSION`, `_ACTIVE_MANIFEST`, `_load_ckpt`, `_state`, the module-level model
  load, `xgb_model`, `cal_params`, `ENS_CAL_COEF/INT`, `thresholds`, `lcnn`,
  `CASCADE_LOW/HIGH`, `LCNN_PLATT_C/I`.
- Scoring: `_sigmoid`, `predict_ensemble`, `lcnn_score`, `ensemble_score_variants`,
  `cascade_score_chunk`, `cw_score`, `get_codec`, `verdict_from_score`,
  `merge_chunks_to_segments`, and `detect(file_path) -> dict`.
- Phase-4 signal module loading (`AUDIOSEAL_AVAILABLE` etc.) — unchanged.

**The one intentional change — startup check:**

Today the smoke-check runs at *import* and can `sys.exit(1)` — so importing the core
for tests or the drift monitor can kill them. In `detector.py`:

- Model loading still happens at import (module-singleton — unchanged behavior).
- The smoke-check body becomes a function **`startup_check() -> bool`** (loads
  nothing new; scores the active bundle's `smoke_check` clip and compares the verdict).
  It is **not** auto-run at import.
- `api.py` calls `startup_check()` in its FastAPI **lifespan startup** and raises
  (fail-fast) if it returns False — so the *server* still fails closed on a bad bundle,
  but importing the core does not force an exit.

**Public surface `detector.py` exposes** (names unchanged so consumers only swap the
import): `detect`, `cascade_score_chunk`, `lcnn_score`, `ensemble_score_variants`,
`predict_ensemble`, `verdict_from_score`, `merge_chunks_to_segments`, `get_codec`,
`CHUNK`, `SR`, `DEVICE`, `CASCADE_LOW`, `CASCADE_HIGH`, `thresholds`, `ACTIVE_VERSION`,
`_ACTIVE_MANIFEST`, `AUDIOSEAL_AVAILABLE`, `METADATA_AVAILABLE`, `C2PA_AVAILABLE`,
`MIC_SIG_AVAILABLE`, `startup_check`.

## 5. The FastAPI App — `api.py`

- `app = FastAPI(title="VoiceGuard V9", version="9")`, with a **lifespan** that runs
  `detector.startup_check()` on startup and raises `RuntimeError` if it fails.
- **CORSMiddleware** with `allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]`
  (equivalent to today's global CORS; FastAPI handles OPTIONS preflight — the manual
  OPTIONS handler is dropped).
- **Routes (parity):**
  - `GET /` → `FileResponse(VoiceGuard_Demo.html)` if present, else 404 text.
  - `GET /ping` → a small Pydantic `PingResponse` model:
    `{status, version, cascade, active_version, active_sha, modules:{watermark, metadata, c2pa, mic_signature}}`.
  - `POST /detect` (`file: UploadFile`): save to a temp file (preserving extension),
    run request-protection (same 429 + `Retry-After` behavior), call
    `detector.detect(tmp)`, attach `filename` + `request_protection`, return the dict.
    On error → `JSONResponse(status_code=500, {"error": str(e)})`; no file → 400.
    Temp file cleaned up in a `finally`.
  - `GET /drift`, `GET /drift/latest`, `GET /drift/history?limit=N`, `GET /drift/baseline`
    — same JSON as today; the small file-reading helpers (`_drift_read_json`,
    `_drift_history`, `_drift_latest_report`) move into `api.py` (they are web-adjacent,
    not scoring), reading `DRIFT_OUTPUT_DIR`.
- **Responses:** return the existing `detect()` dict directly (FastAPI serializes it);
  do **not** define a full response model for it now.
- **Free deliverable:** FastAPI auto-serves `/docs` (Swagger) and `/openapi.json`.
- **Run:** `uvicorn api:app --host 0.0.0.0 --port 7860`.

## 6. Repointing Consumers

- `drift_monitor_3.py`: `import server` → `import detector`; every `server.X`
  reference (`server.CHUNK`, `server.CASCADE_LOW/HIGH`, `server.lcnn_score`,
  `server.ensemble_score_variants`, `server.thresholds`, `server.ACTIVE_VERSION`)
  becomes `detector.X`. No logic change.
- `tests/test_golden.py`: `import server` → `import detector`; `server.detect` →
  `detector.detect`. Pinned scores must stay identical.

## 7. Parity Contract (the correctness gate)

Every response must stay identical to the Flask version:

- `/detect`: same keys and values — `verdict, score, pct, duration, chunks, codec,
  silence_ratio, lcnn, w2v, aasist, rawnet, ensemble, xgb, cascade{...}, policy,
  model_version, sha256, elapsed, timestamp, explanation, segments, flagged_segments,
  watermark, metadata, c2pa, mic_signature, filename, request_protection`.
- `/ping`: same fields.
- `/drift*`: same JSON.
- CORS headers present; rate-limit still returns 429 + `Retry-After`.
- The golden clips must produce the **same pinned verdicts and scores** as today.

## 8. Data Flow

```
POST /detect (multipart file)
  api.py: save temp → request_protection.check_request → detector.detect(tmp)
        → attach filename + protection → return dict (FastAPI → JSON)
GET  /ping     → detector globals → PingResponse
GET  /drift*   → read output/*.json(l) → JSON
startup        → FastAPI lifespan → detector.startup_check() (fail-fast)
```

## 9. Error Handling

- Missing file → `400 {"error": "No file provided"}`.
- Rate-limited → `429 {"error": "Rate limit exceeded", "retry_after_sec", "anomalies"}`
  + `Retry-After` header.
- `detect()` raises → `500 {"error": str(e)}`; temp file still cleaned up.
- Bad active bundle at startup → lifespan raises → server refuses to start (fail-closed),
  matching today's `sys.exit(1)` behavior.
- Missing drift files → the drift routes return `{"available": false}` / 404 as today.

## 10. Testing

1. **Golden test (repointed):** `python tests/test_golden.py` against `detector.detect`
   — the 4 pinned clips must match exactly (proves the core moved cleanly).
2. **API-parity test (new, `tests/test_api.py`):** use FastAPI/starlette `TestClient`
   (in-process, no real server) to hit `/ping`, `/detect` (a bundled golden clip),
   `/drift/baseline`, and assert: `/ping` fields present + `version=="V9"`; `/detect`
   returns the expected `verdict` for the clip and carries the full key set; `/drift`
   routes return JSON. This is the REST-parity gate.
3. **Drift smoke:** a `--quick` drift run against the repointed core completes and
   writes its outputs (confirms `detector` import works for the monitor).

## 11. Files, Requirements, Rollout

- **Create:** `detector.py`, `api.py`, `tests/test_api.py`.
- **Back up:** copy the current Flask `server.py` → `server_flask_backup.py` (retain for
  rollback); `server.py` is retired from the run path.
- **Modify:** `drift_monitor_3.py`, `tests/test_golden.py` (import repoint);
  `requirements.txt` (+ `fastapi`, `uvicorn[standard]`, `python-multipart`).
- **Run:** `uvicorn api:app --host 0.0.0.0 --port 7860` (replaces `python server.py`).
- **Docs to update later (noted, not in this slice):** the RUNBOOK / any "python
  server.py" references → `uvicorn api:app`.

## 12. Assumptions

- Single active model bundle, loaded per worker (unchanged from today).
- `python-multipart` is required by FastAPI for `UploadFile` form parsing.
- The demo HTML and golden fixtures already live in the repo and are unchanged.
- Behavior parity is defined by the golden test + the API-parity test; if either moves,
  the migration is not done.

# VoiceGuard API Reference

*Phase 8 documentation. The REST API is served by `api.py` (FastAPI) — `uvicorn api:app`. Interactive
docs are also available live at **`/docs`** (Swagger UI).*

---

## Overview

- **Base URL:** `http://<host>:7860` (behind nginx/TLS in production — see the deployment runbook).
- **Detection is asynchronous.** `POST /detect` **enqueues** a job and returns `202` immediately; a
  separate **worker process** runs the (heavy) inference. Clients **submit then poll**:
  `POST /detect` → `202 {job_id}` → poll `GET /jobs/{job_id}` until `status` is `done`/`error`.
- **Two processes must run:** the API (`uvicorn api:app`) **and** the worker (`python worker.py`),
  sharing the same env. Without the worker, jobs never leave `queued`.
- **Content:** requests are `multipart/form-data` (the audio file); responses are JSON.

## Authentication

All detection and drift endpoints require an **API key** as a Bearer token:
```
Authorization: Bearer vg_XXXXXXXXXXXXXXXX
```
- Keys are minted with `python auth.py create --client "<name>"` (prints the `vg_...` key **once**;
  only a SHA-256 hash is stored). Revoke with `python auth.py revoke <key_id>`.
- Missing/invalid key → **401**. Per-key rate limiting applies (configurable;
  `VOICEGUARD_REQUEST_PROTECTION=0` disables it). Over the limit → **429** with `Retry-After`.
- Open (no key): `GET /` and `GET /ping`.

---

## Endpoints

### `GET /`
Serves the browser demo UI. Open.

### `GET /ping`
Health + capability check. Open.
```json
{ "status": "ready", "version": "V9", "cascade": true,
  "active_version": "v9h", "active_sha": "…",
  "modules": { "watermark": true, "metadata": true, "c2pa": true, "mic_signature": true } }
```

### `POST /detect`  🔒
Submit an audio file for analysis. `multipart/form-data`, field name **`file`**.
- **202 Accepted:**
  ```json
  { "job_id": "j_ab12…", "status": "queued", "status_url": "/jobs/j_ab12…" }
  ```
- **Errors:** `400` no file · `413` file too large (`VOICEGUARD_MAX_UPLOAD_MB`, default 25) ·
  `429` rate limited · `401` unauthorized.
```bash
curl -X POST http://localhost:7860/detect \
  -H "Authorization: Bearer $KEY" -F "file=@sample.mp3"
```

### `GET /jobs/{job_id}`  🔒
Poll a job. **Owner-only** — a key that didn't submit the job gets **404** (no existence leak).
- `status: "queued" | "running" | "done" | "error"`.
- On `done`, includes the full detection **`result`** (schema below). On `error`, includes `error`.
```json
{ "job_id": "j_ab12…", "status": "done", "created_at": "…", "finished_at": "…",
  "status_url": "/jobs/j_ab12…", "result": { … } }
```

### `GET /drift`  🔒
Composite drift + retrain state (for dashboards): `{ available, output_dir, baseline, latest,
history, alert_state, retrain }`. The `retrain` field is the continual-learning trigger state
(`retrain_needed`, reasons, metrics snapshot). See also `GET /drift/latest`, `GET /drift/history?limit=N`,
`GET /drift/baseline`.

---

## Detection result schema (`result` on a `done` job)

| Field | Type | Meaning |
|---|---|---|
| `verdict` | str | `AUTO_REAL` / `REVIEW` / `LIKELY_FAKE` / `AUTO_FAKE` |
| `score` | float | calibrated p(synthetic), 0–1 |
| `confidence` | float | 0–1, decisiveness × cross-chunk agreement |
| `pct`, `duration`, `chunks`, `codec`, `silence_ratio` | — | summary fields |
| `lcnn`, `w2v`, `aasist`, `rawnet`, `ensemble` | float\|null | per-model mean % (null if not escalated) |
| `cascade` | obj | `{stage1_chunks, stage2_chunks, resolution_pct, band}` |
| `audit_id` | str | `aud_…` chain-of-custody id (also in the audit log) |
| `sha256` | str | hash of the exact bytes analyzed |
| `explanation` | obj | prosody: `{observations[{title,detail,lean}], measurements, summary}` |
| `segments`, `flagged_segments` | list | timestamped chunk verdicts (`start_sec/end_sec/score/verdict`) |
| `heatmap` | obj\|null | Grad-CAM: `{target, chunk_range_sec, values, freq_hz, time_sec, png_base64}` |
| `shap` | obj\|null | XGBoost fusion contributions `{aasist, wav2vec, rawnet, base, …}` |
| `adversarial` | obj\|null | attack risk `{flag, confidence, threshold, signals, latency_ms}` |
| `watermark`, `metadata`, `c2pa`, `mic_signature` | obj | Phase-4 forensic signals (each `{available, findings, lean, summary, …}`) |
| `model_version`, `timestamp`, `elapsed` | — | provenance |

A completed `result` can be rendered into a court-ready report with `forensic_report.py`
(`--result <job.json>` or `--file <audio>`).

## Status codes
`200` OK · `202` job accepted · `400` bad request (no file) · `401` unauthorized ·
`404` job not found / not owner · `413` payload too large · `429` rate limited · `500` server error.

## Notes
- Detection latency lives in the **worker** (seconds; +~8 s on real-verdict clips when the adversarial
  monitor is on) — the API stays fast because it only enqueues.
- The active model bundle is shown at `GET /ping` (`active_version`) and managed via
  `bundle_registry.py`; see `docs/RUNBOOK-model-flow.md`.

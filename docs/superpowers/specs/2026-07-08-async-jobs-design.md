# Async Job Pattern (C2) — Design Spec

- **Date:** 2026-07-08
- **Status:** Approved (design); pending implementation plan
- **Scope:** Sub-project C2 of Phase 7 hardening
- **Depends on:** C1 (API-key auth — provides the authenticated client `key_id` that owns each job)
- **Follow-ups (later specs):** C3 data governance (retention/purge, at-rest encryption, consent, audit-log wiring); C4 deployment + load/pen test

---

## 1. Context & Problem

`POST /detect` runs multi-second CPU inference synchronously, so it cannot both return a full
result and meet the plan's `<200ms p95` submission gate. C2 makes detection **asynchronous**:
submission returns immediately with a job id; a separate worker runs the inference; the client
polls for the result. This is the standard enterprise pattern for slow inference and is the
interaction model chosen during Phase 7 brainstorming (async job pattern, SQLite queue +
separate worker process, submit-and-poll — no webhooks this slice).

## 2. Goals / Non-Goals

**Goals:**
1. `POST /detect` → **`202 {job_id, status, status_url}`** in a few ms (save upload + enqueue only).
2. A durable **SQLite job queue** (`jobs.py`) both the API and worker use.
3. A separate **`worker.py`** process that claims jobs, runs `detector.detect`, and stores the
   result — with crash recovery (orphaned `running` jobs requeued on start).
4. **`GET /jobs/{job_id}`** — authenticated, **owner-only** (404 otherwise); returns the full,
   unchanged `detect()` result when `done`.
5. Per-key rate limiting still applies at submission (C1). Multiple workers never double-process
   a job (atomic claim).

**Non-Goals (later slices):**
- Webhooks / push delivery (submit-and-poll only).
- Retention/purge of old job rows + orphaned input files; at-rest encryption of `result_json`
  and uploads; consent; audit-log wiring — **C3**.
- Making the API model-free (the API keeps importing `detector`, so `/ping`/`active_version`/
  the startup smoke-check are unchanged; the lighter model-free API is a noted follow-up).
- Multi-node scaling / external broker.
- No new dependencies — `sqlite3` is stdlib.

## 3. The Job Store — `jobs.py`

Framework-agnostic (no FastAPI import). DB path from `VOICEGUARD_JOBS_DB` (default `jobs.db`),
opened in **WAL** mode for concurrent reader (API) + writer (worker). Path read per call so tests
use a temp DB.

**Schema (`jobs` table):**

| column | type | notes |
|---|---|---|
| `job_id` | TEXT PK | `j_` + `secrets.token_hex(8)` |
| `client` | TEXT | owner client name (from the API key record) |
| `key_id` | TEXT | owner key id (ownership check) |
| `status` | TEXT | `queued` \| `running` \| `done` \| `error` |
| `created_at` / `started_at` / `finished_at` | TEXT | ISO timestamps (nullable) |
| `input_path` | TEXT | path to the saved upload the worker reads |
| `result_json` | TEXT | the `detect()` result as JSON (when `done`) |
| `error` | TEXT | message (when `error`) |

**API:**
- `init_db()` — create the table + indexes if absent; enable WAL. Called on import (idempotent).
- `enqueue(client: str, key_id: str, input_path: str) -> str` — INSERT a `queued` job, return `job_id`.
- `claim_next() -> dict | None` — **atomic**: `BEGIN IMMEDIATE`; select the oldest `queued` job;
  `UPDATE … SET status='running', started_at=? WHERE job_id=? AND status='queued'`; keep it only
  if `rowcount == 1` (else another worker took it — return None/retry). Returns the claimed job dict.
- `complete(job_id: str, result: dict) -> None` — `status='done'`, `result_json=json.dumps(result)`, `finished_at`.
- `fail(job_id: str, error: str) -> None` — `status='error'`, `error`, `finished_at`.
- `get(job_id: str) -> dict | None` — read one job (row → dict, `result_json` parsed back to `result`).
- `requeue_stale() -> int` — reset any `running` jobs (orphaned by a worker crash) back to `queued`;
  returns the count. Called by the worker on startup.

## 4. `api.py` Changes

- **`POST /detect`** (unchanged: `require_api_key` + 400 missing-file + 413 oversize + per-key
  rate limit). Then, instead of scoring inline:
  - Save the upload to a **unique temp file under `jobs_input/`** (persists until the worker
    consumes it — not the old auto-deleted temp).
  - `job_id = jobs.enqueue(client["client"], client["key_id"], input_path)`.
  - Return **`202 {"job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}`**.
  - (No `detector.detect` call here anymore — that moves to the worker.)
- **`GET /jobs/{job_id}`** (new; `require_api_key`, owner-only):
  - `job = jobs.get(job_id)`; if `job is None` **or** `job["key_id"] != client["key_id"]` → **404**
    `{"error": "job not found"}` (same response for missing and not-owned — no existence leak).
  - Else return a status-shaped body:
    - `queued`: `{job_id, status, created_at, status_url}`
    - `running`: `{job_id, status, created_at, started_at}`
    - `done`: `{job_id, status, created_at, finished_at, result: <full detect() dict>}`
    - `error`: `{job_id, status, created_at, finished_at, error: <message>}`
- The API still `import detector` (unchanged `/ping`, `active_version`, lifespan `startup_check`).

## 5. `worker.py` — the Worker Process

- On start: `jobs.requeue_stale()` (crash recovery), log the active bundle version.
- `process_one() -> bool` — `job = jobs.claim_next()`; if None, return False. Else:
  `try: result = detector.detect(job["input_path"]); jobs.complete(job_id, result)`;
  `except Exception as e: jobs.fail(job_id, str(e))`; `finally: delete job["input_path"]`.
  Returns True (a job was processed).
- `run_forever(poll_interval=0.5)` — loop `process_one()`; when it returns False, `sleep(poll_interval)`.
- `if __name__ == "__main__": requeue_stale(); run_forever()`.
- `process_one()` is a standalone function so tests drive one job without spawning the process.
- The worker loads the models (via `import detector`) — it owns inference.
- **Run:** `python worker.py` (a second process alongside `uvicorn api:app`; systemd unit in C4).

## 6. Data Flow

```
POST /detect (Bearer key, file)
  auth → 400/413 hardening → per-key rate limit
  save upload → jobs_input/<tmp>.<ext>
  jobs.enqueue(client, key_id, path) → job_id
  → 202 {job_id, status:"queued", status_url}

worker.py loop:
  jobs.claim_next() (atomic) → running
  detector.detect(input_path) → jobs.complete(result)   [or jobs.fail(error)]
  delete input_path

GET /jobs/{id} (Bearer key)
  auth → jobs.get(id) → owner check (else 404) → status body (result when done)
```

## 7. Error Handling

- Missing/oversize upload → 400/413 at submission (before enqueue) — C1 behavior.
- Worker exception during scoring → job `status=error` with the message; input file still deleted.
- `GET /jobs/{id}` for a missing or other-owner job → **404** (identical body — no existence leak).
- Corrupt/missing DB → `init_db` recreates the table; `get`/`claim_next` on an empty DB return
  None; never crash the request path.
- Atomic claim guarantees a job is processed at most once even with multiple workers.

## 8. Testing

- **`tests/test_jobs.py` (temp `VOICEGUARD_JOBS_DB`):** `enqueue`→`get` (queued); `claim_next`→
  running + `started_at` set; `complete`→done + `result` round-trips; `fail`→error + message;
  `requeue_stale` resets a running job; **claim-once** (two `claim_next` calls on one queued job:
  first returns it, second returns None).
- **`tests/test_api.py` (extends the C1 auth fixtures):**
  - `POST /detect` with a valid key → **202**, body has `job_id` + `status:"queued"`.
  - `POST /detect` without a key → 401 (C1, unchanged).
  - `GET /jobs/{id}` without a key → 401; with a *different* client's key → **404**; owner → 200.
  - **End-to-end:** submit → `import worker; worker.process_one()` → poll `GET /jobs/{id}` →
    `status:"done"` with `result["verdict"] == "LIKELY_FAKE"` (score ~0.8167).
  - (The C1 tests that asserted `/detect` returned a verdict at 200 are updated to the 202 flow.)
- Golden test untouched (`detector.detect` is called directly; scoring unchanged).

## 9. Files

- **Create:** `jobs.py`, `worker.py`, `tests/test_jobs.py`.
- **Modify:** `api.py` (`/detect` → enqueue+202, add `GET /jobs/{id}`), `tests/test_api.py`
  (async `/detect` + `/jobs` tests), `.gitignore` (+`jobs.db`, `jobs.db-*`, `jobs_input/`),
  `docs/RUNBOOK-model-flow.md` (add "run the worker").
- **No dependency changes.**

## 10. Assumptions

- One SQLite DB shared by the API process and the worker process on the same host (WAL handles
  concurrent access). Multi-host would need an external broker (out of scope).
- `jobs_input/` and `jobs.db` are per-environment runtime state (git-ignored).
- Retention (purging old `done`/`error` rows and any orphaned inputs) is deferred to C3; for C2 the
  worker deletes each job's input file after processing, but completed rows accumulate until C3.

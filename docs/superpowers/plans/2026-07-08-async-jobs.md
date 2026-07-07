# Async Job Pattern (C2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make detection asynchronous — `POST /detect` returns `202 + job_id` in milliseconds by enqueuing a job to a SQLite queue; a separate `worker.py` process runs `detector.detect` and stores the result; `GET /jobs/{job_id}` (owner-only) returns it.

**Architecture:** New `jobs.py` (SQLite queue, WAL) is shared by `api.py` (enqueue + read) and `worker.py` (atomic claim + complete/fail). `api.py`'s `/detect` saves the upload under `jobs_input/`, enqueues, and returns 202; a new `GET /jobs/{id}` polls. The API still imports `detector` (unchanged `/ping`/smoke-check); only the worker does the inference. Golden test untouched (it calls `detector.detect` directly).

**Tech Stack:** Python 3.12, FastAPI, stdlib `sqlite3`, pytest. Reuses `detector`, `auth`, `request_protection`. No new dependencies.

## Global Constraints

- **Commits are the human's.** Implementers do NOT `git add`/`git commit` — leave changes in the working tree. Each task ends with verification.
- **No new dependencies** — `sqlite3` is stdlib.
- Job DB path from `VOICEGUARD_JOBS_DB` (default `jobs.db`), **read per call**; SQLite **WAL** mode; job inputs saved under `VOICEGUARD_JOBS_INPUT` (default `<repo>/jobs_input`).
- `POST /detect` (authenticated) → **`202 {job_id, status:"queued", status_url}`**; the 400/413/429 hardening from C1 runs **before** enqueue.
- `GET /jobs/{job_id}` (authenticated, **owner-only**): missing OR other-owner → **404** `{"error":"job not found"}` (no existence leak); `done` → full `detector.detect` result under `result`.
- **Atomic claim:** a queued job is processed at most once even with multiple workers.
- `job_id` = `"j_" + secrets.token_hex(8)`.
- The API keeps `import detector` (unchanged `/ping`, `active_version`, lifespan smoke-check).
- Golden clips' pinned scores must not change (scoring is untouched).
- Platform: Windows, Git Bash; run tests with `python -m pytest`. Importing `api`/`worker`/`detector` loads ~380 MB of models (~8s).

---

## File Structure

- Create `jobs.py` — SQLite job store: `init_db`, `enqueue`, `claim_next`, `complete`, `fail`, `get`, `requeue_stale`.
- Create `worker.py` — `process_one()` (claim → detect → complete/fail → delete input), `run_forever()`, `requeue_stale` on start.
- Create `tests/test_jobs.py`, `tests/test_worker.py`.
- Modify `api.py` — `/detect` → save-to-`jobs_input` + enqueue + 202; add `GET /jobs/{id}`.
- Modify `tests/test_api.py` — async `/detect` (202) + `/jobs` tests + end-to-end.
- Modify `.gitignore` — `jobs.db`, `jobs.db-*`, `jobs_input/`.
- Modify `docs/RUNBOOK-model-flow.md` — "run the worker".

---

## Task 1: `jobs.py` — SQLite job store + unit tests

**Files:**
- Create: `jobs.py`
- Create: `tests/test_jobs.py`

**Interfaces:**
- Produces:
  - `enqueue(client: str, key_id: str, input_path: str) -> str` (job_id).
  - `claim_next() -> dict | None` (atomically marks oldest queued job `running`, returns it).
  - `complete(job_id: str, result: dict) -> None`; `fail(job_id: str, error: str) -> None`.
  - `get(job_id: str) -> dict | None` (job dict; `result_json` parsed back into `result`).
  - `requeue_stale() -> int`; `init_db() -> None`.

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_jobs.py
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEGUARD_JOBS_DB", str(tmp_path / "jobs.db"))
    import jobs as jobs_mod
    return jobs_mod


def test_enqueue_and_get(jobs):
    jid = jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    j = jobs.get(jid)
    assert j["status"] == "queued" and j["client"] == "Acme" and j["key_id"] == "k_1"
    assert j["input_path"] == "/tmp/x.wav" and jid.startswith("j_")


def test_claim_marks_running(jobs):
    jid = jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    claimed = jobs.claim_next()
    assert claimed["job_id"] == jid and claimed["status"] == "running" and claimed["started_at"]


def test_claim_once(jobs):
    jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    assert jobs.claim_next() is not None
    assert jobs.claim_next() is None          # no more queued after the first claim


def test_complete_roundtrips_result(jobs):
    jid = jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    jobs.claim_next()
    jobs.complete(jid, {"verdict": "LIKELY_FAKE", "score": 0.82})
    j = jobs.get(jid)
    assert j["status"] == "done" and j["result"]["verdict"] == "LIKELY_FAKE" and j["finished_at"]


def test_fail_records_error(jobs):
    jid = jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    jobs.claim_next()
    jobs.fail(jid, "boom")
    j = jobs.get(jid)
    assert j["status"] == "error" and j["error"] == "boom"


def test_requeue_stale(jobs):
    jid = jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    jobs.claim_next()                          # now running
    assert jobs.requeue_stale() == 1
    assert jobs.get(jid)["status"] == "queued"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_jobs.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'jobs'`.

- [ ] **Step 3: Implement `jobs.py`**

```python
#!/usr/bin/env python3
"""jobs.py — SQLite-backed async job queue for VoiceGuard (Phase 7 / C2).

Shared by api.py (enqueue + read) and worker.py (claim + complete/fail).
DB path from $VOICEGUARD_JOBS_DB (default jobs.db), WAL mode, read per call.
"""
import os, sqlite3, json, secrets
from datetime import datetime, timezone


def _db_path():
    return os.environ.get("VOICEGUARD_JOBS_DB", "jobs.db")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    conn = sqlite3.connect(_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                client TEXT, key_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT, started_at TEXT, finished_at TEXT,
                input_path TEXT, result_json TEXT, error TEXT
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)")
        conn.commit()
    finally:
        conn.close()


def _row_to_job(row):
    if row is None:
        return None
    d = dict(row)
    if d.get("result_json"):
        try:
            d["result"] = json.loads(d["result_json"])
        except Exception:
            d["result"] = None
    d.pop("result_json", None)
    return d


def enqueue(client, key_id, input_path):
    init_db()
    job_id = "j_" + secrets.token_hex(8)
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO jobs(job_id, client, key_id, status, created_at, input_path) "
            "VALUES (?,?,?,?,?,?)",
            (job_id, client, key_id, "queued", _now(), input_path))
        conn.commit()
    finally:
        conn.close()
    return job_id


def claim_next():
    init_db()
    conn = _connect()
    try:
        conn.isolation_level = None                  # manual transaction control
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT job_id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        job_id = row["job_id"]
        cur = conn.execute(
            "UPDATE jobs SET status='running', started_at=? WHERE job_id=? AND status='queued'",
            (_now(), job_id))
        claimed = cur.rowcount == 1
        conn.execute("COMMIT")
        if not claimed:
            return None                              # another worker grabbed it
        return _row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
    finally:
        conn.close()


def complete(job_id, result):
    conn = _connect()
    try:
        conn.execute("UPDATE jobs SET status='done', result_json=?, finished_at=? WHERE job_id=?",
                     (json.dumps(result), _now(), job_id))
        conn.commit()
    finally:
        conn.close()


def fail(job_id, error):
    conn = _connect()
    try:
        conn.execute("UPDATE jobs SET status='error', error=?, finished_at=? WHERE job_id=?",
                     (str(error), _now(), job_id))
        conn.commit()
    finally:
        conn.close()


def get(job_id):
    init_db()
    conn = _connect()
    try:
        return _row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
    finally:
        conn.close()


def requeue_stale():
    init_db()
    conn = _connect()
    try:
        cur = conn.execute("UPDATE jobs SET status='queued', started_at=NULL WHERE status='running'")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `python -m pytest tests/test_jobs.py -q`
Expected: PASS (6 passed).

---

## Task 2: `worker.py` — the worker + a real-clip test

**Files:**
- Create: `worker.py`
- Create: `tests/test_worker.py`

**Interfaces:**
- Consumes: `jobs` (Task 1), `detector.detect`, `detector.ACTIVE_VERSION`.
- Produces: `process_one() -> bool` (processes one job); `run_forever(poll_interval=0.5)`.

- [ ] **Step 1: Write the failing worker test**

```python
# tests/test_worker.py
import os, sys, shutil
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_process_one_scores_a_real_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEGUARD_JOBS_DB", str(tmp_path / "jobs.db"))
    import jobs, worker
    # copy a golden fake clip to a temp input the worker will consume + delete
    inp = str(tmp_path / "in.mp3")
    shutil.copy2(os.path.join(REPO, "tests/golden_clips/fake_noizai_a4cd.mp3"), inp)
    jid = jobs.enqueue("Acme", "k_1", inp)

    assert worker.process_one() is True
    j = jobs.get(jid)
    assert j["status"] == "done"
    assert j["result"]["verdict"] == "LIKELY_FAKE"
    assert abs(j["result"]["score"] - 0.8167) < 1e-3
    assert not os.path.exists(inp)          # worker deleted the input
    assert worker.process_one() is False    # queue now empty
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_worker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'worker'`.

- [ ] **Step 3: Implement `worker.py`**

```python
#!/usr/bin/env python3
"""worker.py — VoiceGuard async detection worker (Phase 7 / C2).

Polls the SQLite job queue, runs detector.detect on each job's input, writes the
result back, and deletes the input file. Run alongside the API:
    python worker.py
"""
import os, time, logging
import jobs
import detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("worker")


def process_one():
    """Claim and process one job. Returns True if a job was processed, else False."""
    job = jobs.claim_next()
    if job is None:
        return False
    job_id, path = job["job_id"], job["input_path"]
    log.info(f"processing {job_id} ({path})")
    try:
        result = detector.detect(path)
        jobs.complete(job_id, result)
        log.info(f"done {job_id}: {result.get('verdict')}")
    except Exception as e:
        jobs.fail(job_id, str(e))
        log.error(f"failed {job_id}: {e}")
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return True


def run_forever(poll_interval=0.5):
    log.info(f"worker started (active bundle {detector.ACTIVE_VERSION}); polling every {poll_interval}s")
    while True:
        if not process_one():
            time.sleep(poll_interval)


if __name__ == "__main__":
    n = jobs.requeue_stale()
    if n:
        log.info(f"requeued {n} stale running job(s)")
    run_forever()
```

- [ ] **Step 4: Run the worker test to verify it passes**

Run: `python -m pytest tests/test_worker.py -q`
Expected: PASS (1 passed) — the job goes `done` with `LIKELY_FAKE` at ~0.8167 and the input file is deleted.

---

## Task 3: `api.py` — `/detect` → 202 + `GET /jobs/{id}` + API tests

**Files:**
- Modify: `api.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `jobs` (Task 1), `worker.process_one` (Task 2, used by the e2e test), `require_api_key` (C1).
- Produces: async `POST /detect` (202) and `GET /jobs/{job_id}` (owner-only).

- [ ] **Step 1: Update `tests/test_api.py` for the async flow (failing until api.py changes)**

Replace the `test_detect_valid_key` test (which asserted a 200 verdict) and add the jobs tests. First, in the `auth_key` fixture, also set the jobs DB + input env and mint a second client key. Change the `auth_key` fixture and add an `other_key` fixture:

```python
@pytest.fixture(scope="module")
def auth_key(tmp_path_factory):
    d = tmp_path_factory.mktemp("c2")
    os.environ["VOICEGUARD_AUTH_KEYS"] = str(d / "keys.json")
    os.environ["VOICEGUARD_JOBS_DB"] = str(d / "jobs.db")
    os.environ["VOICEGUARD_JOBS_INPUT"] = str(d / "jobs_input")
    import auth
    _, key = auth.create_key("test-client")
    return key


@pytest.fixture(scope="module")
def other_key(auth_key):
    import auth
    _, key = auth.create_key("other-client")
    return key
```

Replace `test_detect_valid_key` with the async version and add the jobs tests:

```python
def test_detect_valid_key_returns_202(client, auth_key):
    with open(FAKE, "rb") as f:
        r = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")},
                        headers=_hdr(auth_key))
    assert r.status_code == 202
    j = r.json()
    assert j["status"] == "queued" and j["job_id"].startswith("j_")
    assert j["status_url"] == f"/jobs/{j['job_id']}"


def test_jobs_requires_key(client):
    assert client.get("/jobs/j_deadbeef").status_code == 401


def test_jobs_wrong_owner_404(client, auth_key, other_key):
    with open(FAKE, "rb") as f:
        job_id = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")},
                             headers=_hdr(auth_key)).json()["job_id"]
    assert client.get(f"/jobs/{job_id}", headers=_hdr(other_key)).status_code == 404   # not owner
    assert client.get(f"/jobs/{job_id}", headers=_hdr(auth_key)).status_code == 200    # owner


def test_detect_async_e2e(client, auth_key):
    with open(FAKE, "rb") as f:
        job_id = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")},
                             headers=_hdr(auth_key)).json()["job_id"]
    # queued before the worker runs
    assert client.get(f"/jobs/{job_id}", headers=_hdr(auth_key)).json()["status"] == "queued"
    # drive the worker once (same VOICEGUARD_JOBS_DB via the fixture env)
    import worker
    assert worker.process_one() is True
    # now done with the full result
    r = client.get(f"/jobs/{job_id}", headers=_hdr(auth_key))
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "done"
    assert j["result"]["verdict"] == "LIKELY_FAKE" and abs(j["result"]["score"] - 0.8167) < 1e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_api.py -q`
Expected: FAIL — `test_detect_valid_key_returns_202` gets 200 (still synchronous); `/jobs/*` routes 404 (not defined yet).

- [ ] **Step 3: Wire jobs into `api.py`**

Add the import (near `import auth`):
```python
import jobs
```
Add the jobs-input dir constant (near `DRIFT_OUTPUT_DIR`):
```python
JOBS_INPUT_DIR = os.environ.get("VOICEGUARD_JOBS_INPUT", os.path.join(detector.BASE, "jobs_input"))
```

Replace the entire `/detect` route (from `@app.post("/detect")` through its `finally:` cleanup) with the async version:

```python
@app.post("/detect", status_code=202)
async def detect_route(request: Request,
                       client: dict = Depends(require_api_key),
                       file: UploadFile | None = File(None)):
    if file is None:
        return JSONResponse(status_code=400, content={"error": "No file provided"})
    max_mb = int(os.environ.get("VOICEGUARD_MAX_UPLOAD_MB", 25))
    os.makedirs(JOBS_INPUT_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, dir=JOBS_INPUT_DIR, delete=False)
    try:
        tmp.write(await file.read())
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
```

Add the `GET /jobs/{job_id}` route immediately after the `/detect` route:

```python
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
```

- [ ] **Step 4: Run the API tests to verify they pass**

Run: `python -m pytest tests/test_api.py -q`
Expected: PASS (all) — `/detect` → 202 + job_id; `/jobs/{id}` 401 without key, 404 for a non-owner, `queued`→`done` (LIKELY_FAKE 0.8167) after `worker.process_one()`.

---

## Task 4: `.gitignore` + runbook + final verification

**Files:**
- Modify: `.gitignore`
- Modify: `docs/RUNBOOK-model-flow.md`

- [ ] **Step 1: Git-ignore the runtime job state**

```bash
python - <<'PY'
p=".gitignore"; s=open(p,encoding="utf-8").read()
add="\n# Async job queue runtime state (per-environment)\njobs.db\njobs.db-wal\njobs.db-shm\njobs_input/\n"
if "jobs.db" not in s:
    open(p,"a",encoding="utf-8").write(add)
print("gitignore updated" if "jobs_input/" in open(p,encoding='utf-8').read() else "NO CHANGE")
PY
```
Expected: `gitignore updated`.

- [ ] **Step 2: Document running the worker**

Append a section to `docs/RUNBOOK-model-flow.md`:

```bash
python - <<'PY'
p="docs/RUNBOOK-model-flow.md"; s=open(p,encoding="utf-8").read()
section = '''
## Async detection: run the worker
The API enqueues jobs; a separate worker process runs the inference.
- **API:**    `uvicorn api:app --host 0.0.0.0 --port 7860`
- **Worker:** `python worker.py`   (same host; shares $VOICEGUARD_JOBS_DB, default jobs.db)
- **Client flow:** `POST /detect` (Bearer key) -> 202 {job_id, status_url};
  poll `GET /jobs/{job_id}` (same key) until status=done, read `result`.
- On DigitalOcean, run both under systemd (one unit each). The worker owns the models;
  it requeues any orphaned `running` jobs on startup.
'''
if "run the worker" not in s:
    open(p,"a",encoding="utf-8").write(section)
print("runbook updated" if "run the worker" in open(p,encoding='utf-8').read() else "NO CHANGE")
PY
```
Expected: `runbook updated`.

- [ ] **Step 3: Final verification — full suite + golden parity**

Run:
```bash
python -m pytest tests/test_jobs.py tests/test_worker.py tests/test_api.py tests/test_auth.py tests/test_detector.py tests/test_golden.py -q
```
Expected: all pass; golden pinned scores unchanged (jobs/worker don't touch scoring).

- [ ] **Step 4: Live 2-process smoke (API + worker)**

Run (mint a key, start both processes, submit, poll):
```bash
export VOICEGUARD_JOBS_DB="live_jobs.db"
export VOICEGUARD_AUTH_KEYS="live_keys.json"
KEY=$(python auth.py create --client smoke | grep -o 'vg_[A-Za-z0-9_-]*')
uvicorn api:app --host 0.0.0.0 --port 7860 &   # API
python worker.py &                              # worker
sleep 25
JOB=$(curl -s -X POST -H "Authorization: Bearer $KEY" -F "file=@tests/golden_clips/fake_noizai_a4cd.mp3" http://localhost:7860/detect | python -c "import json,sys;print(json.load(sys.stdin)['job_id'])")
echo "submitted job: $JOB"
sleep 8
curl -s -H "Authorization: Bearer $KEY" http://localhost:7860/jobs/$JOB | python -c "import json,sys;d=json.load(sys.stdin);print('status:',d['status'],'verdict:',d.get('result',{}).get('verdict'))"
# stop both processes, then:
rm -f live_jobs.db live_jobs.db-* live_keys.json
rm -rf jobs_input
```
Expected: `submitted job: j_...` then `status: done verdict: LIKELY_FAKE`. Stop the API + worker.

---

## Self-Review

**Spec coverage:**
- §3 `jobs.py` (schema, WAL, enqueue/claim_next-atomic/complete/fail/get/requeue_stale, per-call path) → Task 1. ✓
- §4 `api.py` (`/detect` → save-to-jobs_input + enqueue + 202; `GET /jobs/{id}` owner-only 404; API still imports detector) → Task 3. ✓
- §5 `worker.py` (process_one, run_forever, requeue_stale on start, deletes input, owns models) → Task 2. ✓
- §6 data flow (submit→202; worker claim→detect→complete/fail; poll) → Tasks 2, 3. ✓
- §7 error handling (400/413 before enqueue; worker exception→error+input deleted; 404 no-leak; atomic claim-once) → Tasks 1-3. ✓
- §8 testing (jobs unit incl claim-once; worker real-clip; api 202/401/404/e2e; golden untouched) → Tasks 1-3. ✓
- §9 files (jobs.py, worker.py, test_jobs, test_worker, api.py, test_api, .gitignore, runbook; no deps) → all tasks. ✓

**Placeholder scan:** no TBD/TODO; every code step complete. ✓

**Type consistency:** `enqueue(client, key_id, input_path)->job_id`, `claim_next()->job dict`, `get()->job dict with result`, `process_one()->bool`, the `/detect` 202 body `{job_id,status,status_url}`, and the owner check `job["key_id"] != client["key_id"]` are consistent across Tasks 1-3 and the tests. `client["client"]`/`client["key_id"]` come from C1's `require_api_key` record. ✓

**Note:** Task 3 changes the `/detect` contract from a synchronous 200-with-result (C1) to 202-with-job_id; the C1 test that asserted the verdict at 200 is replaced by `test_detect_valid_key_returns_202` + the e2e. The golden test uses `detector.detect` directly and is unaffected.

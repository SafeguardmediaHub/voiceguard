# Load Test + Runtime Hardening (C4a) — Design Spec

- **Date:** 2026-07-08
- **Status:** Approved (design); pending implementation plan
- **Scope:** Sub-project C4a of Phase 7 hardening (first slice of C4)
- **Depends on:** C1 (auth), C2 (async jobs — the thing under test)
- **Follow-ups (later C4 slices):** C4b deployment config (nginx TLS 1.3, systemd); C4c penetration test (incl. rate-limit validation)

---

## 1. Context & Problem

Phase 7 Task 5 requires a load test proving **100 req/min at <200ms p95**. After C2, `POST /detect`
no longer runs inference — it saves the upload and enqueues a job (milliseconds), so the `<200ms`
gate applies to **submission latency** and is now achievable (inference happens in the worker,
off the request path). This slice: (1) a stdlib load generator that proves submission p95 < 200ms
under concurrency, and (2) the deferred `requeue_stale` age filter so running a second worker /
restarting a worker under load can't double-process an in-flight job.

## 2. Goals / Non-Goals

**Goals:**
1. A **no-dependency** (stdlib) concurrent load generator that fires `POST /detect` at a running
   API and **exits 0 only if submission p95 < 200ms** (and achieved rate ≥ 100 req/min).
2. Report end-to-end (submit→done) latency as an **informational** number (worker-bound) — not the gate.
3. `jobs.requeue_stale(older_than_seconds=600)` — requeue only *stale* running jobs (crash orphans),
   not a job a live worker just claimed — safe for multi-worker / rolling restart.
4. An env toggle to disable per-key rate limiting for the load run (so 100/min from one key isn't
   429'd by the 30/min limiter).

**Non-Goals (later slices / out of scope):**
- nginx TLS + systemd deployment config (C4b); penetration/security-suite testing incl. rate-limit
  behavior as a *feature* (C4c).
- GPU / multi-worker throughput scaling to sustain 100/min *end-to-end* (documented as a known limit).
- No new dependencies — the generator uses `urllib` + manual multipart + `concurrent.futures` (all stdlib).

## 3. Runtime Hardening

**`jobs.py` — age-filtered requeue.** Replace `requeue_stale()` with
`requeue_stale(older_than_seconds: int = 600) -> int`:
- Compute `cutoff = (now_utc - older_than_seconds).isoformat()`.
- `UPDATE jobs SET status='queued', started_at=NULL WHERE status='running' AND started_at IS NOT NULL
  AND started_at < ?` (ISO-8601 UTC strings compare lexicographically = chronologically).
- `worker.py`'s startup call stays `jobs.requeue_stale()` (default 600s) — so a starting worker only
  reclaims jobs orphaned by a crash >10min ago, never a peer worker's fresh in-flight job.
- Existing `test_requeue_stale` is updated to the new semantics (see §6).

**`detector.py` — env-driven rate-limit flag.** Change
`REQUEST_PROTECTION_ENABLED = True` to
`REQUEST_PROTECTION_ENABLED = os.environ.get("VOICEGUARD_REQUEST_PROTECTION", "1") != "0"`
(default **on** — unchanged production behavior). The load run sets `VOICEGUARD_REQUEST_PROTECTION=0`
to measure raw submission throughput. (Rate limiting is *exercised as a feature* in C4c.)

## 4. The Load Generator — `scripts/loadtest.py`

Pure stdlib. Structured so the statistics/gating logic is unit-testable without a server:

- `percentile(sorted_values: list[float], p: float) -> float` — pure (nearest-rank).
- `evaluate(latencies_ms: list[float], elapsed_s: float, p95_ms: float, min_rps: float) -> dict` —
  pure; returns `{n, p50, p95, p99, rps, passed}` where `passed = (p95 < p95_ms) and (rps >= min_rps)`.
- `submit_once(url, key, clip_bytes, filename) -> float` — one `POST /detect` via `urllib` with a
  hand-built multipart body + `Authorization: Bearer <key>`; returns the submission latency in ms
  (raises on non-202).
- `run_load(url, key, clip, n, concurrency) -> (latencies_ms, elapsed_s)` — fire `n` submissions
  across a `ThreadPoolExecutor(concurrency)`; collect latencies.
- `main()` — CLI: `--url`, `--key`, `--clip`, `--requests` (default 200), `--concurrency` (default 20),
  `--p95-ms` (default 200), `--min-rps` (default 1.67 = 100/min); runs, prints a p50/p95/p99/RPS
  table, and `sys.exit(0 if passed else 1)`.

Design notes:
- Firing 200 submissions at concurrency 20 both proves p95 under concurrency and over-delivers the
  100/min rate (achieved RPS will be far higher — headroom), without flooding tens of thousands of
  files (each submission creates one job + one input file).
- Only submission latency is gated. The wrapper (below) additionally polls one job to `done` and
  prints the end-to-end time as informational.

## 5. Running It (the C4a deliverable)

The generator assumes a running API + a key. The full local run (documented in the runbook, and
executed in the final verification step):
1. `export VOICEGUARD_JOBS_DB=loadtest_jobs.db VOICEGUARD_AUTH_KEYS=loadtest_keys.json
   VOICEGUARD_JOBS_INPUT=loadtest_input VOICEGUARD_REQUEST_PROTECTION=0`
2. `KEY=$(python auth.py create --client loadtest | grep -o 'vg_[...]')`
3. Start `uvicorn api:app --port 7860` and `python worker.py`.
4. `python scripts/loadtest.py --url http://localhost:7860 --key "$KEY"
   --clip tests/golden_clips/fake_noizai_a4cd.mp3` → prints the table, exits 0 on pass.
5. Clean up the throwaway DB/keys/input dir.

## 6. Testing

- **`tests/test_loadtest.py` (pure, no server):** `percentile` correctness (e.g. p95 of 1..100);
  `evaluate` PASS case (all latencies < 200ms, rps high → `passed True`) and FAIL case
  (p95 ≥ 200ms → `passed False`; low rps → `passed False`).
- **`tests/test_jobs.py` (updated):**
  - Existing `test_requeue_stale` → call `requeue_stale(older_than_seconds=0)` to requeue a
    just-claimed job (proves the requeue path).
  - New `test_requeue_stale_leaves_fresh`: a freshly-claimed `running` job is **not** requeued by
    the default `requeue_stale()` (returns 0, stays `running`).
  - New `test_requeue_stale_reclaims_old`: a `running` job whose `started_at` is set (directly) to a
    timestamp > threshold in the past **is** requeued (returns 1, back to `queued`).
- **Live load run** (final step): submission p95 < 200ms at ≥100 req/min against a real api+worker —
  the actual C4a gate.

## 7. Error Handling

- `submit_once` on a non-202 (e.g., 401 bad key, 429) raises; the generator counts errors and, if any
  submission failed, prints them and fails the run (a 200/500 under load is a real failure).
- The generator never leaves the API/worker running — those are started/stopped by the wrapper/runbook.
- `requeue_stale` with a malformed/missing `started_at` skips that row (the `IS NOT NULL` guard).

## 8. Files

- **Create:** `scripts/loadtest.py`, `tests/test_loadtest.py`.
- **Modify:** `jobs.py` (`requeue_stale` age filter), `tests/test_jobs.py` (updated + 2 new tests),
  `detector.py` (env-driven `REQUEST_PROTECTION_ENABLED`), `.gitignore`
  (`loadtest_jobs.db*`, `loadtest_keys.json`, `loadtest_input/`), `docs/RUNBOOK-model-flow.md`
  (load-test section).
- **No dependency changes.**

## 9. Assumptions

- The load test runs on the same host as the API + worker (localhost); it measures the app's
  submission latency, not network transit (nginx/TLS overhead is a C4b concern).
- 100 req/min is the plan's stated rate; the generator proves p95 under a much higher achieved rate
  (headroom), which subsumes the 100/min requirement for submission latency.
- Sustained 100/min *end-to-end* on one CPU worker is not a goal — the async design decouples
  submission from completion; end-to-end throughput scales with worker count / GPU (C4b/ops).

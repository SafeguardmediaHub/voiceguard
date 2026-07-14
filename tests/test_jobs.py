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


def test_requeue_stale_zero_threshold_requeues(jobs):
    import time
    jid = jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    jobs.claim_next()                                  # running, started_at = now
    time.sleep(0.01)                                   # age past the 0s cutoff (avoid the now==now race)
    assert jobs.requeue_stale(older_than_seconds=0) == 1   # threshold 0 → requeue all running
    assert jobs.get(jid)["status"] == "queued"


def test_requeue_stale_leaves_fresh(jobs):
    jid = jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    jobs.claim_next()                                  # just-claimed, started_at = now
    assert jobs.requeue_stale() == 0                   # default 600s → fresh job untouched
    assert jobs.get(jid)["status"] == "running"


def test_requeue_stale_reclaims_old(jobs):
    import sqlite3
    jid = jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    jobs.claim_next()
    # backdate started_at to simulate a worker that crashed long ago
    conn = sqlite3.connect(os.environ["VOICEGUARD_JOBS_DB"])
    conn.execute("UPDATE jobs SET started_at='2000-01-01T00:00:00+00:00' WHERE job_id=?", (jid,))
    conn.commit()
    conn.close()
    assert jobs.requeue_stale() == 1                   # older than 600s → reclaimed
    assert jobs.get(jid)["status"] == "queued"


def test_self_heals_if_db_deleted_midprocess(jobs):
    # init_db() is memoized per path; if the DB file vanishes after that first call,
    # a naive memoized init would skip re-creating the table and every op would fail
    # 'no such table' for the rest of the process. _with_schema must self-heal.
    jobs.enqueue("Acme", "k_1", "/tmp/x.wav")          # creates + memoizes the schema
    db = os.environ["VOICEGUARD_JOBS_DB"]
    for suffix in ("", "-wal", "-shm"):                # drop the DB out from under it
        try:
            os.remove(db + suffix)
        except FileNotFoundError:
            pass
    jid = jobs.enqueue("Acme", "k_1", "/tmp/y.wav")    # would raise without self-heal
    assert jobs.get(jid)["status"] == "queued"


def test_claim_next_atomic_under_concurrency(jobs):
    import threading
    jobs.enqueue("Acme", "k_1", "/tmp/x.wav")
    winners, lock = [], threading.Lock()

    def claim():
        c = jobs.claim_next()          # each thread opens its own connection
        if c is not None:
            with lock:
                winners.append(c["job_id"])

    threads = [threading.Thread(target=claim) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1           # exactly one claimer wins the single queued job

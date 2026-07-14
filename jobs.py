#!/usr/bin/env python3
"""jobs.py — SQLite-backed async job queue for VoiceGuard (Phase 7 / C2).

Shared by api.py (enqueue + read) and worker.py (claim + complete/fail).
DB path from $VOICEGUARD_JOBS_DB (default jobs.db), WAL mode, read per call.
"""
import os, sqlite3, json, secrets, functools
from datetime import datetime, timezone, timedelta


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


_initialized = set()   # DB paths whose schema has been created this process


def init_db():
    # Memoized per DB path: creating the schema is idempotent but not free
    # (a connection + CREATE + commit); doing it on every enqueue/get was a
    # per-request cost under load. First call per path does the work; rest no-op.
    path = _db_path()
    if path in _initialized:
        return
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
    _initialized.add(path)


def _with_schema(fn):
    """Ensure the schema exists before the op; if the DB file was deleted/recreated
    mid-process (memoized init_db would otherwise skip re-creating the table), catch
    the 'no such table' error, re-init once, and retry — so callers self-heal instead
    of failing permanently for the rest of the process."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        init_db()
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e).lower():
                raise
            _initialized.discard(_db_path())     # force a real re-create
            init_db()
            return fn(*args, **kwargs)
    return wrapper


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


@_with_schema
def enqueue(client, key_id, input_path):
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


@_with_schema
def claim_next():
    conn = _connect()
    try:
        conn.isolation_level = None                  # manual transaction control
        conn.execute("BEGIN IMMEDIATE")
        try:
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
        except Exception:
            conn.execute("ROLLBACK")                 # never leave the write txn open on error
            raise
        if not claimed:
            return None                              # another worker grabbed it
        return _row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
    finally:
        conn.close()


@_with_schema
def complete(job_id, result):
    conn = _connect()
    try:
        conn.execute("UPDATE jobs SET status='done', result_json=?, finished_at=? WHERE job_id=?",
                     (json.dumps(result), _now(), job_id))
        conn.commit()
    finally:
        conn.close()


@_with_schema
def fail(job_id, error):
    conn = _connect()
    try:
        conn.execute("UPDATE jobs SET status='error', error=?, finished_at=? WHERE job_id=?",
                     (str(error), _now(), job_id))
        conn.commit()
    finally:
        conn.close()


@_with_schema
def get(job_id):
    conn = _connect()
    try:
        return _row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
    finally:
        conn.close()


@_with_schema
def requeue_stale(older_than_seconds=600):
    """Requeue running jobs orphaned by a crash — only those whose started_at is
    older than the threshold, so a peer worker's freshly-claimed job is left alone."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)).isoformat()
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE jobs SET status='queued', started_at=NULL "
            "WHERE status='running' AND started_at IS NOT NULL AND started_at < ?",
            (cutoff,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()

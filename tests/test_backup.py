# tests/test_backup.py
"""Backup unit tests. No weights, no network — runs in the fast CI tier."""
import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy import backup


def _make_db(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, status TEXT)")
    con.execute("INSERT INTO jobs VALUES ('j1', 'done')")
    con.execute("INSERT INTO jobs VALUES ('j2', 'queued')")
    con.commit()
    return con                      # left OPEN: a live writer is the realistic case


def test_snapshot_is_consistent_while_the_db_is_open(tmp_path):
    src = tmp_path / "jobs.db"
    con = _make_db(str(src))
    dest = backup.snapshot_sqlite(str(src), str(tmp_path / "snap.db"))
    con.close()

    rows = sqlite3.connect(dest).execute("SELECT job_id, status FROM jobs ORDER BY job_id").fetchall()
    assert rows == [("j1", "done"), ("j2", "queued")]


def test_encrypt_round_trip(tmp_path):
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    src = tmp_path / "secret.json"
    src.write_text('{"key_id": "k_abc"}', encoding="utf-8")

    enc = backup.encrypt_file(str(src), str(tmp_path / "secret.json.enc"), key)
    blob = open(enc, "rb").read()
    assert b"k_abc" not in blob                       # ciphertext, not plaintext

    assert Fernet(key.encode()).decrypt(blob).decode() == '{"key_id": "k_abc"}'


def test_run_backup_uploads_both_artifacts_under_a_dated_prefix(tmp_path, fake_s3):
    data = tmp_path / "data"
    data.mkdir()
    _make_db(str(data / "jobs.db")).close()
    (data / "auth_keys.json").write_text(json.dumps([{"key_id": "k_1"}]), encoding="utf-8")

    keys = backup.run_backup(fake_s3, "bkt", "voiceguard/model_store",
                             str(data), str(tmp_path / "out"), now="2026-07-21")

    assert sorted(keys) == [
        "voiceguard/model_store/backups/2026-07-21/auth_keys.json",
        "voiceguard/model_store/backups/2026-07-21/jobs.db",
    ]
    assert all(("bkt", k) in fake_s3.store for k in keys)


def test_run_backup_encrypts_when_a_key_is_supplied(tmp_path, fake_s3):
    from cryptography.fernet import Fernet

    fkey = Fernet.generate_key().decode()
    data = tmp_path / "data"
    data.mkdir()
    _make_db(str(data / "jobs.db")).close()
    (data / "auth_keys.json").write_text('[{"key_id": "k_1"}]', encoding="utf-8")

    keys = backup.run_backup(fake_s3, "bkt", "p", str(data), str(tmp_path / "out"),
                             fernet_key=fkey, now="2026-07-21")

    assert sorted(keys) == ["p/backups/2026-07-21/auth_keys.json.enc",
                            "p/backups/2026-07-21/jobs.db.enc"]
    stored = fake_s3.store[("bkt", "p/backups/2026-07-21/auth_keys.json.enc")]
    assert b"k_1" not in stored
    assert Fernet(fkey.encode()).decrypt(stored) == b'[{"key_id": "k_1"}]'


def test_run_backup_includes_the_tamper_evident_audit_log(tmp_path, fake_s3):
    """The audit log is the one artifact whose whole purpose is durable evidence;
    a droplet loss must not be able to destroy it."""
    data = tmp_path / "data"
    (data / "governance").mkdir(parents=True)
    _make_db(str(data / "jobs.db")).close()
    # write_bytes, not write_text: the latter translates \n -> \r\n on Windows and
    # the backup must be a byte-for-byte copy (the hash chain is over exact bytes).
    (data / "governance" / "audit_log.jsonl").write_bytes(
        b'{"seq":1,"verdict":"AUTO_FAKE"}\n')

    keys = backup.run_backup(fake_s3, "bkt", "p", str(data), str(tmp_path / "out"),
                             now="2026-07-21")

    assert "p/backups/2026-07-21/audit_log.jsonl" in keys
    assert fake_s3.store[("bkt", "p/backups/2026-07-21/audit_log.jsonl")] == \
        b'{"seq":1,"verdict":"AUTO_FAKE"}\n'


def test_missing_audit_log_is_not_fatal(tmp_path, fake_s3):
    """A droplet that has served no detections yet has no audit log."""
    data = tmp_path / "data"
    data.mkdir()
    _make_db(str(data / "jobs.db")).close()

    keys = backup.run_backup(fake_s3, "bkt", "p", str(data), str(tmp_path / "out"), now="2026-07-21")
    assert keys == ["p/backups/2026-07-21/jobs.db"]


def test_missing_auth_keys_is_not_fatal(tmp_path, fake_s3):
    """A droplet with no keys issued yet must still back up the queue."""
    data = tmp_path / "data"
    data.mkdir()
    _make_db(str(data / "jobs.db")).close()

    keys = backup.run_backup(fake_s3, "bkt", "p", str(data), str(tmp_path / "out"), now="2026-07-21")
    assert keys == ["p/backups/2026-07-21/jobs.db"]

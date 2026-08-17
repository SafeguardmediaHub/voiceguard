# tests/test_governance.py
"""Tamper-evident audit log (governance.AuditLog) — the mechanism H6 wires into
the live /detect path. Weight-free: governance.py imports no torch at module top,
so this runs in the CI-fast tier."""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import governance


def test_append_and_verify_chain(tmp_path):
    log = governance.AuditLog(path=tmp_path / "audit_log.jsonl")
    for i in range(3):
        log.append(audit_id=f"aud_{i}", intake_sha256="a" * 64,
                   model_versions={"bundle": "v9h"}, score=0.5 + i * 0.1,
                   verdict="AUTO_REAL")
    ok, details = log.verify_chain()
    assert ok, details
    assert details["n_entries"] == 3
    # Each entry links to the previous one's hash (genesis for the first).
    entries = log.read_all()
    assert entries[0]["prev_hash"] == governance.AuditLog.GENESIS_HASH
    assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
    assert entries[2]["prev_hash"] == entries[1]["entry_hash"]


def test_detects_in_place_edit(tmp_path):
    p = tmp_path / "audit_log.jsonl"
    log = governance.AuditLog(path=p)
    log.append(audit_id="aud_0", intake_sha256="a" * 64,
               model_versions={"bundle": "v9h"}, score=0.1, verdict="AUTO_REAL")
    log.append(audit_id="aud_1", intake_sha256="b" * 64,
               model_versions={"bundle": "v9h"}, score=0.9, verdict="AUTO_FAKE")

    # Tamper: flip the first entry's verdict without recomputing its hash.
    lines = p.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["verdict"] = "AUTO_FAKE"
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, details = governance.AuditLog(path=p).verify_chain()
    assert not ok
    assert details["broken_at_seq"] == 1


def test_detects_deleted_entry(tmp_path):
    p = tmp_path / "audit_log.jsonl"
    log = governance.AuditLog(path=p)
    for i in range(3):
        log.append(audit_id=f"aud_{i}", intake_sha256="a" * 64,
                   model_versions={"bundle": "v9h"}, score=0.1, verdict="AUTO_REAL")
    lines = p.read_text(encoding="utf-8").splitlines()
    del lines[1]                                   # remove the middle entry
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, _ = governance.AuditLog(path=p).verify_chain()
    assert not ok


def test_governance_dir_is_env_configurable(tmp_path, monkeypatch):
    # The live deployment must be able to point the audit log at a real writable
    # dir — the module default must not be the hardcoded Kaggle path.
    monkeypatch.setenv("VOICEGUARD_GOVERNANCE_DIR", str(tmp_path / "gov"))
    reloaded = importlib.reload(governance)
    try:
        assert str(tmp_path / "gov") in str(reloaded.GOVERNANCE_DIR)
        assert str(reloaded.AUDIT_LOG_PATH).startswith(str(tmp_path / "gov"))
        assert "kaggle" not in str(reloaded.GOVERNANCE_DIR).lower()
    finally:
        monkeypatch.delenv("VOICEGUARD_GOVERNANCE_DIR", raising=False)
        importlib.reload(governance)               # restore module default for other tests


def test_default_governance_dir_is_not_kaggle(monkeypatch):
    monkeypatch.delenv("VOICEGUARD_GOVERNANCE_DIR", raising=False)
    reloaded = importlib.reload(governance)
    assert "kaggle" not in str(reloaded.GOVERNANCE_DIR).lower()


_CONCURRENT_APPENDER = """
import sys
sys.path.insert(0, %r)
import governance
log = governance.AuditLog(path=%r)
for i in range(%d):
    log.append(audit_id="aud_%%s_%%d" %% (%r, i), intake_sha256="a" * 64,
               model_versions={"bundle": "v9h"}, score=0.5, verdict="AUTO_REAL")
"""


def test_concurrent_appends_keep_the_chain_intact(tmp_path):
    """docker-compose.prod.yml documents `--scale worker=2` as safe. The job queue
    is safe at 2 (BEGIN IMMEDIATE), but the audit chain's read-last-hash-then-append
    is not unless it is locked — and a broken chain is unrecoverable evidence loss."""
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = tmp_path / "audit_log.jsonl"
    n_procs, n_each = 4, 15

    procs = [subprocess.Popen(
        [sys.executable, "-c",
         _CONCURRENT_APPENDER % (repo, str(log_path), n_each, f"p{i}")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for i in range(n_procs)]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err

    log = governance.AuditLog(path=log_path)
    ok, details = log.verify_chain()
    assert ok, details
    assert details["n_entries"] == n_procs * n_each
    # Every entry got a distinct sequence number — no two writers claimed the same slot.
    seqs = sorted(e["seq"] for e in log.read_all())
    assert seqs == list(range(1, n_procs * n_each + 1))


def _seed_chain(gov_dir, tamper=False):
    log = governance.AuditLog(path=gov_dir / "audit_log.jsonl")
    for i in range(2):
        log.append(audit_id=f"aud_{i}", intake_sha256="a" * 64,
                   model_versions={"bundle": "v9h"}, score=0.5, verdict="AUTO_REAL")
    if tamper:
        p = gov_dir / "audit_log.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines()
        d = json.loads(lines[0]); d["verdict"] = "AUTO_FAKE"
        lines[0] = json.dumps(d, sort_keys=True, separators=(",", ":"))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_verify_chain_cli(gov_dir):
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "VOICEGUARD_GOVERNANCE_DIR": str(gov_dir)}
    return subprocess.run([sys.executable, os.path.join(repo, "governance.py"), "verify-chain"],
                          capture_output=True, text=True, env=env)


def test_verify_chain_cli_exits_zero_when_clean(tmp_path):
    gov_dir = tmp_path / "gov"; gov_dir.mkdir()
    _seed_chain(gov_dir, tamper=False)
    proc = _run_verify_chain_cli(gov_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED" in proc.stdout


def test_verify_chain_cli_exits_nonzero_when_tampered(tmp_path):
    # The whole point of an "EXIT GATE" is that a broken chain fails the process,
    # not just prints FAILED — otherwise CI / a cron audit check silently passes.
    gov_dir = tmp_path / "gov"; gov_dir.mkdir()
    _seed_chain(gov_dir, tamper=True)
    proc = _run_verify_chain_cli(gov_dir)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAILED" in proc.stdout

#!/usr/bin/env python3
"""deploy/backup.py — nightly backup of VoiceGuard's critical state to Spaces.

Backs up the two things a droplet loss would otherwise destroy:
  * auth_keys.json — losing it revokes every client's API key
  * jobs.db        — the job queue and its results

jobs.db runs in WAL mode, so it is captured with SQLite's online backup API
rather than a file copy: a copy taken mid-write is not guaranteed consistent.

Optional client-side encryption (Fernet / AES-128-CBC + HMAC) via
$VOICEGUARD_BACKUP_KEY, so the bucket never holds plaintext key hashes.

Run from cron on the droplet — see docs/RUNBOOK-deploy.md:
    python deploy/backup.py --data-dir /var/lib/docker/volumes/voiceguard_vg-data/_data
"""
import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import remote_store

ARTIFACTS = ("jobs.db", "auth_keys.json")


def snapshot_sqlite(src_path, dest_path):
    """Consistent copy of a live (WAL-mode) SQLite database. Returns dest_path."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(dest_path)
        try:
            src.backup(dst)          # online backup API: safe against concurrent writers
        finally:
            dst.close()
    finally:
        src.close()
    return dest_path


def encrypt_file(src_path, dest_path, key):
    """Fernet-encrypt src_path to dest_path. `key` is a urlsafe-base64 Fernet key."""
    from cryptography.fernet import Fernet

    token = Fernet(key.encode() if isinstance(key, str) else key)
    with open(src_path, "rb") as f:
        blob = token.encrypt(f.read())
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(blob)
    return dest_path


def run_backup(client, bucket, prefix, data_dir, out_dir, fernet_key=None, now=None):
    """Snapshot, optionally encrypt, and upload each artifact. Returns object keys."""
    stamp = now or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs(out_dir, exist_ok=True)
    uploaded = []

    for name in ARTIFACTS:
        src = os.path.join(data_dir, name)
        if not os.path.exists(src):
            # A droplet with no keys issued yet has no auth_keys.json. Not an error.
            continue

        staged = os.path.join(out_dir, name)
        if name.endswith(".db"):
            snapshot_sqlite(src, staged)
        else:
            shutil.copy2(src, staged)

        upload_name = name
        if fernet_key:
            staged = encrypt_file(staged, staged + ".enc", fernet_key)
            upload_name = name + ".enc"

        key = f"{prefix}/backups/{stamp}/{upload_name}"
        remote_store.upload_file(client, bucket, key, staged)
        uploaded.append(key)

    return uploaded


def main(argv=None):
    p = argparse.ArgumentParser(description="Back up VoiceGuard state to Spaces")
    p.add_argument("--data-dir", required=True,
                   help="host path of the vg-data volume (contains jobs.db, auth_keys.json)")
    p.add_argument("--out-dir", default="/tmp/voiceguard-backup",
                   help="scratch dir for snapshots before upload")
    args = p.parse_args(argv)

    client = remote_store.make_client()
    bucket, prefix = remote_store.bucket_prefix()
    keys = run_backup(client, bucket, prefix, args.data_dir, args.out_dir,
                      fernet_key=os.environ.get("VOICEGUARD_BACKUP_KEY"))
    shutil.rmtree(args.out_dir, ignore_errors=True)

    if not keys:
        print("WARNING: nothing backed up — is --data-dir correct?", file=sys.stderr)
        return 1
    for k in keys:
        print(f"uploaded s3://{bucket}/{k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

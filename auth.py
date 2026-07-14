#!/usr/bin/env python3
"""auth.py — VoiceGuard API-key authentication (Phase 7 / C1).

Keys are stored HASHED (SHA-256) in a JSON file; the plaintext is shown once at
creation and never stored. Store path from $VOICEGUARD_AUTH_KEYS (default
auth_keys.json), read per call.

CLI:
    python auth.py create --client "Acme Bank"
    python auth.py list
    python auth.py revoke <key_id>
"""
import os, sys, json, hashlib, hmac, secrets, argparse
from datetime import datetime, timezone


def _path():
    return os.environ.get("VOICEGUARD_AUTH_KEYS", "auth_keys.json")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(plaintext):
    return hashlib.sha256(plaintext.encode()).hexdigest()


def _read():
    p = _path()
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []                       # missing/corrupt store → no keys (all invalid)


def _write(records):
    p = _path()
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, p)                  # atomic


def create_key(client, scopes=None):
    plaintext = "vg_" + secrets.token_urlsafe(32)
    key_id = "k_" + secrets.token_hex(8)  # 64-bit: avoids revoke targeting a colliding id
    rec = {"key_id": key_id, "client": client, "key_sha256": _hash(plaintext),
           "created_at": _now(), "active": True, "scopes": list(scopes or [])}
    records = _read()
    records.append(rec)
    _write(records)
    return key_id, plaintext


def verify_key(plaintext):
    if not plaintext:
        return None
    h = _hash(plaintext)
    for rec in _read():
        if rec.get("active") and hmac.compare_digest(rec.get("key_sha256", ""), h):
            return {k: v for k, v in rec.items() if k != "key_sha256"}
    return None


def list_keys():
    return [{k: v for k, v in rec.items() if k != "key_sha256"} for rec in _read()]


def revoke_key(key_id):
    records = _read()
    changed = False
    for rec in records:
        if rec.get("key_id") == key_id and rec.get("active"):
            rec["active"] = False
            changed = True
    if changed:
        _write(records)
    return changed


def main(argv=None):
    p = argparse.ArgumentParser(description="VoiceGuard API-key management")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--client", required=True)
    c.add_argument("--scope", action="append", default=[])
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("key_id")
    args = p.parse_args(argv)
    if args.cmd == "create":
        key_id, plaintext = create_key(args.client, args.scope)
        print(f"key_id: {key_id}")
        print(f"API key (shown ONCE — store it now): {plaintext}")
    elif args.cmd == "list":
        for rec in list_keys():
            state = "active " if rec["active"] else "REVOKED"
            print(f"  {rec['key_id']}  {state}  {rec['client']}  {rec['created_at'][:19]}")
    elif args.cmd == "revoke":
        print("revoked" if revoke_key(args.key_id) else "no active key with that id")


if __name__ == "__main__":
    main()

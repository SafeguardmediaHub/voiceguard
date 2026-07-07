# API-Key Authentication (C1) — Design Spec

- **Date:** 2026-07-07
- **Status:** Approved (design); pending implementation plan
- **Scope:** Sub-project C1 of Phase 7 hardening (the first slice)
- **Follow-ups (separate specs):** C2 async job pattern; C3 data governance (consent, retention, at-rest encryption, audit-log wiring); C4 deployment + load/pen test

---

## 1. Context & Problem

The VoiceGuard API (`api.py`, FastAPI) currently has **no authentication** — any caller can hit
`POST /detect` and the `GET /drift*` operational endpoints. `request_protection.py` provides
per-IP rate limiting + anomaly detection, but there is no notion of *who* is calling.

Phase 7 Task 4 requires an **authenticated REST API**. This slice adds API-key authentication
(the standard B2B pattern for a machine-consumed detection API), the smallest foundational
piece — the async job pattern (C2) will attach job ownership to the authenticated caller, so
identity must exist first.

## 2. Goals / Non-Goals

**Goals:**
1. Per-client **API keys**, stored **hashed**, with a management CLI (create/list/revoke).
2. A FastAPI dependency that requires `Authorization: Bearer <key>` on the protected endpoints
   and rejects missing/invalid/revoked keys with **401**.
3. Protected: `POST /detect`, `GET /drift`, `/drift/latest`, `/drift/history`, `/drift/baseline`.
   Open: `GET /ping` (health), `GET /` (demo).
4. **Per-client rate limiting** — the existing `request_protection` source switches from IP → key.
5. Input hardening: **file-size limit** (413) and **missing-file → 400** (fixes a deferred migration Minor).

**Non-Goals (later slices; the key record reserves a `scopes` field but it is NOT enforced here):**
- Scope/role enforcement (any active key may call any protected endpoint — flat model).
- TLS / transport encryption (C4, nginx).
- At-rest encryption, consent, retention, audit-log wiring (C3).
- The async job pattern (C2).
- No new dependencies — `secrets`, `hashlib`, `hmac`, `json` are stdlib.

## 3. The Key Store & `auth.py`

**Storage.** A JSON file (default `auth_keys.json`, overridable via `VOICEGUARD_AUTH_KEYS`),
read on each verification so tests can point it at a temp store. One record per key:

```json
{"key_id": "k_ab12cd34", "client": "Acme Bank",
 "key_sha256": "<hex sha256 of the plaintext key>",
 "created_at": "2026-07-07T12:00:00Z", "active": true, "scopes": []}
```

The **plaintext key is never stored** — shown once at creation, unrecoverable (GitHub/Stripe pattern).
SHA-256 (unsalted) is appropriate because keys are 256-bit high-entropy random tokens, not
low-entropy passwords (no rainbow-table risk); lookups use a constant-time compare.

**Key format:** `vg_` + `secrets.token_urlsafe(32)` (≈43 urlsafe chars).

**`auth.py` API:**
- `create_key(client: str, scopes: list[str] | None = None) -> tuple[str, str]` — returns
  `(key_id, plaintext_key)`; generates, hashes, appends the record, returns the plaintext once.
- `verify_key(plaintext: str) -> dict | None` — SHA-256 the input, return the matching **active**
  record (via `hmac.compare_digest` on the hex digests) or `None`.
- `list_keys() -> list[dict]` — records without the hash (key_id, client, created_at, active, scopes).
- `revoke_key(key_id: str) -> bool` — set `active=False` (atomic rewrite: temp file + `os.replace`);
  returns whether a key was changed.

**CLI (`python auth.py ...`):**
- `create --client "NAME" [--scope S ...]` — prints the plaintext key ONCE + the key_id.
- `list` — table of key_id / client / active / created_at.
- `revoke <key_id>` — deactivate.

## 4. `api.py` Integration

- Declare `security = HTTPBearer(auto_error=False)` (so `/docs` shows an **Authorize** button and
  a missing header yields our 401, not FastAPI's default 403).
- Dependency:

```python
def require_api_key(creds: HTTPAuthorizationCredentials | None = Depends(security)) -> dict:
    if creds is None or (creds.scheme or "").lower() != "bearer" or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing or malformed API key "
                            "(use 'Authorization: Bearer <key>')")
    rec = auth.verify_key(creds.credentials)
    if rec is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    return rec
```

- Apply `client: dict = Depends(require_api_key)` to `POST /detect` and the four `GET /drift*` routes.
  Leave `GET /ping` and `GET /` unauthenticated.
- **Per-client rate limiting:** in `/detect`, pass `client["key_id"]` (not `request.client.host`) as
  the `source` to `get_protection().check_request(source, file_hash)`.

## 5. Input Hardening (folds in deferred migration Minors)

- **Missing file → 400:** declare `file: UploadFile | None = File(None)`; if `file is None`, return
  `400 {"error": "No file provided"}` (replaces FastAPI's default 422 for the missing-file case).
- **File-size limit:** `MAX_UPLOAD_MB = int(os.environ.get("VOICEGUARD_MAX_UPLOAD_MB", 25))`. After
  reading the upload to the temp file, if its size exceeds the limit, delete it and return
  `413 {"error": "File too large", "max_mb": MAX_UPLOAD_MB}`.

## 6. Data Flow

```
POST /detect  (Authorization: Bearer vg_...)
  require_api_key → verify_key → client record (or 401)
  save upload → size check (413) / missing (400)
  request_protection.check_request(client.key_id, file_hash) → 429 if limited
  detector.detect(tmp) → result (+ filename, request_protection) → 200
GET /drift*   (Authorization: Bearer vg_...) → require_api_key → JSON (or 401)
GET /ping, /  → open (no key)
```

## 7. Error Handling

- No/'malformed key → 401 `{"detail": "Missing or malformed API key ..."}`.
- Invalid/revoked key → 401 `{"detail": "Invalid or revoked API key"}`.
- Missing file (authenticated) → 400 `{"error": "No file provided"}`.
- Oversize file → 413 `{"error": "File too large", "max_mb": N}`.
- Rate-limited → 429 + `Retry-After` (unchanged, now per-key).
- `auth_keys.json` missing/corrupt → treated as "no keys" (all keys invalid → 401); never crashes.

## 8. Testing

- **`tests/test_auth.py` (unit, temp store via `VOICEGUARD_AUTH_KEYS`):** `create_key`→`verify_key`
  round-trip returns the client record; a wrong key → `None`; after `revoke_key`, `verify_key` → `None`;
  `list_keys` omits the hash; the plaintext is never written to the store file.
- **`tests/test_api.py` (updated — existing tests now need a key):** a module fixture points
  `VOICEGUARD_AUTH_KEYS` at a temp file and creates one key. Assertions: `/detect` and `/drift`
  → **401** with no key, **401** with a bad/revoked key, **200** with the valid key; `/ping` → **200**
  with no key; missing-file-with-valid-key → **400**; (optionally) an oversize upload → **413**.

## 9. Files

- **Create:** `auth.py`, `tests/test_auth.py`.
- **Modify:** `api.py` (dependency + apply to routes + per-key rate-limit + size/400 hardening);
  `tests/test_api.py` (auth fixture + key on protected calls); `.gitignore` (+`auth_keys.json`).
- **No dependency changes.**

## 10. Assumptions

- Keys are issued out-of-band by an operator via the CLI; there is no self-service signup endpoint.
- `auth_keys.json` is per-environment deployment state (git-ignored), not committed.
- Rate-limit identity is the API key; unauthenticated open endpoints (`/ping`, `/`) are not rate-limited by key.

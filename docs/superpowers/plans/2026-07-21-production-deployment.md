# VoiceGuard Production Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Commit policy for this repo:** the owner writes their own commits. At every "Commit" step, run the `git add` shown and then **present the commit message for approval — do not run `git commit`** unless explicitly told to.

**Goal:** Take VoiceGuard from "80% of a Compose deployment" to a running, CI-deployed, firewalled service on a single DigitalOcean Droplet that the backend can call over authenticated TLS.

**Architecture:** One droplet (4 vCPU / 8 GB) runs three containers from Docker Compose: `caddy` (reverse proxy, internal TLS, request-size cap) → `api` (gunicorn + uvicorn workers) and `worker` (queue consumer). `api` and `worker` run the **same image** pulled from DigitalOcean Container Registry and share state through one named volume (`vg-data`: SQLite job queue, auth keys, transient uploads). The ~387 MB `v9h` model bundle is **baked into the image**, so the production host holds no Spaces credentials. GitHub Actions on push to `main` runs tests → pulls the bundle from Spaces → builds → pushes to DOCR → SSHes to the droplet and rolls the stack.

**Tech Stack:** Docker + Docker Compose v2, Caddy 2 (internal CA / `tls internal`), Python 3.13, FastAPI + gunicorn/uvicorn, SQLite (WAL), DigitalOcean Droplet + VPC + Cloud Firewall + Container Registry + Spaces, GitHub Actions.

**Source spec:** `docs/superpowers/specs/2026-07-18-voiceguard-deployment-design.md`

## Global Constraints

- **Python 3.13 everywhere.** `requirements.txt` pins `audioop-lts`, `standard-aifc`, `standard-chunk`, `standard-sunau` (3.13 stdlib backports), `numpy==2.4.6`, `torch==2.12.0`. CI currently says 3.12 — that is a bug fixed in Task 2.
- **Active model bundle is `v9h`.** Confirmed by `model_store/ACTIVE.json` (seq 3, reason: "v9fixed no better than v9h; keep v9h"). Never hardcode a different version.
- **CPU-only.** `detector.py` sets `DEVICE = torch.device("cpu")`. No GPU, no CUDA images.
- **No Spaces credentials on the production droplet.** `SPACES_*` are build-time (GitHub Actions) and backup-time only. `deploy/docker-entrypoint.sh` already skips the remote pull when `SPACES_BUCKET` is unset — keep it unset in the production `.env`.
- **Upload cap 25 MB**, enforced in three places: Caddy `max_size`, `VOICEGUARD_MAX_UPLOAD_MB=25` (read at `api.py:141`), and the backend client.
- **The api container is never published to a public interface.** Only `caddy` binds a host port, and only on the VPC private IP.
- **Never commit** `.env`, `auth_keys.json`, `jobs.db`, or `*.pt`. All are already in `.gitignore`; Task 1 adds them to `.dockerignore`.
- **Test tiers:** `VOICEGUARD_CI_FAST=1` makes `tests/conftest.py` skip collection of every weights-loading test module. New tests that do **not** import `detector`/`api` must pass in the fast tier.
- **File paths in this plan are relative to the repo root** (`C:\Users\Michael Ologungbara\Downloads\voice_guard 0ffline`).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `.dockerignore` | Keep secrets, runtime state, and dev artifacts out of the build context | 1 |
| `tests/test_docker_context.py` | Regression guard: sensitive filenames stay excluded | 1 |
| `.github/workflows/ci.yml` | Test tiers (3.12 → 3.13) + new build/push/deploy job | 2, 5 |
| `docker-compose.prod.yml` | Production stack: caddy + api + worker, DOCR image, healthcheck, log caps | 3 |
| `.env.example` | Documented template for the droplet's `.env` | 3 |
| `deploy/Caddyfile` | Reverse proxy, internal TLS, 25 MB body cap, timeouts | 4 |
| `deploy/backup.py` | Nightly `jobs.db` + `auth_keys.json` → Spaces, Fernet-encrypted | 6 |
| `tests/test_backup.py` | Backup unit tests against the in-memory `FakeS3` fixture | 6 |
| `deploy/bootstrap-droplet.sh` | One-shot droplet provisioning (Docker, dirs, `.env`, cron, hosts) | 7 |
| `docs/RUNBOOK-deploy.md` | Operator runbook: deploy, key management, rollback, restore, firewall | 7 |
| `scripts/voiceguard_client.py` | Reference backend client: submit + poll | 8 |
| `tests/test_client.py` | Client tests against an injected fake session | 8 |
| `api.py` | CORS tightened to an env-driven allowlist | 8 |

Unchanged and already correct: `Dockerfile`, `deploy/docker-entrypoint.sh`, `bundle_registry.py`, `auth.py`, `jobs.py`, `worker.py`.

`docker-compose.yml` (dev, `build: .`) and `docker-compose.handoff.yml` (offline tester stack) stay as they are. Production is a **new** file so the two use cases don't fight over one document.

---

### Task 1: Keep secrets and runtime state out of the image

**Why:** `Dockerfile:31` runs `COPY . .`. `.dockerignore` currently excludes weights and media but **not** `auth_keys.json` (live API-key hashes), `jobs.db` (6.9 MB of real job history), `.env`, or the nested duplicate `voice_guard 0ffline/` directory. Everything in that list would be pushed to DOCR as an image layer. Fix this before the first registry push — a layer that has been pushed cannot be un-pushed.

**Files:**
- Modify: `.dockerignore`
- Test: `tests/test_docker_context.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: a `.dockerignore` that excludes `auth_keys.json`, `jobs.db`, `*.db`, `*.db-wal`, `*.db-shm`, `.env`, `.env.*`, `jobs_input/`, `*.docx`, `.pytest_cache/`, `voice_guard 0ffline/`. Task 3 and Task 5 depend on the image being clean.

- [ ] **Step 1: Write the failing test**

Create `tests/test_docker_context.py`:

```python
# tests/test_docker_context.py
"""Guard: nothing sensitive may enter the Docker build context.

Dockerfile does `COPY . .`, so anything not excluded by .dockerignore is baked
into a layer that gets pushed to DigitalOcean Container Registry. This test is a
regression fence around that — it needs no weights, so it runs in the fast tier.
"""
import fnmatch
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Top-level names that must never be copied into the image.
FORBIDDEN = [
    "auth_keys.json",     # SHA-256 hashes of live client API keys
    "jobs.db",            # real job history, ~7 MB
    "jobs.db-wal",
    "jobs.db-shm",
    ".env",
    ".env.production",
    "jobs_input",         # transient uploads
    "AI_Audio_Detector_Phase_Plan.docx",
    ".pytest_cache",
    "voice_guard 0ffline",  # nested duplicate copy of the whole project
]

# Files the image genuinely needs — a too-broad exclusion would break the build.
REQUIRED = [
    "api.py",
    "worker.py",
    "detector.py",
    "requirements.txt",
    "VoiceGuard_LiveDemo (2).html",   # served by GET / (api.py:108)
    "model_store/ACTIVE.json",
    "model_store/registry.jsonl",
]


def _rules():
    """(pattern, is_negation) pairs from .dockerignore, comments and blanks dropped."""
    out = []
    with open(os.path.join(REPO, ".dockerignore"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("!"):
                out.append((line[1:].rstrip("/"), True))
            else:
                out.append((line.rstrip("/"), False))
    return out


def _excluded(path, rules):
    """Docker semantics: last matching rule wins; a directory pattern covers its children."""
    verdict = False
    for pattern, negated in rules:
        if fnmatch.fnmatch(path, pattern) or path.startswith(pattern + "/"):
            verdict = not negated
    return verdict


def test_sensitive_paths_are_excluded():
    rules = _rules()
    leaked = [p for p in FORBIDDEN if not _excluded(p, rules)]
    assert leaked == [], f"these would be baked into the image: {leaked}"


def test_required_paths_are_not_excluded():
    rules = _rules()
    dropped = [p for p in REQUIRED if _excluded(p, rules)]
    assert dropped == [], f".dockerignore is too broad, these are needed: {dropped}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_docker_context.py -v
```

Expected: `test_sensitive_paths_are_excluded` FAILS listing `auth_keys.json`, `jobs.db`, `jobs.db-wal`, `jobs.db-shm`, `.env`, `.env.production`, `jobs_input`, `AI_Audio_Detector_Phase_Plan.docx`, `.pytest_cache`, `voice_guard 0ffline`. `test_required_paths_are_not_excluded` PASSES.

- [ ] **Step 3: Add the exclusions**

In `.dockerignore`, immediately after the `venv/` line and before the `# weights & data` comment block, insert:

```
# Runtime state & secrets — MUST NOT enter the image (Dockerfile does `COPY . .`,
# and image layers get pushed to DOCR). Runtime reads these from the /data volume
# instead: VOICEGUARD_JOBS_DB, VOICEGUARD_AUTH_KEYS, VOICEGUARD_JOBS_INPUT.
auth_keys.json
*.db
*.db-wal
*.db-shm
jobs_input/
.env
.env.*
!.env.example
# Dev artifacts with no runtime role.
*.docx
.pytest_cache/
voice_guard 0ffline/
```

Note `!.env.example` — Task 3 adds that file and it is safe (no secrets); the `.env.*` glob would otherwise swallow it.

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_docker_context.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Verify the demo HTML still ships**

The `*.docx` rule is narrow, but confirm the file `GET /` serves is untouched:

```bash
python -c "
import sys; sys.path.insert(0, 'tests')
from test_docker_context import _rules, _excluded
print('LiveDemo excluded?', _excluded('VoiceGuard_LiveDemo (2).html', _rules()))"
```

Expected: `LiveDemo excluded? False`

- [ ] **Step 6: Commit**

```bash
git add .dockerignore tests/test_docker_context.py
```

Proposed message:

```
fix(docker): keep secrets and runtime state out of the build context

auth_keys.json, jobs.db and .env were being baked into the image by
`COPY . .` and would have been pushed to DOCR. Adds a regression test.
```

---

### Task 2: Align CI with the image's Python version

**Why:** `Dockerfile:7` is `python:3.13-slim` and its comment states `audioop-lts`, `numpy 2.4` and `torch 2.12` require 3.13 — `requirements.txt` confirms (`audioop-lts==0.2.2`, `standard-aifc==3.13.0`, `numpy==2.4.6`, `torch==2.12.0`). `.github/workflows/ci.yml` pins `python-version: "3.12"` in both jobs, so the test gate either fails outright or validates a dependency resolution that production never uses. Also commit the pending Dockerfile/`.dockerignore` bake-in changes, which are still unstaged — CI builds from git, so until they land the built image would have no model in it.

**Files:**
- Modify: `.github/workflows/ci.yml:14` and `.github/workflows/ci.yml:34` (the two `python-version` keys)

**Interfaces:**
- Consumes: Task 1's `.dockerignore`.
- Produces: a green `fast` job on Python 3.13. Task 5 adds a third job to this same file and assumes both existing jobs pass.

- [ ] **Step 1: Confirm the versions currently disagree**

```bash
grep -n "python-version" .github/workflows/ci.yml
grep -n "^FROM" Dockerfile
```

Expected: two lines showing `"3.12"`, and `FROM python:3.13-slim`.

- [ ] **Step 2: Bump both jobs to 3.13**

In `.github/workflows/ci.yml`, in **both** the `fast` and `weights` jobs, change:

```yaml
        with:
          python-version: "3.12"
          cache: pip
```

to:

```yaml
        with:
          python-version: "3.13"          # must match Dockerfile (audioop-lts, numpy 2.4, torch 2.12)
          cache: pip
```

- [ ] **Step 3: Verify the fast tier resolves and passes locally on 3.13**

```bash
python --version
python -m pytest -q
```

with `VOICEGUARD_CI_FAST=1` set:

```bash
VOICEGUARD_CI_FAST=1 python -m pytest -q
```

Expected: PASS. `tests/conftest.py` skips collection of `test_api.py`, `test_detector.py`, `test_worker.py`, `test_golden.py`, `test_gradcam.py`, so no weights load. If local Python is not 3.13, note it and rely on the CI run in Step 5 instead — do not silently proceed on a different interpreter.

- [ ] **Step 4: Commit the version fix together with the pending bake-in changes**

The working tree already carries the model bake-in work (`Dockerfile` switched to 3.13 + `COPY model_store`, `.dockerignore` allowlisting `v9h`, and the untracked `docker-compose.handoff.yml`). These belong in this commit — without them CI would build a model-less image.

```bash
git add .github/workflows/ci.yml Dockerfile .dockerignore docker-compose.handoff.yml
```

Proposed message:

```
build: bake the v9h bundle into the image, align CI on Python 3.13

The image now ships the active bundle so production needs no Spaces
credentials. CI was pinned to 3.12 while the image is 3.13, which
requirements.txt requires (audioop-lts, standard-aifc, numpy 2.4, torch 2.12).
```

- [ ] **Step 5: Push and confirm CI is green**

```bash
git push
gh run watch
```

Expected: the `fast` job passes on 3.13. Do not start Task 3 until it does — every later task builds on this image.

---

### Task 3: Production Compose stack

**Why:** `docker-compose.yml` is a development file. It builds from source on the host (design §7 requires pulling an immutable image from DOCR), publishes `7860` on `0.0.0.0` (design §5 calls VPC-private binding "the one load-bearing security decision"), has no healthcheck (§10), has no log rotation (§10), and mounts `vg-models:/app/model_store` — which **silently defeats the bake-in decision (§6)**: Docker seeds a named volume from the image only while the volume is empty, so after the first `up` every redeploy keeps serving the *old* model even when the new image contains a new one.

**Files:**
- Create: `docker-compose.prod.yml`
- Create: `.env.example`
- Leave alone: `docker-compose.yml`, `docker-compose.handoff.yml`

**Interfaces:**
- Consumes: the clean image from Tasks 1–2.
- Produces:
  - Compose service names `caddy`, `api`, `worker` — Task 5's deploy step and Task 7's runbook reference these by name.
  - Env vars read from `.env`: `VOICEGUARD_IMAGE` (full DOCR ref incl. tag), `VG_BIND_IP` (droplet VPC private IP), `VG_PORT` (default `8443`), `WORKERS`, `VOICEGUARD_MAX_UPLOAD_MB`, `VOICEGUARD_ALLOWED_ORIGINS`.
  - Named volumes `vg-data`, `caddy-data`, `caddy-config`. There is deliberately **no** `vg-models` volume.
  - Task 4 supplies `deploy/Caddyfile`, which this file bind-mounts.

- [ ] **Step 1: Write `.env.example`**

Create `.env.example`:

```bash
# VoiceGuard production environment — copy to .env on the droplet, chmod 600.
# NEVER commit .env. `.dockerignore` and `.gitignore` both exclude it.

# Full DOCR image reference. CI rewrites this to a git-SHA tag on every deploy;
# pin it by hand to roll back (see docs/RUNBOOK-deploy.md).
VOICEGUARD_IMAGE=registry.digitalocean.com/YOUR-REGISTRY/voiceguard:latest

# The droplet's PRIVATE VPC IP — NOT 0.0.0.0. Caddy binds here and nowhere else,
# so the service is unreachable from the public internet. `ip -4 addr show eth1`.
VG_BIND_IP=10.116.0.0
VG_PORT=8443

# gunicorn uvicorn workers. 3 on a 4 vCPU droplet leaves headroom for the worker
# container (the V9 ensemble escalation path is CPU-bound).
WORKERS=3

# Upload cap, enforced here and again in deploy/Caddyfile. Keep the two in sync.
VOICEGUARD_MAX_UPLOAD_MB=25

# CORS allowlist (comma-separated). Empty = no browser origin allowed, which is
# correct for this server-to-server deployment. See Task 8.
VOICEGUARD_ALLOWED_ORIGINS=

# SPACES_* are deliberately ABSENT. The model bundle is baked into the image, so
# the entrypoint skips the remote pull and production holds no Spaces credentials.
# The nightly backup job (deploy/backup.py) gets its own credentials from
# /etc/voiceguard/backup.env, not from this file.
```

- [ ] **Step 2: Write `docker-compose.prod.yml`**

Create `docker-compose.prod.yml`:

```yaml
# VoiceGuard — production stack for a single DigitalOcean Droplet.
# Spec: docs/superpowers/specs/2026-07-18-voiceguard-deployment-design.md
#
#   docker compose -f docker-compose.prod.yml pull
#   docker compose -f docker-compose.prod.yml up -d
#
# api and worker run the SAME image and share state through vg-data (SQLite job
# queue + auth_keys.json + transient uploads). The ~387 MB v9h bundle is baked
# into the image — there is deliberately no model volume, because a named volume
# is seeded from the image only while empty and would pin the model at whatever
# version first started, making every later model update invisible.
#
# The project name is pinned so the named volumes are always voiceguard_vg-data
# etc. regardless of the directory this file sits in. The backup cron and the
# runbook reference /var/lib/docker/volumes/voiceguard_vg-data/_data by that
# exact name — deriving it from the directory would silently break them.
name: voiceguard

services:
  caddy:
    image: caddy:2-alpine
    # The ONLY published port, and only on the private VPC interface. Binding
    # 0.0.0.0 here would expose the detector to the public internet.
    ports:
      - "${VG_BIND_IP}:${VG_PORT}:8443"
    volumes:
      - ./deploy/Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data        # internal CA + issued certs; MUST persist across deploys
      - caddy-config:/config
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped
    logging: &logging
      driver: json-file
      options: { max-size: "10m", max-file: "3" }

  api:
    image: ${VOICEGUARD_IMAGE}
    command: ["api"]
    # No `ports:` — reachable only from caddy on the internal compose network.
    expose: ["7860"]
    env_file: [.env]
    volumes:
      - vg-data:/data
    healthcheck:
      # python-slim has no curl; urllib is already in the interpreter.
      test: ["CMD", "python", "-c",
             "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/ping', timeout=5).status == 200 else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3
      # api.py's lifespan runs detector.startup_check(), which loads ~387 MB and
      # classifies a fixture before the app accepts traffic. Generous start_period
      # keeps a cold start from being reported as a failure.
      start_period: 180s
    restart: unless-stopped
    logging: *logging

  worker:
    image: ${VOICEGUARD_IMAGE}
    command: ["worker"]
    env_file: [.env]
    volumes:
      - vg-data:/data          # same volume: consumes the queue api writes to
    depends_on: [api]
    # 1 replica to start. jobs.claim_next uses BEGIN IMMEDIATE, so scaling to 2 is
    # safe if the queue backs up: `docker compose -f docker-compose.prod.yml up -d --scale worker=2`
    restart: unless-stopped
    logging: *logging

volumes:
  vg-data:
  caddy-data:
  caddy-config:
```

- [ ] **Step 3: Verify Compose parses and resolves**

Task 4 has not created `deploy/Caddyfile` yet, so the bind mount target does not exist — that is fine, `config` only validates the document.

`env_file: [.env]` is resolved at validation time, and `--env-file` does **not**
satisfy it — Compose looks for a literal `.env`. So validate with a throwaway one,
guarding against clobbering a real `.env` if this is ever run on the droplet:

```bash
[ -e .env ] && { echo "real .env present — aborting"; exit 1; }
cp .env.example .env
VOICEGUARD_IMAGE=example/voiceguard:test VG_BIND_IP=10.0.0.1 VG_PORT=8443 \
  docker compose -f docker-compose.prod.yml config
rm -f .env
```

Expected: the fully-resolved YAML prints with `10.0.0.1:8443:8443` under `caddy.ports`, and **no** `vg-models` volume anywhere in the output. Exit code 0.

- [ ] **Step 4: Verify no service publishes a public port**

```bash
VOICEGUARD_IMAGE=example/voiceguard:test VG_BIND_IP=10.0.0.1 VG_PORT=8443 \
  docker compose -f docker-compose.prod.yml config | grep -n "published\|0.0.0.0"
```

Expected: exactly one `published: "8443"` under caddy, and **no** occurrence of `0.0.0.0`.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.prod.yml .env.example
```

Proposed message:

```
feat(deploy): production Compose stack

DOCR image instead of a host build, VPC-private bind, healthcheck on
/ping with a 180s start period for bundle load, and json-file log caps.
Drops the model volume, which would have pinned the baked-in bundle at
its first-started version.
```

---

### Task 4: Caddy reverse proxy with internal TLS

**Why:** Design §5 wants a proxy for request-size limits, timeouts, and one clean ingress, and §12.2 leaves TLS open. Decision taken: **terminate TLS now** with Caddy's internal CA, so bearer tokens are encrypted even inside the VPC. This also satisfies the Phase 7 gate "encryption and auth verified".

Certificates are issued for the hostname `voiceguard.internal`, not for the raw IP — hostname SANs are better supported by HTTP clients and survive a droplet IP change. The backend maps that name to the VoiceGuard VPC IP in its `/etc/hosts` (Task 7 documents this).

**Files:**
- Create: `deploy/Caddyfile`

**Interfaces:**
- Consumes: the `caddy` service and the `caddy-data` volume from Task 3.
- Produces:
  - The backend's base URL: `https://voiceguard.internal:8443`.
  - The internal root CA at `/data/caddy/pki/authorities/local/root.crt` inside the caddy container — Task 7's runbook exports it and Task 8's client trusts it.
  - A 25 MB request-body cap that returns `413` before the upload ever reaches `api.py`.

- [ ] **Step 1: Write the Caddyfile**

Create `deploy/Caddyfile`:

```caddyfile
# VoiceGuard ingress. Terminates TLS with Caddy's internal CA so bearer tokens are
# encrypted even on the private VPC, then proxies to the api container.
#
# The CA lives in the caddy-data volume — that volume must persist across deploys,
# otherwise a new CA is generated and every backend that trusted the old root
# starts failing verification.
{
	# No admin API: it would otherwise listen on 2019 inside the container.
	admin off
	# Internal CA only — never attempt a public ACME challenge from a droplet with
	# no public ingress.
	auto_https disable_redirects
}

voiceguard.internal:8443 {
	tls internal

	# First of three enforcement points for the upload cap (the others are
	# VOICEGUARD_MAX_UPLOAD_MB in .env and the backend client). Rejecting here
	# means an oversized body never touches the Python process.
	request_body {
		max_size 25MB
	}

	reverse_proxy api:7860 {
		# The detector's ensemble escalation path is CPU-bound and slow; /detect is
		# async (202 + poll) so requests are short, but keep read/write generous
		# enough that a large multipart upload on a slow link is not cut off.
		transport http {
			read_timeout 180s
			write_timeout 180s
			dial_timeout 5s
		}
		# Preserve the caller's identity for api-side logging.
		header_up X-Forwarded-For {remote_host}
	}

	log {
		output stdout
		format console
	}
}
```

- [ ] **Step 2: Validate the Caddyfile syntax**

```bash
docker run --rm -v "$(pwd)/deploy/Caddyfile:/etc/caddy/Caddyfile:ro" \
  caddy:2-alpine caddy validate --config /etc/caddy/Caddyfile
```

Expected: `Valid configuration` on the last line, exit code 0. If it reports an adapter warning about formatting, run `caddy fmt --overwrite` via the same container and re-validate.

- [ ] **Step 3: Smoke-test the proxy against a stub upstream**

This proves TLS issuance and the body cap without needing the real 387 MB image.

```bash
docker network create vg-smoke
docker run -d --name api --network vg-smoke --network-alias api \
  -w /srv python:3.13-slim sh -c \
  'mkdir -p /srv && echo ok > /srv/ping && python -m http.server 7860'
docker run -d --name caddy --network vg-smoke -p 127.0.0.1:8443:8443 \
  --add-host voiceguard.internal:127.0.0.1 \
  -v "$(pwd)/deploy/Caddyfile:/etc/caddy/Caddyfile:ro" caddy:2-alpine
sleep 5
docker exec caddy wget -qO- --no-check-certificate https://voiceguard.internal:8443/ping
```

Expected: `ok`.

- [ ] **Step 4: Verify the body cap returns 413**

```bash
docker exec caddy sh -c 'dd if=/dev/zero of=/tmp/big bs=1M count=30 2>/dev/null; \
  wget -qS --no-check-certificate --post-file=/tmp/big \
  https://voiceguard.internal:8443/ 2>&1 | head -3'
```

Expected: a `413 Request Entity Too Large` status line — the 30 MB body is rejected above the 25 MB cap.

- [ ] **Step 5: Confirm the root CA is where the runbook says it is**

```bash
docker exec caddy ls -l /data/caddy/pki/authorities/local/root.crt
```

Expected: the file exists and is non-empty. Task 7's runbook exports exactly this path.

- [ ] **Step 6: Tear down the smoke test**

```bash
docker rm -f caddy api
docker network rm vg-smoke
```

- [ ] **Step 7: Commit**

```bash
git add deploy/Caddyfile
```

Proposed message:

```
feat(deploy): Caddy ingress with internal TLS

Terminates TLS with Caddy's internal CA for voiceguard.internal so bearer
tokens are encrypted on the VPC too, and caps request bodies at 25 MB
before they reach the Python process.
```

---

### Task 5: CI — build, push to DOCR, deploy to the droplet

**Why:** `.github/workflows/ci.yml` has only test jobs today. Design §7 steps 2–5 (pull bundle → build → push DOCR → SSH deploy + healthcheck) do not exist. This task adds them as a third job gated on the tests.

**Prerequisite (manual, do before Step 1):** create these GitHub Actions repository secrets. The `SPACES_*` five already exist for the `weights` job.

| Secret | Value |
|---|---|
| `DOCR_TOKEN` | DO API token with registry read/write. DOCR takes the token as **both** username and password. |
| `DOCR_REGISTRY` | e.g. `registry.digitalocean.com/your-registry` |
| `DROPLET_HOST` | The droplet's **public** IP or hostname (SSH ingress only; the app port stays VPC-private) |
| `DROPLET_USER` | e.g. `deploy` |
| `DROPLET_SSH_KEY` | Private half of the deploy keypair created in Task 7 |
| `DROPLET_SSH_KNOWN_HOSTS` | Output of `ssh-keyscan -H <DROPLET_HOST>` — pins the host key so the deploy cannot be MITM'd |

**Files:**
- Modify: `.github/workflows/ci.yml` (append a `deploy` job)

**Interfaces:**
- Consumes: `fast` and `weights` jobs (Task 2); `docker-compose.prod.yml` service names (Task 3); `deploy/Caddyfile` (Task 4); `/opt/voiceguard` layout and the `deploy` user (Task 7).
- Produces: images tagged `${DOCR_REGISTRY}/voiceguard:<git-sha>` and `:latest`. Task 7's rollback procedure pins the SHA tag.

- [ ] **Step 1: Append the deploy job**

Add to the end of `.github/workflows/ci.yml`:

```yaml
  deploy:                          # build the immutable image, push to DOCR, roll the droplet
    # Only the default branch, and only after both test tiers pass. Never on a PR:
    # the job holds registry and SSH credentials.
    if: github.event_name == 'push' && github.ref == format('refs/heads/{0}', github.event.repository.default_branch)
    needs: [fast, weights]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: pip

      # Weights are not in git (.gitignore excludes *.pt); the Dockerfile bakes
      # model_store/v9h, so the bundle must be in the build context BEFORE build.
      - name: Install registry deps
        run: pip install boto3

      - name: Pull active bundle from Spaces
        run: python bundle_registry.py pull --active
        env:
          SPACES_KEY:      ${{ secrets.SPACES_KEY }}
          SPACES_SECRET:   ${{ secrets.SPACES_SECRET }}
          SPACES_ENDPOINT: ${{ secrets.SPACES_ENDPOINT }}
          SPACES_REGION:   ${{ secrets.SPACES_REGION }}
          SPACES_BUCKET:   ${{ secrets.SPACES_BUCKET }}

      - name: Verify the bundle landed and is the expected version
        run: |
          python bundle_registry.py active
          test -d model_store/v9h || { echo "::error::model_store/v9h missing after pull"; exit 1; }

      - uses: docker/setup-buildx-action@v3

      - name: Log in to DigitalOcean Container Registry
        uses: docker/login-action@v3
        with:
          registry: registry.digitalocean.com
          username: ${{ secrets.DOCR_TOKEN }}     # DOCR: token is both user and password
          password: ${{ secrets.DOCR_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.DOCR_REGISTRY }}/voiceguard:${{ github.sha }}
            ${{ secrets.DOCR_REGISTRY }}/voiceguard:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Deploy to the droplet
        env:
          IMAGE: ${{ secrets.DOCR_REGISTRY }}/voiceguard:${{ github.sha }}
        run: |
          set -euo pipefail
          mkdir -p ~/.ssh
          echo "${{ secrets.DROPLET_SSH_KEY }}" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key
          echo "${{ secrets.DROPLET_SSH_KNOWN_HOSTS }}" > ~/.ssh/known_hosts
          ssh -i ~/.ssh/deploy_key "${{ secrets.DROPLET_USER }}@${{ secrets.DROPLET_HOST }}" \
            "IMAGE='$IMAGE' bash -s" <<'REMOTE'
          set -euo pipefail
          cd /opt/voiceguard

          # Record the currently-deployed image so a failed rollout can be undone.
          grep '^VOICEGUARD_IMAGE=' .env > .env.previous || true

          # Point .env at the new immutable SHA tag (never :latest — a SHA tag is
          # what makes rollback a one-line change).
          sed -i "s|^VOICEGUARD_IMAGE=.*|VOICEGUARD_IMAGE=${IMAGE}|" .env

          git -C /opt/voiceguard/src fetch --depth 1 origin main
          git -C /opt/voiceguard/src checkout -f FETCH_HEAD    # refresh Caddyfile + compose file

          docker compose -f docker-compose.prod.yml pull
          docker compose -f docker-compose.prod.yml up -d --remove-orphans
          docker image prune -f
          REMOTE

      - name: Healthcheck
        run: |
          set -euo pipefail
          ssh -i ~/.ssh/deploy_key "${{ secrets.DROPLET_USER }}@${{ secrets.DROPLET_HOST }}" \
            "bash -s" <<'REMOTE'
          set -euo pipefail
          cd /opt/voiceguard
          # startup_check() loads ~387 MB and classifies a fixture before the app is
          # ready, so poll rather than checking once.
          for i in $(seq 1 40); do
            if docker compose -f docker-compose.prod.yml exec -T api python -c \
                 "import urllib.request,sys,json; \
                  d=json.load(urllib.request.urlopen('http://localhost:7860/ping',timeout=5)); \
                  sys.exit(0 if d.get('active_version')=='v9h' else 1)"; then
              echo "healthy: api reports active bundle v9h"
              docker compose -f docker-compose.prod.yml ps
              exit 0
            fi
            sleep 15
          done
          echo "::error::api did not become healthy within 10 minutes"
          docker compose -f docker-compose.prod.yml logs --tail 80 api
          exit 1
          REMOTE
```

- [ ] **Step 2: Validate the workflow syntax before pushing**

```bash
gh workflow view CI 2>/dev/null || true
python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"
```

Expected: `yaml ok`. (A YAML parse error is the single most common failure here — the heredocs inside `run:` blocks are indentation-sensitive.)

- [ ] **Step 3: Confirm the job is correctly gated**

```bash
python -c "
import yaml
w = yaml.safe_load(open('.github/workflows/ci.yml'))
d = w['jobs']['deploy']
print('needs:', d['needs'])
print('if:', d['if'])
assert d['needs'] == ['fast', 'weights']
assert 'pull_request' not in d['if']
print('gating ok')"
```

Expected: `gating ok`, with `needs: ['fast', 'weights']`.

- [ ] **Step 4: Commit (do not push yet)**

Task 7 provisions the droplet this job SSHes into. Pushing before that exists gives a red build.

```bash
git add .github/workflows/ci.yml
```

Proposed message:

```
ci: build, push to DOCR, and deploy to the droplet on main

Pulls the active bundle into the build context, builds an immutable
SHA-tagged image, pushes to DOCR, then SSHes in to roll the stack and
polls /ping until it reports the v9h bundle.
```

---

### Task 6: Nightly encrypted backup of `jobs.db` and `auth_keys.json`

**Why:** design §8 — `auth_keys.json` is critical (losing it revokes every client) and lives only in the `vg-data` Docker volume on a single droplet. `jobs.db` must be captured with SQLite's backup API, not `cp`: the database runs in WAL mode, so a file copy taken mid-write is not guaranteed consistent.

Written as an importable Python module rather than a shell script so it can be tested against the existing `FakeS3` fixture in `tests/conftest.py`, and so it reuses `remote_store.py` instead of duplicating S3 config parsing.

**Files:**
- Create: `deploy/backup.py`
- Test: `tests/test_backup.py`

**Interfaces:**
- Consumes: `remote_store.make_client(env)`, `remote_store.bucket_prefix(env)`, `remote_store.upload_file(client, bucket, key, local_path)`; the `fake_s3` fixture from `tests/conftest.py`.
- Produces:
  - `snapshot_sqlite(src_path, dest_path) -> str`
  - `encrypt_file(src_path, dest_path, key) -> str`
  - `run_backup(client, bucket, prefix, data_dir, out_dir, fernet_key=None, now=None) -> list[str]` (returns uploaded object keys)
  - Object key layout `"{prefix}/backups/{YYYY-MM-DD}/{name}"`. Task 7's runbook restores from exactly this layout.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backup.py`:

```python
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


def test_missing_auth_keys_is_not_fatal(tmp_path, fake_s3):
    """A droplet with no keys issued yet must still back up the queue."""
    data = tmp_path / "data"
    data.mkdir()
    _make_db(str(data / "jobs.db")).close()

    keys = backup.run_backup(fake_s3, "bkt", "p", str(data), str(tmp_path / "out"), now="2026-07-21")
    assert keys == ["p/backups/2026-07-21/jobs.db"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_backup.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'deploy'`.

- [ ] **Step 3: Write the implementation**

Create `deploy/__init__.py` (empty file — makes `deploy` importable as a package):

```python
```

Create `deploy/backup.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_backup.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Confirm the new tests run in the fast CI tier**

`deploy/backup.py` imports `remote_store`, which imports boto3 only lazily inside `make_client` — so it must not pull weights or require boto3 at import.

```bash
VOICEGUARD_CI_FAST=1 python -m pytest -q
```

Expected: PASS, with `tests/test_backup.py` included in the count (it is not in `conftest.py`'s `collect_ignore` list).

- [ ] **Step 6: Commit**

```bash
git add deploy/__init__.py deploy/backup.py tests/test_backup.py
```

Proposed message:

```
feat(deploy): encrypted nightly backup of jobs.db and auth_keys.json

Uses SQLite's online backup API (jobs.db is WAL-mode, so a file copy is
not consistent) and optional Fernet encryption before upload to Spaces.
```

---

### Task 7: Droplet bootstrap and operator runbook

**Why:** design §13 asks for both, and Task 5's deploy job assumes `/opt/voiceguard` with a `.env`, a checked-out `src/`, and a `deploy` user that can drive Docker. Nothing creates that today. This task also captures the DO-console steps (VPC, firewall, monitoring) that cannot be scripted from the repo.

**Files:**
- Create: `deploy/bootstrap-droplet.sh`
- Create: `docs/RUNBOOK-deploy.md`

**Interfaces:**
- Consumes: `docker-compose.prod.yml` (Task 3), `deploy/Caddyfile` (Task 4), `deploy/backup.py` (Task 6), the CI deploy contract (Task 5).
- Produces: on the droplet — `/opt/voiceguard/.env`, `/opt/voiceguard/src` (git checkout), `/opt/voiceguard/docker-compose.prod.yml` and `deploy/Caddyfile` as symlinks into `src`, `/etc/voiceguard/backup.env` (mode 600, holds `SPACES_*` + `VOICEGUARD_BACKUP_KEY`), and a nightly cron entry.

- [ ] **Step 1: Write the bootstrap script**

Create `deploy/bootstrap-droplet.sh`:

```bash
#!/usr/bin/env bash
# Provision a fresh Ubuntu 24.04 droplet to run VoiceGuard. Idempotent — safe to re-run.
#
#   ssh root@<droplet-public-ip>
#   curl -fsSL https://raw.githubusercontent.com/<org>/<repo>/main/deploy/bootstrap-droplet.sh | bash -s -- <repo-url>
#
# Afterwards, finish the manual steps in docs/RUNBOOK-deploy.md §2 (VPC, firewall,
# DOCR login, .env values, first API key).
set -euo pipefail

REPO_URL="${1:?usage: bootstrap-droplet.sh <git-repo-url>}"
APP_DIR=/opt/voiceguard
DEPLOY_USER=deploy

echo "==> installing docker engine + compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl git python3 python3-venv
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker            # survives reboots (design §4)

echo "==> creating the ${DEPLOY_USER} user"
id -u "$DEPLOY_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$DEPLOY_USER"
usermod -aG docker "$DEPLOY_USER"
install -d -m 0700 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "/home/${DEPLOY_USER}/.ssh"
touch "/home/${DEPLOY_USER}/.ssh/authorized_keys"
chown "$DEPLOY_USER:$DEPLOY_USER" "/home/${DEPLOY_USER}/.ssh/authorized_keys"
chmod 600 "/home/${DEPLOY_USER}/.ssh/authorized_keys"

echo "==> laying out ${APP_DIR}"
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$APP_DIR"
if [ ! -d "${APP_DIR}/src/.git" ]; then
  sudo -u "$DEPLOY_USER" git clone --depth 1 "$REPO_URL" "${APP_DIR}/src"
fi
# Compose reads the file and the Caddyfile from the checkout, so `git checkout` in
# the CI deploy step is all it takes to roll config changes.
ln -sfn "${APP_DIR}/src/docker-compose.prod.yml" "${APP_DIR}/docker-compose.prod.yml"
ln -sfn "${APP_DIR}/src/deploy" "${APP_DIR}/deploy"

if [ ! -f "${APP_DIR}/.env" ]; then
  cp "${APP_DIR}/src/.env.example" "${APP_DIR}/.env"
  chown "$DEPLOY_USER:$DEPLOY_USER" "${APP_DIR}/.env"
  chmod 600 "${APP_DIR}/.env"
  VPC_IP="$(ip -4 -o addr show eth1 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
  if [ -n "$VPC_IP" ]; then
    sed -i "s|^VG_BIND_IP=.*|VG_BIND_IP=${VPC_IP}|" "${APP_DIR}/.env"
    echo "    detected VPC IP ${VPC_IP} -> VG_BIND_IP"
  else
    echo "    !! no eth1 found — set VG_BIND_IP in ${APP_DIR}/.env by hand"
  fi
  echo "    !! set VOICEGUARD_IMAGE in ${APP_DIR}/.env before the first deploy"
fi

echo "==> backup credentials + nightly cron"
install -d -m 0700 /etc/voiceguard
if [ ! -f /etc/voiceguard/backup.env ]; then
  cat > /etc/voiceguard/backup.env <<'EOF'
# Backup-only credentials. These are the ONLY Spaces credentials on this host and
# they are not readable by the containers — the model bundle is baked into the image.
SPACES_KEY=
SPACES_SECRET=
SPACES_ENDPOINT=
SPACES_REGION=
SPACES_BUCKET=
# `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
# STORE THIS OFF-HOST. Without it the backups cannot be decrypted.
VOICEGUARD_BACKUP_KEY=
EOF
  chmod 600 /etc/voiceguard/backup.env
  echo "    !! fill in /etc/voiceguard/backup.env"
fi

if [ ! -d /opt/voiceguard-backup-venv ]; then
  python3 -m venv /opt/voiceguard-backup-venv
  /opt/voiceguard-backup-venv/bin/pip install --quiet boto3 cryptography
fi

cat > /etc/cron.d/voiceguard-backup <<EOF
# Nightly 02:30 UTC backup of jobs.db + auth_keys.json to Spaces (design §8).
SHELL=/bin/bash
30 2 * * * root set -a; . /etc/voiceguard/backup.env; set +a; \
/opt/voiceguard-backup-venv/bin/python ${APP_DIR}/src/deploy/backup.py \
--data-dir /var/lib/docker/volumes/voiceguard_vg-data/_data \
>> /var/log/voiceguard-backup.log 2>&1
EOF
chmod 644 /etc/cron.d/voiceguard-backup

cat > /etc/logrotate.d/voiceguard <<'EOF'
/var/log/voiceguard-backup.log {
    weekly
    rotate 8
    compress
    missingok
    notifempty
}
EOF

echo
echo "==> bootstrap complete. Remaining manual steps: docs/RUNBOOK-deploy.md §2"
```

- [ ] **Step 2: Make it executable and shell-check it**

```bash
chmod +x deploy/bootstrap-droplet.sh
bash -n deploy/bootstrap-droplet.sh && echo "syntax ok"
```

Expected: `syntax ok`. If `shellcheck` is available, run `shellcheck deploy/bootstrap-droplet.sh` and fix anything at error level.

- [ ] **Step 3: Write the runbook**

Create `docs/RUNBOOK-deploy.md`:

````markdown
# VoiceGuard — Deployment Runbook

Companion to `docs/superpowers/specs/2026-07-18-voiceguard-deployment-design.md`.
For model promotion/rollback see `docs/RUNBOOK-model-flow.md`; this document covers
the droplet, the stack, and the backend integration.

## 1. What runs where

| Piece | Where |
|---|---|
| `caddy`, `api`, `worker` | One DO droplet, 4 vCPU / 8 GB, Docker Compose |
| Image | `registry.digitalocean.com/<registry>/voiceguard:<git-sha>`, model `v9h` baked in |
| State | Docker volume `voiceguard_vg-data` → `/data` (jobs.db, auth_keys.json, uploads) |
| Ingress | Caddy on `${VG_BIND_IP}:8443`, TLS via internal CA, **VPC private only** |
| Backups | Nightly 02:30 UTC → Spaces `<prefix>/backups/<date>/` |

## 2. First-time provisioning

1. Create the droplet (Ubuntu 24.04, 4 vCPU / 8 GB) **in the same region and VPC as
   the backend**. Note its private IP (`ip -4 addr show eth1`).
2. `ssh root@<public-ip>` then:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/<org>/<repo>/main/deploy/bootstrap-droplet.sh \
     | bash -s -- https://github.com/<org>/<repo>.git
   ```
3. **DO Cloud Firewall** (this is the load-bearing control — the app has no other
   network protection):
   - Inbound TCP `22` — your admin IP, plus whatever GitHub Actions egress you allow.
   - Inbound TCP `8443` — **source: the backend droplet's private IP only**.
   - Everything else denied. Confirm from any other host that `nc -vz <public-ip> 8443`
     fails.
4. **DOCR login** so `docker compose pull` works:
   ```bash
   sudo -u deploy docker login registry.digitalocean.com -u <DO_API_TOKEN> -p <DO_API_TOKEN>
   ```
5. **Deploy key** for CI:
   ```bash
   ssh-keygen -t ed25519 -f /tmp/vg_deploy -N ""
   cat /tmp/vg_deploy.pub >> /home/deploy/.ssh/authorized_keys
   cat /tmp/vg_deploy          # -> GitHub secret DROPLET_SSH_KEY, then shred /tmp/vg_deploy*
   ssh-keyscan -H <public-ip>  # -> GitHub secret DROPLET_SSH_KNOWN_HOSTS
   ```
6. Fill in `/opt/voiceguard/.env` (`VOICEGUARD_IMAGE`, `VG_BIND_IP`) and
   `/etc/voiceguard/backup.env` (Spaces creds + a generated `VOICEGUARD_BACKUP_KEY`
   — **store that key off-host**).
7. First start:
   ```bash
   cd /opt/voiceguard
   docker compose -f docker-compose.prod.yml pull
   docker compose -f docker-compose.prod.yml up -d
   docker compose -f docker-compose.prod.yml logs -f api    # wait for startup_check
   ```
8. **DO Monitoring**: add alerts on CPU > 80% for 5 min, memory > 85%, disk > 80%.
   The ensemble escalation path is CPU-bound, so CPU saturation is the first symptom
   of a queue backing up.

## 3. Issue an API key for the backend

```bash
cd /opt/voiceguard
docker compose -f docker-compose.prod.yml exec api python auth.py create --client "backend"
```

The plaintext key is printed **once** — store it in the backend's secret manager
immediately. Only the SHA-256 hash is kept, in `/data/auth_keys.json`.

```bash
docker compose -f docker-compose.prod.yml exec api python auth.py list
docker compose -f docker-compose.prod.yml exec api python auth.py revoke <key_id>
```

## 4. Give the backend TLS trust

Caddy issues certs from its own internal CA. Export the root once:

```bash
docker compose -f docker-compose.prod.yml exec caddy \
  cat /data/caddy/pki/authorities/local/root.crt > voiceguard-root.crt
```

On the **backend** host:

```bash
echo "<voiceguard-vpc-ip>  voiceguard.internal" >> /etc/hosts
cp voiceguard-root.crt /usr/local/share/ca-certificates/voiceguard-root.crt
update-ca-certificates
```

Base URL for the backend: `https://voiceguard.internal:8443`.

> The `caddy-data` volume holds that CA. If it is ever deleted, Caddy generates a
> **new** root and every backend that trusted the old one starts failing TLS
> verification — re-export and re-install after any such event.

## 5. Routine deploy

Push to `main`. CI runs tests → pulls the bundle → builds → pushes to DOCR → SSHes
in, rewrites `VOICEGUARD_IMAGE` to the new SHA, pulls, and polls `/ping` until it
reports `active_version: v9h`. Watch it with `gh run watch`.

## 6. Rollback

```bash
cd /opt/voiceguard
sed -i "s|^VOICEGUARD_IMAGE=.*|VOICEGUARD_IMAGE=registry.digitalocean.com/<registry>/voiceguard:<previous-sha>|" .env
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

`.env.previous` in the same directory holds the image line from before the last CI
deploy. Rolling back the **model** rather than the code is a different procedure —
see `docs/RUNBOOK-model-flow.md` (`bundle_registry.py rollback`), then rebuild.

## 7. Restore from backup

```bash
# 1. Stop the stack so nothing writes during the restore.
cd /opt/voiceguard && docker compose -f docker-compose.prod.yml down

# 2. Fetch and decrypt (needs VOICEGUARD_BACKUP_KEY from off-host storage).
aws s3 cp --endpoint-url $SPACES_ENDPOINT \
  s3://$SPACES_BUCKET/<prefix>/backups/<date>/auth_keys.json.enc .
/opt/voiceguard-backup-venv/bin/python -c "
from cryptography.fernet import Fernet
import os
k = Fernet(os.environ['VOICEGUARD_BACKUP_KEY'].encode())
open('auth_keys.json','wb').write(k.decrypt(open('auth_keys.json.enc','rb').read()))"

# 3. Put it back in the volume and restart.
cp auth_keys.json /var/lib/docker/volumes/voiceguard_vg-data/_data/auth_keys.json
docker compose -f docker-compose.prod.yml up -d
```

Verify the restore before telling clients anything: `auth.py list` should show the
expected `key_id`s as active.

## 8. Scaling the worker

If `/jobs/{id}` polls sit in `queued` for long stretches, add a second consumer.
`jobs.claim_next` uses `BEGIN IMMEDIATE`, so concurrent workers are safe:

```bash
docker compose -f docker-compose.prod.yml up -d --scale worker=2
```

Past 2, CPU on a 4 vCPU droplet becomes the ceiling — that is the point at which
the design's "split SQLite out to managed Postgres and add a second droplet" step
is the right move rather than more vertical scaling.

## 9. Triage

| Symptom | Check |
|---|---|
| Backend gets connection refused | `docker compose ps`; firewall rule source IP; `VG_BIND_IP` matches `ip -4 addr show eth1` |
| `api` restarting in a loop | `logs api` — `startup_check()` fails closed on a bad bundle. Roll back the image. |
| Jobs stuck `queued` | `logs worker`; is the worker container up? Is CPU pinned? |
| 413 on upload | Body over 25 MB — cap is enforced in `deploy/Caddyfile` **and** `VOICEGUARD_MAX_UPLOAD_MB` |
| 429 with `Retry-After` | Rate limiter in `request_protection.py`. Expected under bursts; the backend must honour it. |
| TLS verification failure | The `caddy-data` volume was recreated → new CA. Re-export the root (§4). |
| Disk filling | `docker system df`; log caps are 10 MB × 3 per container; check `/data/jobs_input` for orphaned uploads |
````

- [ ] **Step 4: Verify every path the runbook references actually exists**

```bash
python - <<'PY'
import os, re
text = open("docs/RUNBOOK-deploy.md", encoding="utf-8").read()
repo_paths = re.findall(r'`(deploy/[\w.\-/]+|docs/[\w.\-/]+|docker-compose\.prod\.yml)`', text)
missing = sorted({p for p in repo_paths if not os.path.exists(p)})
print("missing:", missing or "none")
PY
```

Expected: `missing: none`.

- [ ] **Step 5: Commit**

```bash
git add deploy/bootstrap-droplet.sh docs/RUNBOOK-deploy.md
```

Proposed message:

```
docs(deploy): droplet bootstrap script and operator runbook

Provisioning, firewall rules, API-key issuance, internal-CA trust for the
backend, deploy, rollback, restore, worker scaling, and triage.
```

- [ ] **Step 6: Provision the droplet, then push Task 5's workflow**

Run the bootstrap and the §2 manual steps against the real droplet. Only once
`docker compose -f docker-compose.prod.yml up -d` serves a healthy `/ping` by hand
should the CI deploy job be pushed:

```bash
git push
gh run watch
```

Expected: `fast`, `weights`, and `deploy` all green; the healthcheck step prints
`healthy: api reports active bundle v9h`.

---

### Task 8: Backend client and CORS tightening

**Why:** design §13 asks for a backend client snippet, and §5 flags `allow_origins=["*"]` (`api.py:32`) for tightening. CORS is a browser mechanism and irrelevant to server-to-server calls, so this is hygiene rather than a vulnerability — but a wildcard in a non-browser-facing service is exactly the kind of thing a penetration test (Phase 7 exit gate) writes up.

**Files:**
- Modify: `api.py:32-33`
- Create: `scripts/voiceguard_client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: the endpoint contract in design §11 — `POST /detect` → `202 {job_id, status, status_url}`; `GET /jobs/{job_id}` → `{status: queued|running|done|error, result?, error?}`; `429` carries `Retry-After`.
- Produces:
  - `api.py` reads `VOICEGUARD_ALLOWED_ORIGINS` (comma-separated; empty ⇒ `[]`).
  - `voiceguard_client.VoiceGuardClient(base_url, api_key, session=None, verify=True)` with `submit(path) -> str` and `wait(job_id, timeout=300, interval=2.0) -> dict` and `detect(path, **kw) -> dict`.

- [ ] **Step 1: Write the failing client tests**

Create `tests/test_client.py`:

```python
# tests/test_client.py
"""Backend-client tests. Uses an injected fake session — no network, no weights."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from voiceguard_client import VoiceGuardClient, VoiceGuardError


class FakeResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    """Replays a scripted list of responses and records the requests it received."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {method} {url}")
        return self._responses.pop(0)

    def post(self, url, **kw):
        return self._next("POST", url, **kw)

    def get(self, url, **kw):
        return self._next("GET", url, **kw)


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "sample.wav"
    p.write_bytes(b"RIFF0000WAVEfmt ")
    return str(p)


def test_submit_sends_bearer_key_and_returns_job_id(clip):
    s = FakeSession([FakeResponse(202, {"job_id": "j-1", "status": "queued",
                                        "status_url": "/jobs/j-1"})])
    c = VoiceGuardClient("https://voiceguard.internal:8443", "vg_secret", session=s)

    assert c.submit(clip) == "j-1"
    method, url, kw = s.calls[0]
    assert (method, url) == ("POST", "https://voiceguard.internal:8443/detect")
    assert kw["headers"]["Authorization"] == "Bearer vg_secret"


def test_wait_polls_until_done(clip):
    s = FakeSession([
        FakeResponse(200, {"job_id": "j-1", "status": "queued"}),
        FakeResponse(200, {"job_id": "j-1", "status": "running"}),
        FakeResponse(200, {"job_id": "j-1", "status": "done",
                           "result": {"verdict": "FAKE", "confidence": 0.91}}),
    ])
    c = VoiceGuardClient("https://voiceguard.internal:8443", "k", session=s)

    assert c.wait("j-1", interval=0)["verdict"] == "FAKE"
    assert len(s.calls) == 3


def test_wait_raises_on_job_error():
    s = FakeSession([FakeResponse(200, {"job_id": "j-1", "status": "error",
                                        "error": "decode failed"})])
    c = VoiceGuardClient("https://x", "k", session=s)

    with pytest.raises(VoiceGuardError, match="decode failed"):
        c.wait("j-1", interval=0)


def test_submit_honours_retry_after_on_429(clip, monkeypatch):
    slept = []
    monkeypatch.setattr("voiceguard_client.time.sleep", slept.append)
    s = FakeSession([
        FakeResponse(429, {"error": "Rate limit exceeded"}, headers={"Retry-After": "3"}),
        FakeResponse(202, {"job_id": "j-2", "status": "queued"}),
    ])
    c = VoiceGuardClient("https://x", "k", session=s)

    assert c.submit(clip) == "j-2"
    assert slept == [3.0]


def test_oversized_file_is_rejected_before_upload(tmp_path):
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (26 * 1024 * 1024))
    s = FakeSession([])                          # no request must be made
    c = VoiceGuardClient("https://x", "k", session=s)

    with pytest.raises(VoiceGuardError, match="25 MB"):
        c.submit(str(big))
    assert s.calls == []


def test_wait_times_out(monkeypatch):
    # started=0.0, then the deadline check after the first poll already exceeds 300s.
    ticks = iter([0.0, 400.0])
    monkeypatch.setattr("voiceguard_client.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("voiceguard_client.time.sleep", lambda _: None)
    s = FakeSession([FakeResponse(200, {"job_id": "j", "status": "running"})])
    c = VoiceGuardClient("https://x", "k", session=s)

    with pytest.raises(VoiceGuardError, match="timed out"):
        c.wait("j", timeout=300, interval=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_client.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'voiceguard_client'`.

- [ ] **Step 3: Write the client**

Create `scripts/voiceguard_client.py`:

```python
#!/usr/bin/env python3
"""voiceguard_client.py — reference client for the backend (design §11).

VoiceGuard is async: POST /detect returns 202 with a job_id, then you poll
GET /jobs/{job_id} until it is done or error.

    from voiceguard_client import VoiceGuardClient
    vg = VoiceGuardClient("https://voiceguard.internal:8443", os.environ["VOICEGUARD_API_KEY"])
    print(vg.detect("clip.wav")["verdict"])

TLS: the service uses Caddy's internal CA. Either install its root on this host
(docs/RUNBOOK-deploy.md §4) or pass verify="/path/to/voiceguard-root.crt".
"""
import os
import time

MAX_UPLOAD_MB = 25          # must match VOICEGUARD_MAX_UPLOAD_MB and deploy/Caddyfile
TERMINAL = ("done", "error")


class VoiceGuardError(RuntimeError):
    """Any non-recoverable failure: rejected upload, job error, or timeout."""


class VoiceGuardClient:
    def __init__(self, base_url, api_key, session=None, verify=True, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verify = verify
        self.timeout = timeout
        if session is None:
            import requests                     # lazy: tests inject a fake session
            session = requests.Session()
        self.session = session

    @property
    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def submit(self, path, max_retries=3):
        """Upload one audio file. Returns the job_id. Honours 429 Retry-After."""
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            # Fail here rather than burning an upload the server will 413 anyway.
            raise VoiceGuardError(f"{path} is {size_mb:.1f} MB, over the {MAX_UPLOAD_MB} MB limit")

        for attempt in range(max_retries):
            with open(path, "rb") as fh:
                resp = self.session.post(
                    f"{self.base_url}/detect",
                    headers=self._headers,
                    files={"file": (os.path.basename(path), fh, "application/octet-stream")},
                    verify=self.verify,
                    timeout=self.timeout,
                )
            if resp.status_code == 429:
                # The server's rate limiter also flags probing-shaped traffic; backing
                # off as instructed is what keeps a client out of that heuristic.
                delay = float(resp.headers.get("Retry-After", 5))
                if attempt == max_retries - 1:
                    raise VoiceGuardError(f"rate limited after {max_retries} attempts")
                time.sleep(delay)
                continue
            if resp.status_code != 202:
                raise VoiceGuardError(f"submit failed ({resp.status_code}): {resp.json()}")
            return resp.json()["job_id"]

        raise VoiceGuardError("submit exhausted retries")

    def wait(self, job_id, timeout=300, interval=2.0):
        """Poll until the job finishes. Returns the result dict, raises on error."""
        started = time.monotonic()
        while True:
            resp = self.session.get(f"{self.base_url}/jobs/{job_id}",
                                    headers=self._headers, verify=self.verify,
                                    timeout=self.timeout)
            if resp.status_code != 200:
                raise VoiceGuardError(f"poll failed ({resp.status_code}): {resp.json()}")
            body = resp.json()
            status = body["status"]
            if status == "done":
                return body["result"]
            if status == "error":
                raise VoiceGuardError(f"job {job_id} failed: {body.get('error')}")
            if time.monotonic() - started > timeout:
                raise VoiceGuardError(f"job {job_id} timed out after {timeout}s (last status: {status})")
            time.sleep(interval)

    def detect(self, path, timeout=300, interval=2.0):
        """submit + wait. Returns the verdict dict."""
        return self.wait(self.submit(path), timeout=timeout, interval=interval)


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Submit one clip to VoiceGuard and print the verdict")
    p.add_argument("path")
    p.add_argument("--url", default=os.environ.get("VOICEGUARD_URL", "https://voiceguard.internal:8443"))
    p.add_argument("--key", default=os.environ.get("VOICEGUARD_API_KEY"))
    p.add_argument("--cacert", default=os.environ.get("VOICEGUARD_CACERT"))
    a = p.parse_args()
    if not a.key:
        raise SystemExit("set --key or $VOICEGUARD_API_KEY")
    client = VoiceGuardClient(a.url, a.key, verify=a.cacert or True)
    print(json.dumps(client.detect(a.path), indent=2))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_client.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Tighten CORS**

In `api.py`, replace lines 32-33:

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
```

with:

```python
# Server-to-server deployment: no browser origin needs access, so the default is
# an empty allowlist. Set VOICEGUARD_ALLOWED_ORIGINS (comma-separated) to serve the
# demo UI at / from a browser on another origin.
_ALLOWED_ORIGINS = [o.strip() for o in
                    os.environ.get("VOICEGUARD_ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])
```

- [ ] **Step 6: Add a CORS test**

Append to `tests/test_api.py` (this module is weights-tier, matching the rest of the file):

```python
def test_cors_is_not_wildcard_by_default(client):
    r = client.get("/ping", headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-origin") != "*"
```

- [ ] **Step 7: Run the weights-tier API tests**

```bash
python -m pytest tests/test_api.py -v
```

Expected: all pass, including `test_cors_is_not_wildcard_by_default`. This loads the real bundle and takes a few minutes.

- [ ] **Step 8: Run the whole suite**

```bash
python -m pytest -q
VOICEGUARD_CI_FAST=1 python -m pytest -q
```

Expected: both PASS. The fast tier must include `test_docker_context.py`, `test_backup.py`, and `test_client.py`.

- [ ] **Step 9: Commit**

```bash
git add api.py scripts/voiceguard_client.py tests/test_client.py tests/test_api.py
```

Proposed message:

```
feat: reference backend client; tighten CORS to an env allowlist

The client handles the async submit/poll contract, honours 429 Retry-After,
and rejects oversized files before upload. CORS defaulted to "*", which has
no purpose in a server-to-server deployment.
```

- [ ] **Step 10: End-to-end verification against the live droplet**

From the **backend** host, after Task 7 §3 issued a key and §4 installed the CA:

```bash
VOICEGUARD_API_KEY=<key> python scripts/voiceguard_client.py /path/to/clip.wav
```

Expected: a JSON verdict. Then confirm the service is genuinely unreachable from
outside the VPC — from any host that is not the backend:

```bash
nc -vz <voiceguard-public-ip> 8443
```

Expected: **connection refused or timeout.** If this succeeds, the firewall rule is
wrong; stop and fix it before handing anything to the client.

---

## Deferred (deliberately not in this plan)

- **Slimming `requirements.txt`.** It pulls `streamlit`, `altair`, `pydeck`, `mlflow`, `Flask`, `torchvision`, and `sentence-transformers` — none of which `api.py`, `worker.py`, or `detector.py` need at runtime. Splitting a `requirements-runtime.txt` would cut image size and deploy time substantially, but it risks breaking an import chain right before a deployment. Do it as its own change once the stack is running and the golden tests can prove nothing regressed.
- **Webhook callbacks** instead of polling (design §11 calls this a future enhancement).
- **Second droplet / managed Postgres** (design §2's stated next step once load outgrows one host).
- **Phase 7 remnants:** penetration test, and the data-retention / consent policy. Both are process work, not deployment code.
- **Phase 8:** the 7-day client acceptance period, which starts once this stack is live.

## Verification checklist

Deployment is done when all of these hold:

- [ ] `VOICEGUARD_CI_FAST=1 python -m pytest -q` passes on Python 3.13
- [ ] `python -m pytest -q` (full, with weights) passes
- [ ] No image layer contains `auth_keys.json`, `jobs.db`, or `.env` (`tests/test_docker_context.py`)
- [ ] A push to `main` builds, pushes to DOCR, deploys, and the healthcheck reports `active_version: v9h`
- [ ] The backend gets a verdict through `https://voiceguard.internal:8443` with a valid bearer key
- [ ] Port 8443 is **not** reachable from any host other than the backend
- [ ] A 26 MB upload is rejected with 413 by Caddy
- [ ] The nightly backup has run at least once and its output decrypts with the stored key
- [ ] `docker compose -f docker-compose.prod.yml down && up -d` recovers cleanly, and the droplet survives a reboot with the stack back up

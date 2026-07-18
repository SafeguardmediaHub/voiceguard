# VoiceGuard — Production Deployment Design

**Date:** 2026-07-18
**Status:** Approved design — pending implementation plan
**Author:** DevOps design session

---

## 1. Context

VoiceGuard is an audio deepfake / voice-clone detector exposed as an async HTTP API.

- **`api.py`** — FastAPI. Client `POST /detect` (Bearer API key) → file saved, job enqueued, returns `202 {job_id, status_url}`. Client polls `GET /jobs/{id}` for the verdict. Also `/ping`, `/drift/*`, and `/` serving a demo HTML page.
- **`worker.py`** — separate process; polls a SQLite queue, runs `detector.detect()`, writes the result back, deletes the input file.
- **`detector.py`** — the ML core. Two-stage cascade: a tiny **LightCNN screener (CPU, ~11 ms)** resolves ~86% of cases; ambiguous ones escalate to the full **V9 ensemble** (AASIST + Wav2Vec2 + RawNet3 fused by XGBoost). **Everything runs on CPU** (`DEVICE = torch.device("cpu")`) — no GPU required.
- **State**: SQLite `jobs.db` (WAL) + `auth_keys.json` + uploaded files, all under a `/data` volume. Model bundle (~387 MB active `v9h`) under `/app/model_store`.
- **Auth**: Bearer keys, SHA-256 hashed, file-backed, managed by the `auth.py` CLI.

**Existing deploy infra** (~80% of a Docker-Compose deployment already built): `Dockerfile` (Python 3.13-slim, bakes wav2vec2 base + the v9h bundle, runs gunicorn+uvicorn), `docker-compose.yml`, a self-contained `docker-compose.handoff.yml`, an entrypoint that can optionally pull the bundle from DigitalOcean Spaces, and a GitHub Actions CI pipeline.

### Decisions captured during design

| Question | Decision |
|----------|----------|
| Hosting target | **Single DigitalOcean Droplet** running Docker Compose |
| Backend topology | Backend runs **separately on DigitalOcean** (another droplet/service) |
| Expected load | **Moderate — steady production** (~tens of req/min, a few concurrent jobs) |
| Build & deploy | **CI/CD via DO Container Registry (DOCR)** |

---

## 2. Why a single Droplet is the correct architecture (not just the easy one)

1. **The api and worker share a filesystem.** They communicate through a SQLite queue (`jobs.db`) and hand off uploaded files through a shared `/data` volume. On one host that's a single Docker named volume. On Cloud Run / App Runner there is no shared persistent disk between containers, and on Kubernetes you'd need a ReadWriteMany network volume — both force rearchitecting the queue (→ Redis / managed queue) and state (→ managed Postgres + object storage) before anything runs.
2. **CPU-only, ~387 MB model baked into the image.** No GPU nodes, no autoscaling group to justify. One right-sized droplet holds the whole thing in memory.
3. **The pieces already exist.** `docker-compose.yml`, the entrypoint, and CI are done.

**Honest tradeoff:** a single droplet is a single point of failure and scales only vertically. For a moderate-traffic client deliverable this is fine. The clean next step if load grows: split the SQLite queue out to managed Postgres and add a second droplet.

---

## 3. Topology

```
                      Public Internet
                            │
                            │ HTTPS (public)
                   ┌────────▼─────────┐
                   │  Backend Droplet │  (your app — already exists)
                   │  public IP + TLS │
                   └────────┬─────────┘
                            │  HTTP over DO VPC (private, free)
                            │  Bearer API key
        ┌───────────────────▼──────────────────────┐
        │   VoiceGuard Droplet  (4 vCPU / 8 GB)     │
        │   NO public exposure — VPC IP only        │
        │                                           │
        │   ┌──────────┐   SQLite queue   ┌────────┐│
        │   │  api     │◄───/data volume─►│ worker ││
        │   │ gunicorn │   + input files  │  x1    ││
        │   │ 3–4 wrk  │                  │        ││
        │   └──────────┘                  └────────┘│
        │        both = same image (DOCR)           │
        │   volumes: vg-data (state), vg-models      │
        └───────────────────────────────────────────┘
                            ▲
                            │ pull image on deploy
                   ┌────────┴─────────┐
                   │ DO Container Reg │◄── GitHub Actions (build+push)
                   └──────────────────┘
```

**The one load-bearing security decision:** VoiceGuard binds to the **VPC private IP only**, and a **DO Cloud Firewall** allows inbound to it *exclusively from the backend droplet*. It is never reachable from the public internet. The backend is the only public surface.

---

## 4. Compute & containers

- **One droplet**, 4 vCPU / 8 GB (DO "Regular" or "Premium Intel"), same region + VPC as the backend.
- **Docker Compose**, two services from **one image**:
  - `api` — gunicorn + 3–4 uvicorn workers. The `--preload` flag (already set in the entrypoint) loads the ~387 MB model once and shares it across workers via copy-on-write.
  - `worker` — 1 replica to start (can bump to 2 later; SQLite `claim_next` uses `BEGIN IMMEDIATE`, so multiple workers are safe).
- `restart: unless-stopped` + Docker enabled on boot → survives reboots. The app already **fails closed** on startup (`startup_check()` refuses to boot a broken bundle).

---

## 5. Networking & security

- **DO VPC** joins both droplets on a private network; VoiceGuard publishes its port on the **private interface only** (not publicly on `0.0.0.0`).
- **DO Cloud Firewall**:
  - Inbound `22` (SSH) — ideally restricted to your admin IP.
  - Inbound app port — **from the backend droplet's private IP only**.
  - Everything else denied.
- **Reverse proxy (Caddy)** in front of the api container: enforces request-size limits, sane timeouts, a single clean ingress, and a one-line TLS upgrade path.
- **TLS stance:**
  - **v1:** plain HTTP over the locked-down VPC — acceptable and standard for service-to-service on a private, firewalled network.
  - **Hardening upgrade (recommended for "no security sacrifice"):** Caddy terminates TLS with an internal cert so the bearer tokens are encrypted in transit too. Both options documented; final pick deferred.
- **CORS**: `api.py` currently uses `allow_origins=["*"]`. Irrelevant for server-to-server (CORS is a browser mechanism), but since this deployment is not browser-facing it will be tightened as hygiene.
- **Auth**: existing Bearer-key system. Keys created via `docker compose exec api python auth.py create --client "backend"`; the hash lives in the `/data` volume.

---

## 6. Model bundle strategy — bake it in

Keep the **model baked into the image** (as the Dockerfile already does). With immutable, registry-based deploys this is the right call: reproducible, no runtime download, no cold-start stall, and **no Spaces credentials on the production droplet at all**.

**Spaces stays as the model source-of-truth for builds** — CI pulls the active bundle at build time. Updating the model = push a new bundle to Spaces → rebuild → redeploy (a versioned rollout via the existing registry/`ACTIVE.json` mechanism).

> Note: model weights are **not** in git (`.gitignore` excludes `*.pt`; only manifests/registry are tracked). CI must run `bundle_registry.py pull --active` to fetch weights into the build context *before* `docker build`.

---

## 7. CI/CD pipeline (GitHub Actions → DOCR → droplet)

On push to `main`:

1. Run the existing **fast test suite** (gate).
2. `bundle_registry.py pull --active` — pull weights from Spaces into the build context (Spaces creds as GH secrets, already wired in `ci.yml`).
3. `docker build` → tag with the git SHA + `latest`.
4. Push to **DO Container Registry (DOCR)**.
5. **Deploy step**: SSH to the droplet (deploy key in GH secrets) → `docker compose pull && docker compose up -d` → healthcheck `/ping`.

Immutable image, no multi-GB build on the production box, rollback = redeploy a previous SHA tag.

---

## 8. State, persistence & backups

- `/data` volume holds `jobs.db` (+ WAL), `auth_keys.json`, and transient uploads.
- `auth_keys.json` is **critical** — losing it revokes every client.
- **Backup**: nightly cron on the droplet copies `auth_keys.json` + a SQLite `.backup` of `jobs.db` to a Spaces bucket (encrypted). Optionally enable DO Droplet weekly snapshots as belt-and-suspenders.
- Uploaded files are deleted by the worker after processing, so `/data` stays small.

---

## 9. Secrets & config

- `.env` on the droplet (`chmod 600`, never in git — already excluded). At runtime it is minimal: `WORKERS`, upload limit, etc. **No Spaces creds needed at runtime** (model baked in) → smaller secret surface.
- GH Actions secrets: DOCR token, Spaces creds (build-time only), droplet SSH deploy key.

---

## 10. Observability & ops

- **Healthcheck**: Compose healthcheck hits `/ping`; the backend can also poll it as a liveness signal.
- **DO Monitoring + alerts** on CPU / memory / disk (the ensemble path is CPU-bound — watch for saturation).
- **Log rotation**: Docker `json-file` driver with size/rotation caps so logs don't fill the disk.
- **Uptime**: an external/backend check on `/ping`.

---

## 11. Backend integration contract

Async, poll-based. The backend does:

1. `POST {VOICEGUARD_URL}/detect` with `Authorization: Bearer <key>` + multipart file → `202 {job_id, status_url}`.
2. Poll `GET /jobs/{job_id}` until `status` is `done` (→ `result` verdict) or `error`.
3. Enforce the 25 MB upload cap on its side too, and handle `429` (rate-limit) with `Retry-After`.

A webhook/callback (push instead of poll) is a small future enhancement, not part of v1.

### Endpoint reference

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET`  | `/ping` | none | Liveness + active bundle info |
| `POST` | `/detect` | Bearer | Submit audio; returns `202 {job_id, status_url}` |
| `GET`  | `/jobs/{job_id}` | Bearer | Poll job status/result (scoped to the caller's key) |
| `GET`  | `/drift`, `/drift/latest`, `/drift/history`, `/drift/baseline` | Bearer | Drift-monitor reads |
| `GET`  | `/` | none | Demo HTML UI |

---

## 12. Judgment calls (open for revision)

1. **Model baked into the image** (not pulled at runtime) — for immutability + no prod secrets.
2. **HTTP over the VPC for v1**, with Caddy+TLS documented as a one-step hardening upgrade.
3. **Single worker replica** to start (scale to 2 if the queue backs up).

---

## 13. Deliverables for the implementation plan

- Production `docker-compose.yml` (api + worker, healthcheck, log rotation, volumes).
- `Caddyfile` (reverse proxy; TLS-ready).
- DO VPC + Cloud Firewall rules.
- GitHub Actions workflow (test → pull bundle → build → push DOCR → SSH deploy).
- Backup cron script (`jobs.db` + `auth_keys.json` → Spaces).
- Droplet bootstrap + operator runbook (create API keys, deploy, roll back, restore).
- Backend client snippet (submit + poll).

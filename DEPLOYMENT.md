# VoiceGuard — DigitalOcean Deployment Runbook

Operator guide for deploying VoiceGuard onto a single DigitalOcean Droplet with
Docker Compose. Design rationale lives in
[`docs/superpowers/specs/2026-07-18-voiceguard-deployment-design.md`](docs/superpowers/specs/2026-07-18-voiceguard-deployment-design.md);
this file is the executable version of it.

**Read [§10 Known gaps](#10-known-gaps) before you promise drift monitoring to
anyone** — the API side is wired, the nightly evaluation needs a validation set
that is not in the image.

---

## 1. What you are deploying

```
                    Public Internet
                          │  HTTPS
                 ┌────────▼─────────┐
                 │ Backend Droplet  │   (your app; the only public surface)
                 └────────┬─────────┘
                          │  HTTPS over the DO VPC, Bearer API key
        ┌─────────────────▼───────────────────────────┐
        │  VoiceGuard Droplet — 4 vCPU / 8 GB, CPU only│
        │  bound to the VPC IP, never 0.0.0.0          │
        │                                              │
        │   caddy :8443 ──► api (gunicorn, 3 workers)  │
        │                     │  SQLite queue on       │
        │                     ▼  the vg-data volume    │
        │                   worker  x1                 │
        └──────────────────────────────────────────────┘
                          ▲ docker pull
                 ┌────────┴─────────┐
                 │ DO Container Reg │◄── built from this repo
                 └──────────────────┘
```

`api` and `worker` are the **same image** with different entrypoint arguments.
They communicate only through the shared `/data` volume (SQLite job queue in WAL
mode + uploaded files), which is why this is one droplet and not a cluster.

The ~387 MB `v9h` model bundle is **baked into the image**. Production therefore
holds no DigitalOcean Spaces credentials at all — Spaces is a build-time source
of truth only.

| Volume | Holds | Losing it means |
|---|---|---|
| `voiceguard_vg-data` | `jobs.db`, `auth_keys.json`, `governance/audit_log.jsonl`, `drift/`, transient uploads | every client API key is revoked; audit chain of custody is broken |
| `voiceguard_caddy-data` | Caddy's internal CA + issued certs | every backend that trusted the old root fails TLS verification until re-pinned |
| `voiceguard_caddy-config` | Caddy autosave config | nothing important |

---

## 2. Prerequisites

On your workstation:

- `doctl` authenticated (`doctl auth init`)
- Docker with buildx
- The repo checked out, and **Spaces credentials** — the model weights are not in
  git (`.gitignore` excludes `*.pt`), so the build context has to be filled from
  Spaces first.

You need these DigitalOcean resources:

- A **VPC** shared with the backend droplet, in the same region.
- A **Container Registry** (DOCR).
- A **Spaces bucket** holding the model bundles (already in use by CI).

Export the Spaces config locally (these are the same five variables CI uses):

```bash
export SPACES_KEY=...       SPACES_SECRET=...
export SPACES_ENDPOINT=https://fra1.digitaloceanspaces.com
export SPACES_REGION=fra1   SPACES_BUCKET=your-bucket
# SPACES_PREFIX defaults to voiceguard/model_store
```

---

## 3. One-time infrastructure

### 3.1 Droplet

```bash
doctl compute droplet create voiceguard-prod \
  --image docker-20-04 \
  --size s-4vcpu-8gb \
  --region fra1 \
  --vpc-uuid <YOUR_VPC_UUID> \
  --ssh-keys <YOUR_SSH_KEY_ID> \
  --enable-monitoring \
  --wait
```

4 vCPU / 8 GB is the floor: gunicorn runs 3 uvicorn workers plus the worker
container, and the escalation path (AASIST + Wav2Vec2 + RawNet3 fused by XGBoost)
is CPU-bound. `--enable-monitoring` gives you the CPU/memory/disk alerts in §9.

Get the **private** VPC address — this is `VG_BIND_IP`, and it is the one value
people most often get wrong:

```bash
ssh root@<public-ip> "ip -4 addr show eth1 | awk '/inet /{print \$2}'"
```

### 3.2 Cloud Firewall

The load-bearing security control. VoiceGuard must never be publicly reachable.

```bash
doctl compute firewall create \
  --name voiceguard-prod \
  --droplet-ids <VOICEGUARD_DROPLET_ID> \
  --inbound-rules "protocol:tcp,ports:22,address:<YOUR_ADMIN_IP>/32 protocol:tcp,ports:8443,address:<BACKEND_PRIVATE_IP>/32" \
  --outbound-rules "protocol:tcp,ports:all,address:0.0.0.0/0 protocol:udp,ports:all,address:0.0.0.0/0"
```

Port 8443 is open **only** to the backend droplet's private IP. Everything else
inbound is denied. Verify from anywhere else that it is closed:

```bash
curl -m 5 https://<VOICEGUARD_PUBLIC_IP>:8443/ping   # must time out
```

### 3.3 Container registry

```bash
doctl registry create your-registry          # if you do not have one yet
doctl registry login                         # local docker credentials
```

---

## 4. Build and push the image

Run from the repo root. **Step 1 is mandatory** — without it `docker build`
copies an empty `model_store` and the container fails its startup check.

```bash
# 1. Fill the build context with the active bundle from Spaces.
python bundle_registry.py pull --active
python bundle_registry.py active          # sanity: should print v9h

# 2. Build and tag with the git SHA (immutable) plus a moving :latest.
SHA=$(git rev-parse --short HEAD)
REG=registry.digitalocean.com/your-registry
docker build -t $REG/voiceguard:$SHA -t $REG/voiceguard:latest .

# 3. Push both tags.
docker push $REG/voiceguard:$SHA
docker push $REG/voiceguard:latest
```

The build bakes the `facebook/wav2vec2-base` weights in as well and sets
`HF_HUB_OFFLINE=1`, so the running container never reaches Hugging Face. Expect
a ~3 GB image and a slow first build; the deps, HF and bundle layers cache
separately from application code, so later builds are fast.

> Rollback is "deploy a previous SHA tag", so never overwrite a SHA tag — always
> build a fresh one.

---

## 5. Configure and start the droplet

### 5.1 Get the deploy files onto the droplet

`docker-compose.prod.yml` bind-mounts `./deploy/Caddyfile`, so the compose file
and the `deploy/` directory must sit together:

```bash
ssh root@<public-ip>
mkdir -p /srv/voiceguard && cd /srv/voiceguard
git clone https://github.com/SafeguardmediaHub/voiceguard.git .
```

(Cloning the repo is the simplest option and keeps `deploy/` in sync. Copying
just `docker-compose.prod.yml`, `deploy/` and `.env.example` works equally well.)

### 5.2 `.env`

```bash
cp .env.example .env
chmod 600 .env
vi .env
```

Set exactly these:

```ini
VOICEGUARD_IMAGE=registry.digitalocean.com/your-registry/voiceguard:<SHA>
VG_BIND_IP=10.x.x.x            # the PRIVATE eth1 address from §3.1 — never 0.0.0.0
VG_PORT=8443
WORKERS=3
VOICEGUARD_MAX_UPLOAD_MB=25    # keep in sync with request_body max_size in deploy/Caddyfile
VOICEGUARD_ALLOWED_ORIGINS=    # empty = no browser origin allowed; correct for server-to-server
```

Do **not** add `SPACES_*` here. The bundle is baked in; the entrypoint only
attempts a Spaces pull when `SPACES_BUCKET` is set, and production should hold no
Spaces credentials.

Do **not** add `VOICEGUARD_DEVICE` — production stays on the default CPU path.
`tests/test_docker_context.py::test_deploy_config_never_sets_the_device_override`
fences this.

### 5.3 First boot

```bash
doctl registry login                                    # on the droplet
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

The api container stays `starting` for up to ~3 minutes: `api.py`'s lifespan runs
`detector.startup_check()`, which loads ~387 MB and classifies a fixture before
accepting traffic. That is why the healthcheck has `start_period: 180s`. Caddy
waits on `service_healthy`, so it will not come up before the api is genuinely
ready.

The startup check **fails closed** — a broken or tampered bundle stops the
container rather than serving degraded verdicts. If the api never turns healthy,
go straight to §11.

```bash
docker compose -f docker-compose.prod.yml logs -f api
curl -sk https://localhost:8443/ping | python3 -m json.tool
```

---

## 6. Issue an API key for the backend

Keys are SHA-256 hashed into `auth_keys.json` on the `vg-data` volume. **The
plaintext key is shown once and never again.**

```bash
docker compose -f docker-compose.prod.yml exec api \
  python auth.py create --client "backend-prod"

docker compose -f docker-compose.prod.yml exec api python auth.py list
# revoke by key id:
# docker compose -f docker-compose.prod.yml exec api python auth.py revoke <key_id>
```

Store the plaintext in the backend's secret manager immediately.

---

## 7. Wire up the backend

### 7.1 Trust Caddy's internal CA

Caddy serves `voiceguard.internal:8443` with a certificate from its **own
internal CA**, so the backend must (a) resolve that hostname and (b) trust that
root. Skipping this is the most common integration failure.

On the VoiceGuard droplet, export the root certificate:

```bash
docker compose -f docker-compose.prod.yml exec caddy \
  cat /data/caddy/pki/authorities/local/root.crt > voiceguard-root.crt
```

On the **backend** droplet:

```bash
echo "<VOICEGUARD_PRIVATE_IP>  voiceguard.internal" >> /etc/hosts
cp voiceguard-root.crt /usr/local/share/ca-certificates/voiceguard-root.crt
update-ca-certificates
curl https://voiceguard.internal:8443/ping        # no -k needed if trust worked
```

> The CA lives in the `voiceguard_caddy-data` volume. **Never delete that
> volume** — a new CA would be generated and the backend would start failing
> verification until you re-export and re-trust.

### 7.2 The request contract

Detection is asynchronous: submit, then poll.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/ping` | none | Liveness + active bundle version |
| `POST` | `/detect` | Bearer | Submit audio → `202 {job_id, status_url}` |
| `GET` | `/jobs/{job_id}` | Bearer | Poll status/result, scoped to the calling key |
| `GET` | `/drift`, `/drift/latest`, `/drift/history`, `/drift/baseline` | Bearer | Drift monitor reads |
| `GET` | `/` | none | Demo HTML page |

```bash
KEY=<plaintext key>
curl -X POST https://voiceguard.internal:8443/detect \
     -H "Authorization: Bearer $KEY" -F "file=@sample.wav"
# → 202 {"job_id":"...","status_url":"/jobs/..."}
curl -H "Authorization: Bearer $KEY" https://voiceguard.internal:8443/jobs/<job_id>
```

`scripts/voiceguard_client.py` is a reference client that already handles polling,
`429` + `Retry-After`, and pre-flight size rejection. The backend must enforce the
25 MB cap on its side too — it is enforced in three places (backend client, Caddy
`request_body max_size 25MiB`, `VOICEGUARD_MAX_UPLOAD_MB`) and they must agree.

---

## 8. Drift monitoring

Drift monitoring has two halves. **The reader is wired; the producer needs a
validation set you must supply** (see §10).

### 8.1 Reader (done automatically)

`DRIFT_OUTPUT_DIR=/data/drift` is set in the Dockerfile, so the `/drift*`
endpoints read from the `vg-data` volume and the entrypoint creates the directory
on boot. Both the api container and any one-off monitor container resolve to the
same path, and reports survive a redeploy.

Until the first run lands, `/drift` answers with empty/`null` sections rather than
erroring — that is expected, not a fault.

### 8.2 Producer (nightly evaluation)

The monitor needs a manifest of labelled validation clips plus the audio itself.
Neither is in the image (`.dockerignore` excludes `models/`, `data/`, `*.wav`,
`*.mp3`), so mount them from the host.

Put the validation set on the droplet, e.g. `/srv/voiceguard/valset/`, containing
the audio and a `val.json` whose `path` fields are the **container** paths:

```json
[
  {"path": "/valset/real/clip_0001.mp3", "label": 0, "source": "real_local"},
  {"path": "/valset/fake/noizai_0001.mp3", "label": 1, "source": "noizai_tts"}
]
```

`label` is `0` = real, `1` = fake; `source` groups clips for the per-source catch
rate alerts. All three keys are required — `_validate_manifest_schema` rejects the
manifest otherwise.

Establish the baseline once:

```bash
cd /srv/voiceguard
docker compose -f docker-compose.prod.yml run --rm \
  -v /srv/voiceguard/valset:/valset:ro \
  -e DRIFT_VAL_MANIFEST=/valset/val.json \
  api python drift_monitor_3.py --init-baseline
```

Then schedule the nightly run (03:30 UTC, off the CI schedule and off peak):

```bash
cat >/etc/cron.d/voiceguard-drift <<'EOF'
30 3 * * * root cd /srv/voiceguard && /usr/bin/docker compose -f docker-compose.prod.yml run --rm -v /srv/voiceguard/valset:/valset:ro -e DRIFT_VAL_MANIFEST=/valset/val.json api python drift_monitor_3.py --run >>/var/log/voiceguard-drift.log 2>&1
EOF
chmod 644 /etc/cron.d/voiceguard-drift
```

A run writes `drift_report_<ts>.json`, appends `drift_log.jsonl`, and updates
`drift_alert_state.json` — all in `/data/drift`, all immediately visible on
`/drift`.

### 8.3 How alerting behaves

| Signal | Threshold | Notes |
|---|---|---|
| Clean ensemble EER | ±3.0 pp | also needs `p < 0.05` on a two-proportion z-test before it counts |
| Deployed (cascade) EER | ±3.0 pp | what production actually ships; can regress while the ensemble looks fine |
| Per-source catch rate | −10 pp | −15 pp for phone-class sources |
| Noiz.ai catch rate | −15 pp | |
| Val manifest hash | any change | you changed the test set — re-baseline deliberately |

A breach does **not** fire immediately. `DRIFT_CONFIRM_RUNS` (default 2) requires
consecutive breaching runs before an alert is confirmed, and a clean run resets
the counter. Only confirmed alerts write `retrain_trigger.json`, which surfaces on
`/drift` as `retrain.retrain_needed`. This behaviour is pinned by
`tests/test_drift_monitor.py` and `tests/test_retrain_trigger.py`.

Optional email on confirmed alerts — add to `.env`:

```ini
DRIFT_SMTP_HOST=...
DRIFT_SMTP_PORT=587
DRIFT_SMTP_USER=...
DRIFT_SMTP_PASS=...
DRIFT_ALERT_TO=ops@example.com,michael@example.com
```

After acting on a trigger:

```bash
docker compose -f docker-compose.prod.yml run --rm api python drift_monitor_3.py --retrain-status
docker compose -f docker-compose.prod.yml run --rm api python drift_monitor_3.py --clear-trigger
```

---

## 9. Backups and monitoring

### 9.1 Nightly encrypted backup

`deploy/backup.py` snapshots `jobs.db` through SQLite's online backup API (a plain
file copy of a WAL-mode database is not consistent), plus `auth_keys.json` and
`governance/audit_log.jsonl`, optionally Fernet-encrypted, to Spaces.

This is the **one** place production needs Spaces credentials, and they stay out
of `.env` deliberately:

```bash
mkdir -p /etc/voiceguard
cat >/etc/voiceguard/backup.env <<'EOF'
SPACES_KEY=...
SPACES_SECRET=...
SPACES_ENDPOINT=https://fra1.digitaloceanspaces.com
SPACES_REGION=fra1
SPACES_BUCKET=your-backup-bucket
VOICEGUARD_BACKUP_KEY=<44-char Fernet key>
EOF
chmod 600 /etc/voiceguard/backup.env
```

Generate the Fernet key with
`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
and **store it outside the droplet** — without it the backups are unreadable.

```bash
cat >/etc/cron.d/voiceguard-backup <<'EOF'
0 2 * * * root set -a; . /etc/voiceguard/backup.env; set +a; /usr/bin/python3 /srv/voiceguard/deploy/backup.py --data-dir /var/lib/docker/volumes/voiceguard_vg-data/_data >>/var/log/voiceguard-backup.log 2>&1
EOF
chmod 644 /etc/cron.d/voiceguard-backup
```

The volume path is `voiceguard_vg-data` because `docker-compose.prod.yml` pins
`name: voiceguard`. Run it once by hand and confirm it prints uploaded keys.

Also enable weekly DO droplet snapshots as a second line of defence.

### 9.2 Monitoring

- DO Monitoring alerts on CPU > 80% for 10 min, memory > 85%, disk > 80%.
- The backend should poll `/ping` as an uptime check.
- Docker logs are capped at 10 MB × 3 files per service by the compose `logging`
  block, so they cannot fill the disk.

---

## 10. Known gaps

Be explicit with stakeholders about these.

1. **The drift validation set is not deployable as-is.** `models/val_v8_fresh.json`
   contains absolute Windows paths from a development laptop. It will not resolve
   inside a Linux container. §8.2 needs a regenerated manifest with `/valset/...`
   paths plus the audio copied to the droplet. Until that exists, `/drift` serves
   empty results and no drift alerting happens.
2. **No automated deploy step.** `.github/workflows/ci.yml` runs tests only —
   there is no build/push/SSH-deploy job, so §4 and §5.3 are manual. The design
   calls for CD; it is not built.
3. **`docs/RUNBOOK-deploy.md` is referenced by `.env.example` but does not
   exist.** This file replaces it.
4. **Single point of failure.** One droplet, vertical scaling only. The clean next
   step if load grows is moving the SQLite queue to managed Postgres and adding a
   second droplet.
5. **Rate limiting is per-process.** `request_protection.py` keeps its counters in
   memory, so the effective limit scales with the number of gunicorn workers and
   does not aggregate across them.

---

## 11. Operations

### Deploy a new version

```bash
cd /srv/voiceguard
sed -i "s|^VOICEGUARD_IMAGE=.*|VOICEGUARD_IMAGE=registry.digitalocean.com/your-registry/voiceguard:$NEW_SHA|" .env
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps        # wait for api = healthy
curl -sk https://localhost:8443/ping
```

### Roll back

Point `VOICEGUARD_IMAGE` at the previous SHA tag and repeat. This is why SHA tags
are never overwritten.

### Update the model bundle

Promotion is a production change and is **attributable by design**:

```bash
python bundle_registry.py promote v9h2 --actor "firstname.lastname" --reason "H1 eval passed"
```

- `--actor` must be a real name. `cli`, `admin`, `root`, `unknown` and anything
  under 3 characters are rejected (`_require_named_actor`) — ISO 42001 change
  control, and it lands in the tamper-evident hash chain in `ACTIVE.json`.
- A version in `BLOCKED_VERSIONS` can never be activated, and `rollback` refuses
  to land on one. `v9` is blocked: its `aasist.pt` is the collapsed from-scratch
  V9 architecture that current `detector.py` cannot load, so activating it would
  crash the service on start.
- Promotion runs the sub-model health gate against `tests/probe_clips/`, which is
  baked into the image precisely so the gate can certify in production. Avoid
  `--skip-health`; it exists for emergencies, not routine use.

Then rebuild (§4) and redeploy (§11) so the new bundle is baked into the image.

```bash
python bundle_registry.py rollback --actor "firstname.lastname" --reason "regression in prod"
python bundle_registry.py active
python bundle_registry.py verify v9h
```

### Restore from backup

```bash
docker compose -f docker-compose.prod.yml down
# decrypt if VOICEGUARD_BACKUP_KEY was set, then:
cp jobs.db auth_keys.json /var/lib/docker/volumes/voiceguard_vg-data/_data/
mkdir -p /var/lib/docker/volumes/voiceguard_vg-data/_data/governance
cp audit_log.jsonl /var/lib/docker/volumes/voiceguard_vg-data/_data/governance/
docker compose -f docker-compose.prod.yml up -d
```

Restore `audit_log.jsonl` byte-for-byte — the tamper-evident hash chain is
computed over exact bytes, and any line-ending translation breaks verification.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| api never becomes healthy, logs show a bundle error | `startup_check()` failing closed on a missing/tampered bundle | `docker compose ... run --rm api python bundle_registry.py verify v9h`; rebuild with a fresh `pull --active` |
| api healthy, Caddy will not start | Caddy waits on `service_healthy` | wait out the 180 s `start_period`; check `logs api` |
| Backend gets a TLS verification error | Caddy's internal root not trusted, or `caddy-data` was recreated | redo §7.1 |
| Backend gets connection refused from outside the VPC | working as designed | traffic must originate from the backend's private IP |
| `413` on an upload the API would accept | MiB vs MB mismatch | `deploy/Caddyfile` must say `25MiB`, matching `VOICEGUARD_MAX_UPLOAD_MB=25` |
| `/drift` always empty | no monitor run yet, or `DRIFT_OUTPUT_DIR` overridden off `/data` | §8.2; `curl .../drift` reports the `output_dir` it read |
| Jobs queue up, results are slow | one worker saturated | `docker compose -f docker-compose.prod.yml up -d --scale worker=2` — `jobs.claim_next` uses `BEGIN IMMEDIATE` and the audit log takes an OS file lock, so extra workers are safe |
| Disk filling | uploads not cleaned, or drift reports accumulating | worker deletes inputs after processing; prune old `drift_report_*` files in `/data/drift` |

### Useful commands

```bash
docker compose -f docker-compose.prod.yml logs -f api worker caddy
docker compose -f docker-compose.prod.yml exec api python auth.py list
docker compose -f docker-compose.prod.yml run --rm api python bundle_registry.py active
docker compose -f docker-compose.prod.yml run --rm api python bundle_registry.py log v9h
curl -sk https://localhost:8443/ping | python3 -m json.tool
```

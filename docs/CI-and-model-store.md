# CI & DigitalOcean Spaces Model Store

Model bundles (the 7 co-calibrated V9 artifacts + `bundle.json`, ~389 MB) are too large for
git. They live in a **DigitalOcean Spaces** bucket and are pulled on demand — verified against
the manifest's SHA-256 — by CI and by the deployment host.

## Bucket layout
```
s3://<SPACES_BUCKET>/<SPACES_PREFIX>/          # SPACES_PREFIX default: voiceguard/model_store
    ACTIVE.json                                 # hash-chained active pointer (verbatim)
    registry.jsonl                              # audit record
    bundles/<version>/ { bundle.json, aasist.pt, wav2vec.pt, rawnet.pt, lcnn.pt,
                         xgb.json, cal.json, thresholds.json }
```

## One-time setup
1. Create a Spaces bucket (region `fra1` recommended — same region as the deploy droplet).
   Generate a Spaces access key/secret.
2. Set these as **GitHub repository secrets** (Settings → Secrets and variables → Actions):
   `SPACES_KEY`, `SPACES_SECRET`, `SPACES_ENDPOINT` (`https://fra1.digitaloceanspaces.com`),
   `SPACES_REGION` (`fra1`), `SPACES_BUCKET`.
3. Populate the bucket from a machine that has the local `model_store/`:
   ```bash
   export SPACES_KEY=... SPACES_SECRET=... SPACES_ENDPOINT=https://fra1.digitaloceanspaces.com \
          SPACES_REGION=fra1 SPACES_BUCKET=<bucket>
   python bundle_registry.py push --active
   ```
   Re-run `push --active` after every future `promote` so Spaces tracks the active bundle.

## Pulling (CI and deploy)
```bash
python bundle_registry.py pull --active     # downloads + verifies the active bundle into model_store/
```
`pull` fails closed: any file whose bytes don't match the manifest SHA-256 aborts with an error
and never activates the bundle.

## CI tiers (`.github/workflows/ci.yml`)
- **fast** — every push + PR. Runs `VOICEGUARD_CI_FAST=1 pytest`: the weights-free tests
  (registry, jobs, auth, loadtest, tracking, remote_store). No secrets, seconds.
- **weights** — default-branch push, manual dispatch, or nightly (03:00 UTC). Pulls the active
  bundle from Spaces (using the secrets) and runs the full suite incl. the golden regression.
  Never runs on pull requests, so secrets are never exposed to fork PRs.

## Local runs
- Weights-free only (no bundle needed): `VOICEGUARD_CI_FAST=1 pytest`
- Everything (needs `model_store/` present locally or a prior `pull`): `pytest`

# CI Pipeline + DigitalOcean Spaces Model Store — Design Spec

- **Date:** 2026-07-08
- **Status:** Approved (design); pending implementation plan
- **Scope:** Phase 1, Task 1 remaining item ("CI pipeline"), plus the shared prerequisite
  it depends on: a remote model-artifact store on DigitalOcean Spaces.
- **Unblocks:** CI weights-tier tests; the C4b DigitalOcean deployment (droplet pulls the
  same bundle the same way).
- **Depends on:** the model-flow hub (`bundle_registry.py`, `governance.py`) and the existing
  test suite (48 tests + golden regression).

---

## 1. Context & Problem

Phase 1's foundation task calls for "MLflow experiment tracking, version control and CI
pipeline." MLflow tracking (`tracking.py`), version control (git), and the GPU training env
(Kaggle) are done; **CI is the outstanding item.**

Two obstacles make CI non-trivial:

1. **The model weights (389 MB) cannot live in GitHub.** The deployable bundle — 7
   co-calibrated V9 artifacts + `bundle.json` — is git-ignored. CI's model-dependent tests
   (and the DigitalOcean droplet) need the weights from somewhere GitHub is not.
2. **`detector.py` loads all models at import.** Any test module that imports `detector`
   (directly or via `api`) loads 380 MB at collection time and crashes on a machine without
   the weights. pytest imports every test file during collection, so a marker deselection
   (`-m "not weights"`) is *not* enough to keep a weights-free runner alive.

The fix is a **DigitalOcean Spaces** (S3-compatible) bucket as the single source of truth for
model bundles, a `pull`/`push` remote backend on the existing registry (with the existing
SHA-256 manifest giving pull-and-verify for free), and a **two-tier** GitHub Actions pipeline
that keeps model-dependent tests off the credential-free fast runner.

## 2. Goals / Non-Goals

**Goals:**
1. A **DigitalOcean Spaces** bucket holding model bundles, mirroring the local `model_store/`
   layout, populated by a `push` command and consumed by a `pull` command.
2. `pull` re-hashes every downloaded file against `bundle.json` and **refuses to install a
   bundle whose bytes don't match** — a corrupt or tampered download never becomes active.
3. A **two-tier GitHub Actions** workflow:
   - **fast** (every `push` + `pull_request`): the credential-free, weights-free tests.
   - **weights** (`push` to the default branch + manual `workflow_dispatch` + nightly
     `schedule`): pull the active bundle from Spaces, run the model tests incl. the golden
     regression.
4. Unit tests for the remote backend that run **without a real bucket and without new test
   dependencies** (injectable client + in-memory fake store).
5. One entrypoint (`python bundle_registry.py pull --active`) reused verbatim by CI and by the
   future C4b droplet provisioning.

**Non-Goals (out of scope / later):**
- The C4b deployment itself (nginx/systemd/gunicorn) — this slice only produces the *fetch*
  the droplet will use.
- A linter/formatter gate in CI — kept out to focus scope on the high-value test guard; a
  `ruff`/`flake8` step can be a fast follow.
- A reduced `requirements-fast.txt` — the fast job installs the full `requirements.txt`
  (pip-cached); trimming the dependency set for speed is a possible later optimization, not now.
- Multi-cloud / multi-bucket — one Spaces bucket, endpoint-swappable via env (works for R2,
  B2, AWS S3 unchanged) but only DO Spaces is documented.

## 3. Remote Store Layout (DigitalOcean Spaces)

S3-compatible bucket, one prefix, mirroring `model_store/`:

```
s3://<SPACES_BUCKET>/<SPACES_PREFIX>/           # SPACES_PREFIX default: "voiceguard/model_store"
    ACTIVE.json                                  # the hash-chained active pointer (verbatim)
    registry.jsonl                               # audit record (dirs are non-portable; see §5)
    bundles/<version>/
        bundle.json
        aasist.pt  wav2vec.pt  rawnet.pt  lcnn.pt
        xgb.json   cal.json    thresholds.json
```

**Config (all via env; never committed):**

| Env var | Meaning | Example / default |
|---|---|---|
| `SPACES_KEY` | Spaces access key | (secret) |
| `SPACES_SECRET` | Spaces secret key | (secret) |
| `SPACES_ENDPOINT` | Regional S3 endpoint | `https://fra1.digitaloceanspaces.com` |
| `SPACES_REGION` | Region name | `fra1` (default) |
| `SPACES_BUCKET` | Bucket (Space) name | e.g. `voiceguard-models` |
| `SPACES_PREFIX` | Key prefix | `voiceguard/model_store` (default) |

**Region default `fra1` (Frankfurt):** well-connected European region with good latency to
Nigeria; the C4b droplet should be provisioned in the *same* region so bundle transfer is
intra-datacenter (fast, egress-free). All fields are env-configurable — switching to `lon1`,
`ams3`, etc. is a config change, no code.

## 4. `remote_store.py` — the boto3 boundary (new)

The **only** module that imports boto3. Keeps S3 mechanics isolated and testable.

```python
def make_client(env=os.environ):
    """boto3 S3 client for DigitalOcean Spaces built from SPACES_* env vars.
    Raises RuntimeError naming the first missing required var."""

def upload_file(client, bucket, key, local_path): ...
def download_file(client, bucket, key, local_path): ...      # creates parent dirs
def download_bytes(client, bucket, key) -> bytes | None:     # None if key absent (404)
    ...
def list_keys(client, bucket, prefix) -> list[str]: ...
```

- Pure I/O, no registry knowledge. The **client is a parameter** everywhere, so tests inject a
  fake (an in-memory object exposing the handful of boto3 methods used); no `moto`, no new dep.
- `download_bytes` distinguishes "absent" (return `None`) from other errors (raise), so callers
  can treat a missing `ACTIVE.json` as "empty remote" rather than a crash.

## 5. `bundle_registry.py` — `push` / `pull` (modify)

Add two methods to `Registry` plus a small remote helper. They orchestrate `remote_store`; they
do **not** import boto3 directly (they receive a client, defaulting to `remote_store.make_client()`).

```python
def push(self, version, client=None):
    """Upload bundle <version>'s files + bundle.json to bundles/<version>/, and upload the
    store's ACTIVE.json and registry.jsonl to the prefix root. Verifies integrity first."""

def push_active(self, client=None):
    """push(get_active())."""

def pull(self, version=None, client=None):
    """Download the active pointer; resolve the target version (arg or ACTIVE's latest);
    download that bundle's 8 files into store_dir/<version>/; validate_bundle() (sha256 vs
    manifest) — raise BundleError on mismatch; ensure a LOCAL registry.jsonl entry exists for
    the version (register_bundle if absent, so `dir` is the correct local abspath); write
    ACTIVE.json verbatim from remote (so verify_active_chain() still holds). Return version."""
```

**Why `pull` rebuilds the local registry entry:** `registry.jsonl` records `dir` as an
**absolute path** on the machine that registered the bundle (`os.path.abspath`). Copying that
verbatim to a fresh CI/droplet machine would point at a nonexistent path, breaking
`verify_integrity` and model loading. So `pull` writes the files locally and calls
`register_bundle(local_dir)` (idempotent guard: skip if the version is already registered),
giving a correct local `dir`. `ACTIVE.json`, by contrast, is the tamper-evident chain and
contains **no paths** — it is written byte-for-byte from the remote so `verify_active_chain()`
stays valid.

**CLI subcommands** (extend `_build_cli`):
- `python bundle_registry.py push <version>` / `push --active`
- `python bundle_registry.py pull [<version>] [--active]` (default `--active`)

## 6. Test tiering

**`conftest.py` (repo root, new) — the real weights-isolation mechanism:**

```python
import os
collect_ignore = []
if os.environ.get("VOICEGUARD_CI_FAST"):
    # These modules import detector -> load 380MB weights at import. On a weights-free
    # runner even pytest COLLECTION of them fails, so we must not import them at all.
    collect_ignore = ["tests/test_api.py", "tests/test_detector.py",
                      "tests/test_worker.py", "tests/test_golden.py"]
```

- **Fast tier** runs with `VOICEGUARD_CI_FAST=1 pytest` → the four model-loading modules are
  never imported. Covers: `test_bundle_registry`, `test_tracking`, `test_jobs`, `test_auth`,
  `test_loadtest`, `test_remote_store`.
- **Weights tier** runs plain `pytest` (all modules) after `pull --active`.
- A `weights` marker is also **registered in `pytest.ini`** and applied (`pytestmark`) to the
  four modules, for local ergonomics (`pytest -m weights` / `-m "not weights"`); but
  `collect_ignore` — not the marker — is what makes the fast runner safe.

**`pytest.ini` (new):** register the `weights` marker (silences unknown-marker warnings) and set
`testpaths = tests`.

## 7. `.github/workflows/ci.yml` (new)

```yaml
name: CI
on:
  push:
  pull_request:
  workflow_dispatch:
  schedule:
    - cron: "0 3 * * *"          # nightly 03:00 UTC = 04:00 WAT

jobs:
  fast:                           # every push + PR; no secrets, no weights
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      - run: pytest -q            # env below deselects weights modules
        env: { VOICEGUARD_CI_FAST: "1" }

  weights:                        # model tests where the bundle can be pulled
    # default-branch push, manual dispatch, or nightly — never on a (fork) PR,
    # so Spaces secrets are never exposed to untrusted PR code.
    if: >-
      github.event_name == 'workflow_dispatch' ||
      github.event_name == 'schedule' ||
      (github.event_name == 'push' && github.ref == format('refs/heads/{0}', github.event.repository.default_branch))
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      - name: Cache model bundle            # key derives from the active version (plan detail)
        uses: actions/cache@v4
        with: { path: model_store, key: bundle-${{ github.sha }} }
      - name: Pull active bundle from Spaces
        run: python bundle_registry.py pull --active
        env:
          SPACES_KEY: ${{ secrets.SPACES_KEY }}
          SPACES_SECRET: ${{ secrets.SPACES_SECRET }}
          SPACES_ENDPOINT: ${{ secrets.SPACES_ENDPOINT }}
          SPACES_REGION: ${{ secrets.SPACES_REGION }}
          SPACES_BUCKET: ${{ secrets.SPACES_BUCKET }}
      - run: pytest -q            # full suite incl. golden regression
```

The `on:` triggers plus the weights-job `if:` encode: the **weights** job fires on
`workflow_dispatch`, on the nightly `schedule`, and on `push` to the **default branch** — but
never on a pull request (secrets must not be exposed to fork PRs). The `fast` job runs on
everything, including PRs. Bundle caching lets a re-run skip the 389 MB download when unchanged;
the exact cache key is a plan detail and correctness never depends on it — `pull` is idempotent
and re-verifies every file's sha256 regardless.

**GitHub repo secrets to configure (documented, not committed):** `SPACES_KEY`,
`SPACES_SECRET`, `SPACES_ENDPOINT`, `SPACES_REGION`, `SPACES_BUCKET`.

## 8. Testing

- **`tests/test_remote_store.py` (fast tier, no bucket, no new deps):**
  - A `FakeS3` in-memory client (dict of `key -> bytes`) implementing the boto3 methods
    `remote_store` uses. `upload_file`/`download_file` round-trip; `download_bytes` returns
    `None` for an absent key; `make_client` raises naming the missing var when a `SPACES_*` is
    unset.
  - **Registry `pull` verify test:** seed the fake store with a bundle + `ACTIVE.json`; `pull`
    into a temp `store_dir` → files present, `verify_integrity` True, `get_active()` correct,
    `verify_active_chain()` True. Then **flip one byte** of a stored file in the fake store →
    `pull` raises `BundleError` (sha256 mismatch) and does not leave an active pointer to a
    corrupt bundle.
  - **`push` round-trip test:** `push` a local bundle into the fake store, then `pull` it into a
    second temp store → identical + verified.
- **Fast-tier collection safety test:** with `VOICEGUARD_CI_FAST=1`, `collect_ignore` excludes
  the four modules (assert via a tiny check that importing the fast set doesn't import
  `detector`). (Kept lightweight — the CI run itself is the real proof.)
- **Weights tier (on GitHub, by the user):** after populating the bucket, the `weights` job's
  green golden regression is the live proof of the real pull + model load.

## 9. Error Handling

- `pull` on sha256 mismatch → `BundleError`, no `ACTIVE.json` update (fail closed).
- `pull` with a missing remote `ACTIVE.json` and no explicit version → `BundleError`
  ("remote store has no active pointer").
- `make_client` with any `SPACES_*` unset → `RuntimeError` naming the first missing var (so a
  misconfigured CI job fails loudly, not with an opaque boto3 error).
- Network/boto3 errors propagate (CI marks the job failed) — no silent skip of the golden
  regression.
- Fast job never references secrets, so it runs unchanged on fork PRs.

## 10. Files

- **Create:** `remote_store.py`, `conftest.py`, `pytest.ini`, `.github/workflows/ci.yml`,
  `tests/test_remote_store.py`, `docs/CI-and-model-store.md`.
- **Modify:** `bundle_registry.py` (`push`/`push_active`/`pull` + CLI subcommands),
  `requirements.txt` (add `boto3`), the four weights test modules (add `pytestmark =
  pytest.mark.weights`).
- **Docs:** `docs/CI-and-model-store.md` — bucket layout, the 5 GitHub secrets, the one-time
  `push --active` to populate Spaces, how the two tiers behave, and the `pull --active`
  entrypoint the C4b droplet will reuse.

## 11. Assumptions

- The repository is (or will be) hosted on **GitHub**, so **GitHub Actions** is the CI provider.
- The user creates the Spaces bucket and sets the 5 repo secrets; the bucket is populated once
  via `push --active` (and again on each future `promote`). CI only ever *pulls*.
- `boto3` is an acceptable new runtime/deploy dependency (it is also what the C4b droplet
  provisioning uses to pull). It is import-isolated to `remote_store.py`.
- The default branch is `main` for the weights-job trigger (the plan reads the actual default
  branch; adjust if the repo uses another).

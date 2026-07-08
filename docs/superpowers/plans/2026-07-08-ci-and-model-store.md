# CI Pipeline + DigitalOcean Spaces Model Store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store model bundles in a DigitalOcean Spaces bucket and add a two-tier GitHub Actions CI that runs fast weights-free tests on every push/PR and a gated golden-regression job that pulls the bundle from Spaces.

**Architecture:** A new `remote_store.py` isolates all boto3 calls behind an injectable client. `bundle_registry.py` gains `push`/`pull` that reuse the existing SHA-256 manifest so a pull is verified byte-for-byte and fails closed on any mismatch. A conftest `collect_ignore` (env-gated) keeps the four model-loading test modules off the credential-free fast runner, since pytest imports every module at collection time.

**Tech Stack:** Python 3.12, boto3 (S3-compatible → DigitalOcean Spaces), pytest, GitHub Actions.

## Global Constraints

- **`boto3` is the only new dependency**, and it is import-isolated to `remote_store.py` (imported lazily inside `make_client`). No other module imports boto3.
- **All Spaces config via `SPACES_*` env vars**; never hardcode or commit secrets. Required: `SPACES_KEY`, `SPACES_SECRET`, `SPACES_ENDPOINT`, `SPACES_REGION`, `SPACES_BUCKET`. Optional: `SPACES_PREFIX` (default `voiceguard/model_store`).
- **Region default `fra1`**; endpoint form `https://fra1.digitaloceanspaces.com`.
- **`pull` fails closed:** any sha256 mismatch → `BundleError`, and the local `ACTIVE.json` is not updated to point at the bad bundle.
- **Fast tier must never import** `test_api.py`, `test_detector.py`, `test_worker.py`, `test_golden.py` (they load 380 MB at import). The mechanism is `collect_ignore` in `tests/conftest.py`, gated by `VOICEGUARD_CI_FAST=1` — not the marker alone.
- **The weights CI job never runs on pull requests** (fork-secret safety).
- **DO NOT `git commit`.** Per standing project rule, the controller snapshots each task and the **user** commits. Every task's final step runs the fast suite and reports DONE — it does not commit.
- Preserve existing behavior: the registry's `register`/`promote`/`rollback`/`verify` CLI and semantics are unchanged; `ACTIVE.json` stays a hash-chained tamper-evident log (`verify_active_chain()` must still pass after a pull).

---

### Task 1: `remote_store.py` — the boto3 boundary + fake fixture

**Files:**
- Create: `remote_store.py`
- Create: `tests/conftest.py` (the `FakeS3` fake + `fake_s3` fixture)
- Create: `tests/test_remote_store.py`
- Modify: `requirements.txt` (add the resolved `boto3` pin)

**Interfaces:**
- Produces (consumed by Task 2):
  - `remote_store.make_client(env=os.environ)` → boto3 S3 client; raises `RuntimeError` naming the first missing required var.
  - `remote_store.bucket_prefix(env=os.environ)` → `(bucket: str, prefix: str)`; prefix defaults to `voiceguard/model_store`, trailing slash stripped.
  - `remote_store.upload_file(client, bucket, key, local_path)` → None.
  - `remote_store.download_file(client, bucket, key, local_path)` → None (creates parent dirs).
  - `remote_store.download_bytes(client, bucket, key)` → `bytes | None` (None if the key is absent).
  - `tests/conftest.py` exposes a `fake_s3` pytest fixture returning a `FakeS3()` (in-memory client with `upload_file(filename, bucket, key)`, `download_file(bucket, key, filename)`, `get_object(Bucket, Key)`).

- [ ] **Step 1: Create the `FakeS3` fake + fixture in `tests/conftest.py`**

```python
# tests/conftest.py
import io
import os
import pytest


class FakeS3:
    """In-memory stand-in for a boto3 S3 client — only the methods remote_store uses.
    Keys are (bucket, key) tuples -> bytes."""

    def __init__(self):
        self.store = {}

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as f:
            self.store[(bucket, key)] = f.read()

    def download_file(self, bucket, key, filename):
        data = self.store[(bucket, key)]                 # KeyError if absent
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "wb") as f:
            f.write(data)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.store:
            raise KeyError((Bucket, Key))
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}


@pytest.fixture
def fake_s3():
    return FakeS3()
```

- [ ] **Step 2: Write the failing tests** in `tests/test_remote_store.py`

```python
# tests/test_remote_store.py
import os
import sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import remote_store


def test_make_client_missing_var_raises(monkeypatch):
    for v in ("SPACES_KEY", "SPACES_SECRET", "SPACES_ENDPOINT", "SPACES_REGION", "SPACES_BUCKET"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(RuntimeError) as e:
        remote_store.make_client()
    assert "SPACES_" in str(e.value)


def test_bucket_prefix_defaults(monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.delenv("SPACES_PREFIX", raising=False)
    assert remote_store.bucket_prefix() == ("vg-bucket", "voiceguard/model_store")


def test_bucket_prefix_custom_strips_slash(monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.setenv("SPACES_PREFIX", "custom/store/")
    assert remote_store.bucket_prefix() == ("vg-bucket", "custom/store")


def test_upload_download_roundtrip(fake_s3, tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"hello-bytes")
    remote_store.upload_file(fake_s3, "b", "k/a.bin", str(src))
    dst = tmp_path / "sub" / "a.bin"
    remote_store.download_file(fake_s3, "b", "k/a.bin", str(dst))
    assert dst.read_bytes() == b"hello-bytes"


def test_download_bytes_present_and_absent(fake_s3, tmp_path):
    src = tmp_path / "p.json"
    src.write_bytes(b'{"x":1}')
    remote_store.upload_file(fake_s3, "b", "k/p.json", str(src))
    assert remote_store.download_bytes(fake_s3, "b", "k/p.json") == b'{"x":1}'
    assert remote_store.download_bytes(fake_s3, "b", "missing/key") is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_remote_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_store'`.

- [ ] **Step 4: Implement `remote_store.py`**

```python
"""remote_store.py — DigitalOcean Spaces (S3-compatible) backend for model bundles.

The ONLY module that imports boto3, and it does so lazily inside make_client so the
module imports fine without boto3 (and tests run against an injected fake client).
Config comes from SPACES_* env vars; see docs/CI-and-model-store.md.
"""
import os

_REQUIRED = ("SPACES_KEY", "SPACES_SECRET", "SPACES_ENDPOINT", "SPACES_REGION", "SPACES_BUCKET")


def make_client(env=os.environ):
    """boto3 S3 client for DigitalOcean Spaces from SPACES_* env vars.
    Raises RuntimeError naming the first missing required var."""
    for name in _REQUIRED:
        if not env.get(name):
            raise RuntimeError(f"missing required env var: {name}")
    import boto3                                          # lazy: only needed for a real client
    return boto3.client(
        "s3",
        region_name=env["SPACES_REGION"],
        endpoint_url=env["SPACES_ENDPOINT"],
        aws_access_key_id=env["SPACES_KEY"],
        aws_secret_access_key=env["SPACES_SECRET"],
    )


def bucket_prefix(env=os.environ):
    """(bucket, prefix) from env; prefix default 'voiceguard/model_store', no trailing slash."""
    return env["SPACES_BUCKET"], env.get("SPACES_PREFIX", "voiceguard/model_store").rstrip("/")


def upload_file(client, bucket, key, local_path):
    client.upload_file(local_path, bucket, key)


def download_file(client, bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    client.download_file(bucket, key, local_path)


def download_bytes(client, bucket, key):
    """Object bytes, or None if the key is absent (404). Other errors propagate."""
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except Exception as e:                                # noqa: BLE001 — normalized below
        if _is_not_found(e):
            return None
        raise


def _is_not_found(e):
    # Real boto3 raises ClientError with response['Error']['Code'] in {NoSuchKey,404,NotFound};
    # the in-memory FakeS3 raises KeyError. Treat both as "absent".
    if isinstance(e, KeyError):
        return True
    code = getattr(e, "response", {}).get("Error", {}).get("Code")
    return code in ("NoSuchKey", "404", "NotFound")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_remote_store.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Add the resolved `boto3` pin to `requirements.txt`**

Run: `pip install boto3` then `pip show boto3 | grep -i version` to get the exact installed version.
Then add one line to `requirements.txt` (keep the file's existing alphabetical style; `boto3` sits after `blinker`):

```
boto3==<the version pip just installed, e.g. 1.40.11>
```

Verify it installs cleanly: `pip install -r requirements.txt` → no errors.

- [ ] **Step 7: Run the fast suite; report DONE (do not commit)**

Run: `VOICEGUARD_CI_FAST=1 python -m pytest tests/test_remote_store.py tests/test_jobs.py tests/test_auth.py -q`
Expected: PASS. Report DONE with the test summary. The controller snapshots; the user commits.

---

### Task 2: `bundle_registry.py` — `push` / `pull` + CLI

**Files:**
- Modify: `bundle_registry.py` (add `import remote_store`; add `push`, `push_active`, `pull` methods to `Registry`; add `push`/`pull` CLI subcommands + `main` handling)
- Modify: `tests/test_bundle_registry.py` (add push/pull tests using the `fake_s3` fixture)

**Interfaces:**
- Consumes (from Task 1): `remote_store.make_client`, `remote_store.bucket_prefix`, `remote_store.upload_file`, `remote_store.download_file`, `remote_store.download_bytes`; the `fake_s3` fixture.
- Consumes (existing): `Registry.register_bundle`, `get_bundle`, `get_active`, `verify_integrity`, `validate_bundle(bundle_dir)`, `BUNDLE_FILES` (7 names), `read_manifest`, `write_manifest`, `self.store_dir`, `self.active_path`, `self.registry_path`.
- Produces (consumed by Tasks 3/4 and CI):
  - `Registry.push(version, client=None)` → version (uploads the 8 bundle files + `ACTIVE.json` + `registry.jsonl`).
  - `Registry.push_active(client=None)` → version.
  - `Registry.pull(version=None, client=None)` → version (downloads + verifies; sets active when the pulled version is the remote-active one).
  - CLI: `python bundle_registry.py push [<version>] [--active]`, `python bundle_registry.py pull [<version>] [--active]`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_bundle_registry.py`

These reuse the `fake_s3` fixture. `_make_bundle` builds a valid dummy bundle (7 files + manifest) in a fresh store, registers and promotes it.

```python
# --- appended to tests/test_bundle_registry.py ---
import bundle_registry
from bundle_registry import Registry, BUNDLE_FILES, BundleError


def _make_bundle(store_dir, version="v9"):
    """Create a valid dummy bundle under store_dir/<version>, register + promote it."""
    bdir = os.path.join(store_dir, version)
    os.makedirs(bdir, exist_ok=True)
    for name in BUNDLE_FILES:
        with open(os.path.join(bdir, name), "wb") as f:
            f.write(f"dummy-{version}-{name}".encode())
    bundle_registry.write_manifest(bdir, {"version": version})
    reg = Registry(store_dir=store_dir)
    reg.register_bundle(bdir)
    reg.promote(version, actor="test", reason="seed")
    return reg


def test_push_then_pull_roundtrip(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.delenv("SPACES_PREFIX", raising=False)
    src_store = str(tmp_path / "src")
    reg = _make_bundle(src_store, "v9")
    reg.push_active(client=fake_s3)

    dst_store = str(tmp_path / "dst")
    reg2 = Registry(store_dir=dst_store)
    pulled = reg2.pull(client=fake_s3)
    assert pulled == "v9"
    assert reg2.get_active() == "v9"
    assert reg2.verify_integrity("v9") is True
    assert reg2.verify_active_chain() is True
    for name in BUNDLE_FILES:
        assert os.path.exists(os.path.join(dst_store, "v9", name))


def test_pull_rejects_tampered_file(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.delenv("SPACES_PREFIX", raising=False)
    src_store = str(tmp_path / "src")
    _make_bundle(src_store, "v9").push_active(client=fake_s3)

    # Flip one uploaded file's bytes in the fake store.
    key = "voiceguard/model_store/bundles/v9/xgb.json"
    fake_s3.store[("vg-bucket", key)] = b"TAMPERED"

    dst_store = str(tmp_path / "dst")
    reg2 = Registry(store_dir=dst_store)
    with pytest.raises(BundleError):
        reg2.pull(client=fake_s3)
    assert reg2.get_active() is None            # never activated a corrupt bundle


def test_pull_no_pointer_raises(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    reg = Registry(store_dir=str(tmp_path / "dst"))
    with pytest.raises(BundleError):
        reg.pull(client=fake_s3)                # empty remote, no version given
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bundle_registry.py -q -k "push or pull"`
Expected: FAIL — `AttributeError: 'Registry' object has no attribute 'push_active'`.

- [ ] **Step 3: Add `import remote_store` and the three methods** to `bundle_registry.py`

At the top, alongside `from governance import ...`:

```python
import remote_store
```

Add these methods to the `Registry` class (e.g. after `verify_active_chain`):

```python
    # ── remote store (DigitalOcean Spaces) ──
    def push(self, version, client=None):
        """Upload bundle <version> (7 files + bundle.json) to Spaces and refresh the remote
        ACTIVE.json + registry.jsonl. Verifies local integrity first — never push a corrupt
        bundle."""
        entry = self.get_bundle(version)
        if entry is None:
            raise BundleError(f"cannot push unregistered version: {version}")
        if not self.verify_integrity(version):
            raise BundleError(f"integrity check failed for {version}; refusing to push")
        if client is None:
            client = remote_store.make_client()
        bucket, prefix = remote_store.bucket_prefix()
        for name in sorted(BUNDLE_FILES | {"bundle.json"}):
            remote_store.upload_file(client, bucket, f"{prefix}/bundles/{version}/{name}",
                                     os.path.join(entry["dir"], name))
        if os.path.exists(self.active_path):
            remote_store.upload_file(client, bucket, f"{prefix}/ACTIVE.json", self.active_path)
        if os.path.exists(self.registry_path):
            remote_store.upload_file(client, bucket, f"{prefix}/registry.jsonl", self.registry_path)
        return version

    def push_active(self, client=None):
        active = self.get_active()
        if active is None:
            raise BundleError("no active version to push")
        return self.push(active, client=client)

    def pull(self, version=None, client=None):
        """Download a bundle from Spaces into store_dir, verify every file's sha256 against its
        manifest, register it locally (correct local dir), and — when it is the remote-active
        version — write ACTIVE.json verbatim so the tamper-evident chain still verifies.
        Fails closed: a sha256 mismatch raises BundleError and never activates the bundle."""
        if client is None:
            client = remote_store.make_client()
        bucket, prefix = remote_store.bucket_prefix()

        active_bytes = remote_store.download_bytes(client, bucket, f"{prefix}/ACTIVE.json")
        remote_active = json.loads(active_bytes)[-1]["version"] if active_bytes else None
        if version is None:
            if remote_active is None:
                raise BundleError("remote store has no active pointer (ACTIVE.json); pass a version")
            version = remote_active

        bundle_dir = os.path.join(self.store_dir, version)
        os.makedirs(bundle_dir, exist_ok=True)
        for name in sorted(BUNDLE_FILES | {"bundle.json"}):
            remote_store.download_file(client, bucket, f"{prefix}/bundles/{version}/{name}",
                                       os.path.join(bundle_dir, name))
        validate_bundle(bundle_dir)                        # raises BundleError on sha256 mismatch

        if self.get_bundle(version) is None:
            self.register_bundle(bundle_dir)               # local entry with correct local dir
        if active_bytes is not None and version == remote_active:
            with open(self.active_path, "wb") as f:        # verbatim: keeps verify_active_chain valid
                f.write(active_bytes)
        return version
```

- [ ] **Step 4: Add the CLI subcommands** in `_build_cli()` (after the `verify` parser):

```python
    ps = sub.add_parser("push"); ps.add_argument("version", nargs="?"); ps.add_argument("--active", action="store_true")
    pl = sub.add_parser("pull"); pl.add_argument("version", nargs="?"); pl.add_argument("--active", action="store_true")
```

And in `main()` (after the `verify` branch):

```python
    elif args.cmd == "push":
        v = reg.push_active() if (args.active or not args.version) else reg.push(args.version)
        print("pushed:", v)
    elif args.cmd == "pull":
        v = reg.pull(None if (args.active or not args.version) else args.version)
        print("pulled:", v, "(active)" if reg.get_active() == v else "")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_bundle_registry.py -q`
Expected: PASS (existing registry tests + the 3 new push/pull tests).

- [ ] **Step 6: Run the fast suite; report DONE (do not commit)**

Run: `VOICEGUARD_CI_FAST=1 python -m pytest tests/test_bundle_registry.py tests/test_remote_store.py tests/test_tracking.py -q`
Expected: PASS. Report DONE with the summary. Controller snapshots; user commits.

---

### Task 3: Test tiering — `pytest.ini`, `collect_ignore`, `weights` markers

**Files:**
- Create: `pytest.ini`
- Modify: `tests/conftest.py` (prepend env-gated `collect_ignore`)
- Modify: `tests/test_api.py`, `tests/test_detector.py`, `tests/test_worker.py`, `tests/test_golden.py` (add `pytestmark = pytest.mark.weights`)

**Interfaces:**
- Consumes: the existing `tests/conftest.py` from Task 1.
- Produces: `VOICEGUARD_CI_FAST=1 pytest` collects only weights-free modules; plain `pytest` collects everything. The `weights` marker is registered (no unknown-marker warnings).

- [ ] **Step 1: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    weights: tests that load model weights (need the bundle pulled from DigitalOcean Spaces)
```

- [ ] **Step 2: Prepend the env-gated `collect_ignore` to `tests/conftest.py`**

Add at the very top of `tests/conftest.py` (paths are relative to the `tests/` dir):

```python
import os as _os

# On a weights-free runner (the CI "fast" tier), these modules must not even be IMPORTED:
# they import detector, which loads ~380MB of weights at import. pytest imports every module
# at collection time, so a marker deselection is not enough — we skip collection entirely.
collect_ignore = []
if _os.environ.get("VOICEGUARD_CI_FAST"):
    collect_ignore = ["test_api.py", "test_detector.py", "test_worker.py", "test_golden.py"]
```

(Leave the existing `FakeS3` / `fake_s3` fixture below it unchanged.)

- [ ] **Step 3: Mark the four model-loading modules**

In each of `tests/test_api.py`, `tests/test_detector.py`, `tests/test_worker.py`, `tests/test_golden.py`, ensure `import pytest` is present and add a module-level marker near the top (after imports):

```python
import pytest
pytestmark = pytest.mark.weights
```

- [ ] **Step 4: Verify the fast tier excludes the weights modules**

Run: `VOICEGUARD_CI_FAST=1 python -m pytest --collect-only -q`
Expected: the listing contains `test_jobs`, `test_auth`, `test_bundle_registry`, `test_tracking`, `test_loadtest`, `test_remote_store` and does **NOT** contain `test_api`, `test_detector`, `test_worker`, `test_golden`.

Run: `python -m pytest --collect-only -q`  (no env)
Expected: the listing **does** include `test_api`, `test_detector`, `test_worker`, `test_golden` (all modules collected).

- [ ] **Step 5: Run the fast tier end-to-end; report DONE (do not commit)**

Run: `VOICEGUARD_CI_FAST=1 python -m pytest -q`
Expected: PASS — only the weights-free modules run, no model weights loaded, no unknown-marker warnings. Report DONE with the summary. Controller snapshots; user commits.

---

### Task 4: GitHub Actions workflow + operator docs

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `docs/CI-and-model-store.md`

**Interfaces:**
- Consumes: `VOICEGUARD_CI_FAST` (Task 3), `python bundle_registry.py pull --active` (Task 2), the 5 `SPACES_*` GitHub secrets.
- Produces: the two-tier CI pipeline and the operator runbook for populating Spaces + configuring secrets.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  push:
  pull_request:
  workflow_dispatch:
  schedule:
    - cron: "0 3 * * *"            # nightly 03:00 UTC = 04:00 WAT

jobs:
  fast:                            # every push + PR; no secrets, no weights
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - run: python -m pytest -q
        env:
          VOICEGUARD_CI_FAST: "1"

  weights:                         # model tests where the bundle can be pulled
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
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Cache model bundle
        uses: actions/cache@v4
        with:
          path: model_store
          key: bundle-${{ github.sha }}
          restore-keys: bundle-
      - name: Pull active bundle from Spaces
        run: python bundle_registry.py pull --active
        env:
          SPACES_KEY: ${{ secrets.SPACES_KEY }}
          SPACES_SECRET: ${{ secrets.SPACES_SECRET }}
          SPACES_ENDPOINT: ${{ secrets.SPACES_ENDPOINT }}
          SPACES_REGION: ${{ secrets.SPACES_REGION }}
          SPACES_BUCKET: ${{ secrets.SPACES_BUCKET }}
      - run: python -m pytest -q            # full suite incl. golden regression
```

- [ ] **Step 2: Verify the workflow YAML parses**

Run: `python -c "import yaml, sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"`
Expected: `yaml ok`. (PyYAML ships transitively via mlflow; if it is somehow absent, `pip install pyyaml` first — it is not added to `requirements.txt`.)

- [ ] **Step 3: Create `docs/CI-and-model-store.md`**

```markdown
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
```

- [ ] **Step 4: Confirm files exist and report DONE (do not commit)**

Run: `ls .github/workflows/ci.yml docs/CI-and-model-store.md`
Run: `VOICEGUARD_CI_FAST=1 python -m pytest -q`  (final fast-suite sanity)
Expected: both files listed; fast suite PASS. Report DONE. Controller snapshots; user commits.

---

## Notes for the executor

- The `weights` CI job and the real Spaces pull are verified **on GitHub** by the user after the
  bucket is populated and secrets are set — they cannot run on the Windows dev box. Local
  verification covers everything else: `remote_store` + registry `push`/`pull` against the fake,
  the tiering (`collect_ignore`), and YAML validity.
- `boto3` is the only new dependency and is import-isolated to `remote_store.py`.
- Do not commit — the controller snapshots each task; the user commits.

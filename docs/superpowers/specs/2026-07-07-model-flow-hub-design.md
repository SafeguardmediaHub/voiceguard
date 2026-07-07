# Model-Flow Hub — Design Spec

- **Date:** 2026-07-07
- **Status:** Approved (design); pending implementation plan
- **Scope:** Sub-project 1 of the VoiceGuard MLOps effort (retrain → drift → deploy pipeline)
- **Deployment target:** DigitalOcean droplet (CPU); retrain runs off-box (Kaggle now, scriptable GPU VM later)

---

## 1. Context & Problem

VoiceGuard V9 is served by `server.py`, which **hardcodes** the paths of its seven
model artifacts (`aasist_v9_best.pt`, `wav2vec_v9_best.pt`, `rawnet3.pt`,
`xgb_v9.json`, `cal_v9_params.json`, `thresholds_v9.json`, `lcnn_screener_v9.pt`).
Consequences:

- **Deploying a new model means editing code** and restarting by hand — no record
  of what is live, no reversible history.
- The existing `governance.ModelRegistry` is append-only and can record checkpoints,
  but has **no "active/production" pointer** and no promote/rollback.
- There is **no experiment tracking** (metrics live in scattered JSON), **no git**,
  and **no CI**.
- The seven artifacts are **co-calibrated**: `xgb_v9`/`cal_v9` are fit on specific
  component-model outputs, `thresholds_v9` are tuned to the set, and the
  peak-normalization convention differs by stage (LCNN raw / ensemble peak-normed).
  They cannot be swapped individually without breaking calibration.

**Goal:** a CPU-only control plane where deploying a new model is
*register → validate → promote → restart* — never editing code — with a recorded,
one-command-reversible history and every run tracked in MLflow.

## 2. Goals / Non-Goals

**Goals (this sub-project):**

1. A **bundle** abstraction: the deployable unit is the full set of seven artifacts
   plus a manifest, versioned and promoted atomically.
2. A **registry** with an explicit active-production pointer plus `promote` /
   `rollback`, tamper-evident (hash-chained), extending `governance.py`.
3. **MLflow experiment tracking** (Phase 1): each bundle's params/metrics/tags
   logged; large binaries stay out of MLflow.
4. `server.py` loads the **active** bundle from the registry (with a safe fallback),
   and reports the active version on `/ping`.
5. **git + `.gitignore`** initialized so code/specs are versioned and large binaries
   are never committed.
6. Tests: golden test runs against the *active* bundle; unit tests for the
   registry/promote/rollback and the startup fallback.

**Non-Goals (handled by later sub-projects, explicitly out of scope here):**

- Retrain job orchestration and the automated validation gate (sub-project 3).
- Porting the drift monitor to V9 and wiring the real dashboard (sub-project 2).
- CI runners and FPR-by-demographic monitoring (sub-projects 4 / 2).
- Object storage: bundles live in a local `model_store/` directory now; the
  registry records paths + hashes so the store is swappable to DO Spaces/S3 later.
- Authentication/RBAC: promotion is a local CLI + OS-level restart, so no network
  auth surface is added in this sub-project.
- Hot-reload endpoint: rejected in favor of promote-and-restart (see §6).

## 3. The Bundle

**Definition.** A bundle is one immutable, versioned directory holding everything
needed to serve a model generation, promoted as a single atomic unit.

**Layout:**

```
model_store/
  <version>/                      # e.g. v9  (or v9.1, v10, …)
    aasist.pt
    wav2vec.pt
    rawnet.pt
    xgb.json
    cal.json
    thresholds.json
    lcnn.pt
    bundle.json                   # manifest (below)
  ACTIVE.json                     # append-only active-pointer log (see §4)
  registry.jsonl                  # append-only bundle registry (see §4)
```

**`bundle.json` schema:**

```json
{
  "version": "v9",
  "created_at": "2026-07-07T12:00:00Z",
  "git_sha": "<commit that produced/registered this>",
  "mlflow_run_id": "<run id, or null>",
  "files": {
    "aasist.pt":       {"sha256": "…", "role": "component"},
    "wav2vec.pt":      {"sha256": "…", "role": "component"},
    "rawnet.pt":       {"sha256": "…", "role": "component"},
    "xgb.json":        {"sha256": "…", "role": "fusion"},
    "cal.json":        {"sha256": "…", "role": "calibration"},
    "thresholds.json": {"sha256": "…", "role": "thresholds"},
    "lcnn.pt":         {"sha256": "…", "role": "screener"}
  },
  "metrics": {
    "val_eer": 0.1325, "studio_fp": 0.12, "noizai_catch": 0.833,
    "per_language_fpr": {"yoruba": 0.02, "igbo": 0.04, "pidgin": 0.0, "arabic": 0.16}
  },
  "train_manifest_hash": "<sha256 of the training manifest>",
  "preprocessing": {
    "ensemble_peak_norm": true,
    "lcnn_peak_norm": false,
    "notes": "LCNN mel is per-sample standardized (scale-invariant); ensemble peak-normalizes."
  },
  "cascade": {"low_thresh": 0.20, "high_thresh": 0.80,
              "platt": {"coef": 10.931, "intercept": -4.148}},
  "verdict_thresholds": {"auto_fake": 0.85, "likely_fake": 0.55, "to_review": 0.30}
}
```

**Atomicity rationale.** Because of co-calibration and the stage-specific
peak-norm convention (both discovered during V9 integration), a bundle is promoted
whole or not at all. The `preprocessing` block travels *with* the model so the
train/serve convention can never drift from the weights again.

## 4. Registry (`registry.py`, extends `governance.py`)

A thin layer over the existing `governance.ModelRegistry` / `AuditLog` patterns.
All state is append-only and hash-chained for tamper-evidence.

**Storage:**

- `model_store/registry.jsonl` — one line per registered bundle: version,
  `created_at`, per-file sha256, path, `bundle.json` snapshot.
- `model_store/ACTIVE.json` — append-only pointer log. Each entry:
  `{seq, version, activated_at, actor, reason, prev_version, entry_sha, prev_sha}`.
  The current active bundle is the last entry; history + hash-chain give an
  auditable, reversible record.

**API:**

| Function | Behavior |
|---|---|
| `register_bundle(dir) -> version` | Validate the dir has all 7 files + `bundle.json`; recompute and verify per-file sha256 against the manifest; reject incomplete bundles and duplicate versions; append to `registry.jsonl`. Does **not** activate. |
| `list_bundles() -> [entry]` | All registered bundles, newest first. |
| `get_bundle(version) -> entry` | Registry entry for a version. |
| `get_active() -> version \| None` | Current active version (last `ACTIVE.json` entry). |
| `promote(version, actor, reason)` | Run `verify_integrity(version)`; if any file changed since registration, **refuse**. Else append a new `ACTIVE.json` entry. Does not restart the server (see §6). |
| `rollback(actor, reason)` | Append an `ACTIVE.json` entry re-pointing to `prev_version` of the current entry. Refuses if there is no prior entry. |
| `verify_integrity(version) -> bool` | Recompute sha256 of every file vs the registered manifest (reuses governance logic). |

`register_bundle` and `promote` also append to the governance `AuditLog` so model
lifecycle events share the existing tamper-evident chain.

## 5. Experiment Tracking (`tracking.py`, MLflow wrapper)

A thin helper so callers never touch the MLflow API directly.

- `log_bundle(bundle_dir, run_name=None) -> run_id` — start/lookup an MLflow run;
  log **params** (arch/preproc flags, cascade band, thresholds), **metrics**
  (val_eer, studio_fp, noizai_catch, per-language FPR), and **tags**
  (version, git_sha, train_manifest_hash). Log the small `bundle.json` as an
  artifact and the `model_store/<version>/` path as a tag/param — **not** the
  weight blobs. Returns the run id, which is written back into `bundle.json`.
- **What is deliberately NOT logged to MLflow:** the `.pt`/`.json` weight files
  (up to ~378 MB each). MLflow tracks metrics + references; `model_store/` owns the
  binaries. This keeps the MLflow artifact store small and DO-friendly.

**MLflow deployment:** a local `mlflow server` with a SQLite backend
(`mlflow.db`) and a filesystem artifact root (`./mlruns`). Configured via
`MLFLOW_TRACKING_URI` (default `http://127.0.0.1:5000`, falling back to a local
`./mlruns` file store if the server is down, so tracking never blocks a promote).
Documented for the DO droplet (run under the same process manager as the server).

## 6. Server Integration (`server.py`)

- **Startup resolution.** On boot, resolve the active bundle via
  `registry.get_active()` and load those seven paths. If the registry does not yet
  exist (pre-migration), fall back to today's hardcoded `models/*_v9*` paths so the
  server keeps working during rollout.
- **Startup smoke-check (the safety property).** After loading the active bundle,
  score a bundled golden clip and assert the verdict/score matches the value
  recorded for that bundle (within tolerance). If loading or the smoke-check fails:
  log `CRITICAL`, fall back to the **previous** active pointer entry, and retry. If
  the fallback also fails, **refuse to start** (fail closed) with a clear message.
  This replaces the rejected in-process hot-swap with an equally safe boot path.
- **`/ping`** reports `active_version` and its short sha alongside the existing
  cascade/module fields.
- **Applying a promotion = restart.** `promote`/`rollback` only move the pointer.
  A new model goes live when the server process restarts:
  - **DO:** a `systemd` unit (e.g. `voiceguard.service`); apply with
    `systemctl restart voiceguard`. The `promote` CLI accepts an optional
    `--restart` that runs an env-configured `VOICEGUARD_RESTART_CMD`.
  - **Local:** re-run `python server.py`.

## 7. Git & Repo Hygiene

- `git init`; add a `.gitignore` **before** the first `git add`.
- **Ignore:** `*.pt`, `*.onnx`, `*.rar`, `*.wav`, `*.m4a`, `*.mp3` (except the small
  tracked golden clips under `tests/golden_clips/`), `model_store/*/` binaries
  (track `bundle.json` and the pointer/registry logs, ignore the weights),
  `mlruns/`, `mlflow.db`, `output/`, `__pycache__/`, large data dirs
  (`data/`, `cv-corpus-*/`, `bias_audit*/`, `studio_clips/`, `v8_finetune_combined/`,
  `adapted_checkpoints/`), and the nested duplicate `voice_guard 0ffline/`.
- **Track:** all `*.py`, `docs/`, `tests/` (incl. golden clips + manifest),
  `requirements.txt`, `.gitignore`, and per-bundle `bundle.json` manifests.
- First commits: (1) `.gitignore` + this spec; (2) code baseline. Use explicit
  paths on the first `git add` (never `git add .`) so no large blob is ever staged.

## 8. Data Flow (end-to-end)

```
Kaggle retrain job
  └─ produces model_store/<version>/ (7 files + bundle.json)
        └─ register_bundle(dir)            → registry.jsonl  (candidate, inactive)
              └─ tracking.log_bundle(dir)  → MLflow run (metrics + refs)
                    └─ [validation gate — sub-project 3]
                          └─ promote(version)   → ACTIVE.json (new pointer)
                                └─ restart server → active bundle live
Rollback: rollback() → ACTIVE.json (prev) → restart server
```

## 9. Error Handling

- `register_bundle`: reject missing files, manifest/hash mismatch, or duplicate
  version — with a specific message naming the offending file.
- `promote`: refuse if `verify_integrity` fails (a registered file changed on disk).
- Server startup: missing/corrupt active bundle → fall back to previous active and
  log `CRITICAL`; both failing → fail closed.
- `tracking.log_bundle`: MLflow unreachable → log a warning and fall back to the
  local file store; never block a register/promote on tracking being down.

## 10. Testing

- **Golden test (extend):** after `promote`/`rollback`, the golden clips must
  produce the pinned verdicts **for the active bundle** — this guards the
  registry-driven load path, not just the hardcoded one.
- **Unit tests (new):**
  - register → promote → `get_active` → rollback round-trip.
  - `promote` refuses on integrity failure (tamper a file, expect refusal).
  - `register_bundle` rejects an incomplete bundle (missing one of the 7 files).
  - Startup fallback: point ACTIVE at a deliberately broken bundle; the server must
    fall back to the previous active and stay up (fail-safe), not crash serving.
  - `ACTIVE.json` hash-chain verifies after a promote/rollback sequence.

## 11. How Later Sub-Projects Plug In

- **Sub-project 2 (drift monitor / dashboard):** the V9 monitor loads `get_active()`
  instead of hardcoded V8 paths, and logs each run to MLflow. The dashboard reads
  MLflow + `drift_log.jsonl`.
- **Sub-project 3 (retrain + gate):** the retrain job emits a bundle dir; the gate
  calls `register_bundle` + `tracking.log_bundle`, compares candidate metrics to the
  active bundle, and only then calls `promote`.
- **Sub-project 4 (CI):** runs the golden + unit tests on every change.

## 12. Assumptions

- One model "family" is active at a time (no multi-tenant serving).
- `model_store/` is local now; path+hash indirection makes object storage a later
  drop-in with no API change.
- A migration step packages the current V9 files in `models/` into
  `model_store/v9/` and registers+promotes it as the first active bundle.

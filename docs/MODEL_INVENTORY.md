# VoiceGuard — Model Inventory Register

**Document ID:** VG-DOC-004
**Version:** 1.0
**Date:** 2026-07-22
**Owner:** Michael Ologungbara
**Classification:** Confidential — internal, audit, and customer due diligence
**Status:** Baseline issue. Integrity verified 2026-07-22. **§9 records two provenance gaps affecting the deployed screener.**
**Review cycle:** On every bundle registration or promotion. Full re-verification quarterly.
**Related:** Model Development History (VG-DOC-001) · Model Card (VG-DOC-002) · Dataset Inventory (VG-DOC-003) · GRC Control & Risk Pack (VG-DOC-005, pending)

---

## 1. Purpose and scope

This is the authoritative register of every model artifact in the VoiceGuard system: what exists, what is deployed, what each artifact is derived from, its cryptographic identity, and its lifecycle status.

It answers the questions an auditor asks about AI assets:

1. **What models do you run in production, exactly?** — identity by hash, not by name
2. **Can you prove the deployed artifact is the one you approved?** — integrity verification
3. **Where did each artifact come from?** — provenance to a source checkpoint and a training run
4. **Who approved its deployment, and when?** — promotion record
5. **Can you get back to a known-good state?** — rollback capability

It supports ISO/IEC 42001 (AI system resource documentation, lifecycle management, change control) and ISO/IEC 27001 (information asset inventory, integrity controls).

**Scope:** all model artifacts under `model_store/` and `models/`. Excludes training data (VG-DOC-003) and application code.

---

## 2. Verification procedure

Every hash in this register is reproducible. To re-verify the deployed bundle:

```bash
python bundle_registry.py active        # -> the active version
python bundle_registry.py verify v9h    # -> OK, or names the failing file
```

`verify` recomputes SHA-256 for every file in the bundle and compares against `bundle.json`. Verification also runs automatically on `pull`, so a corrupted or substituted artifact cannot enter a build.

**Verification evidence, this issue:**

| Check | Command | Result | Date |
|---|---|---|---|
| Active bundle | `bundle_registry.py active` | `v9h` | 2026-07-22 |
| Integrity — all 7 files | `bundle_registry.py verify v9h` | **OK** | 2026-07-22 |
| Golden regression | `pytest tests/test_golden.py` | **1 passed** | 2026-07-22 |
| Promotion chain | `ACTIVE.json` hash chain | 4 entries, intact | 2026-07-22 |

---

## 3. Register A — deployable bundles

A *bundle* is the unit of deployment: 7 files, individually hashed, promoted and rolled back atomically.

### A1. `v9h` — **ACTIVE / PRODUCTION**

| Field | Value |
|---|---|
| **Status** | **ACTIVE** |
| Registered | 2026-07-12T12:06:08 UTC |
| Activated | 2026-07-12T12:06:11 UTC (seq 1); **reconfirmed 2026-07-14T16:54:20** (seq 3) |
| Promoted by | actor `route-b`; reconfirmed by actor `eval` |
| Recorded reason | "V8 AASIST + hybrid fusion + CASCADE_LOW 0.10" |
| Bundle notes | "Route B: V8 AASIST + hybrid XGBoost fusion (xgb/cal_v9_hybrid); CASCADE_LOW 0.20→0.10." |
| Total size | ~405 MB |
| Cascade | `low_thresh` 0.10 · `high_thresh` 0.80 |
| Verdict thresholds | `auto_fake` 0.85 · `likely_fake` 0.55 · `to_review` 0.30 |
| Integrity | **Verified OK**, 2026-07-22 |

**Full SHA-256 manifest** — the cryptographic definition of what is in production:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `aasist.pt` | 1,194,519 | `bf6fb21a18dbf4f12b6d5fae32aade92ad28e4a7f26f989f5c29356900bc5c88` |
| `cal.json` | 66 | `504795f1b54bf5bad1030e693c36afa2745e512726eed30b2313d7debb7344a3` |
| `lcnn.pt` | 1,164,747 | `f0f920043de1f4206c5192a4f4c06fe30908e38365409a71c6b21bec5104b3b8` |
| `rawnet.pt` | 24,401,380 | `8d8fba7687bdaf37467a64c53660846fa2c056e14995a39e66b7041d6994560f` |
| `thresholds.json` | 66 | `ea17384d9b7049712c48dbb01a0e5775e84273552605938b01ca0035e5f77cab` |
| `wav2vec.pt` | 378,431,557 | `37c916e8aa13f52a08f4599573ba380bb85c6dc58562602738bf2bd63e4eb8fc` |
| `xgb.json` | 111,777 | `446a768aaa0f7e1308e2f18ae1bbe616bd3ce5652e7646bf434dd1b5f5685064` |

### A2. `v9` — SUPERSEDED

| Field | Value |
|---|---|
| **Status** | Superseded (retained for rollback) |
| Registered / activated | 2026-07-07T10:23:53 / 10:23:55 UTC |
| Promoted by | actor `migration` — "initial V9 bundle" |
| Cascade | `low_thresh` **0.20** · `high_thresh` 0.80 |
| **Known defect** | Its `aasist.pt` is the collapsed V9 retrain — standalone **AUC 0.5258** (VG-DOC-001 §12.1) |

⚠️ **Do not roll back to `v9` without accepting the AASIST collapse.** The registry permits it; this register records why it should not be done. Rolling back would restore a bundle in which one of three ensemble members contributes no signal.

### A3. `v9fixed` — REGISTERED, EVALUATED, REJECTED

| Field | Value |
|---|---|
| **Status** | **Rejected** — promoted then rolled back |
| Registered / activated | 2026-07-14T13:28:43 / 13:28:47 UTC (seq 2) |
| **Deactivated** | 2026-07-14T16:54:20 UTC (seq 3) — **3h 26m in production state** |
| Promoted by | actor `aasist-retrain` — "proper AASIST retrain + refit — candidate" |
| Rejected by | actor `eval` — "v9fixed no better than v9h; keep v9h" |
| Bundle notes | "Proper AASIST retrain (aasist_v9_fixed, V8-arch, balanced) + refitted fusion (xgb/cal_v9_fixed); lcnn CASCADE_LOW 0.10 from v9h." |

This bundle contains the **corrected** AASIST retrain — class-balanced, V8 architecture, per the recipe in `docs/AASIST-V9-retrain-recipe.md`. It is technically the "right" fix for the V9 collapse, and it did not outperform the `v9h` workaround. Retained, registered, and available.

**No quantitative comparison was recorded** — the rejection reason is qualitative ("no better than"). See §10 item 3.

---

## 4. Cross-bundle change matrix

Derived by comparing SHA-256 across all three bundle manifests. This proves precisely what changed between versions — and what did not.

| File | `v9` | `v9h` | `v9fixed` | Verdict |
|---|---|---|---|---|
| `rawnet.pt` | `8d8fba76…` | `8d8fba76…` | `8d8fba76…` | **IDENTICAL — never changed** |
| `wav2vec.pt` | `37c916e8…` | `37c916e8…` | `37c916e8…` | **IDENTICAL — never changed** |
| `thresholds.json` | `ea17384d…` | `ea17384d…` | `ea17384d…` | **IDENTICAL — never changed** |
| `lcnn.pt` | `ea5db300…` | `f0f92004…` | `f0f92004…` | 2 distinct — changed at `v9h`, unchanged since |
| `aasist.pt` | `7c7f4536…` | `bf6fb21a…` | `c5872bdc…` | **3 distinct — different in every bundle** |
| `cal.json` | `fd4cad2a…` | `504795f1…` | `512286d7…` | 3 distinct |
| `xgb.json` | `74cd7feb…` | `446a768a…` | `ce1eff1a…` | 3 distinct |

**What this independently confirms:**

1. **RawNet3 has never been retrained** — byte-identical across every bundle, corroborating VG-DOC-001 §9.2 with cryptographic evidence rather than prose.
2. **Wav2Vec2 V9 has been stable** since the V9 retrain — the same weights serve all three bundles.
3. **Verdict thresholds have never changed** since V8 — 0.85/0.55/0.30 are constant across the entire bundle history.
4. **AASIST is the axis of variation.** Every bundle carries a different AASIST, and each fusion (`xgb.json`, `cal.json`) was refit to match — exactly as the recipe requires (VG-DOC-001 §12.2, §4 of the recipe).
5. **`v9h` and `v9fixed` share an LCNN**, confirming the `CASCADE_LOW` 0.10 change was made once at `v9h` and carried forward.

---

## 5. Register B — deployed component detail

Every artifact in the **active** bundle, with role and provenance.

| # | Artifact | Role | Version | Params | Provenance |
|---|---|---|---|---|---|
| B1 | `lcnn.pt` | Stage-1 CPU screener | V9 (retuned) | 287,234 | ⚠️ **Unresolved — §9.1** |
| B2 | `aasist.pt` | Ensemble — spectral/graph-attention | **V8** | ~285 K | `models/aasist_v8.pt` **[hash-confirmed]** |
| B3 | `wav2vec.pt` | Ensemble — SSL representation | V9 | ~95 M frozen + 213 K trainable | V9 head retrain |
| B4 | `rawnet.pt` | Ensemble — raw waveform | **V8** | ~6 M | `models/rawnet3.pt` **[hash-confirmed]** |
| B5 | `xgb.json` | Fusion | v9 hybrid | — | `models/hybrid_ensemble_config.json` refit |
| B6 | `cal.json` | Platt calibration | v9 hybrid | — | `models/cal_v9_hybrid.json` |
| B7 | `thresholds.json` | Verdict banding | V8-era, unchanged | — | Constant since V8 |

**Ensemble fusion weights (V9 reference):** AASIST 0.402 · Wav2Vec2 0.445 · RawNet3 0.153.

**Operational note on B1.** `detector.py:528-529` reads `CASCADE_LOW` / `CASCADE_HIGH` **from inside the LCNN checkpoint** (`_lcnn_ck["cascade_thresholds"]`), not from a config file. Cascade routing is therefore a property of the model artifact itself — which is why changing the early-out threshold at `v9h` required a **new** `lcnn.pt` rather than a config edit. Good for reproducibility (routing travels with the model, hashed); a trap for anyone expecting to retune routing without re-bundling.

---

## 6. Register C — source checkpoints (`models/`)

Working artifacts from which bundles are assembled. **Not deployed directly; not integrity-managed by the registry.**

| Artifact | Size | Status | Used in |
|---|---|---|---|
| `aasist_v8.pt` | 1,194,519 | **In production** | `v9h` B2 **[hash-confirmed]** |
| `aasist_v9_best.pt` | 3,530,616 | Superseded — collapsed | `v9` **[hash-confirmed]** |
| `aasist_v9_fixed.pt` | 1,198,557 | Candidate — rejected | `v9fixed` **[hash-confirmed]** |
| `rawnet3.pt` | 24,401,380 | **In production** | all bundles **[hash-confirmed]** |
| `lcnn_screener_v9.pt` | 1,167,583 | Superseded | `v9` **[hash-confirmed]** |
| `lcnn_student_v9.pt` | 1,167,544 | **Not deployed** | — see §9.2 |
| `cal_v8_params.json` | 68 | Historical | V8 |
| `cal_v9_params.json` | 67 | Superseded | `v9` |
| `cal_v9_hybrid.json` | 66 | **In production** | `v9h` B6 |
| `cal_v9_fixed.json` | 66 | Rejected | `v9fixed` |
| `hybrid_ensemble_config.json` | 691 | **In production** | `v9h` fusion config |

**Note the AASIST size anomaly.** `aasist_v9_best.pt` is 3,530,616 bytes against ~1,195,000 for the V8 and fixed variants — roughly 3×, for the same architecture. This is consistent with the collapsed checkpoint carrying optimizer state that the others do not. Not itself a defect, but a useful tell: **bundle file sizes are not uniform for the same component**, so size is not a substitute for hash identity.

---

## 7. Register D — supporting artifacts

| Artifact | Size | Purpose | Status |
|---|---|---|---|
| `teacher_scores_v9_train.json` | 1,313,428 | V9 ensemble soft labels, 4,720 entries | Retained |
| `ensemble_scores_v9.json` | 165,968 | Ensemble feature scores | Retained |
| `cascade_bias_audit_results.json` | 260,154 | Per-sample cascade bias-audit results | Retained |
| `ablation_study.json` | 1,810 | Component ablation | Retained |
| **`lcnn_v9_results.json`** | **0 bytes** | LCNN training results + calibration | ⚠️ **EMPTY — data lost (§9.3)** |
| `tests/golden_manifest.json` | — | Pinned regression baseline, 4 clips | **Active control** |
| `drift_baseline.json` | 2,730 | Drift-monitor baseline | Active |
| `model_store/ACTIVE.json` | 1,492 | Hash-chained promotion log | **Active control** |
| `model_store/registry.jsonl` | 6,008 | Append-only bundle registry | **Active control** |

---

## 8. Integrity and lifecycle controls

| Control | Implementation | Status |
|---|---|---|
| **Per-file integrity** | SHA-256 per file in `bundle.json`; recomputed on `verify` and on `pull` | ✅ Verified 2026-07-22 |
| **Tamper-evident promotion log** | `ACTIVE.json` — hash-chained (`prev_sha` → `entry_sha`), append-only | ✅ 4 entries intact |
| **Append-only registry** | `registry.jsonl` | ✅ 3 bundles |
| **Atomic pointer write** | `ACTIVE.json` written temp + `os.replace` | ✅ |
| **Fail-closed startup** | `api.py` lifespan → `detector.startup_check()`; refuses to boot if the active bundle cannot classify its fixture | ✅ |
| **Deterministic inference** | Same file → same score; pinned by `tests/test_golden.py` | ✅ 1 passed |
| **Rollback** | `bundle_registry.py rollback` — one command | ✅ Exercised in production (seq 3) |
| **Immutable deployment** | Bundle baked into a SHA-tagged container image | ✅ (VG-DOC deployment plan) |
| **Per-sub-model health check** | — | ❌ **ABSENT — see §9.4** |
| **Named human approver** | — | ❌ **ABSENT — see §9.5** |

**The rollback control has been exercised under real conditions**, not merely tested: `v9fixed` was promoted and rolled back within 3h 26m, with both transitions recorded. That is stronger evidence than a documented procedure.

---

## 9. Provenance and control gaps

> **Update 2026-07-23:** §9.1 and §9.4 have since been **resolved** — see the resolution notes inline below. Retained here in full because they document real gaps that existed and how they were closed (evidence for the fix, and for VG-DOC-006 X2/X3).

### 9.1 🔴→✅ The deployed screener cannot be traced to a source checkpoint

`v9h`'s `lcnn.pt` (`f0f92004…`, 1,164,747 bytes) **matches no artifact in `models/`**. The nearest candidate, `lcnn_screener_v9.pt`, hashes to `ea5db300…` and is 1,167,583 bytes — that is `v9`'s LCNN, not `v9h`'s.

The `v9h` checkpoint was evidently produced by rewriting `cascade_thresholds.low_thresh` from 0.20 to 0.10 inside the checkpoint during bundling, but **no script, notebook cell, or source artifact recording that transformation is retained.**

**Why this matters:** the LCNN screener resolves **~86% of all production traffic**. The artifact making the large majority of production decisions has:

- no reproducible derivation from a retained source,
- no recorded training run of its own (it inherits `v9`'s training, with an undocumented post-hoc edit),
- and, per VG-DOC-001 §16.6, a headline EER that is a best-of-50 pick from an unstable run.

An auditor asking *"show me how you produced the model that makes 86% of your decisions"* cannot currently be answered. **This is the most significant gap in this register.**

**Remediation:** write and commit a `retune_cascade.py` that takes a source checkpoint plus new thresholds and emits a new checkpoint deterministically; re-derive `v9h`'s LCNN from `lcnn_screener_v9.pt` and confirm it reproduces `f0f92004…`. If it does not reproduce, the provenance question becomes materially more serious.

> **✅ RESOLVED 2026-07-23.** `scripts/retune_cascade.py` was written and run. Result: `v9h`'s `lcnn.pt` is **content-identical** to `models/lcnn_screener_v9.pt` retuned to `low_thresh=0.10` (every weight tensor and every metadata field equal). Provenance is proven. It did **not** reproduce byte-for-byte — but that turned out to be because **`torch.save` is non-deterministic** (re-saving identical content yields different bytes), not a model difference. Verify with:
> ```
> python scripts/retune_cascade.py --src models/lcnn_screener_v9.pt --low 0.10 \
>     --match-content model_store/v9h/lcnn.pt --check-only
> ```
> Consequence for this register: file-SHA is valid for **tamper detection** but not for **provenance** — see §9.7.

### 9.2 🟠 Documented "identical" checkpoints are not identical

`Handoff_Summary_V9_Phase7.md` states that `lcnn_screener_v9.pt` and `lcnn_student_v9.pt` are *"identical checkpoints, different roles"*. They are not:

| Artifact | Size | SHA-256 (first 24) |
|---|---|---|
| `lcnn_screener_v9.pt` | 1,167,583 | `ea5db3005b6cd05025445db9` |
| `lcnn_student_v9.pt` | 1,167,544 | `20a188fa1723b91bb53620d4` |

Different size, different hash. The distillation exit gate (VG-DOC-001 §11.1) evaluated *"the student"* — and it is now unclear which artifact that referred to. The documentation must be corrected and the student's role clarified: it is **not deployed** in any bundle.

### 9.3 🟠 LCNN training results lost

`models/lcnn_v9_results.json` is **0 bytes**. This should hold the LCNN training history and cascade calibration. Combined with §9.1, the deployed screener has neither a reproducible derivation nor a retained training record. The Platt coefficients survive only because they are embedded in the checkpoint and in `bundle.json`.

### 9.4 🔴→✅ No per-sub-model health check

All integrity controls verify **file identity**; all behavioural controls (golden regression, startup smoke check, drift monitor) verify **ensemble output**. Nothing verifies that each sub-model still *contributes signal*.

This is precisely how the AASIST V9 collapse reached production: the file was intact, hashes matched, the bundle loaded, the ensemble produced plausible verdicts — while one of three members had AUC 0.5258. The check that eventually found it (`aasist_probe.py`) exists and is run manually.

**Remediation:** add a per-sub-model probe to the promotion gate, asserting standalone AUC above a floor and non-degenerate score spread on a fixed probe set. This is a small piece of work against a demonstrated production failure.

> **✅ RESOLVED 2026-07-23 (check built; enforcement pending).** `scripts/submodel_health.py` scores all four models on a fixed probe set. The primary gate is **output spread** — a collapsed model emits a near-constant value (the AASIST V9 signature was softmax std ≈ 0.003) — with AUC as a secondary *weak-but-alive* warning that does not fail the gate (so a deliberately down-weighted member like RawNet3 does not trip it). Verified: **PASS on `v9h`** (exit 0), and a collapsed model at spread ≈ 0.003 fails against the 0.05 floor. **Still open:** wiring it into the promotion path (VG-DOC-006 H4). A by-product finding: deployed **AASIST V8 scores AUC 0.5403** on a studio probe — alive but weak on clean audio (VG-DOC-006 F1).

### 9.5 🟠 No named human approver

Promotion actors are process labels — `migration`, `route-b`, `aasist-retrain`, `eval` — not identified people. ISO/IEC 42001 change control expects an accountable, named approver per production change. The chain records *that* a change was approved and *why*, but not *by whom*.

**Remediation:** require a named actor (`--actor "michael.ologungbara"`) on `promote` / `rollback`.

### 9.6 🟡 Source checkpoints are not integrity-managed

Artifacts in `models/` carry no manifest or hash record. They are the inputs to every bundle. A silent corruption or substitution there would propagate into the next bundle and be faithfully hashed as legitimate. Consider extending the registry to cover source checkpoints.

### 9.7 🟡 File-SHA proves tamper, not provenance (found 2026-07-23)

Establishing §9.1 surfaced that **`torch.save` is not byte-deterministic** — re-serialising identical checkpoint content produces different bytes (zip metadata / pickle memo ordering). Therefore **no checkpoint in this system is byte-reproducible from its inputs**, and the registry's per-file SHA-256 cannot answer *"was this derived correctly from source?"* — only *"has this exact file changed since registration?"*.

Two distinct integrity concerns, only one currently covered:

| Concern | Question | Mechanism | Status |
|---|---|---|---|
| Tamper detection | Has this artifact changed since it was registered? | file SHA-256 | ✅ In place |
| Provenance | Was this artifact derived correctly from its source? | **content** hash (weights + metadata) | ⚠️ Only via `retune_cascade.py --match-content`, ad hoc |

**Remediation:** adopt a content-hash (canonical over sorted tensors + metadata) as the provenance primitive, alongside the existing file-SHA for tamper detection.

---

## 10. Required actions

| # | Action | Owner | Priority |
|---|---|---|---|
| 1 | Re-derive and document `v9h` `lcnn.pt` provenance (§9.1) | ML | **Critical** |
| 2 | Add a per-sub-model health check to the promotion gate (§9.4) | ML/Eng | **Critical** |
| 3 | Record the quantitative `v9fixed` vs `v9h` comparison that justified the rollback | ML | High |
| 4 | Correct the screener/student "identical" claim; clarify the student's status (§9.2) | ML | High |
| 5 | Require a named human approver on promote/rollback (§9.5) | Eng | High |
| 6 | Regenerate or formally retire `lcnn_v9_results.json` (§9.3) | ML | Medium |
| 7 | Extend integrity management to `models/` source checkpoints (§9.6) | Eng | Medium |
| 8 | Mark `v9` as rollback-blocked in the registry, given the known AASIST collapse | ML | Medium |
| 9 | Record model cards per bundle, not only for the active one | ML | Low |

---

## 11. Register summary

| Class | Count | Detail |
|---|---|---|
| Deployable bundles | 3 | 1 active (`v9h`), 1 superseded (`v9`), 1 rejected (`v9fixed`) |
| Files per bundle | 7 | Each individually SHA-256 pinned |
| Distinct model weights in production | 4 | LCNN · AASIST (V8) · Wav2Vec2 (V9) · RawNet3 (V8) |
| Source checkpoints | 11 | `models/` — not integrity-managed |
| Active production footprint | ~405 MB | Dominated by `wav2vec.pt` at 378 MB |
| Integrity status | **Verified** | All 7 files, 2026-07-22 |
| Open critical gaps | **2** | §9.1 screener provenance · §9.4 sub-model health check |

---

## 12. Document change log

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-07-22 | Michael Ologungbara | Baseline. All bundles and source checkpoints enumerated with verified SHA-256. Integrity verification run and evidenced. Cross-bundle change matrix establishes cryptographically that RawNet3, Wav2Vec2 and the verdict thresholds have never changed. Two critical gaps raised: deployed LCNN has no traceable source (§9.1); no per-sub-model health check (§9.4). Documentation error corrected: screener and student checkpoints are not identical (§9.2). |

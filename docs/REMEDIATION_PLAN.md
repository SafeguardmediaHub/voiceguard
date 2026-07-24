# VoiceGuard — Remediation Plan

**Document ID:** VG-DOC-006
**Version:** 1.0
**Date:** 2026-07-23
**Owner:** Michael Ologungbara
**Classification:** Confidential — internal
**Status:** Live tracker. Two items already resolved (§X2, §X3).
**Related:** VG-DOC-001 (Development History) · VG-DOC-003 (Dataset Inventory) · VG-DOC-004 (Model Inventory) · VG-DOC-005 (GRC Control Pack)

---

## 0. What this is

The single consolidated list of everything the four documentation documents surfaced that needs fixing — deduplicated, prioritised, with an owner and a status. It is the working tracker; the four source documents hold the detail and the evidence.

**The one-line summary of the whole picture:** the engineering is ahead of the record-keeping. Of the ten highest-rated issues, six are documentation, provenance, or measurement problems — not broken code. The system largely works; the proof of *how* and *how well* is what needs strengthening. That is the cheaper problem to have.

**Status key:** ✅ Done · 🔧 In progress · ⬜ Not started · 🔵 Needs external party (counsel)

---

## 1. Priority tiers at a glance

| Tier | Meaning | Items |
|---|---|---|
| **P0 — Blocking** | Must resolve before enterprise sale | C1, C2 |
| **P1 — High** | Fix before relying on published claims or the next promotion | H1–H9 |
| **P2 — Medium** | Fix in the normal course of hardening | M1–M9 |
| **Resolved** | Fixed during this documentation pass | X1, X2, X3 |

---

## 2. Resolved during this pass

| ID | Item | Evidence |
|---|---|---|
| **X1** | **Full development history reconstructed (V1→v9h) and all metrics re-sourced to the training notebook.** Four documentation errors caught in the process (see H-tier). | VG-DOC-001 v1.2 |
| **X2** | **Per-sub-model health check built and validated.** `scripts/submodel_health.py` scores all four models on a fixed probe set; hard-fails on *collapse* (near-zero output spread — the exact AASIST V9 signature), warns on *weakness*. Verified: PASS on the deployed `v9h` (exit 0), and a *simulated* collapse (constant-output model, spread 0.0000, AUC 0.500) is caught. Closes GRC G1/R4. | `scripts/submodel_health.py` |
| **X3** | **Deployed screener provenance closed.** `scripts/retune_cascade.py` proves the `v9h` screener is `models/lcnn_screener_v9.pt` retuned to `low_thresh=0.10` — *content-identical* to `model_store/v9h/lcnn.pt`. Closes GRC G2/R3. Surfaced a new finding: torch.save is non-deterministic (M9). | `scripts/retune_cascade.py` |
| **X4** | **Health gate wired into promotion (H4).** `bundle_registry.py promote` runs the gate on the candidate (via `$VOICEGUARD_FORCE_BUNDLE`, no live-pointer flip) and refuses promotion on a collapse. Fail-closed. Two bugs found and fixed during testing (numpy-bool JSON, multi-line JSON parsing); one new finding: `v9` cannot be loaded by current code at all (M7). | `bundle_registry.py`, `detector.py` |

---

## 3. P0 — Blocking (before enterprise sale)

### C1 — Training-data licensing 🔵🔴
**Risk:** GRC R1. `real_studio` (494 clips) sourced from YouTube via `yt-dlp`, URL lists committed. Exposure on platform ToS, third-party copyright, and speaker consent. **Structural** — it is the bucket that fixed studio FP from 80% to 12%; removing it regresses the flagship result.
**Fix:** Legal review first (counsel). If untenable, re-source `real_studio` from permissive audio — Common Voice v26 (CC0, already held, 27,929 Nigerian-language clips) + LibriVox public-domain — and retrain.
**Owner:** Counsel → ML · **Blocks:** enterprise sale · **Note:** cost rises the longer current weights stay in production.

### C2 — Non-commercial datasets in a commercial product 🔵🔴
**Risk:** GRC R2. WaveFake and XTTS (Coqui Public Model Licence, non-commercial) are baked into shipped weights.
**Fix:** Legal review of each licence against commercial use. Remediation is the same retrain path as C1.
**Owner:** Counsel · **Blocks:** enterprise sale.

---

## 4. P1 — High

### H1 — Re-measure the deployed bundle ⬜
**The single highest-value measurement task.** Published headline metrics (EER 2.43%, catch 99.2%, studio FP 12.0%) were measured on **V9**, not the deployed **`v9h`** — and they differ precisely in AASIST, the sub-model later found inert. Every performance claim is currently unsupported for what ships.
**Fix:** Re-run the bias audit and held-out studio/noiz.ai evaluation on `v9h`; restate VG-DOC-001 §10/§14 and MODEL_CARD §4. **[GRC R5]**

### H2 — A published number is wrong ⬜
V9 val EER **13.25% was computed with an inverted softmax**; the correct value is **16.19%** (notebook [420]). It propagated into both handoff summaries and the model card.
**Fix:** Correct 13.25% → 16.19% everywhere; note the correction. **[GRC R6]**

### H3 — Run an independent benchmark ⬜
No public benchmark has ever been run. ASVspoof 2021 LA harness is built (`scripts/asvspoof_eval.py`); ASVspoof 2019 is already a *training* set so cannot serve as independent.
**Fix:** Run ASVspoof 2021 LA; publish the EER. **[GRC R10/G6]**

### H4 — Wire the sub-model health check into the promotion gate ✅
**RESOLVED 2026-07-24.** `bundle_registry.py promote` now runs the health gate on the **candidate** before flipping the pointer. Implementation: `detector.py` honours `$VOICEGUARD_FORCE_BUNDLE` so the gate loads the candidate (not the active bundle) with no live-pointer flip; `bundle_registry._run_health_gate()` runs `submodel_health.py` in a subprocess against it. Fail-closed: a collapsed sub-model → exit 1, promote refused; missing probe data → exit 2, "cannot certify" (override with `--skip-health`, which records `[health-gate:skipped]` in the reason). Verified end-to-end: passes `v9h`; a simulated collapse (constant-output model, spread 0.0000) is caught and blocks promotion; all four CLI paths correct. **[GRC G1/R4 — closed]**
**Follow-on:** the health gate belongs in GRC **Policy 6** (model-validation acceptance criteria) as a written requirement, not only a code path.

### H5 — Reconcile the recall discrepancy ⬜
An internal finding that V9 misses ~half of TTS/multilingual fakes conflicts with the 99.2% catch figure. Unresolved; concerns the core capability claim.
**Fix:** Define one evaluation protocol; measure; reconcile or retract. **[GRC R8]**

### H6 — Wire the audit log into `/detect` ⬜
`governance.py` provides a tamper-evident audit log; live detections do not write to it.
**Fix:** Emit an audit-log entry per detection on the live path. **[GRC G7]**

### H7 — Restate bias-audit scope honestly ⬜
Parity PASS rests on 5 of 7 languages; Pidgin was tested with **Nigerian-English** TTS; Hausa train/test fakes came from one generation run; 25 of 50 Hausa test files are corrupt.
**Fix:** Add scope caveats to the bias-audit report and every summary that quotes "parity PASS". Longer term, generate proper Yoruba/Igbo/Pidgin fakes. **[GRC R9]**

### H8 — Named human approver on promotion ⬜
Promotion actors are process labels (`route-b`, `eval`), not people. ISO 42001 change control expects an accountable named approver.
**Fix:** Require `--actor "name"` on `promote`/`rollback` in `bundle_registry.py`. **[GRC G4/R14]**

### H9 — Screener reproducibility ⬜
The deployed screener's 8.18% EER is a best-of-50 from a run oscillating 8–57%. It drives ~86% of traffic. Whether that number reproduces on retrain is unknown.
**Fix:** Re-run the screener training; test whether 8.18% (or near) reproduces. If not, restate the screener's true performance band. **[GRC R16]**

---

## 5. P2 — Medium

| ID | Item | Fix | GRC |
|---|---|---|---|
| M1 | Backup encryption is opt-in | Confirm `VOICEGUARD_BACKUP_KEY` is set in production, else `auth_keys.json` backs up as plaintext | R12 |
| M2 | `jobs.db` grows without rotation | Add a retention/rotation policy and job | G12/R20 |
| M3 | 25 of 50 Hausa test fakes corrupt | Regenerate | VG-DOC-003 §J |
| M4 | `models/` source checkpoints not integrity-managed | Extend the registry to hash source checkpoints | R19/G6-sec |
| M5 | No penetration test | Commission one (Phase 7 exit gate) | G13 |
| M6 | Legal explainability template unreviewed | Non-technical reviewer sign-off (Phase 6 exit gate) | G14 |
| M7 | **`v9` cannot be loaded by the current `detector.py` at all** — its `aasist.pt` is the from-scratch V9 architecture (`bn0`, `sinc.hamming`, `sinc.n_` shape [1,63]) while the current AASIST class expects the V8 shape. Rolling back to `v9` would **crash on startup**, not merely serve a weak model. Found 2026-07-24 while testing the health gate. | Mark `v9` rollback-blocked in the registry (now mandatory, not advisory) | VG-DOC-004 §9 |
| M8 | "identical" screener/student checkpoints are not identical; `lcnn_v9_results.json` is 0 bytes | Correct the doc; regenerate or formally retire the results file | VG-DOC-004 §9.2/§9.3 |
| M9 | **torch.save is non-deterministic** (found during X3) — no checkpoint is byte-reproducible from inputs | Adopt content-hashing for provenance (weights + metadata), keeping file-SHA for tamper detection only. `retune_cascade.py --match-content` is the pattern | new |

---

## 6. New findings from this pass (beyond the four documents)

Two came out of building X2 and X3:

**F1 — AASIST V8 is weak on clean audio, not just V9.** The sub-model health check (X2) measured the deployed AASIST V8 at **AUC 0.5403** on a studio-heavy probe — barely above the *collapsed* V9's 0.5258. It is not collapsed (spread 0.40, so it varies with input), but the MODEL_CARD §1 claim that "V8 AASIST is genuinely discriminative" does not hold on clean/studio audio, which is exactly the distribution the whole `v9h` recovery was meant to serve. **Worth investigating** whether the ensemble's studio performance actually depends on AASIST at all, or whether Wav2Vec2 (AUC 0.69 here) is carrying it. Feeds H1.

**F2 — torch.save is non-deterministic (M9).** Re-saving identical checkpoint content produces different bytes. This is *why* `v9h`'s screener had no byte-reproducible source — it was never negligence, it is a property of the serializer. The correct integrity model separates two concerns: **file-SHA for tamper detection** (has this exact artifact changed since it was registered?) and **content-hash for provenance** (was this artifact derived correctly from its source?). The registry currently does only the first. `retune_cascade.py`'s `--match-content` demonstrates the second.

---

## 7. Suggested sequence

1. **Start C1/C2 (legal) now** — it runs on counsel's clock, not yours, so start it first even though the fix comes later.
2. **H1 (re-measure on v9h)** — unblocks every honest performance claim. Nothing customer-facing is solid until this is done.
3. **H4 (enforce the sub-model gate)** — X2 is built; wiring it in prevents a repeat of the exact failure that motivated this whole pass. Small.
4. **H2, H7 (correct the wrong/overstated numbers)** — cheap, and they stop wrong figures spreading further.
5. **H3 (ASVspoof)** — the number enterprise buyers ask for.
6. Everything else as capacity allows.

**Note:** H1, H2, H5, H7, H9 will change numbers in VG-DOC-001/003/004. After acting on them, the four documents need a revision pass to match measured reality.

---

## 8. Change log

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-07-23 | Michael Ologungbara | Baseline. Consolidated from VG-DOC-001/003/004/005. X2 (sub-model health check) and X3 (screener provenance) resolved during this pass; two new findings (F1 AASIST-V8 weakness on clean audio, F2 torch.save non-determinism) recorded. |

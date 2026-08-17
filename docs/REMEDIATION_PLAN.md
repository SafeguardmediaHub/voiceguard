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
| **P1 — High** | Fix before relying on published claims or the next promotion | H1, H2, H3, ~~H4~~ ✅, H5, ~~H6~~ ✅, H7, ~~H8~~ ✅, H9 |
| **P2 — Medium** | Fix in the normal course of hardening | M1–M6, ~~M7~~ ✅, M8, M9 |
| **D — Deployment mechanics** | Environment gaps that silently undid controls marked ✅ | ~~D1–D7~~ ✅, D8, ~~D9~~ ✅ |
| **✅ Resolved** | Fixed during this pass | X1, X2, X3, X4 (+ H4) |

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

### H4 — Wire the sub-model health check into the promotion gate ✅ DONE
**RESOLVED 2026-07-24.** `bundle_registry.py promote` now runs the health gate on the **candidate** before flipping the pointer. Implementation: `detector.py` honours `$VOICEGUARD_FORCE_BUNDLE` so the gate loads the candidate (not the active bundle) with no live-pointer flip; `bundle_registry._run_health_gate()` runs `submodel_health.py` in a subprocess against it. Fail-closed: a collapsed sub-model → exit 1, promote refused; missing probe data → exit 2, "cannot certify" (override with `--skip-health`, which records `[health-gate:skipped]` in the reason). Verified end-to-end: passes `v9h`; a simulated collapse (constant-output model, spread 0.0000) is caught and blocks promotion; all four CLI paths correct. **[GRC G1/R4 — closed]**
**Follow-on:** the health gate belongs in GRC **Policy 6** (model-validation acceptance criteria) as a written requirement, not only a code path.

### H5 — Reconcile the recall discrepancy ⬜
An internal finding that V9 misses ~half of TTS/multilingual fakes conflicts with the 99.2% catch figure. Unresolved; concerns the core capability claim.
**Fix:** Define one evaluation protocol; measure; reconcile or retract. **[GRC R8]**

### H6 — Wire the audit log into `/detect` ✅ DONE
**RESOLVED 2026-07-27.** Every live detection now emits one entry into the tamper-evident, hash-chained `governance.AuditLog` (`governance/audit_log.jsonl`). Implementation: `detector._write_audit_log()` was rewritten from a plain, editable jsonl line into a `governance.AuditLog().append()` — recording the **full** intake SHA-256 (not the truncated display hash), the active bundle version, and each sub-model checkpoint's SHA-256 as `model_versions` (the true version identity — the registry tracks weights by content hash, not an integer). `governance.py`'s hardcoded `/kaggle/working/governance` path is now `$VOICEGUARD_GOVERNANCE_DIR` (repo-relative default) so it is writable in the live deployment. Never-fatal: an audit-write failure does not fail a paid detection. `detect(..., audit=False)` skips the internal startup smoke-check so it neither pollutes the chain-of-custody log nor interleaves appends across concurrent API boots. Single-writer invariant holds: real detections run only in the sequential `worker.py`. Verified end-to-end on real weights — a live `detect()` writes exactly one chain-verifiable entry (`audit_id`/`verdict`/full-sha/bundle match the response); the smoke-check writes none; `python governance.py verify-chain` PASSES on the live log and FAILS on an edited entry. This makes the "tamper-evident audit log" reference in `forensic_report.py` true. **[GRC G7 — closed]**
**Follow-on (done, same pass):** `governance.py verify-chain` now exits non-zero on a broken chain (it previously printed `FAILED` but exited 0, silently passing any CI/cron audit gate). Verified via CLI subprocess tests: exit 0 clean, exit 1 tampered.

### H7 — Restate bias-audit scope honestly ⬜
Parity PASS rests on 5 of 7 languages; Pidgin was tested with **Nigerian-English** TTS; Hausa train/test fakes came from one generation run; 25 of 50 Hausa test files are corrupt.
**Fix:** Add scope caveats to the bias-audit report and every summary that quotes "parity PASS". Longer term, generate proper Yoruba/Igbo/Pidgin fakes. **[GRC R9]**

### H8 — Named human approver on promotion ✅ DONE
**RESOLVED 2026-07-27.** `bundle_registry.py` now rejects generic/placeholder actors (`cli`, `admin`, `root`, blanks, anything <3 chars) on both `promote` and `rollback` via `_require_named_actor()`. A real name (`--actor "firstname.lastname"`) is required and lands in the tamper-evident chain. Verified: `cli`/`ab` rejected, `michael.ologungbara`/`Jane Doe` accepted. **[GRC G4/R14 — closed]**

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
| M7 ✅ | **`v9` cannot be loaded by the current `detector.py` at all** — its `aasist.pt` is the from-scratch V9 architecture (`bn0`, `sinc.hamming`, `sinc.n_` shape [1,63]) while the current AASIST class expects the V8 shape. Rolling back to `v9` would **crash on startup**. **DONE 2026-07-27:** `BLOCKED_VERSIONS` in `bundle_registry.py` refuses both promoting `v9` and rolling back *into* `v9`, with the reason surfaced. Verified. | ✅ resolved | VG-DOC-004 §9 |
| M8 | "identical" screener/student checkpoints are not identical; `lcnn_v9_results.json` is 0 bytes | Correct the doc; regenerate or formally retire the results file | VG-DOC-004 §9.2/§9.3 |
| M9 | **torch.save is non-deterministic** (found during X3) — no checkpoint is byte-reproducible from inputs | Adopt content-hashing for provenance (weights + metadata), keeping file-SHA for tamper detection only. `retune_cascade.py --match-content` is the pattern | new |

---

## 5a. D-tier — deployment mechanics (2026-08-09 pass)

Found by working the deploy stack rather than the model story. Four of these
silently undid controls this plan already marks ✅ — the control existed in code
but not in the environment it had to run in.

| ID | Item | Status |
|---|---|---|
| **D1** | **CI was red on `main`.** The H8 (named actor) and M7 (`BLOCKED_VERSIONS`) fixes landed without test updates: 4 failures in `test_bundle_registry.py`. Worse, `test_promote_refuses_unregistered` and `test_promote_refuses_on_integrity_failure` still *passed* — but only because `_require_named_actor()` raises first, so neither reached the guard it exists to test. Both H8 and M7 were themselves untested. **DONE:** tests fixed, and 10 added covering the named-actor and blocked-version guards directly (including rollback *into* `v9`, the dangerous path). | ✅ |
| **D2** | **The audit log was ephemeral in production.** `$VOICEGUARD_GOVERNANCE_DIR` was set nowhere — not the Dockerfile, compose, or `.env.example` — so H6's tamper-evident chain wrote to `/app/governance` in the container's writable layer and was destroyed by every redeploy. **DONE:** `VOICEGUARD_GOVERNANCE_DIR=/data/governance` (the `vg-data` volume), created in the entrypoint. | ✅ |
| **D3** | **The developer's local audit log was baked into the image.** `.dockerignore` did not exclude `governance/` and the Dockerfile does `COPY . .`, so production's chain-of-custody would have continued from laptop test detections. **DONE:** excluded, with a regression fence in `test_docker_context.py`. | ✅ |
| **D4** | **Backup covered the wrong files.** `deploy/backup.py` backed up `jobs.db` + `auth_keys.json` only — not `audit_log.jsonl`. The one artifact whose entire purpose is durable evidence was the one a droplet loss would destroy. **DONE:** added to `ARTIFACTS`. | ✅ |
| **D5** | **Bundle integrity was never verified at load.** The registry hashes artifacts at *promote* and *pull* time; the serving process trusted whatever was on disk at startup, and `torch.load(weights_only=False)` unpickles — so write access to `model_store/` meant code execution in the API process. **DONE:** `detector._verify_bundle_before_load()` fails closed *before* the first `torch.load` (verifying after it would be theatre). Reuses a new `Registry.integrity_problems()`, which reports against `registry.jsonl` rather than the bundle's own `bundle.json` — an attacker rewriting weights could rewrite the latter alongside them. Cost: **7.6 s** for the 386.5 MB `v9h` bundle, inside the 180 s healthcheck `start_period`. | ✅ |
| **D6** | **The audit chain had no writer lock.** `AuditLog.append` is read-last-hash → compute → write, unserialized, while `docker-compose.prod.yml` explicitly documented `--scale worker=2` as safe. True for the job queue (`BEGIN IMMEDIATE`), false for the chain. Reproduced: 4 processes × 15 appends → 59 of 60 entries, chain broken at seq 13, reported as *tampering* — indistinguishable from a real attack. **DONE:** `governance.chain_lock()` (thread lock + `fcntl`/`msvcrt` file lock) around the critical section, plus `fsync` before release. Regression test spawns concurrent writers and asserts the chain verifies. | ✅ |
| **D7** | **Three stale copies of the golden expectation.** `test_detector.py` and `test_worker.py` hardcoded `LIKELY_FAKE / 0.8167` — V9-era values — so the **weights tier was red too**, failing against a `v9h` bundle behaving exactly as its own baseline says it should. **DONE:** both now read `tests/golden_manifest.json`, per the precedent in `19ccce3`. | ✅ |
| **D8** | **Rate limiting does not aggregate across workers.** `request_protection.py` holds token buckets in per-process memory; the API runs `gunicorn -w 3`, so the effective limit is ~3× the configured 30/min, distributed non-deterministically. The module's own docstring flags it. **OPEN** — needs shared state (Redis) or an ingress-level limit in Caddy. | ⬜ |
| **D9** | **The promotion health gate could not run in the deployed image.** `scripts/submodel_health.py` probes `studio_clips`, `bias_audit_fakes`, `studio_fake_test`, `bias_audit/` — all `.dockerignore`d, so `promote` in production always exited 2 ("cannot certify") and the only way through was `--skip-health`, the exact habit H4 exists to prevent. H4 was real, but only on a developer machine. **DONE 2026-08-09 — see §5b.** | ✅ |

---

## 5b. D9 in detail — making the health gate enforceable in production

Three separate faults had to be fixed; only the first was the one originally identified.

**The probe set could never have decoded anyway.** `sweep_cascade._load`, which the gate uses to read every probe clip, hardcoded `C:/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe` and discarded ffmpeg's return code. In the Linux container that path does not exist, so every clip failed to decode and was silently skipped. Now uses `D.FFMPEG` (honouring `$VOICEGUARD_FFMPEG`) and raises on a non-zero exit.

**An environment fault was reported as a total model collapse.** A probe set that will not decode yields zero spread for every model — byte-identical to a collapse. The gate would have announced `COLLAPSED: lcnn, aasist, wav2vec, rawnet` when the real cause was a missing binary. It fails closed either way, but an operator reading that would conclude the gate is broken and reach for `--skip-health`. `run_health_check` now raises a distinct environment error, which `bundle_registry` maps to CANNOT CERTIFY; a new `HEALTH_ERROR` sentinel carries the reason past the model-loading noise so the operator sees it instead of the transformers banner.

**A real-audio-only probe set silently breaks the collapse gate.** This one was not anticipated. Collapse is measured as near-zero output *spread*, and on an all-real set a *correct* model produces a tight cluster by design — it confidently says "real" to every clip. Measured on the CC0 set: **LCNN spread 0.0076, RawNet3 0.0001 — both healthy, both reported COLLAPSED.** Spread only separates healthy from collapsed when the probe set spans both classes.

**What ships.** `tests/probe_clips/` — 15 Common Voice v26 clips (**CC0-1.0**, 572 KB, 15 distinct speakers across ha/ig/yo), plus the two fake clips *already* in the image for the golden regression, referenced rather than copied. The shipped set therefore adds no new redistributed audio, which was the constraint given C1/C2 are still with counsel. Built deterministically by `scripts/build_probe_set.py`; every clip's SHA-256 is in a manifest and fenced by a test.

**Honest scope.** Two fake clips cannot support an AUC — on them the four models "score" 0.87–1.00 against 0.49–0.69 on the full corpora, which would read as near-perfect and flatly contradict F1. The gate withholds AUC below `MIN_FAKE_FOR_AUC` (10) and reports `n/a`. **The COLLAPSE tier — the control that matters, and the one that would have caught AASIST V9 — is fully enforced**, and is demonstrated against the shipped set: a simulated saturated sub-model is caught (spread 0.0000) with no false positives on the other three. The AUC weak-warning tier, which never failed a promotion, is inert in production until a larger labelled set can ship. Probe selection prefers the local corpora when present, so nothing changes on a dev machine — verified identical (AASIST 0.5403).

---

**Verified:** fast tier 94 passed; full suite 120 passed.

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

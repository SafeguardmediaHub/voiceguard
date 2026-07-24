# VoiceGuard — Model Development History

**Document ID:** VG-DOC-001
**Version:** 1.1
**Date:** 2026-07-22
**Owner:** Michael Ologungbara
**Classification:** Confidential — client and internal distribution
**Status:** Complete lineage V1 → v9h. Open evidence gaps tracked in §16.
**Review cycle:** On every bundle promotion, and at minimum quarterly.
**Related documents:** `docs/MODEL_CARD.md` (VG-DOC-002), Dataset Inventory (VG-DOC-003, pending), Model Inventory Register (VG-DOC-004, pending), GRC Control & Risk Pack (VG-DOC-005, pending).

---

## 1. Purpose and scope

This document is the auditable record of how the VoiceGuard audio deepfake detector was built: every training run that produced a shipped or candidate model, the data each was trained on, the measured results, and the decisions — including the rejections and failures — that connect them.

It exists to answer four questions an enterprise customer, auditor, or regulator will ask:

1. **Where did this model come from?** — lineage from the first prototype to the deployed bundle.
2. **What was it trained on?** — datasets, sizes, composition, and how composition changed between versions.
3. **How well does it work, and how do you know?** — measured metrics, the sets they were measured on, and the limits of those measurements.
4. **What went wrong, and how did you catch it?** — failed approaches, regressions, and the controls that detected them.

It supports ISO/IEC 42001 clauses on AI system lifecycle, data governance, and performance evaluation, and provides the development-history evidence that ISO/IEC 27001 change management expects. It is **not** a model card (see `docs/MODEL_CARD.md`), an API specification (`docs/API_REFERENCE.md`), or an operating procedure (`docs/RUNBOOK-model-flow.md`).

**Product positioning.** VoiceGuard is sold as an enterprise **platform**: the API is general-purpose and customers apply it to their own use cases (fraud/KYC, media verification, evidence triage). Performance below is characterised on the vendor's evaluation sets; a customer's own distribution may differ materially. §17 states this limitation formally.

---

## 2. Evidentiary basis

Every factual claim below is traceable to an artifact. Claims resting on narrative handoff notes rather than machine-readable output are marked.

| Source | Type | Evidences |
|---|---|---|
| `voiceguard-training-notebook (3).ipynb` | **Primary — 425 cells with retained execution output** | All training runs and measured metrics, V1 → V9 (§4–§12) |
| `model_store/ACTIVE.json` | Hash-chained promotion log | Bundle activation history, actors, reasons (§3, §13) |
| `model_store/registry.jsonl` | Append-only registry | Bundle registration and file manifests |
| `model_store/v9h/bundle.json` | Bundle manifest | Deployed thresholds, calibration, per-file SHA-256 |
| `docs/new_docs/Handoff_Summary_Post_Phase5_6_7.md` | Narrative | V8 baseline, Phase 5, studio-FP discovery, fine-tune rejection |
| `docs/new_docs/Handoff_Summary_V9_Final.md` | Narrative | V9 retrain, bias audit |
| `Handoff_Summary_V9_Phase7.md` | Narrative | Cascade + distillation |
| `docs/AASIST-V9-retrain-recipe.md` | Technical diagnosis | Root cause of the AASIST V9 collapse (§12) |
| `tests/golden_manifest.json` | Pinned regression baseline | Current deterministic behaviour (§13) |
| `docs/new_docs/VoiceGuard_V9_Bias_Audit_Report.docx` | Formal report | Phase 6 bias audit (§10) |

**Evidentiary strength improved at v1.1.** The training notebook — supplied 2026-07-22 — carries retained cell outputs for every training run. Metrics in this document are now sourced to notebook cell numbers (shown as `[cell N]`) rather than to summary prose. **Two material discrepancies between the notebook and the handoff notes were found and are recorded in §16.1 and §16.2.**

---

## 3. Version lineage at a glance

```
V1 ─▶ V2 ─▶ V3 ─▶ V4 ─▶ V5 ─▶ V6/V6b ─▶ V7 ─▶ V8 ─▶ V9 ─▶ v9h ─▶ [v9fixed]
 │                  │      │      │        │      │      │      │        │
 │                  │      │      │        │      │      │      │        └ candidate,
 │                  │      │      │        │      │      │      │          REJECTED
 │                  │      │      │        │      │      │      └ CURRENT PRODUCTION
 │                  │      │      │        │      │      └ AASIST later found collapsed
 │                  │      │      │        │      └ HARDER VAL SET — metrics not
 │                  │      │      │        │        comparable across this boundary
 │                  │      │      │        └ Wav2Vec-only gains; AASIST/RawNet3 reverted
 │                  │      │      └ superseded twice, reverted twice
 │                  │      └ first ensemble that worked end-to-end (6/6 real-world)
 │                  └ acoustic-only
 └ real-world accuracy 0/6
```

| Version | Status | Defining change |
|---|---|---|
| V1 | Superseded | First working detector. Real-world accuracy **0/6**. |
| V2–V3 | Superseded | Acoustic-only stream. V3 standalone EER 4.00%. |
| V4 | Superseded | **First production-grade ensemble.** Modern-TTS training → real-world **6/6**. |
| V5 | Partially adopted | Wav2Vec improved; AASIST + RawNet3 retrains **did not improve and were reverted**. |
| V6 / V6b | Reverted | Two Wav2Vec variants; neither retained over V4/V5. |
| V7 | Superseded | **Hard validation set introduced.** Metrics before/after are not comparable. |
| V8 | Superseded | Production baseline. Calibrated ensemble EER 7.41%. |
| V9 | Superseded | Full retrain on expanded manifest. AASIST later found collapsed. |
| **v9h** | **ACTIVE** | AASIST reverted to V8; hybrid fusion refit; screener early-out widened. |
| v9fixed | Rejected | Corrected AASIST retrain. Promoted, evaluated, rolled back same day. |

Promotion record, verbatim from the hash-chained `ACTIVE.json`:

| Seq | Transition | Timestamp (UTC) | Actor | Recorded reason |
|---|---|---|---|---|
| 0 | → `v9` | 2026-07-07T10:23:55 | `migration` | initial V9 bundle |
| 1 | `v9` → `v9h` | 2026-07-12T12:06:11 | `route-b` | V8 AASIST + hybrid fusion + CASCADE_LOW 0.10 |
| 2 | `v9h` → `v9fixed` | 2026-07-14T13:28:47 | `aasist-retrain` | proper AASIST retrain + refit — candidate |
| 3 | `v9fixed` → `v9h` | 2026-07-14T16:54:20 | `eval` | v9fixed no better than v9h; keep v9h |

Seq 2–3 merit an auditor's attention: a candidate was promoted, evaluated against the incumbent, found not to improve on it, and rolled back the same day — both actions recorded in a tamper-evident chain. That is change control working as designed.

---

## 4. V1 – V7 — the pre-V8 lineage

Reconstructed from the training notebook's retained outputs. Previously undocumented.

### 4.1 V1 – V3: getting to a working detector

Initial training used **ASVspoof 2019 LA** (7,740 real) and **WaveFake** (157,066 fake) — a train split of **164,806 files**, 4,000 validation, a **24,844-file held-out set**, and a **71,237-file "living test set"** `[cells 57–85]`.

The class imbalance is stark — roughly **20:1 fake:real**. That imbalance shaped everything that followed and is the same failure mode that resurfaced in V9 (§12.2).

| Model | EER | Source |
|---|---|---|
| AASIST V1 | 5.60% | `[163]` |
| RawNet3 | 11.68% | `[163]` |
| Wav2Vec V3 | 2.90% | `[154]` |

**V1's real-world accuracy was 0/6** `[177]` — it worked on benchmarks and failed on real audio. Recognising that gap early, and measuring it, is what drove every subsequent version.

A key intervention in this era: **VCTK** was added after spectral analysis showed ASVspoof reals were unusually "clean" (flatness 0.0105 vs VCTK 0.0164) `[96–100]`. A 21,695-file VCTK manifest (mic1, max 200/speaker) was built explicitly to teach the model to accept clean real speech. **This is the same "clean = fake" failure mode that later produced the 65.5% studio false-positive rate (§7)** — it was identified, partially mitigated, and then re-emerged at scale.

### 4.2 V4 — the first production-grade ensemble

| Component | Result | Source |
|---|---|---|
| Wav2Vec V4 | EER **1.53%** (epoch 5) | `[156]` |
| AASIST | EER 5.60% / 5.65%, AUC 98.23% | `[158]` |
| RawNet3 | EER 11.68% / 11.03%, AUC 95.61% | `[158]` |
| XGBoost ensemble | **OOF EER 1.22%**, AUC 99.93% (5-fold) | `[158]` |
| 100-file benchmark | EER **4.00%**, AUC 99.36%, FPR **0.0%**, FNR 18% | `[176]`, `[204]` |
| Real-world | **6/6** (from V1's 0/6) | `[177]` |

Artifacts: `best_model_v4.pt`, `xgb_v4.pkl`. V4 became the reference every later version was measured against.

### 4.3 V5 — selective adoption, two reverts

Two distinct V5 efforts appear.

**First attempt** `[173–175]`: Wav2Vec improved 1.53% → **1.22%**, OOF EER 0.80%. Then *"V4 reloaded"* `[175]` — not retained.

**Second attempt** `[226–231]`, on a much larger corpus — Wav2Vec V5 *"trained on 696,280 files"* `[231]`:

| Model | Outcome | Decision |
|---|---|---|
| Wav2Vec V5 | EER 1.53% (epoch 9) | **Adopted** |
| AASIST V5 | Best 15.33%, unstable (33.9% → 55.6% swings) `[227]`; second run returned 5.60% at epoch 0 | **Rejected** — "kept — V5 did not improve" `[231]` |
| RawNet3 V5 | Best 11.68% at epoch 0; all epochs worse (20.9–25.6%) `[230]` | **Rejected** — "kept — V5 did not improve" `[231]` |
| XGBoost V5 | OOF EER **0.98%** | Adopted |

Benchmark comparison: *"V5 wins on benchmark EER (0% vs 4%)"*, with false negatives V4=3 vs V5=1 — but *"V5 catches more YT TTS but misses Ana"* `[233]`.

**This is a documented instance of per-component acceptance gates working.** Two of three retrains were rejected on evidence and the previous weights retained.

### 4.4 V6 and V6b — both reverted

| Variant | Wav2Vec EER | OOF EER | Outcome |
|---|---|---|---|
| V6 | 1.31% | 0.86%, AUC 99.96% | *"V4 reloaded"* `[192]` — reverted |
| V6b | 1.95% | 1.33%, AUC 99.93% | Benchmark EER 4.00%, AUC 99.56% `[203]` — not retained |

Neither improved on V4/V5 in a way that justified adoption.

### 4.5 V7 — the measurement changes, not just the model

**The most important interpretive event in the project's history.** V7 introduced a substantially harder validation set. Measured on it, the existing V4 models performed far worse than their headline numbers:

| Model | On V4-era val | **On V7 val** | Source |
|---|---|---|---|
| AASIST V4 | 5.60% EER | **31.92% EER**, AUC 76.75% | `[266]` |
| Wav2Vec V4 | 1.53% EER | **24.58% EER**, AUC 82.06% | `[271]` |

Retrained against it:

| Model | Result | Source |
|---|---|---|
| AASIST V7 | 15.54% → best **12.74%** | `[266]`, `[268]` |
| Wav2Vec V7 | → **14.90%** | `[274]` |

**Every metric before V7 and every metric after it are measured on different populations and must never be compared directly.** The apparent "regression" from V4's 1.53% to V8's 12.69% is overwhelmingly a change of *measurement*, not of model quality — the models got better while the yardstick got honest. Any performance narrative that plots V4 → V8 as a regression is wrong, and §14 flags the boundary explicitly.

---

## 5. V8 — the production baseline

### 5.1 Architecture

| Sub-model | Parameters | XGBoost weight | Role |
|---|---|---|---|
| AASIST | ~285 K | 0.501 | Graph-attention spectral detector |
| Wav2Vec2 (head) | ~95 M frozen + classifier | 0.414 | Self-supervised representation |
| RawNet3 | ~6 M | 0.086 | Raw-waveform detector |

Fusion: XGBoost → logistic (Platt) calibration → probability → verdict banding.
**Deployed thresholds, unchanged through to today:** `auto_fake ≥ 0.85`, `likely_fake ≥ 0.55`, `to_review ≥ 0.30`, else `auto_real`.

### 5.2 Training runs

| Model | Best val EER | Val AUC | Source |
|---|---|---|---|
| AASIST V8 | **0.1298** (12.98%), 13 epochs | 0.9325 | `[290]` |
| Wav2Vec V8 | **0.1269** (12.69%), 10 epochs | 0.9317 | `[292]` |

Wav2Vec improved monotonically across all 10 epochs — a healthy training curve, in contrast to the oscillation seen in V5 AASIST and later in the LCNN (§11.2).

### 5.3 Ensemble results

| Metric | Value | Source |
|---|---|---|
| **Calibrated AUC** | **0.9790** | `[294]` |
| **Calibrated EER** | **0.0741** (7.41%) | `[294]` |
| Clean ensemble EER | 0.0933 – 0.1000 | `[305]`, `[343]` |
| Overall evaluation set | 2,933 samples | `[295]` |

Training data: ~16,400 samples across 7 source buckets; working manifest `train_v8_fresh.json` (4,400–4,438 entries), validation `val_v8_fresh.json` (1,200).

**Composition was predominantly amateur recordings** — the defining weakness (§7).

---

## 6. Phase 5 — adversarial robustness (a register of negative results)

Phase 5 tested V8 against deliberate evasion. **Most of it failed, and the failures were documented rather than buried.** Retained in full because a customer's security team will ask exactly these questions.

### 6.1 Measured attack results `[323]`, `[327]`

| Model | Clean EER | Under FGSM | Under PGD |
|---|---|---|---|
| AASIST | 0.1450 | **0.9750** | **1.0000** |
| Wav2Vec2 | 0.1400 | — | — |
| RawNet3 | 0.3550 | — | — |
| **Ensemble** | **0.1000** | **0.4600** | **0.4800** |

AASIST under PGD reaches EER 1.0000 — *worse than chance*, meaning the attack reliably inverts its decision. The ensemble degrades to ~0.48 — better than any single model, but not usable.

### 6.2 Attempted mitigations

| Mitigation | Result | Source |
|---|---|---|
| Adversarial training | AASIST adv EER 0.9750 → **0.1250**, but clean EER 0.1450 → 0.2150 and ensemble clean 0.1000 → 0.1300. **Net rejected** — clean-performance cost too high. | `[323]` |
| Adversarial training vs PGD | PGD EER **unchanged at 1.0000** across all epochs. Complete failure. | `[327]` |
| Input randomisation | σ=0.005 → EER 0.1000 → **0.3300**. Collapsed clean accuracy. | `[329]` |
| Effects augmentation | Catch rate regressed 81% → 50%. Checkpoint discarded. | Handoff |
| Adversarial-input detector | AUC **0.7361** (n=720), clean FP 29.5%. **No deployable operating point.** | `[354]` |
| Longer attack budgets | Ensemble EER 0.1000 → 0.9600 at 4.1 min, 0.9300 at 40.2 min | `[328]` |

**Distribution shift:** phone-effected fakes catch 40% (n=5) vs 100% on all other effect classes; concentrated in AASIST.

**The honest position, which must be carried into every customer claim:** VoiceGuard is **not robust to a knowledgeable adversary** crafting gradient-based perturbations. The advisory adversarial monitor flags *suspicion*; it does not prevent evasion. Stated in `MODEL_CARD.md` §7 and must remain in every security questionnaire response.

**Retained from Phase 5:** server defences (`request_protection.py`), drift monitor with phone-class fingerprinting, few-shot adaptation pipeline.

---

## 7. The studio/broadcast false-positive discovery

The most consequential finding in the project's history, and the reason V9 exists.

**Method.** 325 real clips (100 audiobook, 100 broadcast news, 125 podcast) sourced via `fetch_studio_clips.py`, evaluated with `studio_fp_check.py`.

**Result — 65.5% false-positive rate on genuine professionally produced audio:**

| Category | FP rate |
|---|---|
| Audiobook | **95.0%** (67% reaching `auto_fake`) |
| Podcast | 56.8% |
| Broadcast news | 47.0% |

**Root cause.** V8's real training data was almost entirely amateur recordings. Professional audio is acoustically closer to TTS — clean, low-noise, full-frequency. V8 had learned **"clean = fake, messy = real"**. The proxy held on its training distribution and failed on any well-produced real audio.

**Note the recurrence.** §4.1 records that this exact failure mode was identified during V1–V3 and partially mitigated by adding VCTK. It re-emerged at scale in V8. A one-off data fix did not durably solve it; only the V9 retrain with `real_studio` did.

**Two points that make the finding trustworthy:**

1. **Tooling artifact ruled out.** First chunking used `ffmpeg -c copy`, producing broken MP3 frames at boundaries. Re-run with `libmp3lame`: 65.5% vs 64.3% — essentially identical.
2. **Safety check preceded the fix.** `studio_processed_fake_test.py` verified V8 catches TTS fakes via genuine synthesis artifacts, not the cleanliness proxy — 100% catch held after studio processing, zero flips, scores slightly *up*. Adding clean reals would not destroy fake detection.

That second step is worth highlighting in any process-maturity discussion: the team tested whether the fix would break something else *before* running it.

---

## 8. Rejected intervention — the combined fine-tune

Before committing to a full retrain, a cheaper option was tried and **correctly rejected by its own gate**. `v8_finetune_combined.py` fine-tuned AASIST with a frozen front-end on 296 new samples + a 300-sample replay buffer.

| Metric | Before | After | Verdict |
|---|---|---|---|
| Noiz.ai catch | 60% | **76.7%** | ✅ +16.7 pp |
| OpenVoice VC | 77.3% | **81.8%** | ✅ +4.5 pp |
| Studio FP | 80% | **84%** | ❌ worse |
| `real_in_the_wild` FP | 39.2% | **51.0%** | ❌ +11.8 pp |

Corroborated in the notebook `[377–379]`: noiz.ai catch 0% → 60% → 76.7% across fine-tune stages, with ensemble EER drifting 0.1000 → 0.1150.

**Gate rejected the candidate.** The studio-FP problem lives in the SincConv front-end and early ResBlocks — which were frozen. Fine-tuning later layers learns new *fake* patterns but cannot override the front-end's "clean = fake" representation. **A full retrain was required.** The fine-tune's gains were expected to carry into the retrain — and did.

---

## 9. V9 — the full retrain

### 9.1 Manifest

`train_v9.json` — **4,746 entries**, 1.07:1 real:fake:

| Bucket | Entries | Purpose |
|---|---|---|
| Original V8 data (7 buckets) | 4,400 | Preserve capability |
| `real_studio` | 226 | Fix "clean = fake" (§7) |
| `fake_noizai` | 70 | Commercial-clone catch rate |
| `fake_hausa_tts` | 50 | Added during bias-audit mitigation (§10) |

Held-out: **105 entries never trained on** (75 studio reals, 30 noiz.ai fakes).

### 9.2 Component training

| Model | Best val EER | Source |
|---|---|---|
| AASIST V9 | **15.00%** at epoch 39 (from 35.68% at epoch 0) | `[385–386]` |
| Wav2Vec2 V9 | **15.83%** at epoch 28 | `[408]` |
| RawNet3 | **Not retrained** — ~1 hr/epoch on T4 with cuDNN disabled | Handoff |

### 9.3 Ensemble refit — three iterations

The notebook records the fusion being refit three times, which the handoff notes compress into a single result:

| Iteration | Val EER | Val AUC | Held-out EER | Source |
|---|---|---|---|---|
| 1 | 22.11% | 0.8738 | 15.62% | `[388]` |
| 2 | 15.33% | 0.9331 | 23.00% | `[391]` |
| **3 (adopted)** | **13.25%** | **0.9408** | **14.33%** | `[394]` |
| Simple average (baseline) | 14.92% | 0.9371 | 18.67% | `[396]` |

Note iteration 2: val improved while held-out *worsened* to 23.00% — a divergence that iteration 3 resolved. The adopted configuration beats the simple-average baseline on both axes, which is the justification for the learned fusion.

Final weights: AASIST **0.402**, Wav2Vec2 **0.445**, RawNet3 **0.153**.
Calibration: `P = sigmoid(4.8246 · xgb_score − 2.1469)`.

### 9.4 Results against goals

**Primary goals achieved** (held-out, never trained on):

| Metric | V8 | V9 |
|---|---|---|
| Studio FP | 80% | **12.0%** ✅ |
| Noiz.ai catch | 60% | **83.3%** ✅ |

**Validation regression, accepted with reason:** val EER 7.41% → 13.25% (but see §16.1 — the true figure is 16.19%). Driven by the un-retrained RawNet3's residual "clean = fake" bias on `real_in_the_wild`, measured at **30.3–31.7% FPR** `[419–420]` — notably worse than the 25.3% recorded in the handoff notes. `real_common_voice` FPR was 2.3–2.7%.

The trade — a val-EER regression in exchange for eliminating a 65.5% real-world false-positive mode — was judged worthwhile. That reasoning is recorded so it can be challenged on review rather than rediscovered.

---

## 10. Phase 6 — bias audit and the Hausa mitigation

### 10.1 Test set

599 samples, **7 languages** — Arabic, English, French, Hausa, Igbo, Pidgin, Yoruba — across 3 language families. Nigerian-language coverage was a client priority.

### 10.2 Progression

| Stage | EER | FPR | Catch | Source |
|---|---|---|---|---|
| Initial subset (n=56 real / 40 fake) | 7.68% | 5.4% | 90.0% | `[402]` |
| Full set, pre-mitigation (n=349 / 250) | **10.36%** | 4.6% | 88.4% | `[403]` |
| **Post-mitigation** | **2.43%** | **5.4%** | **99.2%** | `[408]` |

Mitigation: 50 Hausa TTS fakes added to training; AASIST + Wav2Vec2 retrained. **Hausa catch 50% → 100%.**

**Catch-rate parity: PASS — zero violations.** Nigerian languages at or below overall FPR: Yoruba 2%, Igbo 4%, Pidgin 0%.

### 10.3 The Arabic finding and its cost

16% FPR at the deployed threshold, attributed to recording quality (broadcast audio), not language bias. The notebook records the full sweep `[405–406]`:

| Threshold | Arabic FPR | Arabic catch |
|---|---|---|
| 0.60–0.70 | 10.0% | 96.0% |
| 0.80 | 8.0% | 90.0% |
| **0.86** | **6.0%** | **86.0%** |
| 0.90 | 6.0% | 66.0% |
| 0.92 | 4.0% | 60.0% |

The 0.86 mitigation costs **10 pp of catch rate** to halve the false-positive rate. It is **not applied in production** — it requires language detection the system does not perform. A global FPR-matching alternative was also measured: matching 4.6% FPR overall drops catch to **80.0%** `[404]`.

**These are the numbers a customer needs to set policy.** The trade is real and should be surfaced in the threshold guidance, not buried.

### 10.4 Coverage limitation

Yoruba and Igbo were tested for **false positives only** — no free TTS engine produced fakes in those languages, so **catch rate for Yoruba and Igbo is untested**. The parity PASS covers fewer languages on the catch axis than the FP axis. This nuance must not be lost when "parity PASS" is quoted.

---

## 11. Phase 7 — cascade screener and distillation

**Objective:** resolve most traffic on a fast CPU model, reserving the ensemble for genuinely ambiguous audio.

### 11.1 V8-era attempts — the gate that kept failing

Four screener configurations were tried against V8, and the notebook records an explicit, automated exit gate for each:

| Attempt | Screener val EER | Stage-1 resolution | Gate (≥80%) | Source |
|---|---|---|---|---|
| 1 | 0.0525 | **83.6%** | **PASSED** | `[356–357]` |
| 2 | 0.0175 | 67.3% | **FAILED** | `[360–361]` |
| 3 | 0.0150 | 69.1% | **FAILED** | `[363–364]` |
| 4 | 0.0125 | — | — | `[369]` |

**The instructive pattern: better screener EER produced *worse* coverage.** A sharper model pushes scores toward the middle of the calibrated band, so fewer chunks clear the confidence thresholds. Optimising the screener's accuracy in isolation actively harmed the cascade's purpose.

A separate distillation gate failed on capability — *"EXIT GATE (EER within 3pp of ensemble): FAILED"*, though size passed at 10.35 MB `[368]`. A later student passed: *"student=0.0500, ensemble=0.2153, student BETTER than teacher"* `[375]`.

That last line deserves scrutiny rather than celebration: the student beat the teacher because **the teacher was weak on that set** — the notebook flags *"Teacher (V8) EER: 0.1930 ⚠ HIGH — teacher confused on these samples"* `[371]`, `[373]`. A student outperforming its teacher is a signal about the evaluation set, not a breakthrough.

### 11.2 V9 screener — the adopted model

**LightCNN:** 4 MaxFeatureMap conv blocks → AdaptiveAvgPool2d → bidirectional GRU(64→128) with **mean-pooling over all timesteps** → dropout → classifier. **287,234 parameters (~1.1 MB)**.

Input: mel spectrogram (n_fft 512, hop 160, win 400, 80 mels, 20–8000 Hz), per-sample normalised.

Trained by knowledge distillation against the V9 ensemble: 70% KL on soft labels (T=4) + 30% hard CE. Teacher scores over 4,720 train / 1,199 val entries; teacher accuracy 97.1% train, 84.4% val.

| Iteration | Change | Best EER | Resolved FPR | Source |
|---|---|---|---|---|
| v1 | Last-GRU-timestep pooling | 9.17% | 1.0% | `[413]` |
| **v2** | **Mean-pool + warmup + Platt** | **8.18%** | **0.9%** | `[414]` |

**The oscillation is severe and visible in the raw logs.** v2's 50 epochs swing between 8.18% and 56.83%, with adjacent epochs differing by 40+ points (e.g. epoch 35: 25.33%, epoch 36: 8.18%, epoch 38: 20.50%) `[414]`. The adopted 8.18% is a **best-checkpoint selection from a highly unstable run**, not a converged value. True performance is estimated at 9–13% EER.

**This is a model-risk item, not a footnote.** A best-of-50 selection from an oscillating run may not reproduce on retraining, and the deployed screener resolves ~86% of all production traffic.

Platt calibration: `P = sigmoid(10.9310 · raw − 4.1481)`.

### 11.3 Cascade performance

**Validation (1,199 entries):** screener EER **8.18%**, stage-1 resolution 86.2%, resolved accuracy 95.6%, resolved FPR/FNR 0.9%/7.9%, latency p50 **11.0 ms** / p95 **12.3 ms** (CPU).

**Bias-audit set (574 of 599 scored)** `[415]`: EER **0.00%**, catch 100%, FPR 0%, stage-1 resolution 96.7% (555 stage 1, 19 stage 2), 34.0 ms average. Parity PASS across all 7 languages.

**Recorded honesty note, carried verbatim from the Phase 7 handoff:** the 0% bias-audit EER *"is partly because that set is easier than the val set (no adversarial `real_in_the_wild` samples). The val EER of 8.18% is the honest performance estimate."* **Customer-facing material must quote 8.18%, never 0%.**

Additional caveat: 25 of 50 Hausa test fakes fail to load (corrupt MP3, consistent across sessions).

---

## 12. The AASIST V9 collapse and the `v9h` recovery

### 12.1 Discovery — measured, not inferred

AASIST V9 loaded cleanly with a strict key match and correct architecture, but was **contributing nothing**. The notebook's probe `[423]` is unambiguous:

```
class0:       fake_mean=0.0003±0.0007   real_mean=0.0005±0.0033   AUC=0.5258
class1:       fake_mean=0.9997±0.0007   real_mean=0.9995±0.0033   AUC=0.4743
logit_margin: fake_mean=-10.76±2.74     real_mean=-11.03±2.85     AUC=0.5258
```

**AUC 0.5258 is coin-flip.** Fake and real audio receive statistically indistinguishable outputs — means differ in the fourth decimal while standard deviations are an order of magnitude larger. The logit margin is saturated at ≈ −11 for everything. The XGBoost fusion, seeing a near-constant feature, had effectively ignored it.

### 12.2 Root cause

Diagnosed in `docs/AASIST-V9-retrain-recipe.md`. Four defects, all in **how it was trained**, not the architecture:

| # | Defect | Effect |
|---|---|---|
| 1 | **No class balancing** — plain `CrossEntropyLoss`, no `weight=`, no `WeightedRandomSampler` | On a real-heavy manifest, loss is minimised by predicting the majority class. **Primary cause.** |
| 2 | **No logit regularisation** — `label_smoothing = 0.0` over 100 epochs | Saturation → uninformative feature for the fusion |
| 3 | **Per-sample `weight` ignored** — in the manifest, fed to neither sampler nor loss | Intended balancing silently dropped |
| 4 | **Validation on a V8 set** — best checkpoint chosen by `val_v8_fresh.json` EER | "Best EER" looks fine while the model fails on the families that matter |

Ruled out: preprocessing mismatch — training peak-normalises exactly as inference does.

**Note the recurrence.** Defect 1 is the same class-imbalance failure present in the original 20:1 ASVspoof/WaveFake corpus (§4.1). It has now caused a material failure twice, four versions apart.

**The control lesson:** the model passed its checkpoint-selection criterion and loaded without error. **Nothing in the automated pipeline caught it.** It was found by targeted probing of the sub-model's score distribution. §17 item 5 records the corresponding gap.

### 12.3 The `v9h` recovery — "Route B"

Rather than block on a corrected retrain, `v9h` restored capability by reconfiguring around the known-good component:

1. **AASIST reverted to the proven V8 checkpoint** (`aasist_v8.pt`).
2. **XGBoost fusion and Platt calibration refit** around V8-AASIST + V9 features (`xgb_v9_hybrid`, `cal_v9_hybrid`).
3. **Screener early-out widened** — `CASCADE_LOW` 0.20 → **0.10**, so more suspected-real audio reaches the full ensemble.

Deployed configuration, from `model_store/v9h/bundle.json`:

| Parameter | Value |
|---|---|
| `CASCADE_LOW` / `CASCADE_HIGH` | 0.10 / 0.80 |
| LCNN Platt | coef 10.9310, intercept −4.1481 |
| Verdict thresholds | 0.85 / 0.55 / 0.30 |
| Bundle files | 7, each SHA-256 pinned |
| Startup smoke check | `fake_noizai_a4cd.mp3` must return `LIKELY_FAKE` or `AUTO_FAKE` |

The last row is a deployed control: the service **fails closed** at startup if the active bundle cannot classify its own fixture (`api.py` lifespan → `detector.startup_check()`).

**Measured `v9h` performance** (`sweep_cascade.py`, 73 fake / 80 real): fake recall **72/73 (98.6%)** at ≥0.55, 100% at ≥0.30; real FP **6.25%** at ≥0.55.

### 12.4 `v9fixed` — candidate evaluated and rejected

On 2026-07-14 a properly retrained AASIST was bundled as `v9fixed`, promoted at 13:28 UTC, evaluated, and **rolled back at 16:54 UTC** — *"v9fixed no better than v9h; keep v9h"*. The bundle is retained and registered. The corrected-retrain work is **done but not yet demonstrably better**.

---

## 13. Current production state

| Property | Value |
|---|---|
| **Active bundle** | **`v9h`** |
| Activated | 2026-07-12, reconfirmed 2026-07-14 after the `v9fixed` rollback |
| Composition | LCNN V9 screener · AASIST **V8** · Wav2Vec2 V9 · RawNet3 V8 · hybrid XGBoost + Platt |
| Integrity | 7 files, per-file SHA-256, verified on pull |
| Determinism | Deterministic on CPU; pinned by `tests/test_golden.py` |
| Serving | Two-stage cascade; ~86% resolve at stage 1 in ~11 ms |
| Fail-closed | Startup smoke check refuses a bundle that cannot classify its fixture |
| Rollback | One command (`bundle_registry.py rollback`); hash-chained pointer log |

**Golden regression baseline** (`tests/golden_manifest.json`, tol 1e-3):

| Clip | Score | Verdict | Truth |
|---|---|---|---|
| `real_studio_037.mp3` | 0.1062 | AUTO_REAL | real |
| `real_studio_055.mp3` | 0.1088 | AUTO_REAL | real |
| `fake_noizai_a4cd.mp3` | 0.9235 | AUTO_FAKE | fake |
| `fake_concert_hall.mp3` | 0.8000 | LIKELY_FAKE | fake |

---

## 14. Consolidated metrics

> **⚠ The V7 boundary.** Metrics from V1–V6 were measured on an easier validation set than V7 onward. **Do not compare across this line.** §4.5 quantifies the shift: V4's AASIST scored 5.60% on the old set and 31.92% on the V7 set — same model, same weights.

| Version | Val EER | Val AUC | Notes |
|---|---|---|---|
| V1 | 5.60% (AASIST) | — | Real-world 0/6 |
| V3 | 2.90% (W2V) | — | Acoustic-only EER 4.00% |
| V4 | 1.53% (W2V) / 1.22% OOF | 99.93% | Real-world 6/6; benchmark EER 4.00% |
| V5 | 1.53% (W2V) / 0.98% OOF | 99.90% | AASIST + RawNet3 retrains rejected |
| V6 / V6b | 1.31% / 1.95% | 99.96% / 99.93% | Both reverted |
| ══════ | ══════ | ══════ | **HARDER VAL SET FROM HERE** |
| V7 | 12.74% (AASIST) / 14.90% (W2V) | 94.46% | |
| V8 | **7.41%** calibrated | **0.9790** | Production baseline |
| V9 | 13.25% → **16.19% corrected** | 0.9408 → 0.9027 | **See §16.1** |
| V9 cascade | **8.18%** (screener) | — | 0.00% on the easier bias set |
| **v9h** | **not re-measured** | — | **See §16.2** |

| Version | Studio FP | Noiz.ai catch | Bias-set EER |
|---|---|---|---|
| V8 | 80% | 60% | — |
| V9 | **12.0%** | **83.3%** | **2.43%** |
| v9h | 6.25% FP ‡ | 98.6% recall ‡ | not re-measured |

‡ Clean-label internal sweep (73 fake / 80 real) at ≥0.55 — a different protocol from the V9 rows.

---

## 15. Adversarial and robustness summary

| Threat | V8 measured | Status |
|---|---|---|
| FGSM | Ensemble EER 0.4600 (AASIST 0.9750) | Not mitigated |
| PGD | Ensemble EER 0.4800 (AASIST 1.0000) | Not mitigated; adversarial training had **zero** effect |
| Extended attack budget | Ensemble EER → 0.9300 at 40 min | Not mitigated |
| Phone/call audio | 40% catch (n=5) | Known weakness |
| Adversarial-input detection | AUC 0.7361, clean FP 29.5% | No deployable operating point |

---

## 16. Evidence gaps and unreconciled findings

Recorded openly because an auditor who finds these unaided will treat every other number as suspect.

### 16.1 The V9 val EER of 13.25% is wrong — a softmax inversion

Notebook `[420]` records:

```
Val EER: 16.19%  (was 13.25% with inverted softmax)
Val AUC: 0.9027
```

**The widely quoted V9 val EER of 13.25% was computed with an inverted softmax.** The corrected figure is **16.19%**, with AUC 0.9027 rather than 0.9408. The 13.25% figure propagated into `Handoff_Summary_V9_Final.md`, `Handoff_Summary_V9_Phase7.md`, and every downstream summary — none of which record the correction.

The corrected number changes the story materially: the V8 → V9 val regression is **7.41% → 16.19%**, not 7.41% → 13.25%. **Every document quoting 13.25% must be corrected.**

### 16.2 Headline metrics were not measured on the deployed bundle

The quoted figures — **EER 2.43%**, catch **99.2%**, studio FP **12.0%** — were measured on the **V9** bundle. The deployed bundle is **`v9h`**, which differs precisely in the sub-model later found to be contributing nothing (§12). `MODEL_CARD.md` §4 presents these V9 numbers under a card describing `v9h`.

The numbers are not necessarily wrong for `v9h` — but they were **not measured on it**, and the direction of the difference is unknown.

**Required before any customer-facing claim:** re-run the bias audit and held-out evaluation against `v9h`; restate §10 and §14. Until then, source claims as *"measured on V9; the deployed v9h configuration differs — see VG-DOC-001 §16.2."*

### 16.3 Unreconciled recall finding

An internal assessment recorded that V9 **misses roughly half of TTS and multilingual fakes**, attributed to the screener early-out plus weak ensemble contribution. This sits in tension with §10's 99.2% catch and §12.3's 98.6% recall.

They may measure different populations — but the discrepancy is **unresolved** and concerns the core capability claim. Note that widening `CASCADE_LOW` to 0.10 in `v9h` was a direct response to the early-out mechanism implicated here.

### 16.4 No independent benchmark

No public-benchmark number exists. The **ASVspoof 2021 LA** harness is built (`scripts/asvspoof_eval.py`) but **has never been run** — despite ASVspoof 2019 being a primary *training* corpus since V1. Every performance figure here is self-reported on internally constructed sets. This is the highest-value evidence gap for enterprise due diligence.

### 16.5 `real_in_the_wild` FPR understated in summaries

Handoff notes record 25.3%; the notebook measures **30.3%** `[419]` and **31.7%** `[420]`. Summaries should be corrected to the measured value.

### 16.6 The adopted screener is a best-of-50 from an unstable run

Per §11.2. The 8.18% is not a converged value, and the screener handles ~86% of production traffic. Reproducibility on retrain is unverified.

---

## 17. Known limitations carried into production

| # | Limitation | Status |
|---|---|---|
| 1 | Not robust to gradient-based adversarial attack (§6, §15) | Accepted, disclosed |
| 2 | Phone/call-quality audio materially weaker (40% catch) | Accepted, disclosed |
| 3 | RawNet3 never retrained; retains "clean = fake" bias; `real_in_the_wild` FPR ~31% | Accepted; needs GPU or cuDNN-compatible GRU |
| 4 | Arabic FPR 16%; mitigation costs 10 pp catch and needs unavailable language detection | Open (§10.3) |
| 5 | Yoruba/Igbo catch rate untested — no TTS available | Open |
| 6 | Screener EER unstable across epochs; deployed value is a best-checkpoint pick | Accepted (§16.6) |
| 7 | 25 of 50 Hausa test fakes are corrupt MP3s | Open — test-data quality |
| 8 | Sub-model collapse not caught by any automated control (§12.2) | **Open control gap** |
| 9 | Verdicts are advisory, not determinative; not an accredited forensic instrument | By design, disclosed |

**Item 8 deserves emphasis for the GRC pack.** Existing controls — golden regression, startup smoke check, drift monitor — all validate the **ensemble's end-to-end output**. None validates that each **sub-model** still contributes signal. A per-sub-model health check asserting score spread and standalone AUC on a fixed probe set would have caught the AASIST collapse in seconds; the probe that eventually found it `[423]` is exactly that test, run manually and too late.

---

## 18. Open items

| # | Item | Owner | Blocks |
|---|---|---|---|
| 1 | Correct the 13.25% → 16.19% figure across all documents | ML | §16.1 — a published number is wrong |
| 2 | Re-measure bias audit + held-out on `v9h`; restate §10/§14 | ML | §16.2 — all external claims |
| 3 | Reconcile the §16.3 recall discrepancy under one protocol | ML | Core capability claim |
| 4 | Run ASVspoof 2021 LA; publish the number | ML | §16.4 — independent validation |
| 5 | Add a per-sub-model health check to the promotion gate | ML/Eng | §17 item 8 |
| 6 | Re-run the screener training to test 8.18% reproducibility | ML | §16.6 |
| 7 | Determine whether `v9fixed` can beat `v9h`, or formally retire it | ML | AASIST remains a V8 component |
| 8 | Emit machine-readable evaluation artifacts | ML | §2 — traceability |
| 9 | Non-technical review of the legal explainability template | Project owner | Phase 6 exit gate |

---

## Appendix A — Training-run register

One row per identifiable training or refit run, with its outcome. This is the "what was tried, what was kept, what was thrown away" record — the evidence that **acceptance was gated, not automatic**. Many runs produced weights that were deliberately discarded; that is the system working, not effort wasted.

**Outcome key:** ✅ **Adopted** (its weights are in a bundle, current or historical) · ↩️ **Reverted** (trained, then the previous weights were restored) · ❌ **Rejected** (failed an explicit gate) · 🔴 **Adopted-then-failed** (adopted, later found defective) · ⏸️ **Abandoned** (not completed).

Metrics are cited to notebook cells (`[N]`). Where a version had several attempts, each is a row.

### A.1 Sub-model runs

| # | Run | Version | Best result | Outcome | Reason / evidence |
|---|---|---|---|---|---|
| 1 | AASIST initial | V1 | EER 5.60% `[163]` | ✅ Adopted | First discriminative AASIST |
| 2 | RawNet3 initial | V1 | EER 11.68% `[163]` | ✅ Adopted | **Never retrained since** — byte-identical in every bundle (VG-DOC-004 §4) |
| 3 | Wav2Vec V3 | V3 | EER 2.90% `[154]` | ✅ Adopted | Superseded by V4 |
| 4 | Wav2Vec V4 | V4 | **EER 1.53%** (epoch 5) `[156]` | ✅ Adopted | Long-lived reference; still the comparison baseline through V6 |
| 5 | Wav2Vec V5 (small) | V5 | EER 1.22% `[173]` | ↩️ Reverted | *"V4 reloaded"* `[175]` — not kept |
| 6 | Wav2Vec V5 (696,280 files) | V5 | EER 1.53% (epoch 9) `[231]` | ✅ Adopted | Kept; large-corpus retrain |
| 7 | **AASIST V5** | V5 | Best 15.33%, unstable 33.9–55.6% `[227]` | ❌ Rejected | *"kept — V5 did not improve"* `[231]` — V4 AASIST retained |
| 8 | **RawNet3 V5** | V5 | Best 11.68% at epoch 0; all epochs worse `[230]` | ❌ Rejected | *"kept — V5 did not improve"* `[231]` — V4 RawNet3 retained |
| 9 | Wav2Vec V6 | V6 | EER 1.31% `[190]` | ↩️ Reverted | *"V4 reloaded"* `[192]` |
| 10 | Wav2Vec V6b | V6b | EER 1.95% `[202]` | ↩️ Reverted | Not retained over V4/V5 |
| 11 | AASIST V7 | V7 | 31.92% → best **12.74%** `[266][268]` | ✅ Adopted | Retrained for the harder val set (§4.5) |
| 12 | Wav2Vec V7 | V7 | 24.58% → **14.90%** `[274]` | ✅ Adopted | Same |
| 13 | AASIST V8 | V8 | **EER 12.98%** (13 epochs) `[290]` | ✅ Adopted | Production baseline; **later reused in v9h** |
| 14 | Wav2Vec V8 | V8 | **EER 12.69%** (10 epochs) `[292]` | ✅ Adopted | Healthy monotonic curve |
| 15 | **AASIST V9** | V9 | Best 15.00% (epoch 39) `[386]` | 🔴 **Adopted-then-failed** | Passed its val gate; later found **collapsed, AUC 0.5258** `[423]` (§12) |
| 16 | Wav2Vec V9 | V9 | Best 15.83% (epoch 28) `[408]` | ✅ Adopted | Still deployed in v9h |
| 17 | **RawNet3 V9** | V9 | — | ⏸️ Abandoned | ~1 hr/epoch on T4 with cuDNN disabled (§9.2) — V8 retained |
| 18 | **AASIST V8 fine-tune (combined)** | — | noiz.ai +16.7pp but studio FP 80%→84% `[377–379]` | ❌ Rejected | Gate rejected — frozen front-end can't fix "clean=fake" (§8) |
| 19 | **AASIST v9fixed** | v9fixed | Balanced retrain, V8-arch | ❌ Rejected | Promoted then rolled back — *"no better than v9h"* (§12.4) |

### A.2 Fusion / calibration runs

| # | Run | Version | Result | Outcome | Reason |
|---|---|---|---|---|---|
| 20 | XGBoost V4 | V4 | OOF EER 1.22% `[158]` | ✅ Adopted | |
| 21 | XGBoost V5 | V5 | OOF EER 0.98% `[231]` | ✅ Adopted | |
| 22 | XGBoost V9 iter 1 | V9 | Val 22.11%, held-out 15.62% `[388]` | ❌ Rejected | Superseded by iter 3 |
| 23 | XGBoost V9 iter 2 | V9 | Val 15.33%, held-out **23.00%** `[391]` | ❌ Rejected | Held-out diverged upward |
| 24 | **XGBoost V9 iter 3** | V9 | Val **13.25%** (→16.19% corrected), held-out 14.33% `[394]` | ✅ Adopted | Beat the simple-average baseline `[396]` |
| 25 | Simple-average fusion | V9 | Val 14.92% `[396]` | ❌ Rejected | Baseline; learned fusion won |
| 26 | Hybrid XGBoost/Platt | v9h | — | ✅ **Adopted (current)** | Refit around V8-AASIST (§12.3) |
| 27 | XGBoost/Platt v9fixed | v9fixed | — | ❌ Rejected | With run 19 |

### A.3 Screener / distillation runs

| # | Run | Result | Gate | Outcome |
|---|---|---|---|---|
| 28 | Screener attempt 1 | EER 0.0525, **83.6%** resolution `[356]` | ≥80% | ✅ **PASSED** |
| 29 | Screener attempt 2 | EER 0.0175, 67.3% `[360]` | ≥80% | ❌ **FAILED** — sharper EER, worse coverage (§11.1) |
| 30 | Screener attempt 3 | EER 0.0150, 69.1% `[363]` | ≥80% | ❌ **FAILED** |
| 31 | Distillation student (early) | Within-3pp EER | 3pp | ❌ **FAILED** `[368]` |
| 32 | Distillation student (later) | student 0.0500 vs ensemble 0.2153 `[375]` | 3pp | ⚠️ PASSED — but against a **weak teacher** (§11.1) |
| 33 | **LCNN v1** | EER 9.17%, band collapsed `[413]` | — | ❌ Rejected | Last-timestep pooling; unusable router |
| 34 | **LCNN v2** | EER **8.18%** (best-of-50, unstable) `[414]` | — | ✅ **Adopted (current screener)** — but see §16.6 |

### A.4 What the register shows

- **34 identifiable runs; ~15 discarded** (rejected, reverted, or abandoned). A roughly 2:1 kept-to-discarded ratio is healthy — it means candidates were actually being tested against a bar, not accepted by default.
- **Every rejection has a recorded reason.** Nine were rejected against an explicit numeric gate (runs 7, 8, 18, 19, 22–25, 29–31).
- **Two adopted models later proved problematic** (run 15, the AASIST collapse; run 34, the unstable screener). Both are the origin of active-control recommendations (§17 items 5, 6).
- **The gap this exposes:** acceptance gates existed for *aggregate* metrics (EER, coverage, held-out) at every stage, yet run 15 passed its gate and still shipped a dead sub-model. The gate measured the wrong thing (§12.2). This register is the evidence for why Policy 6 in the GRC pack (per-sub-model validation) is needed.

---

## 19. Artifact index

**Primary record:** `voiceguard-training-notebook (3).ipynb` — 425 cells with retained outputs, V1 → V9.

**Training/evaluation:** `build_v9_manifest.py` · `retrain_aasist_v9.py` · `retrain_wav2vec_v9.py` · `refit_ensemble_v9.py` · `generate_teacher_scores_v9.py` · `train_lcnn_distill_v9_cell.py` · `cascade_bias_audit_eval_cell.py` · `generate_bias_fakes.py` · `v8_finetune_combined.py` · `studio_fp_check.py` · `studio_processed_fake_test.py` · `fetch_studio_clips.py` · `scripts/asvspoof_eval.py` · `sweep_cascade.py` · `aasist_probe.py`

**Governance/serving:** `bundle_registry.py` · `governance.py` · `drift_monitor_3.py` · `detector.py` · `api.py` · `worker.py` · `forensic_report.py`

**Bundles:** `model_store/v9/` · `model_store/v9h/` (active) · `model_store/v9fixed/`

**Reports:** `VoiceGuard_V9_Bias_Audit_Report.docx` · `Phase5_Findings_VoiceGuard_V8.docx` · `Phase6_Asymmetric_Thresholds_Cost_Rationale.docx` · `Phase6_Legal_Explainability_Report_Template.docx`

---

## 20. Document change log

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-07-22 | Michael Ologungbara | Baseline. V8 → V9 → v9h → v9fixed. Pre-V8 lineage open. |
| 1.1 | 2026-07-22 | Michael Ologungbara | Training notebook incorporated. **§4 complete (V1–V7).** All metrics re-sourced to notebook cells. New findings: softmax-inversion error invalidating the published 13.25% (§16.1); AASIST collapse quantified at AUC 0.5258 (§12.1); V7 measurement-boundary identified (§4.5); screener instability characterised (§11.2); `real_in_the_wild` FPR understated (§16.5); Arabic threshold trade quantified (§10.3). |
| 1.2 | 2026-07-23 | Michael Ologungbara | Added **Appendix A — training-run register**: 34 identifiable runs with adopted/reverted/rejected/abandoned outcome and cell-cited evidence. Makes the accepted-vs-discarded distinction explicit and evidences that acceptance was gated. |

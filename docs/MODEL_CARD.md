# VoiceGuard Model Card — bundle `v9h`

*Phase 8 documentation. This card describes the model that ships in the active bundle. It follows
the standard model-card structure (details · intended use · data · evaluation · limitations ·
ethics · maintenance).*

---

## 1. Model details
- **Name / version:** VoiceGuard cascade detector, active bundle **`v9h`** (derived from V9).
- **Type:** Audio deepfake / synthetic-speech detector. Input: an audio file; output: a calibrated
  probability of synthesis `[0,1]`, a plain-language verdict, and a forensic explainability payload.
- **Architecture — two-stage cascade** (applied per 4 s chunk):
  - **Stage 1 — LCNN screener** (mel-spectrogram CNN). Confident chunks (`p ≤ CASCADE_LOW` or
    `p ≥ CASCADE_HIGH`) resolve here (fast, CPU).
  - **Stage 2 — 3-model ensemble** on uncertain chunks: **AASIST + Wav2Vec2 + RawNet3 → XGBoost
    fusion → logistic (Platt) calibration.** Chunk scores are aggregated with confidence-weighting.
- **Cascade thresholds (v9h):** `CASCADE_LOW = 0.10`, `CASCADE_HIGH = 0.80`.
- **Verdict thresholds:** `auto_fake ≥ 0.85`, `likely_fake ≥ 0.55`, `to_review ≥ 0.30`, else `auto_real`.
- **Determinism:** fully deterministic (eval mode, input-randomization off, CPU) — the same file
  produces the same score on repeat, which the golden regression (`tests/test_golden.py`) pins.

### Sub-models in `v9h`
| Model | Version in v9h | Role | Notes |
|---|---|---|---|
| LCNN screener | V9 | stage-1 gate | resolves confident chunks; `CASCADE_LOW` lowered 0.20→0.10 in v9h |
| AASIST | **V8 architecture** (`aasist_v8.pt`) | ensemble | reverted from the V9 retrain, which had collapsed (see §7); V8 AASIST is discriminative |
| Wav2Vec2 | V9 (head retrained, base frozen) | ensemble | strongest general fake detector |
| RawNet3 | V8 | ensemble | XGBoost down-weights it (~0.15) |
| XGBoost + Platt | **hybrid refit** (`xgb_v9_hybrid` / `cal_v9_hybrid`) | fusion | refit over the V8-AASIST + V9 ensemble features |

**`v9h` provenance:** V9's from-scratch AASIST retrain (on a narrow noizai+Hausa manifest) collapsed
to predicting "real" for everything, dropping general fake recall. `v9h` reverts AASIST to the
proven V8 architecture, refits the XGBoost fusion around it, and lowers the screener's "real"
early-out (`CASCADE_LOW` 0.20→0.10) so more suspected-reals reach the ensemble. Rollback to `v9`
is preserved via the model registry.

## 2. Intended use
- **In scope:** flagging whether an audio recording shows characteristics consistent with
  AI-generated / cloned / converted speech, as **one advisory signal** among others.
- **Out of scope / must NOT be used for:** speaker identification, voice biometric matching, or
  determining *who* is speaking or their intent. It is **not** an accredited forensic instrument;
  outputs are advisory (see the legal report template, `forensic_report.py`).
- **Client priority:** Nigeria-based; Nigerian-language audio (Yoruba, Hausa, Igbo, Pidgin) is a
  first-class target.

## 3. Training data
- **Manifest:** `train_v9.json` — ~4,746 entries, ~1.07:1 real:fake balance. Sources:
  original V8 data (4,400 across 7 source buckets), studio reals (226), noiz.ai fakes (70), Hausa
  TTS fakes (50, added during bias-audit mitigation).
- **Provenance:** the trained weights are the product IP; large weights are **not** in git — they
  live in a DigitalOcean Spaces bucket and are pulled + SHA-256-verified via `bundle_registry.py
  pull --active` (see `docs/CI-and-model-store.md`).
- Preprocessing: 16 kHz mono; the stage-2 ensemble peak-normalizes (matches its training/calibration);
  the LCNN mel is per-sample standardized (scale-invariant).

## 4. Evaluation
- **Bias-audit set** (599 entries, 7 languages): overall **EER 2.43%**, catch **99.2%**, FPR 5.4%.
  **Catch-rate parity PASS — zero violations across languages** (Yoruba 2% FPR, Igbo 4%, Pidgin 0%).
  Arabic FPR elevated (16%) — a *recording-quality* artifact, not language bias (see the Bias Audit
  Report). See `docs/new_docs/VoiceGuard_V9_Bias_Audit_Report.docx`.
- **Held-out studio set:** studio false-positive **12.0%** (V8 was 80%), noiz.ai catch **83.3%**
  (V8 was 60%) — the V9 goals.
- **v9h clean-label internal sweep** (73 fake / 80 real, correct fake/real partitioning): fake
  recall **72/73 (98.6%)** at `≥0.55`, 100% at `≥0.30`; real FP **6.25%** at `≥0.55`. `sweep_cascade.py`.
- **ASVspoof 2021 LA benchmark:** harness ready (`scripts/asvspoof_eval.py`, per-attack + `--split
  eval`); the standard EER number is **pending a run** on the ASVspoof data.
- **Calibration (V9 reference):** `P = sigmoid(4.8246·xgb − 2.1469)`; `v9h` ships the hybrid-refit
  calibration (`cal_v9_hybrid.json`).

## 5. Additive forensic signals (advisory; never change the verdict)
Each detection also returns: **Grad-CAM heatmap** (which frequencies triggered detection) + **SHAP**
(which model drove the score) + **timestamp segment flagging** + **prosody observations** +
**adversarial-attack risk** (`adversarial_monitor_v2`) + **AudioSeal watermark** + **metadata
forensics** + **C2PA Content Credentials** (`c2pa_validation`) + **mic-signature** + `audit_id` +
`confidence`. Any failure degrades to a null field; detection never breaks.

## 6. Ethical considerations & bias
- Multilingual **catch-rate parity passes** with no Nigerian-language bias.
- A "likely synthetic" result is **not proof** of fraud/forgery — it is one technical signal to be
  weighed with others. A "no indication" result does not guarantee authenticity.
- The system is not independently certified as a forensic tool; the legal report template discloses
  this explicitly.

## 7. Limitations
- **Deliberate adversarial manipulation** may evade detection (mitigated, not eliminated, by the
  advisory adversarial monitor).
- **Phone / call-quality audio** is harder to analyze than studio audio; treat with caution.
- **Language/accent coverage** is an ongoing testing area; Arabic showed elevated FPR from recording
  quality.
- Historical note: the V9 from-scratch **AASIST retrain collapsed** (catastrophic forgetting on a
  narrow manifest); `v9h` fixed this by reverting AASIST to V8 + refitting the fusion. A proper
  V9-architecture AASIST retrain recipe exists (`docs/AASIST-V9-retrain-recipe.md`).

## 8. Maintenance & continual improvement
- **Drift monitor** (`drift_monitor_3.py`) re-scores a living test set on a schedule, detects EER /
  per-source catch-rate drift, and on **confirmed** drift fires a **retrain trigger**
  (`retrain_trigger.json` + `$VOICEGUARD_RETRAIN_CMD`), surfaced on `GET /drift`.
- **Model registry** (`bundle_registry.py`): hash-chained active pointer, `promote` / `rollback`,
  `push`/`pull` to DigitalOcean Spaces. Bundles are integrity-verified (SHA-256) on pull.
- Retraining runs externally (GPU/Kaggle); the trigger's command hook launches it, and a candidate is
  benchmarked (ASVspoof + sweep + golden) before promotion.

## 9. References
- API: `docs/API_REFERENCE.md` · Ops runbook: `docs/RUNBOOK-model-flow.md` · CI + model store:
  `docs/CI-and-model-store.md` · Legal report: `forensic_report.py` +
  `docs/new_docs/Phase6_Legal_Explainability_Report_Template.docx` · Bias audit:
  `docs/new_docs/VoiceGuard_V9_Bias_Audit_Report.docx`.

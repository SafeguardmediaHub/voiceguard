# VoiceGuard V8 — Handoff Summary (Post-Phase 5/6/7 Work)

> **For starting a new Claude conversation to continue this work.**
> Paste this entire document into your first message.

---

## Project context

**Engagement:** VoiceGuard V8 audio deepfake detection. Multi-phase client deliverable (NGN 500,000 per phase) plus Master's research component. Currently mid-Phase 7, with Phase 6 bias audit still open and a V8 retrain planned.

**Current priority:** Full AASIST retrain with expanded training data (studio reals + noiz.ai fakes), then cascade/distillation, then bias audit, then project wrap-up.

---

## V8 architecture (production reference)

**Ensemble:** AASIST (~285K params, XGB weight 0.501) + Wav2Vec2 head (~95M frozen + small classifier, weight 0.414) + RawNet3 (~6M, weight 0.086), fused via XGBoost + logistic calibration.

**Performance baseline:** Trained on ~16,400 samples across 7 source buckets. Val AUC 0.979, EER 7.41%. Clean ensemble EER on 200-sample balanced eval: **10.0%**.

**Verdict thresholds (deployed):** auto_fake ≥ 0.85, likely_fake ≥ 0.55, to_review ≥ 0.30, otherwise auto_real.

### Critical model details (for any retraining)

**AASIST v8 architecture:** SincConv 70-ch kernel 127 → BN → SELU → MaxPool(3) → 6 ResBlocks channels [32,32,64,64,128,128] strides [1,2,1,2,1,2] (ALL blocks have downsample) → AdaptiveAvgPool1d(64) → GAT(128→64) → GAT(64→64) → classifier Linear(64→64) SELU Dropout Linear(64→2). Takes 3D input (B,1,T).

**RawNet3 v8 architecture:** SincConvRaw(128, kernel 513) → BN → SELU → MaxPool(3) → 4 RawResBlocks [128,128,256,256] strides [1,2,1,2] with FMS → 2-layer GRU(256) → AttentionPooling → classifier Linear(256→128) SELU Dropout(0.25) Linear(128→2). Wrapper key: `model_state`. cuDNN must be disabled for GRU backward.

**Wav2Vec head:** `Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')` frozen + classifier `Sequential(Linear(768→256) ReLU Dropout(0.3) Linear(256→64) ReLU Dropout(0.15) Linear(64→2))`. Attribute is `classifier` not `head`. Takes 2D input (B,T).

---

## Kaggle environment

**Dataset:** `voiceguard-v8-artefacts` at `/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts/`

Contents: `aasist_v8.pt`, `wav2vec_v8.pt`, `rawnet3.pt`, `xgb_v8.json`, `cal_v8_params.json`, `thresholds_v8.json`, `val_v8_fresh.json` (1,200 entries), `train_v8_fresh.json` (4,438 entries).

**Noiz.ai samples:** `/kaggle/input/datasets/michaelologungbara/noizai-auidos` (16 .mp3 files)

**V9 train/test data:** `/kaggle/input/datasets/michaelologungbara/v9-train-test/` — contains:
- `new_samples/new_samples/fake/` (70 noiz.ai fakes for training)
- `new_samples/new_samples/real/` (226 studio clips for training)
- `held_out/held_out/fake/` (30 noiz.ai fakes, never trained on)
- `held_out/held_out/real/` (75 studio clips, never trained on)

Note: double-nested directory structure (`new_samples/new_samples/`).

---

## Phase 5 — COMPLETED

All tasks delivered, documented, findings doc updated. Key results:

- **Adversarial robustness (Tasks 1-3):** V8 broken by FGSM/PGD at all tested ε. Two adversarial training stages failed (gradient masking, non-convergence). Input randomization collapsed clean accuracy. All documented as honest negative results.
- **Distribution-shift vulnerability:** Phone-effected fakes (noiz.ai phone_like): 40% catch rate. All other effect classes: 100%. Concentrated in AASIST.
- **Effects augmentation:** Failed, regressed catch rate from 81% to 50%. Checkpoint discarded.
- **Server defenses (Task 4):** Delivered — rate limiting, duplicate detection, burst pattern flags.
- **Drift monitor (Task 5):** Built, patched with phone-class fingerprinting (v2), baseline established. Per-class breakdown: phone_like 40%, reverb_heavy 100%, clean_or_studio 100%, unclassified 100%.
- **Few-shot adaptation pipeline (Task 6):** Built (`fewshot_adapt.py`), logic-verified, ready for use.
- **Adversarial detection classifier:** Built, trained on 3,600 entries (300 source × 7 epsilons), rigorously evaluated. **No deployable operating point** — AUC 0.736, clean FP 29.5%, no threshold achieves both acceptable FP and useful catch rate. Documented as honest negative result in Phase 5 findings doc Section 5.

**Findings document:** `Phase5_Findings_VoiceGuard_V8.docx` — updated with Section 5 (adversarial detection classifier final characterization).

---

## Phase 6 — PARTIALLY COMPLETED

### Task 1-2: Governance & Institutional Layer — DONE

**`governance.py`** — fully built, tested, and verified:
- **Model registry:** SHA-256 intake hashing, versioned entries. All three production checkpoints registered (aasist_v8, wav2vec_v8, rawnet3).
- **Tamper-evident audit log:** Hash-chained, append-only. Tested against both attack patterns (in-place edit and deletion — both correctly detected).
- **Deterministic inference:** EXIT GATE PASSED — `0.16522590559706776` bit-identical across 20 CPU runs. Settings: PYTHONHASHSEED=42, CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn.deterministic=True, use_deterministic_algorithms=True.
- Windows UTF-8 encoding fix applied (cp1252 crash on report save).
- `--metadata-file` option added to avoid Windows cmd.exe JSON quoting issues.
- Note: audit log not yet wired into `server.py`'s actual `/detect` route for live logging.

### Tasks 3-4: Bias Audit — BLOCKED (data sourcing)

**`bias_audit_discovery.py`** — ran on Kaggle. Found:
- Existing Common Voice data is a flattened English-only Kaggle repackage (vedant2022), not the official Mozilla multilingual structure. No demographic metadata TSV available.
- edge-tts not installed on Kaggle; OpenAI API key not configured.

**`bias_audit_sourcing.py`** — built but blocked:
- AfriSpeech-200 loading fails: HF `datasets` library fully dropped script-based loading. Fix: `pip install "datasets==2.19.0"` (not yet tested).
- edge-tts Yoruba voices (`yo-NG-AdesuwaNeural`, `yo-NG-AyodeleNeural`) returned "No audio was received" errors — voices may not exist. Need to verify with `edge-tts --list-voices | findstr "yo-"`.
- ElevenLabs voice IDs for Hausa/Igbo not yet configured.

**Status:** Script infrastructure built, but actual multilingual audio sourcing not yet completed. Revisit after retrain.

### Task 5: Asymmetric Operating Thresholds — DONE

**`Phase6_Asymmetric_Thresholds_Cost_Rationale.docx`** — delivered. Four use cases (telephone banking fraud, CEO/CFO impersonation, KYC liveness, insurance claims), each with a per-score-band action mapping and cost-function rationale. Uses existing V8 thresholds (0.30/0.55/0.85), varies ACTION per band per use case.

### Task 6: Legal Explainability Template — DONE (pending non-technical review)

**`Phase6_Legal_Explainability_Report_Template.docx`** — delivered. 9 sections + appendix feedback form. Designed for non-technical audience (judge, compliance officer, adjuster). Includes known-limitations table, chain-of-custody section linking to governance.py, and structured reviewer feedback form for the exit gate.

**EXIT GATE NOT YET SATISFIED:** requires actual review by a non-technical person. Template is ready; the review step is on Michael.

---

## Phase 7 — IN PROGRESS

### Tasks 1-3: Cascade Screener + Distillation — PAUSED (pending retrain)

**`cascade_and_distillation.py`** — built with LCNN architecture (MFM activation, parameterized width_mult), includes:
- Stage-1 screener training (plain CE with label smoothing + weight decay)
- Cascade router with configurable thresholds
- Threshold sweep tool (`--sweep-thresholds`)
- Knowledge distillation student training (CE + KD vs teacher scores)

**Training runs completed (3 iterations):**

1. **width_mult=1, 2000 samples, no regularization:** 82.2% stage-1 accuracy, 83.6% coverage. Severe overfitting — train loss collapsed to 0.0003 while held-out accuracy was only ~82%. Threshold sweep showed NO margin achieving ≥95% accuracy.

2. **width_mult=1, 4000 samples, label_smoothing=0.05, weight_decay=1e-4:** 92.1% accuracy, 67.3% coverage. Regularization fixed overconfidence (loss plateaued at ~0.12, not 0.0003). Sweep: 96% accuracy achievable only at 45.5% coverage; 100% accuracy only at 8.5% coverage.

3. **width_mult=2, 4000 samples, same regularization:** 93.2% accuracy, 69.1% coverage. Marginal improvement over width_mult=1. Training loss hit same ~0.12 floor.

**Conclusion:** The cascade screener is paused pending the V8 retrain. It should be retrained/distilled against the updated V8, not the current one.

**Known bugs fixed during development:**
- BatchNorm2d channel mismatch (32 vs 24) — caught by forward pass test
- `fc1_out` formula wrong in parameterized LCNN (80*w vs correct 160*w) — caught by exact param-count verification
- Missing `cache_path` assignment in `cmd_sweep_thresholds`
- OOM in `evaluate_detector_breakdown` (unbatched forward pass on 3600 samples) — fixed to batch

### Tasks 4-6: API, Load Testing, Pen Test — NOT STARTED

---

## Edge Case Investigations — COMPLETED (major finding)

### Studio/Broadcast Audio False-Positive Check

**`studio_fp_check.py`** + **`fetch_studio_clips.py`** — sourced 325 clips (100 audiobook, 100 broadcast_news, 125 podcast) via yt-dlp + ffmpeg chunking.

**MAJOR FINDING: 65.5% overall false-positive rate on real studio/broadcast audio.**
- Audiobook: **95.0%** FP (67% hitting auto_fake)
- Podcast: **56.8%** FP
- Broadcast news: **47.0%** FP

**Root cause (confirmed):** V8's training data is predominantly amateur recordings. Professionally produced audio is acoustically closer to TTS output (clean, low-noise, full-frequency). V8 has learned "clean = fake, messy = real" as a proxy. This proxy works on its training distribution but fails on any well-produced real audio.

**Safety check completed before planning fix:** `studio_processed_fake_test.py` confirmed V8 catches TTS fakes using genuine synthesis artifacts, NOT the cleanliness proxy — 100% catch rate held after studio processing (0 flips, scores went slightly UP). Adding studio reals to training is safe.

Note: first chunking attempt used `-c copy` in ffmpeg which produced broken MP3 frames at chunk boundaries. Fixed to re-encode with libmp3lame. Re-run confirmed results were real (65.5% vs original 64.3% — essentially identical), ruling out tooling artifact.

### Combined Fine-Tune Attempt — GATE REJECTED

**`v8_finetune_combined.py`** — attempted fine-tuning AASIST (frozen front-end) with 296 new samples (70 noiz.ai fakes + 226 studio reals) + 300 replay buffer.

**Results:**
- Noiz.ai catch rate: 60% → **76.7%** (+16.7pp) ✓
- OpenVoice VC: 77.3% → **81.8%** (+4.5pp) ✓
- Studio FP: 80% → **84%** (WORSE) ✗
- real_in_the_wild FP: 39.2% → **51.0%** (+11.8pp) ✗

**Gate correctly rejected.** Reason: the studio FP problem lives in SincConv + early ResBlocks (the spectral representation layer), which were frozen. Fine-tuning the last 2 ResBlocks + GAT can learn new fake patterns but cannot override the front-end's "clean = fake" representation. A full retrain is needed.

---

## What needs to happen next (agreed plan)

### 1. Full AASIST retrain with expanded training data — NEXT

**What to build:**
- Rebuild `train_v8_fresh.json` with two new source buckets:
  - `real_studio` (~226 clips, capped like other real buckets)
  - `fake_noizai` (~70 clips, capped like other fake buckets)
- Keep exact same AASIST architecture and training procedure as V8
- Only variable: expanded data composition
- Retrain AASIST from scratch (NOT fine-tune — early layers need to change)
- Wav2Vec and RawNet3: keep as-is unless AASIST retrain results suggest otherwise
- Re-fit XGBoost + calibration after AASIST retrain (since AASIST's score distribution will shift)

**Validation (same held-out sets used in the fine-tune attempt):**
- `val_v8_fresh.json` per-source breakdown (no existing bucket regresses)
- Held-out studio reals (75 clips): FP rate must improve dramatically vs baseline 80%
- Held-out noiz.ai fakes (30 clips): catch rate must improve vs baseline 60%
- Run `drift_monitor.py` before/after

### 2. Cascade + Distillation — AFTER retrain

Retrain the LCNN screener and distilled student against the UPDATED V8, not the current one. Scripts are ready (`cascade_and_distillation.py`), just need to be re-run post-retrain.

### 3. Bias Audit — AFTER retrain

Unblock the data sourcing issues (AfriSpeech, edge-tts voices, ElevenLabs config), then run the actual audit against the retrained model. Scripts are ready (`bias_audit_sourcing.py`), blocked on data access.

### 4. Project wrap-up

- Wire audit log into `server.py` for live detection logging
- Get the legal explainability template reviewed by a non-technical person
- Update Phase 5/6 findings documents with retrain results
- Phase 7 Tasks 4-6 (API hardening, load testing, pen test)

---

## Files delivered across this conversation

**Phase 5 completion:**
- `drift_monitor.py` (v2 — patched with phone-class fingerprinting)
- `fewshot_adapt.py` (few-shot adaptation pipeline)
- `adversarial_detector.py` (adversarial detection classifier + ROC sweep)
- `Phase5_Findings_VoiceGuard_V8.docx` (updated with Section 5)

**Phase 6:**
- `governance.py` (model registry, audit log, determinism verification)
- `bias_audit_discovery.py` (multilingual data discovery)
- `bias_audit_sourcing.py` (test set construction — blocked on data access)
- `Phase6_Asymmetric_Thresholds_Cost_Rationale.docx`
- `Phase6_Legal_Explainability_Report_Template.docx`

**Phase 7:**
- `cascade_and_distillation.py` (LCNN screener + router + student — paused pending retrain)

**Edge case investigation:**
- `studio_fp_check.py` (studio/broadcast FP evaluation)
- `fetch_studio_clips.py` (yt-dlp + ffmpeg batch downloader/chunker)
- `studio_processed_fake_test.py` (safety check: do studio-processed fakes still get caught)
- `v8_finetune_combined.py` (combined fine-tune — gate rejected, leading to full retrain decision)

---

## Key decisions and findings across this conversation

- **Phone-effect vulnerability confirmed** at 40% catch rate on noiz.ai phone_like (n=5), stable across drift monitor runs
- **Adversarial detection classifier: no deployable operating point** (AUC 0.736, clean FP 29.5%). Documented as honest negative result.
- **Deterministic inference exit gate PASSED** (20 CPU runs, bit-identical)
- **Studio/broadcast FP: 65.5% overall** — a major, previously uncharacterized blind spot. Root cause: V8 learned "clean = fake" proxy from amateur-only real training data.
- **Studio processing does NOT degrade fake catch rate** (100% held, 0 flips) — confirmed V8 uses genuine synthesis artifacts, not cleanliness proxy, for fake detection. Adding studio reals is safe.
- **Fine-tuning cannot fix the studio FP problem** — frozen front-end can't override the spectral representation. Full retrain needed.
- **Fine-tuning DID improve noiz.ai catch rate** (+16.7pp) and OpenVoice VC (+4.5pp) — these gains should carry over to the full retrain with more data.

---

## How to continue in the new conversation

Paste this whole document as the first message. Then say what you want next. The agreed plan is:

**Step 1:** Build the manifest-construction script for the full AASIST retrain (adding `real_studio` and `fake_noizai` buckets to `train_v8_fresh.json`, with per-bucket caps).

**Step 2:** Run the retrain, validate against held-out sets + existing val manifest.

**Step 3:** If retrain passes validation, re-run cascade/distillation against the updated model.

**Step 4:** Revisit bias audit data sourcing.

**Step 5:** Wrap up remaining Phase 6/7 deliverables.

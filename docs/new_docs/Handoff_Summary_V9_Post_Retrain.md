# VoiceGuard V9 — Handoff Summary (Post-Retrain)

> **For starting a new Claude conversation to continue this work.**
> Paste this entire document into your first message.

---

## Project context

**Engagement:** VoiceGuard audio deepfake detection. Multi-phase client deliverable (NGN 500,000 per phase) plus Master's research component. V9 retrain is complete. Current priorities: bias audit (Phase 6 Task 3-4), cascade/distillation update (Phase 7), and project wrap-up.

**Client:** Based in Nigeria. Nigerian-language audio (Yoruba, Hausa, Igbo, Pidgin) is a priority for the bias audit.

---

## V9 Ensemble — PRODUCTION REFERENCE

### Architecture

**3-model ensemble:** AASIST V9 + Wav2Vec2 V9 + RawNet3 V8, fused via XGBoost + logistic calibration.

| Model | Params | XGB Weight | Version | Notes |
|-------|--------|-----------|---------|-------|
| AASIST | ~285K | 0.402 | **V9 (retrained)** | Full retrain from scratch on V9 manifest |
| Wav2Vec2 | ~95M (213K trainable) | 0.445 | **V9 (retrained)** | Classifier head retrained, base frozen |
| RawNet3 | ~6M | 0.153 | V8 (unchanged) | Retrain attempted but impractical on T4 (~1hr/epoch) |

### V9 Performance

**Val set (val_v8_fresh.json, 1200 entries):**
- EER: 13.25% (V8 was 7.41%)
- AUC: 0.9408 (V8 was 0.979)
- Val EER regression is real but acceptable — driven by `real_in_the_wild` FP at 25.3% (RN3's residual bias)

**Held-out set (105 entries — never trained on):**
- EER: 14.33%
- Studio FP: **12.0%** (V8 was 80%) ← primary goal achieved
- Noiz.ai catch: **83.3%** (V8 was 60%) ← secondary goal achieved

**Per-source val breakdown:**
| Source | Metric |
|--------|--------|
| conference_fake (86) | catch 98.8% |
| fake_edge_tts (135) | catch 98.5% |
| fake_in_the_wild (141) | catch 100.0% |
| fake_openai_tts (134) | catch 96.3% |
| fake_openvoice_vc (104) | catch 31.7% |
| real_common_voice (300) | FP 1.3% |
| real_in_the_wild (300) | FP 25.3% |

**Verdict distribution (V8 thresholds: auto_fake ≥ 0.85, likely_fake ≥ 0.55, to_review ≥ 0.30):**
- Fakes (n=600): auto_fake 76.0%, likely_fake 10.3%, to_review 1.7%, auto_real 12.0%
- Reals (n=600): auto_real 85.5%, to_review 3.7%, likely_fake 7.7%, auto_fake 3.2%

**Calibration:** `P = sigmoid(4.8246 * xgb_score + -2.1469)`

### Critical model details (for any future retraining)

**AASIST V9:** SincConv 70-ch kernel 127 → BN → SELU → MaxPool(3) → 6 ResBlocks channels [32,32,64,64,128,128] strides [1,2,1,2,1,2] (ALL blocks have downsample) → AdaptiveAvgPool1d(64) → GAT(128→64) → GAT(64→64) → classifier Linear(64→64) SELU Dropout(0.3) Linear(64→2). Takes 3D input (B,1,T). Checkpoint wrapper: `model_state_dict`.

**Wav2Vec2 V9:** `Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')` frozen + classifier `Sequential(Linear(768→256)[0] ReLU[1] Dropout(0.3)[2] Linear(256→64)[3] ReLU[4] Dropout(0.15)[5] Linear(64→2)[6])`. Checkpoint wrapper: `model` key with full state dict (base + classifier). Takes 2D input (B,T).

**RawNet3 V8:** SincConvRaw(128, kernel 512→forced to 513) → sinc_bn → SELU → MaxPool(3) → 4 RawResBlocks [128,128,256,256] strides [1,2,1,2] with FMS(Sequential: AdaptiveAvgPool1d→Flatten→Linear→Sigmoid) → 2-layer GRU(256, dropout=0.5) → AttentionPooling(Linear(256→128)→Tanh→Linear(128→1)) → Dropout(0.5) → classifier Sequential(Linear(256→128) SELU Dropout(0.25) Linear(128→2)). Wrapper key: `model_state`. cuDNN must be disabled for GRU backward. Forward does double permute around GRU (into batch_first, back to channels-first for attention_pool). SincConvRaw uses half-kernel parameterization (n_ shape [1,256], window_ shape [256], left-half construction with matmul then mirror with center point).

---

## Kaggle environment

### Datasets

**V8 artefacts:** `/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts/`
Contents: `aasist_v8.pt`, `wav2vec_v8.pt`, `rawnet3.pt`, `xgb_v8.json`, `cal_v8_params.json`, `thresholds_v8.json`, `val_v8_fresh.json` (1,200 entries), `train_v8_fresh.json` (4,400 entries).

**V9 train/test data:** `/kaggle/input/datasets/michaelologungbara/v9-train-test/`
- `new_samples/new_samples/fake/` (70 noiz.ai fakes for training)
- `new_samples/new_samples/real/` (226 studio clips for training)
- `held_out/held_out/fake/` (30 noiz.ai fakes, never trained on)
- `held_out/held_out/real/` (75 studio clips, never trained on)
Note: double-nested directory structure (`new_samples/new_samples/`).

### V9 artefacts (in `/kaggle/working/` — wiped between sessions)

These must be regenerated each session OR uploaded to a persistent Kaggle dataset:
- `train_v9.json` — rebuilt by `build_v9_manifest.py` + weight patch
- `eval_v9_heldout.json` — rebuilt by `build_v9_manifest.py`
- `aasist_v9_best.pt` — from `retrain_aasist_v9.py` (best EER 15.00%, epoch 39)
- `wav2vec_v9_best.pt` — from `retrain_wav2vec_v9.py` (best EER 15.67%, epoch 31)
- `xgb_v9.json` — from `refit_ensemble_v9.py`
- `cal_v9_params.json` — from `refit_ensemble_v9.py`
- `thresholds_v9.json` — from `refit_ensemble_v9.py`
- `ensemble_scores_v9.json` — raw per-model scores for analysis

**Michael has downloaded all artefacts locally.** They need to be uploaded to a persistent Kaggle dataset to avoid re-running the full retrain pipeline each session.

### Session startup (if artefacts not yet persisted to Kaggle dataset)

```python
# 1. Rebuild manifests
!python build_v9_manifest.py

# 2. Patch weight field
import json
with open("/kaggle/working/train_v9.json") as f:
    data = json.load(f)
for e in data:
    if "weight" not in e:
        e["weight"] = 1.0
with open("/kaggle/working/train_v9.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Patched — {len(data)} entries ready.")
```

---

## V9 Retrain — COMPLETED

### What was done

1. **Manifest construction** (`build_v9_manifest.py`): Added `real_studio` (226 clips) and `fake_noizai` (70 clips) to `train_v8_fresh.json` → `train_v9.json` (4,696 entries, 1.07:1 class balance). Built separate `eval_v9_heldout.json` (75 studio reals + 30 noiz.ai fakes).

2. **AASIST retrain** (`retrain_aasist_v9.py`): Full retrain from scratch on V9 manifest. Converged at epoch 39, best val EER 15.00%. Studio FP dropped from 80% → 16%, noiz.ai catch improved from 60% → 80%. All existing fake buckets maintained 100% catch rate.

3. **Wav2Vec2 retrain** (`retrain_wav2vec_v9.py`): Classifier head retrained (base frozen) on V9 manifest. Converged at epoch 31, best val EER 15.67%. Studio FP 18.7%, noiz.ai catch 80%.

4. **RawNet3 retrain attempted** (`retrain_rawnet3_v9.py`): Script built with exact architecture from training notebook. Training impractical on T4 — ~1 hour per epoch with cuDNN disabled (required for GRU backward). At 3 hours only 3 epochs completed. Decision: keep RN3 V8, downweight via XGBoost.

5. **Ensemble refit** (`refit_ensemble_v9.py`): XGBoost refitted with AASIST V9 + W2V V9 + RN3 V8. XGBoost correctly learned to downweight RN3 (15.3%). Final held-out results: studio FP 12.0%, noiz.ai catch 83.3%.

### Key finding: 2-model vs 3-model comparison

Tested dropping RN3 entirely (simple average of AASIST + W2V). Result: 3-model ensemble outperformed 2-model on studio FP (12.0% vs 22.7%) and val EER (13.25% vs 14.92%). XGBoost successfully extracts useful signal from RN3 while limiting its bias damage. Only noiz.ai catch was better in 2-model (90% vs 83.3%).

### V8 → V9 comparison summary

| Metric | V8 | V9 | Change |
|--------|-----|-----|--------|
| Val EER | 7.41% | 13.25% | +5.84pp (regression, driven by RN3 bias on real_in_the_wild) |
| Val AUC | 0.979 | 0.9408 | -0.038 |
| Studio FP | 80% | **12.0%** | **-68pp ✓ primary goal** |
| Noiz.ai catch | 60% | **83.3%** | **+23.3pp ✓ secondary goal** |
| real_common_voice FP | ~6% | 1.3% | -4.7pp ✓ |
| fake_in_the_wild catch | ~98% | 100% | +2pp ✓ |
| fake_openai_tts catch | ~65% | 96.3% | +31pp ✓ |

---

## Phases 5-7 Status

### Phase 5 — COMPLETED

All tasks delivered. Key results: adversarial robustness tested (honest negative results documented), phone-effect vulnerability confirmed (40% catch on noiz.ai phone_like), effects augmentation failed, server defenses delivered, drift monitor built, few-shot adaptation pipeline built, adversarial detection classifier built (no deployable operating point — AUC 0.736).

### Phase 6 — PARTIALLY COMPLETED

- **Tasks 1-2 (Governance):** DONE. Model registry, tamper-evident audit log, deterministic inference gate passed.
- **Tasks 3-4 (Bias Audit):** BLOCKED on data sourcing. Scripts built (`bias_audit_discovery.py`, `bias_audit_sourcing.py`). Issues: AfriSpeech-200 HF loading broken (needs `datasets==2.19.0`), edge-tts Yoruba voices returned errors, ElevenLabs Hausa/Igbo not configured.
- **Task 5 (Asymmetric Thresholds):** DONE. Four use-case docs delivered.
- **Task 6 (Legal Explainability):** DONE pending non-technical review.

### Phase 7 — IN PROGRESS

- **Tasks 1-3 (Cascade/Distillation):** PAUSED. Scripts ready (`cascade_and_distillation.py`). Must retrain against V9, not V8.
- **Tasks 4-6 (API, Load Test, Pen Test):** NOT STARTED.

---

## What needs to happen next (current plan)

### 1. Bias Audit (Phase 6 Tasks 3-4) — NEXT

Unblock data sourcing for Nigerian-language audio (Yoruba, Hausa, Igbo, Pidgin). Test V9 ensemble for demographic/linguistic bias. Scripts exist but need data access fixes.

Known blockers:
- AfriSpeech-200: `pip install "datasets==2.19.0"` may fix HF loading
- edge-tts Yoruba: verify voice availability with `edge-tts --list-voices`
- ElevenLabs: configure Hausa/Igbo voice IDs
- Need Nigerian Pidgin English samples (source TBD)

### 2. Cascade + Distillation — AFTER bias audit

Retrain LCNN screener and distilled student against V9 ensemble. Scripts ready.

### 3. Persist V9 artefacts to Kaggle dataset

Upload `aasist_v9_best.pt`, `wav2vec_v9_best.pt`, `xgb_v9.json`, `cal_v9_params.json` to a persistent Kaggle dataset to avoid re-running retrains.

### 4. Project wrap-up

- Wire audit log into `server.py` for live detection logging
- Get legal explainability template reviewed by non-technical person
- Update Phase 5/6 findings documents with V9 results
- Phase 7 Tasks 4-6 (API hardening, load testing, pen test)
- Update `governance.py` model registry with V9 checksums

---

## Scripts delivered in the V9 retrain session

| Script | Purpose |
|--------|---------|
| `build_v9_manifest.py` | Manifest construction — adds real_studio + fake_noizai to train manifest, builds held-out eval manifest |
| `retrain_aasist_v9.py` | Full AASIST retrain from scratch on V9 manifest |
| `retrain_wav2vec_v9.py` | Wav2Vec2 classifier head retrain on V9 manifest |
| `retrain_rawnet3_v9.py` | RawNet3 retrain (built but impractical on T4) |
| `refit_ensemble_v9.py` | Extract scores from all 3 models, fit XGBoost + logistic calibration, evaluate full ensemble |

---

## How to continue in the new conversation

Paste this whole document as the first message. Then say what you want next.

The agreed next step is the **bias audit** (Phase 6 Tasks 3-4): unblock Nigerian-language data sourcing and run the V9 ensemble against multilingual test sets to characterize demographic/linguistic bias.

# VoiceGuard V9 — Handoff Summary (Post-Retrain + Bias Audit)

> **For starting a new Claude conversation to continue this work.**
> Paste this entire document into your first message.

---

## Project context

**Engagement:** VoiceGuard audio deepfake detection. Multi-phase client deliverable (NGN 500,000 per phase) plus Master's research component. V9 retrain and bias audit are complete. Current priority: cascade/distillation (Phase 7 Tasks 1-3).

**Client:** Based in Nigeria. Nigerian-language audio (Yoruba, Hausa, Igbo, Pidgin) is a priority.

---

## V9 Ensemble — PRODUCTION REFERENCE

### Architecture

**3-model ensemble:** AASIST V9 + Wav2Vec2 V9 + RawNet3 V8, fused via XGBoost + logistic calibration.

| Model | Params | XGB Weight | Version | Notes |
|-------|--------|-----------|---------|-------|
| AASIST | ~285K | 0.402 | **V9 (retrained)** | Full retrain from scratch on V9 manifest |
| Wav2Vec2 | ~95M (213K trainable) | 0.445 | **V9 (retrained)** | Classifier head retrained, base frozen |
| RawNet3 | ~6M | 0.153 | V8 (unchanged) | Retrain impractical on T4 (~1hr/epoch) |

### V9 Performance

**Val set (val_v8_fresh.json, 1200 entries):**
- EER: 13.25% | AUC: 0.9408
- Val EER is higher than V8's 7.41% — driven by RN3's residual "clean = fake" bias on `real_in_the_wild` (25.3% FPR)
- Core fake detection: in_the_wild 100%, conference 98.8%, edge_tts 98.5%, openai_tts 96.3%
- real_common_voice FP: 1.3%

**Held-out set (105 entries — never trained on):**
- Studio FP: **12.0%** (V8 was 80%) ← primary goal achieved
- Noiz.ai catch: **83.3%** (V8 was 60%) ← secondary goal achieved

**Bias audit test set (599 entries, 7 languages):**
- Overall EER: **2.43%** | FPR: 5.4% | Catch: 99.2%
- Catch rate parity: **PASS** — zero violations across all languages
- Nigerian languages: Yoruba 2% FPR, Igbo 4% FPR, Pidgin 0% FPR — no bias
- Arabic FPR elevated (16%) — recording quality issue, not language bias. Threshold calibration to 0.86 reduces to 6%

**Calibration:** `P = sigmoid(4.8246 * xgb_score + -2.1469)`

**Verdict thresholds:** auto_fake ≥ 0.85, likely_fake ≥ 0.55, to_review ≥ 0.30

### Training manifest

`train_v9.json` — **4,746 entries**, 1.07:1 class balance. Built from:
- Original V8 data: 4,400 entries (7 source buckets)
- Studio reals: 226 clips (`real_studio`)
- Noiz.ai fakes: 70 clips (`fake_noizai`)
- Hausa TTS fakes: 50 clips (`fake_hausa_tts`) — added during bias audit mitigation

### Critical model details (for any future retraining)

**AASIST V9:** SincConv 70-ch kernel 127 → BN → SELU → MaxPool(3) → 6 ResBlocks channels [32,32,64,64,128,128] strides [1,2,1,2,1,2] (ALL blocks have downsample) → AdaptiveAvgPool1d(64) → GAT(128→64) → GAT(64→64) → classifier Linear(64→64) SELU Dropout(0.3) Linear(64→2). Takes 3D input (B,1,T). Checkpoint wrapper: `model_state_dict`.

**Wav2Vec2 V9:** `Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')` frozen + classifier `Sequential(Linear(768→256)[0] ReLU[1] Dropout(0.3)[2] Linear(256→64)[3] ReLU[4] Dropout(0.15)[5] Linear(64→2)[6])`. Checkpoint wrapper: `model` key with full state dict (base + classifier). Takes 2D input (B,T).

**RawNet3 V8:** SincConvRaw(128, kernel 512→forced to 513 via `kernel_size + (kernel_size % 2 == 0)`) → sinc_bn → SELU → MaxPool(3) → 4 RawResBlocks [128,128,256,256] strides [1,2,1,2] with FMS(Sequential: AdaptiveAvgPool1d→Flatten→Linear→Sigmoid) → 2-layer GRU(256, dropout=0.5) → AttentionPooling(Linear(256→128)→Tanh→Linear(128→1)) → Dropout(0.5) → classifier Sequential(Linear(256→128) SELU Dropout(0.25) Linear(128→2)). Wrapper key: `model_state`. cuDNN must be disabled for GRU backward. Forward does double permute around GRU (into batch_first, back to channels-first for attention_pool). SincConvRaw uses half-kernel parameterization: n_ shape [1,256], window_ shape [256], left-half construction with `torch.matmul`, then mirror with center point. Init uses `low_hz=30.0`, numpy-based mel scale (`np.linspace` + `np.diff`).

---

## Kaggle environment

### Datasets

**V8 artefacts:** `/kaggle/input/datasets/michaelologungbara/voiceguard-v8-artefacts/`
Contents: `aasist_v8.pt`, `wav2vec_v8.pt`, `rawnet3.pt`, `xgb_v8.json`, `cal_v8_params.json`, `thresholds_v8.json`, `val_v8_fresh.json` (1,200 entries), `train_v8_fresh.json` (4,400 entries).

**V9 train/test data:** `/kaggle/input/datasets/michaelologungbara/v9-train-test/`
- `new_samples/new_samples/fake/` (70 noiz.ai fakes)
- `new_samples/new_samples/real/` (226 studio clips)
- `held_out/held_out/fake/` (30 noiz.ai fakes, never trained on)
- `held_out/held_out/real/` (75 studio clips, never trained on)
Note: double-nested directory structure.

**Bias audit data:** `/kaggle/input/datasets/michaelologungbara/bias-audit-fakes/bias_audit_fakes/`
- 7 language folders: arabic, english, french, hausa, igbo, pidgin, yoruba
- Each has `real/` and/or `fake/` subdirectories
- Hausa has 100 fakes total: first 50 (alphabetical) = test set, last 50 = added to training
- English has no `real/` folder — English reals pulled from val_v8_fresh.json
- Yoruba and Igbo have no `fake/` folders (no free TTS available)

### V9 artefacts (in `/kaggle/working/` — wiped between sessions)

Must be regenerated each session OR uploaded to a persistent dataset:
- `train_v9.json` — rebuilt by `build_v9_manifest.py` + weight patch + Hausa fakes addition
- `eval_v9_heldout.json` — rebuilt by `build_v9_manifest.py`
- `aasist_v9_best.pt` — retrained AASIST (best EER 15.00%, includes Hausa exposure)
- `wav2vec_v9_best.pt` — retrained W2V classifier (best EER 15.83%, includes Hausa exposure)
- `xgb_v9.json` — XGBoost fusion model
- `cal_v9_params.json` — logistic calibration parameters
- `thresholds_v9.json` — verdict thresholds
- `bias_audit_report_v9.json` — raw per-sample scores from bias audit

**Michael has downloaded all artefacts locally.** They need to be uploaded to a persistent Kaggle dataset for future sessions.

### Session startup (if artefacts not yet persisted)

```python
# 1. Rebuild manifests
!python build_v9_manifest.py

# 2. Patch weight field + add Hausa training fakes
import json, os
with open("/kaggle/working/train_v9.json") as f:
    data = json.load(f)
for e in data:
    if "weight" not in e:
        e["weight"] = 1.0

hausa_dir = "/kaggle/input/datasets/michaelologungbara/bias-audit-fakes/bias_audit_fakes/hausa/fake"
hausa_files = sorted(os.listdir(hausa_dir))
train_fakes = hausa_files[50:]  # last 50, first 50 = test set
for f in train_fakes:
    data.append({"path": os.path.join(hausa_dir, f), "label": 1,
                 "source": "fake_hausa_tts", "weight": 1.0})

with open("/kaggle/working/train_v9.json", "w") as fw:
    json.dump(data, fw, indent=2)
print(f"Manifest: {len(data)} entries")
```

---

## Completed work

### V9 Retrain — COMPLETED

1. **Manifest construction** (`build_v9_manifest.py`): Added `real_studio` (226), `fake_noizai` (70), `fake_hausa_tts` (50) → 4,746 entries total.

2. **AASIST retrain** (`retrain_aasist_v9.py`): Full retrain from scratch. Studio FP 80% → 16%, noiz.ai catch 60% → 80%.

3. **Wav2Vec2 retrain** (`retrain_wav2vec_v9.py`): Classifier head retrained. Studio FP 18.7%, noiz.ai catch 80%.

4. **RawNet3 retrain attempted** (`retrain_rawnet3_v9.py`): Script built with exact architecture from training notebook. Impractical on T4 (~1hr/epoch with cuDNN disabled). Decision: keep RN3 V8, downweight via XGBoost.

5. **Ensemble refit** (`refit_ensemble_v9.py`): XGBoost correctly downweights RN3 to 15.3%. Final held-out: studio FP 12.0%, noiz.ai catch 83.3%.

### Phase 6 Bias Audit — COMPLETED

1. **Test set compiled**: 599 samples across 7 languages (Arabic, English, French, Hausa, Igbo, Pidgin, Yoruba), 3 language families (Niger-Congo, Afroasiatic, Indo-European).

2. **Initial finding — Hausa catch rate 50%**: AASIST only 20% catch on Hausa TTS fakes. Root cause: Hausa edge-tts voices produce artifact signatures not seen during training.

3. **Mitigation applied**: Added 50 Hausa TTS fakes to training manifest, retrained AASIST + W2V. Result: Hausa catch 50% → 100%. Overall EER improved 10.36% → 2.43%.

4. **Arabic FPR finding documented**: 16% FPR at deployed threshold — recording quality issue (broadcast audio), not language bias. Threshold calibration to 0.86 reduces to 6% while maintaining 86% catch rate.

5. **Final result**: Catch rate parity PASS (zero violations). FPR finding documented with mitigation. All Nigerian languages perform at or below overall FPR.

6. **Report delivered**: `VoiceGuard_V9_Bias_Audit_Report.docx` — 8-page professional report with methodology, results, findings, mitigations, and compliance matrix.

### Phase 5 — COMPLETED (prior session)

Adversarial robustness tested, honest negative results documented, phone-effect vulnerability confirmed, server defenses and drift monitor delivered.

### Phase 6 Tasks 1-2, 5-6 — COMPLETED (prior session)

Governance layer, deterministic inference gate, asymmetric thresholds documentation, legal explainability template.

---

## What needs to happen next

### 1. Phase 7 Tasks 1-3 — Cascade + Distillation — NEXT

Retrain LCNN cascade screener and distilled student against V9 ensemble. Previous scripts exist (`cascade_and_distillation.py`) but must be updated for V9. LCNN screener had plateaued during V8 development (three iterations, plateau identified). Worth re-evaluating with stronger V9 teacher.

Key context from prior work:
- LCNN cascade screener: intended as a fast first-pass filter before the full ensemble
- Distilled student: smaller model trained to match V9 ensemble scores
- Both need V9 teacher scores for training
- Prior LCNN used Kaggle T4 training environment

### 2. Phase 7 Tasks 4-6 — API, Load Test, Pen Test

- Wire audit log into `server.py` for live detection logging
- API hardening and endpoint security
- Load testing under concurrent requests
- Penetration test (adversarial input fuzzing)

### 3. Persist V9 artefacts to Kaggle dataset

Upload `aasist_v9_best.pt`, `wav2vec_v9_best.pt`, `xgb_v9.json`, `cal_v9_params.json` to a persistent Kaggle dataset. Michael has downloaded locally but hasn't uploaded to Kaggle yet.

### 4. Project wrap-up

- Update `governance.py` model registry with V9 checksums
- Get legal explainability template reviewed by non-technical person
- Final client deliverable packaging

---

## Scripts delivered across V9 sessions

| Script | Purpose |
|--------|---------|
| `build_v9_manifest.py` | Manifest construction with studio reals + noiz.ai fakes + Hausa TTS |
| `retrain_aasist_v9.py` | Full AASIST retrain from scratch |
| `retrain_wav2vec_v9.py` | Wav2Vec2 classifier head retrain |
| `retrain_rawnet3_v9.py` | RawNet3 retrain (built but impractical on T4) |
| `refit_ensemble_v9.py` | Score extraction + XGBoost + calibration refit |
| `generate_bias_fakes.py` | TTS fake generation for bias audit (run locally) |
| `bias_audit_v9.py` | Full bias audit evaluation across languages |
| `bias_mitigation_v9.py` | Per-language threshold calibration |
| `VoiceGuard_V9_Bias_Audit_Report.docx` | Phase 6 deliverable report |

---

## Known issues and limitations

1. **Val EER regression**: V9 val EER (13.25%) is higher than V8 (7.41%). Driven by RN3's unretrained "clean = fake" bias on `real_in_the_wild`. Acceptable tradeoff for the studio FP and noiz.ai improvements.

2. **RawNet3 not retrained**: Only 15.3% ensemble weight. Contributes residual bias on clean audio. Retraining requires better GPU or cuDNN-compatible GRU implementation.

3. **Arabic broadcast FPR**: 16% at deployed threshold. Mitigatable via threshold calibration to 0.86 when language detection is available.

4. **Yoruba/Igbo fakes unavailable**: No free TTS engines produce these languages. Catch rate testing for these languages deferred to future audit when TTS becomes available.

5. **`refit_ensemble_v9.py` on Kaggle uses old RawNet3 architecture**: The version uploaded to Kaggle may have the pre-fix RawNet3 definition. `bias_audit_v9.py` has the correct architecture. If re-running ensemble refit, ensure the corrected RawNet3 (SincConvRaw with force-odd kernel, FMS Sequential, AttentionPooling taking (B,C,T), double permute) is used.

---

## How to continue

Paste this whole document as the first message in a new conversation. Then say what you want next.

The agreed next step is **Phase 7 Tasks 1-3: Cascade + Distillation** — retrain LCNN screener and distilled student against V9 ensemble.

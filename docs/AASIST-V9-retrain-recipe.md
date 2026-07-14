# AASIST V9 Retrain Recipe

**Goal:** retrain AASIST so it actually discriminates (instead of predicting "real" for everything with saturated logits), then fold it back into the ensemble.

**Where it runs:** Kaggle GPU (the training paths are `/kaggle/...`; this offline repo can't train). Script: `retrain_aasist_v9.py`.

---

## 1. Diagnosis — why AASIST V9 collapsed (confirmed from the code)

The checkpoint loads cleanly (strict, all keys match) and the architecture is fine. The failure is in **how it was trained**:

| # | Defect | Evidence in `retrain_aasist_v9.py` | Effect |
|---|--------|-----------------------------------|--------|
| 1 | **No class balancing** | `criterion = nn.CrossEntropyLoss(...)` with **no `weight=`** (L640); `DataLoader(..., shuffle=True)` with **no `WeightedRandomSampler`** (L595) | If `train_v9.json` is real-heavy, the model minimizes loss by predicting the majority class → **collapses to "real"** for everything. This is the primary cause. |
| 2 | **No logit regularization** | `label_smoothing = 0.0` (L78) + 100 epochs | Model becomes **overconfident → logits saturate (±26)** → `softmax[:,1]` (p_fake) ≈ 0.0000 for real *and* fake → the XGBoost fusion sees a near-constant feature and ignores AASIST entirely. |
| 3 | **Per-sample `weight` ignored** | manifest carries `weight` (L317) but it feeds neither the sampler nor the loss | Any intended balancing/curriculum in the manifest is dropped. |
| 4 | **Thin augmentation** | dataset only does random crop (L361-364); peak-norm is applied (L372, and matches inference — good) | Poor robustness to unseen fake families (edge-tts, Nigerian-language TTS). |
| 5 | **Val = a V8 set** | `val_manifest = .../val_v8_fresh.json` (L47); "best" chosen by that EER (L690) | "Best EER" can look fine while the model is weak on the families that actually fail. |

**Not a cause:** preprocessing is consistent — training peak-normalizes (L372) exactly like inference, so no train/test mismatch.

**Net:** AASIST has faint residual signal (~68% single-threshold separation) buried under a majority-class collapse and softmax saturation. The fix is a balanced, regularized retrain — plus measuring against the real weak spots first (Option 4).

---

## 2. Fixes — concrete changes to `retrain_aasist_v9.py`

Keep the architecture V8-exact. Change only training.

**(A) Balance the classes — `WeightedRandomSampler` (preferred).** Replace the train `DataLoader` (L595-597):
```python
from torch.utils.data import WeightedRandomSampler
labels = [int(e["label"]) for e in train_ds.entries]
n0, n1 = labels.count(0), labels.count(1)
class_w = {0: 1.0 / max(n0, 1), 1: 1.0 / max(n1, 1)}
# optionally fold in the manifest's per-sample weight:
sample_w = [class_w[int(e["label"])] * float(e.get("weight", 1.0)) for e in train_ds.entries]
sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
train_loader = DataLoader(
    train_ds, batch_size=cfg["batch_size"], sampler=sampler,   # NOTE: sampler replaces shuffle
    num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
```
(Alternative if you prefer weighted loss instead of a sampler:
`criterion = nn.CrossEntropyLoss(weight=torch.tensor([1/n0, 1/n1], device=device).float(), label_smoothing=0.05)` — but the sampler also diversifies fake families per batch, so prefer it.)

**(B) Add label smoothing** to stop saturation (so p_fake stays informative for the fusion). L78:
```python
"label_smoothing": 0.05,     # was 0.0
```
If fakes remain hard, consider focal loss (γ≈2) instead of CE — but start with (A)+(B).

**(C) Select "best" on a representative, balanced val set** (see §3). Keep the `val_eer < best_eer` selection but point `val_manifest` at a set that includes the weak families; optionally select on **macro-EER across sources** rather than pooled EER, so one easy family can't mask a failing one.

**(D) Print class counts at startup** (sanity — confirm the imbalance): after loading `train_ds`, log `Counter(int(e["label"]) for e in train_ds.entries)`.

**(E) Stronger augmentation (optional, helps coverage):** add mild Gaussian noise / SpecAugment-style masking / codec-simulation to the training crop, so the model generalizes to edge-tts and phone-codec fakes.

---

## 3. Option 4 — measure the weak spots FIRST (drives the data)

Before/alongside retraining, build a **labeled eval manifest that includes the failing families** so we retrain and validate against reality, not a stale V8 val set:

- Sources to include as **fake**: edge-tts (`studio_fake_test/processed_fakes`, `NoteGPT`-style), and the Nigerian-language TTS in `bias_audit_fakes/` (hausa, igbo, yoruba, pidgin, arabic).
- Sources as **real**: `studio_clips/`, `bias_audit/real`.
- Report **fake-recall per source/language** (not just pooled EER) — the current V9 metrics only measured multilingual *false-positive* rate on reals, never *fake recall* on those languages.
- Use `sweep_cascade.py` / `check_audio.py` (repo root) to establish the **baseline** per-family recall before retrain, so we can prove the retrain helped.

Retrain the training manifest to include adequate fake counts from these families (data augmentation / collection), since a balanced sampler only helps if the fakes are actually present.

---

## 4. After the retrain — refit the ensemble (required)

AASIST's output distribution will change, so the XGBoost fusion (fit on the old saturated feature) MUST be refit:
1. Regenerate ensemble features (AASIST/Wav2Vec/RawNet scores) on the training set with the new AASIST.
2. Run `refit_ensemble_v9.py` → new `xgb_v9.json` + re-Platt-calibrate (`cal_v9_params.json`).
3. Re-bundle (7 files) and register/promote via `bundle_registry.py` (then push to Spaces).

---

## 5. Acceptance — do not promote unless all hold

- **AASIST no longer saturates:** on held-out real+fake, `softmax[:,1]` shows real spread (not ~0 for everything); margin separation ≫ 0.68 (target AASIST standalone AUC ≥ ~0.85). Verify with an `aasist_probe.py`-style check.
- **Per-family fake recall improves** on the §3 eval set (especially edge-tts + Nigerian languages).
- **Studio false-positive rate not regressed** (was ~0.12) after the fusion refit.
- **Golden regression** still passes through the registry load.

---

## 6. Run order (Kaggle)

1. Build `train_v9.json` (balanced-source, weak families present) + a representative `val`/`eval` manifest (§3).
2. Apply the §2 patches to `retrain_aasist_v9.py`.
3. `python retrain_aasist_v9.py --epochs 100` (watch: val EER should drop AND train fake-recall should be non-trivial from early epochs — if the model is still all-real, the sampler/loss didn't take).
4. Download `aasist_v9_best.pt`; verify §5 acceptance (AASIST standalone) before touching the fusion.
5. Refit fusion (§4), re-bundle, re-validate (§5 full), promote.

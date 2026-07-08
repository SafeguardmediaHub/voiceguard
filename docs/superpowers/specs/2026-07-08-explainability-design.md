# Phase 3 Explainability — Design Spec

- **Date:** 2026-07-08
- **Status:** Approved (design); pending implementation plan
- **Scope:** Phase 3 explainability — the model-based signals still missing from the detector,
  plus the structured response fields and a UI refresh to render them.
- **Depends on:** the V9 cascade (`detector.py`), the async API/worker (`api.py`/`worker.py`),
  auth (C1). Builds on the existing per-chunk scoring and `explainability.py` (prosody).

---

## 1. Context & Problem

Phase 3 calls for: Grad-CAM spectrogram heatmaps, SHAP feature importance, timestamp-level
segment flagging, and a structured API response (`score, confidence, flagged_segments, heatmap,
audit_id`), reviewed on 20 diverse samples.

**Already present (do not rebuild):**
- **Timestamp segment flagging** — `detector.merge_chunks_to_segments()`, per-chunk
  `start_sec/end_sec/score/verdict`, and the `segments` + `flagged_segments` response fields.
- **Prosody/acoustic explanation** — `explainability.py` (F0, jitter, shimmer, HNR, spectral
  flatness, high-freq ratio, pauses → `observations`/`measurements`/`summary`), wired as the
  `explanation` response field.

**Genuine gaps this slice fills:**
1. **Grad-CAM heatmap** (which frequencies/time triggered detection) — none today.
2. **SHAP feature importance** (which model drove the fused score) — none today.
3. **`audit_id` + `confidence`** structured fields — none today (only `sha256`/`score`).
4. **UI rendering** — the demo renders none of these and still uses the pre-auth, synchronous
   `/detect` flow.

## 2. Goals / Non-Goals

**Goals:**
1. A **Grad-CAM heatmap** from the **LCNN screener**, computed on the **peak (max-score) chunk**,
   returned as a canonical numeric array **and** a rendered PNG.
2. **SHAP** per-model attribution for the XGBoost fusion, via XGBoost's **native `pred_contribs`**
   (no new dependency), on the peak stage-2 chunk.
3. `audit_id` (chain-of-custody UUID) and `confidence` (0–1) response fields, plus an append-only
   audit log line per detection.
4. Explainability is **strictly additive** — the verdict/score are unchanged (the golden
   regression must stay green), and any explainability failure degrades to a null field without
   breaking detection.
5. UI (`VoiceGuard_LiveDemo (2).html`) updated to the auth + async-poll flow and rendering the
   heatmap, SHAP, flagged segments, and prosody; `api.py`'s `/` repointed to it.
6. A scripted **acceptance harness** that dumps a heatmap PNG + explainability JSON per clip over
   ~20 diverse samples into an output folder for review.

**Non-Goals (later / out of scope):**
- AASIST/stage-2 attribution (LCNN Grad-CAM ships now; AASIST is a possible later extension).
- Per-segment heatmaps (one heatmap for the peak chunk this slice).
- A PDF/legal forensic report template (Phase 6).
- A `shap` library dependency (native `pred_contribs` only).
- The C4b deployment (pended).

## 3. Architecture

All new signals are computed inside the **worker's `detector.detect()`** path (never the API
event loop), and added to the existing response dict. Two new modules keep model-coupled code
out of `detector.py`'s core and make the pure logic unit-testable without loading models:

- **`explain_signals.py`** — pure functions (no torch/model imports): `compute_confidence`,
  `new_audit_id`, `shap_from_contribs`. Fast-tier testable.
- **`gradcam.py`** — LCNN Grad-CAM (torch + matplotlib). Weights-tier.

`detector.detect()` orchestrates: it already scores every chunk; it additionally remembers the
peak chunk, then calls `gradcam` + the XGBoost `pred_contribs` + `explain_signals`, wraps each in
try/except, and attaches `heatmap`, `shap`, `audit_id`, `confidence` to the response.

## 4. Components

### 4.1 `explain_signals.py` (new, pure)
```python
def new_audit_id() -> str:
    """'aud_' + uuid4().hex — unique per detection, for chain of custody."""

def compute_confidence(chunk_scores: list[float], final_score: float) -> float:
    """0–1 confidence = decisiveness * agreement, rounded to 3 dp.
      decisiveness = abs(final_score - 0.5) * 2            # 0 at the fence, 1 at extremes
      agreement    = 1 - min(1.0, pstdev(chunk_scores) / 0.5)  # 1 when chunks agree
    Single chunk -> stdev 0 -> agreement 1. Empty -> 0.0."""

def shap_from_contribs(contribs_row) -> dict:
    """Map XGBoost pred_contribs output [c_aasist, c_wav2vec, c_rawnet, bias] to
    {'aasist': float, 'wav2vec': float, 'rawnet': float, 'base': float}. The feature
    order matches predict_ensemble's [aasist, wav2vec, rawnet]."""
```

### 4.2 `gradcam.py` (new)
```python
def lcnn_gradcam(lcnn_model, mel_fn, wav_1d, mel_freqs_hz, sec_per_frame,
                 chunk_range_sec, max_time_cols=128) -> dict:
    """Grad-CAM over the LCNN's last conv block (block4) for one CHUNK-length waveform.

    Detection runs under torch.no_grad(); this re-runs a single forward WITH gradients:
      - forward hook on lcnn_model.block4 saves activations A (B,64,F4,T4);
      - backward hook saves gradients G = d(fake_logit)/dA;
      - alpha_k = mean over (F4,T4) of G[:,k]; CAM = ReLU(sum_k alpha_k * A_k);
      - normalize CAM to 0..1; bilinear-upsample to the mel grid (N_MELS x T');
      - if T' > max_time_cols, average-pool the time axis down to max_time_cols.

    Returns:
      {'target': 'lcnn',
       'chunk_range_sec': list(chunk_range_sec),
       'values':   [[float 0..1]],            # rows = mel bins, cols = time frames
       'freq_hz':  [float],                   # len == n rows (mel band-center Hz)
       'time_sec': [float],                   # len == n cols (frame start within the chunk)
       'png_base64': str}                     # matplotlib Agg render, labeled axes + colorbar
    Fake-class logit index is 1 (matches softmax[...,1] used as p_fake)."""
```
- The PNG is rendered with `matplotlib` using the `Agg` backend (import-time `matplotlib.use("Agg")`
  inside `gradcam.py`), a perceptually-ordered colormap (e.g. `magma`), y-axis = frequency (Hz),
  x-axis = time within the chunk (s), title = the chunk's absolute time range.

### 4.3 `detector.py` changes
- Precompute `MEL_FREQS_HZ` (list of 80 mel band-center frequencies) once at module load, from the
  same params used to build `_MEL_FB` (htk mel scale over `[f_min, f_max]`, `SR=16000`).
  `sec_per_frame = HOP_LENGTH / SR = 160/16000 = 0.01`.
- In `detect()`, retain per-chunk model scores so the **peak chunk** (max `score`) and the **peak
  stage-2 chunk** (max `score` among `stage == 2`, carrying `aasist/wav2vec/rawnet`) are known.
- After the final `score`/`verdict`:
  - `audit_id = explain_signals.new_audit_id()`
  - `confidence = explain_signals.compute_confidence([c['score'] for c in chunk_results], score)`
  - `heatmap`: `try: gradcam.lcnn_gradcam(lcnn, wav_to_mel, peak_wav_1d, MEL_FREQS_HZ, HOP_LENGTH/SR, (peak_start, peak_end)) except Exception: None`
  - `shap`: if a peak stage-2 chunk exists,
    `contribs = xgb_model.get_booster().predict(xgb_lib.DMatrix(np.array([[s_a,s_w,s_r]])), pred_contribs=True)[0]`
    → `explain_signals.shap_from_contribs(contribs)` plus `chunk_range_sec` and a `note` that the
    values are margin-space (pre-Platt-calibration) contributions; else `None`.
  - Append one line to the audit log (see 4.4).
- Add to the response dict: `"audit_id"`, `"confidence"`, `"heatmap"`, `"shap"` (keeping all
  existing fields). All must be JSON-safe (the API already runs `_json_safe` to null out NaN/Inf).

### 4.4 Audit log (chain of custody)
Append one JSON line per detection to `output/audit_log.jsonl` (dir from the existing
`DRIFT_OUTPUT_DIR`/output convention, `encoding='utf-8'`), non-fatal on failure:
`{audit_id, sha256, verdict, score, timestamp}`. The persisted job record already stores the full
result; this is a compact, append-only forensic trail keyed by `audit_id`.

### 4.5 UI — `VoiceGuard_LiveDemo (2).html`
- Add an **API-key** field; persist in `localStorage`; send `Authorization: Bearer <key>`.
- Switch to the **async flow**: `POST /detect` → `202 {job_id, status_url}` → poll
  `GET /jobs/{job_id}` (~1s interval) until `status` is `done`/`error`; render `result`.
- Render: verdict + `score` + `confidence`; the **heatmap** (`<img src="data:image/png;base64,…">`
  from `heatmap.png_base64`, with the chunk range shown); a **SHAP** per-model contribution bar
  (aasist/wav2vec/rawnet, signed); the **flagged-segment** timeline (`flagged_segments`); and the
  **prosody** observations (`explanation`). Show `audit_id`.
- Handle key-missing (401) and stage-1-only (`shap`/`heatmap` null) gracefully.
- Repoint `api.py`'s `/` route to serve `VoiceGuard_LiveDemo (2).html`.

## 5. Response Schema Additions
```jsonc
{
  // ...all existing fields unchanged (verdict, score, segments, flagged_segments, explanation, …)
  "audit_id": "aud_1a2b3c…",
  "confidence": 0.87,
  "heatmap": {                       // null on failure
    "target": "lcnn",
    "chunk_range_sec": [12.0, 16.0],
    "values": [[0.0, 0.1, …], …],    // rows=mel bins, cols=time (<=128)
    "freq_hz": [31.0, …, 7800.0],
    "time_sec": [0.0, 0.01, …],
    "png_base64": "iVBOR…"
  },
  "shap": {                          // null when every chunk resolved at stage 1
    "aasist": 0.42, "wav2vec": 1.13, "rawnet": -0.08, "base": -0.30,
    "chunk_range_sec": [12.0, 16.0],
    "note": "margin-space contributions from the XGBoost fusion (pre-calibration)"
  }
}
```

## 6. Testing

**Fast tier (`explain_signals`, no models):**
- `compute_confidence`: decisive+agreeing → high; fence score (0.5) → ~0; disagreeing chunks
  (high stdev) → lowered; single chunk → agreement 1; empty → 0.0.
- `shap_from_contribs`: maps the 4-element row to the named dict in the right order.
- `new_audit_id`: `aud_` prefix, unique across calls.

**Weights tier (need the bundle):**
- `gradcam.lcnn_gradcam` on a real clip: `values` shape == `len(freq_hz) × len(time_sec)`, all in
  `[0,1]`, `freq_hz` length == N_MELS, `time_sec` length ≤ 128, `png_base64` decodes to a valid PNG.
- `detect()` response contains the four new fields, JSON-serializable; `heatmap`/`shap` shapes as
  specified; `shap` null on a stage-1-only clip.
- **Golden regression stays green** — `verdict`/`score` byte-identical to the recorded values
  (proves explainability is additive).

**Acceptance harness (`scripts/explain_acceptance.py`, weights tier / manual):**
- Input: a folder or manifest of ~20 diverse clips (real + fake, multiple languages/codecs).
- For each: run `detect()`, write `<audit_id>.png` (the heatmap) and `<audit_id>.json`
  (`{file, verdict, score, confidence, shap, flagged_segments}`) into an output folder.
- Prints a summary table. Used for the "reviewed on 20 samples" gate (human eyeballs the folder).

## 7. Error Handling

- Grad-CAM / SHAP / prosody each wrapped in try/except in `detect()` → the field becomes `null`
  (SHAP/heatmap) or the existing empty-explanation fallback; **detection never fails** for an
  explainability error.
- `gradcam` restores the model to `eval()`/no-grad state and removes its hooks in a `finally` so a
  failure can't leave gradient hooks attached to the shared LCNN model.
- Audit-log write failure is caught and ignored (logged), never blocking the response.
- Heatmap `values` are finite floats (NaN/Inf → 0) so `_json_safe` has nothing to strip; PNG is
  standard base64.

## 8. Files
- **Create:** `explain_signals.py`, `gradcam.py`, `tests/test_explain_signals.py`,
  `tests/test_gradcam.py`, `scripts/explain_acceptance.py`.
- **Modify:** `detector.py` (MEL_FREQS_HZ, peak-chunk tracking, the four new fields + audit log),
  `tests/test_detector.py` (assert new fields + JSON-safety), `VoiceGuard_LiveDemo (2).html` (auth
  + async poll + rendering), `api.py` (`/` serves LiveDemo(2)).
- **Weights markers:** `tests/test_gradcam.py` gets `pytestmark = pytest.mark.weights` and is added
  to `tests/conftest.py`'s `collect_ignore` (it loads the LCNN). `tests/test_explain_signals.py`
  stays in the fast tier.
- **No new dependencies** (torch, xgboost, matplotlib, pillow all already present).

## 9. Assumptions
- The LCNN fake-class index is 1 (consistent with `softmax(...)[0,1]` used as p_fake throughout).
- One heatmap per detection (peak chunk) satisfies the "heatmap" requirement; per-segment heatmaps
  are a later extension.
- SHAP margin-space contributions (pre-calibration) are acceptable for relative attribution; the
  `note` field states this so the number isn't misread as a calibrated probability delta.
- The "20 diverse samples" are supplied by the user (or drawn from `tests/golden_clips` + studio
  clips); the harness doesn't ship its own dataset.

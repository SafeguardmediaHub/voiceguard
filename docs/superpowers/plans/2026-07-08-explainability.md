# Phase 3 Explainability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model-based explainability (LCNN Grad-CAM heatmap + XGBoost SHAP) plus `audit_id`/`confidence` to the detector response, and a UI that renders it — without changing any verdict or score.

**Architecture:** Two new modules — `explain_signals.py` (pure: confidence, audit id, SHAP mapping) and `gradcam.py` (LCNN Grad-CAM, torch + matplotlib). `detector.detect()` tracks the peak chunk, calls both (each wrapped in try/except → null on failure), and attaches four new response fields. The UI (`VoiceGuard_LiveDemo (2).html`) switches to the auth + async-poll flow and renders the new signals.

**Tech Stack:** Python 3.12, torch, xgboost (native `pred_contribs`), matplotlib (Agg), pytest.

## Global Constraints

- **Additive only:** the `verdict` and `score` must be byte-identical to today. The golden regression (`tests/test_golden.py`) MUST stay green — it is the proof.
- **Never break detection:** each explainability signal (heatmap, SHAP, audit log) is wrapped in try/except; on failure the field is `null` (or skipped) and detection returns normally.
- **No new dependencies** — torch, xgboost, matplotlib, pillow are all already installed.
- **Grad-CAM needs gradients:** `gradcam.lcnn_gradcam` must NOT be called inside a `torch.no_grad()` context; it removes its hooks and restores the model in a `finally`.
- **Fake-class logit index is 1** (matches `softmax(...)[0,1]` used as p(fake) throughout `detector.py`).
- **SHAP via XGBoost native `pred_contribs`** — do NOT add the `shap` library.
- **Test tiers:** `tests/test_explain_signals.py` is fast-tier (pure, no models). `tests/test_gradcam.py` is weights-tier — add `pytestmark = pytest.mark.weights` AND add `"test_gradcam.py"` to `tests/conftest.py`'s `collect_ignore` list.
- **JSON-safe:** all new fields are finite floats / plain lists / strings (the API also runs `_json_safe`).
- **DO NOT `git commit`.** The controller snapshots each task; the **user** commits. Each task's final step runs its tests and reports DONE — it does not commit.

---

### Task 1: `explain_signals.py` — pure signals (confidence, audit id, SHAP mapping)

**Files:**
- Create: `explain_signals.py`
- Create: `tests/test_explain_signals.py`

**Interfaces:**
- Produces (consumed by Task 3):
  - `new_audit_id() -> str` (`"aud_" + uuid4().hex`)
  - `compute_confidence(chunk_scores: list[float], final_score: float) -> float`
  - `shap_from_contribs(contribs_row) -> dict` with keys `aasist, wav2vec, rawnet, base`

- [ ] **Step 1: Write the failing tests** — `tests/test_explain_signals.py`

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import explain_signals as es


def test_new_audit_id_prefix_and_unique():
    a, b = es.new_audit_id(), es.new_audit_id()
    assert a.startswith("aud_") and len(a) > 4
    assert a != b


def test_confidence_decisive_and_agreeing():
    assert es.compute_confidence([0.95, 0.96, 0.94], 0.95) > 0.85


def test_confidence_fence_score_zero():
    assert es.compute_confidence([0.5, 0.5], 0.5) == 0.0


def test_confidence_disagreement_lowers():
    assert es.compute_confidence([0.1, 0.9], 0.9) < es.compute_confidence([0.9, 0.9], 0.9)


def test_confidence_single_chunk_is_decisiveness():
    assert es.compute_confidence([0.8], 0.8) == round(abs(0.8 - 0.5) * 2, 3)


def test_confidence_empty_zero():
    assert es.compute_confidence([], 0.9) == 0.0


def test_shap_from_contribs_mapping():
    assert es.shap_from_contribs([0.42, 1.13, -0.08, -0.30]) == {
        "aasist": 0.42, "wav2vec": 1.13, "rawnet": -0.08, "base": -0.30}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_explain_signals.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'explain_signals'`.

- [ ] **Step 3: Implement `explain_signals.py`**

```python
"""explain_signals.py — pure explainability helpers (no torch / no model imports).

Fast-tier testable: confidence, chain-of-custody id, and the SHAP contribution mapping.
"""
import uuid
from statistics import pstdev


def new_audit_id():
    """'aud_' + uuid4().hex — unique per detection, for chain of custody."""
    return "aud_" + uuid.uuid4().hex


def compute_confidence(chunk_scores, final_score):
    """0-1 confidence = decisiveness * agreement (rounded 3dp).
      decisiveness = |final_score - 0.5| * 2                  (0 at the fence, 1 at extremes)
      agreement    = 1 - min(1.0, pstdev(chunk_scores) / 0.5) (1 when chunks agree)
    Empty chunk_scores -> 0.0; a single chunk -> stdev 0 -> agreement 1."""
    if not chunk_scores:
        return 0.0
    decisiveness = abs(final_score - 0.5) * 2.0
    spread = pstdev(chunk_scores) if len(chunk_scores) > 1 else 0.0
    agreement = 1.0 - min(1.0, spread / 0.5)
    return round(decisiveness * agreement, 3)


def shap_from_contribs(contribs_row):
    """Map XGBoost pred_contribs output [c_aasist, c_wav2vec, c_rawnet, bias] to a named
    dict. Feature order matches detector.predict_ensemble's [aasist, wav2vec, rawnet]."""
    return {
        "aasist":  round(float(contribs_row[0]), 4),
        "wav2vec": round(float(contribs_row[1]), 4),
        "rawnet":  round(float(contribs_row[2]), 4),
        "base":    round(float(contribs_row[3]), 4),
    }
```

Note: the test uses exact values (0.42 etc.) that are already 4-dp — `round(..., 4)` leaves them unchanged, so the equality holds.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_explain_signals.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Run the fast suite; report DONE (do not commit)**

Run: `VOICEGUARD_CI_FAST=1 python -m pytest -q`
Expected: PASS (previous fast tier + 7 new). Report DONE with the summary. Controller snapshots; user commits.

---

### Task 2: `gradcam.py` — LCNN Grad-CAM heatmap

**Files:**
- Create: `gradcam.py`
- Create: `tests/test_gradcam.py`
- Modify: `tests/conftest.py` (add `"test_gradcam.py"` to the `collect_ignore` list)

**Interfaces:**
- Produces (consumed by Task 3):
  - `lcnn_gradcam(lcnn_model, mel_fn, wav_1d, mel_freqs_hz, sec_per_frame, chunk_range_sec, max_time_cols=128) -> dict`
    returning `{target, chunk_range_sec, values, freq_hz, time_sec, png_base64}`.
- Consumes (existing, passed in by the caller — NOT imported here): the loaded LCNN model
  (`lcnn`), the mel transform (`wav_to_mel`), `N_MELS`, `HOP_LENGTH`, `SR`, `CHUNK`.

- [ ] **Step 1: Add `test_gradcam.py` to the fast-tier ignore list** in `tests/conftest.py`

Change the `collect_ignore` list (added in the CI slice) to include the new module:

```python
if _os.environ.get("VOICEGUARD_CI_FAST"):
    collect_ignore = ["test_api.py", "test_detector.py", "test_worker.py",
                      "test_golden.py", "test_gradcam.py"]
```

- [ ] **Step 2: Write the failing test** — `tests/test_gradcam.py`

```python
import os, sys, base64
import pytest
pytestmark = pytest.mark.weights            # loads the LCNN model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import detector
import gradcam


def test_lcnn_gradcam_shape_range_and_png():
    wav = torch.zeros(detector.CHUNK)                      # a valid CHUNK-length input
    freqs = [float(i) for i in range(detector.N_MELS)]     # dummy Hz axis (echoed through)
    hm = gradcam.lcnn_gradcam(detector.lcnn, detector.wav_to_mel, wav,
                              freqs, detector.HOP_LENGTH / detector.SR, (0.0, 4.0))
    assert hm["target"] == "lcnn"
    assert hm["chunk_range_sec"] == [0.0, 4.0]
    assert len(hm["values"]) == len(hm["freq_hz"]) == detector.N_MELS
    assert len(hm["time_sec"]) <= 128
    assert all(len(row) == len(hm["time_sec"]) for row in hm["values"])
    flat = [v for row in hm["values"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)
    assert base64.b64decode(hm["png_base64"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_gradcam_leaves_no_hooks_and_restores_eval():
    wav = torch.zeros(detector.CHUNK)
    freqs = [float(i) for i in range(detector.N_MELS)]
    gradcam.lcnn_gradcam(detector.lcnn, detector.wav_to_mel, wav, freqs,
                         detector.HOP_LENGTH / detector.SR, (0.0, 4.0))
    # forward + full-backward hooks removed (register_full_backward_hook uses _full_backward_hooks)
    assert len(detector.lcnn.block4._forward_hooks) == 0
    assert len(detector.lcnn.block4._full_backward_hooks) == 0
    assert detector.lcnn.training is False
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_gradcam.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gradcam'`.

- [ ] **Step 4: Implement `gradcam.py`**

```python
"""gradcam.py — Grad-CAM heatmap over the LCNN screener's last conv block (block4).

Model-agnostic via dependency injection: detect() passes the loaded LCNN model and the mel
transform, so this module never imports detector (no circular import). Renders a labeled PNG
via matplotlib's Agg backend. MUST be called outside any torch.no_grad() context.
"""
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _render_png(cam, freq_hz, time_sec, chunk_range_sec):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    x1 = time_sec[-1] if len(time_sec) > 1 else time_sec[0] + 0.01
    im = ax.imshow(cam, aspect="auto", origin="lower", cmap="magma",
                   extent=[time_sec[0], x1, freq_hz[0], freq_hz[-1]])
    ax.set_xlabel("time within chunk (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"LCNN Grad-CAM  [{chunk_range_sec[0]:.1f}s-{chunk_range_sec[1]:.1f}s]")
    fig.colorbar(im, ax=ax, label="attribution")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def lcnn_gradcam(lcnn_model, mel_fn, wav_1d, mel_freqs_hz, sec_per_frame,
                 chunk_range_sec, max_time_cols=128):
    """Grad-CAM over lcnn_model.block4 for one CHUNK-length waveform (1-D tensor)."""
    activations, gradients = {}, {}
    h1 = lcnn_model.block4.register_forward_hook(lambda m, i, o: activations.__setitem__("A", o))
    h2 = lcnn_model.block4.register_full_backward_hook(lambda m, gi, go: gradients.__setitem__("G", go[0]))
    try:
        lcnn_model.eval()
        mel = mel_fn(wav_1d).clone().detach().requires_grad_(True)   # (1,1,80,T')
        lcnn_model.zero_grad(set_to_none=True)
        logits = lcnn_model(mel)                                     # (1,2)
        logits[0, 1].backward()                                      # fake-class logit

        A, G = activations["A"], gradients["G"]                      # (1,64,F4,T4)
        alpha = G.mean(dim=(2, 3), keepdim=True)                     # (1,64,1,1)
        cam = F.relu((alpha * A).sum(dim=1, keepdim=True))           # (1,1,F4,T4)

        _, _, n_mels, t_full = mel.shape
        cam = F.interpolate(cam, size=(n_mels, t_full), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()                   # (80, T')
        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        rng = float(cam.max() - cam.min())
        cam = (cam - cam.min()) / rng if rng > 1e-12 else np.zeros_like(cam)

        if cam.shape[1] > max_time_cols:                             # average-pool the time axis
            idx = np.linspace(0, cam.shape[1], max_time_cols + 1).astype(int)
            cam = np.stack([cam[:, idx[i]:idx[i + 1]].mean(axis=1) for i in range(max_time_cols)], axis=1)

        n_cols = cam.shape[1]
        time_sec = [round(i * (t_full / n_cols) * sec_per_frame, 4) for i in range(n_cols)]
        freq_hz = [round(float(f), 1) for f in mel_freqs_hz]
        return {
            "target": "lcnn",
            "chunk_range_sec": [round(float(chunk_range_sec[0]), 2), round(float(chunk_range_sec[1]), 2)],
            "values": [[round(float(v), 4) for v in row] for row in cam],
            "freq_hz": freq_hz,
            "time_sec": time_sec,
            "png_base64": _render_png(cam, freq_hz, time_sec, chunk_range_sec),
        }
    finally:
        h1.remove()
        h2.remove()
        lcnn_model.zero_grad(set_to_none=True)
        lcnn_model.eval()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_gradcam.py -q`
Expected: PASS (2 passed). (First run downloads/loads models — allow time.)

- [ ] **Step 6: Confirm fast tier still ignores it; report DONE (do not commit)**

Run: `VOICEGUARD_CI_FAST=1 python -m pytest --collect-only -q 2>NUL | findstr test_gradcam` (or on bash: `... | grep test_gradcam`)
Expected: empty (no output) — `test_gradcam` is excluded from the fast tier.
Report DONE. Controller snapshots; user commits.

---

### Task 3: Wire explainability into `detector.detect()`

**Files:**
- Modify: `detector.py`
- Modify: `tests/test_detector.py`

**Interfaces:**
- Consumes (Task 1): `explain_signals.new_audit_id`, `compute_confidence`, `shap_from_contribs`.
- Consumes (Task 2): `gradcam.lcnn_gradcam`.
- Consumes (existing): `lcnn`, `wav_to_mel`, `xgb_model`, `xgb_lib`, `N_MELS`, `HOP_LENGTH`, `SR`,
  `BASE`, `chunk_results` (each has `start_sec/end_sec/score/verdict/stage`), and per-chunk stage-2
  scores `r['aasist']/r['wav2vec']/r['rawnet']`.
- Produces: response fields `audit_id`, `confidence`, `heatmap`, `shap`; an `output/audit_log.jsonl`
  append; a module-level `MEL_FREQS_HZ` and `_write_audit_log`.

- [ ] **Step 1: Write the failing test additions** — append to `tests/test_detector.py`

(This module is weights-tier; it already imports `detector` and runs `detect()` on a golden clip.
Add a test that reuses that path. Use the existing golden clip constant/path in the file; the clip
below is the known fake sample.)

```python
def test_detect_has_explainability_fields():
    r = detector.detect("tests/golden_clips/fake_noizai_a4cd.mp3")
    # additive fields present
    assert r["audit_id"].startswith("aud_")
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["heatmap"]["target"] == "lcnn"
    assert len(r["heatmap"]["values"]) == detector.N_MELS
    assert len(r["heatmap"]["freq_hz"]) == detector.N_MELS
    # shap is present (this clip escalates) or explicitly null; if present, has the 4 keys
    if r["shap"] is not None:
        assert {"aasist", "wav2vec", "rawnet", "base"} <= set(r["shap"].keys())
    # whole response is JSON-serializable
    import json
    json.dumps(r)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_detector.py::test_detect_has_explainability_fields -q`
Expected: FAIL — `KeyError: 'audit_id'`.

- [ ] **Step 3: Add imports + `MEL_FREQS_HZ` + `_write_audit_log`** to `detector.py`

Near the top imports (after `from input_randomization import randomize_chunk`, line ~50):

```python
import explain_signals
import gradcam
```

Immediately after the `_MEL_FB = _melscale_fbanks(N_FFT // 2 + 1, 20, 8000, N_MELS, SR)` line (~330):

```python
# Mel band-center frequencies (Hz) for the Grad-CAM heatmap's frequency axis — the same
# htk mel points as _melscale_fbanks (f_min=20, f_max=8000), inner 80 centers.
_MEL_M_PTS   = torch.linspace(_hz_to_mel_htk(20), _hz_to_mel_htk(8000), N_MELS + 2)
MEL_FREQS_HZ = [round(_mel_to_hz_htk(m.item()), 1) for m in _MEL_M_PTS[1:-1]]
```

Add this module-level helper (e.g. just below `verdict_from_score`):

```python
def _write_audit_log(audit_id, sha256, verdict, score):
    """Append one chain-of-custody line to output/audit_log.jsonl. Never fatal."""
    try:
        out_dir = os.environ.get("DRIFT_OUTPUT_DIR", os.path.join(BASE, "output"))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "audit_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "audit_id": audit_id, "sha256": sha256, "verdict": verdict,
                "score": round(score, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }) + "\n")
    except Exception as e:
        print(f"  audit log write failed: {e}")
```

- [ ] **Step 4: Track the peak chunks in the scoring loop** in `detect()`

Just before the `for chunk_idx, chunk in enumerate(chunks):` loop, initialize:

```python
    peak_score, peak_wav_1d, peak_range = -1.0, None, (0.0, 0.0)
    peak2_score, peak2_feats, peak2_range = -1.0, None, (0.0, 0.0)
```

Inside the loop, immediately after `r = cascade_score_chunk(wav_1d)` (line ~717):

```python
        if r['score'] > peak_score:
            peak_score, peak_wav_1d = r['score'], wav_1d
            peak_range = (chunk_start_sec, chunk_end_sec)
        if r['stage'] == 2 and r['score'] > peak2_score:
            peak2_score = r['score']
            peak2_feats = (r['aasist'], r['wav2vec'], r['rawnet'])
            peak2_range = (chunk_start_sec, chunk_end_sec)
```

- [ ] **Step 5: Compute the signals after the final verdict** (after `verdict = verdict_from_score(score)`, ~line 739)

```python
    audit_id   = explain_signals.new_audit_id()
    confidence = explain_signals.compute_confidence([c['score'] for c in chunk_results], score)

    heatmap = None
    if peak_wav_1d is not None:
        try:
            heatmap = gradcam.lcnn_gradcam(lcnn, wav_to_mel, peak_wav_1d, MEL_FREQS_HZ,
                                           HOP_LENGTH / SR, peak_range)
        except Exception as e:
            print(f"  gradcam failed: {e}")

    shap = None
    if peak2_feats is not None:
        try:
            dm = xgb_lib.DMatrix(np.array([list(peak2_feats)], dtype=float))
            contribs = xgb_model.get_booster().predict(dm, pred_contribs=True)[0]
            shap = explain_signals.shap_from_contribs(contribs)
            shap["chunk_range_sec"] = [round(peak2_range[0], 2), round(peak2_range[1], 2)]
            shap["note"] = "margin-space contributions from the XGBoost fusion (pre-calibration)"
        except Exception as e:
            print(f"  shap failed: {e}")

    _write_audit_log(audit_id, sha256, verdict, score)
```

- [ ] **Step 6: Add the four fields to the response dict**

In the `return { ... }` dict, add alongside the existing fields (e.g. after `"explanation": explanation,`):

```python
        "audit_id":     audit_id,
        "confidence":   confidence,
        "heatmap":      heatmap,
        "shap":         shap,
```

- [ ] **Step 7: Run the new test + the golden regression**

Run: `python -m pytest tests/test_detector.py::test_detect_has_explainability_fields tests/test_golden.py -q`
Expected: PASS. The golden test passing proves `verdict`/`score` are unchanged (additive-only).

- [ ] **Step 8: Report DONE (do not commit)**

Report DONE with the test summary (new field test + golden green). Controller snapshots; user commits.

---

### Task 4: `scripts/explain_acceptance.py` — 20-sample dump harness

**Files:**
- Create: `scripts/explain_acceptance.py`

**Interfaces:**
- Consumes: `detector.detect()` (Task 3 fields `audit_id`, `heatmap.png_base64`, `confidence`, `shap`).

- [ ] **Step 1: Implement the harness**

```python
"""explain_acceptance.py — run detect() over a folder of clips and dump the explainability
outputs (heatmap PNG + summary JSON) for the Phase 3 20-sample review.

Usage: python scripts/explain_acceptance.py --clips tests/golden_clips --out explain_out
"""
import argparse, base64, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder of audio files, or a glob")
    ap.add_argument("--out", default="explain_out")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    pattern = os.path.join(args.clips, "*") if os.path.isdir(args.clips) else args.clips
    files = [f for f in sorted(glob.glob(pattern)) if os.path.isfile(f)]

    n = 0
    for f in files:
        try:
            r = detector.detect(f)
        except Exception as e:
            print(f"  {os.path.basename(f):40s} ERROR {e}")
            continue
        aid = r["audit_id"]
        hm = r.get("heatmap")
        if hm and hm.get("png_base64"):
            with open(os.path.join(args.out, aid + ".png"), "wb") as fp:
                fp.write(base64.b64decode(hm["png_base64"]))
        with open(os.path.join(args.out, aid + ".json"), "w", encoding="utf-8") as fp:
            json.dump({"file": os.path.basename(f), "verdict": r["verdict"],
                       "score": r["score"], "confidence": r["confidence"],
                       "shap": r.get("shap"), "flagged_segments": r.get("flagged_segments")},
                      fp, indent=2)
        n += 1
        print(f"  {os.path.basename(f):40s} {r['verdict']:12s} "
              f"score={r['score']:.3f} conf={r['confidence']:.3f} -> {aid}")
    print(f"\n{n} clips -> {args.out}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the golden clips (verification)**

Run: `python scripts/explain_acceptance.py --clips tests/golden_clips --out explain_out`
Expected: prints one line per clip and `N clips -> explain_out/`; `explain_out/` contains an
`aud_*.png` (heatmap) and `aud_*.json` per clip. Open one PNG to eyeball the heatmap.

- [ ] **Step 3: Report DONE (do not commit)**

Report DONE with the printed table + confirmation that PNG/JSON files were written. Controller snapshots; user commits. (`explain_out/` is throwaway — do not track it; note it for `.gitignore`.)

---

### Task 5: UI — `VoiceGuard_LiveDemo (2).html` (auth + async poll + render) and repoint `/`

**Files:**
- Modify: `VoiceGuard_LiveDemo (2).html`
- Modify: `api.py` (`/` route serves this file)

**Interfaces:**
- Consumes: the API — `POST /detect` (Bearer key) → `202 {job_id, status_url}`; `GET /jobs/{id}`
  (Bearer key) → `{status, result}`; and the response fields `verdict, score, confidence, audit_id,
  heatmap.png_base64, shap, flagged_segments, explanation`.

**Context for the implementer:** This is an existing ~700-line demo page that currently does a
single synchronous `fetch(SERVER+'/detect', {method:'POST', body:form})` (around line 632) with no
auth and expects an immediate result. Read the file first to find: the upload/submit handler, the
`SERVER` constant, and the result-rendering area (the element that shows the verdict/scores). Wire
the snippets below into that existing structure (reuse existing element ids where present; add new
containers where noted).

- [ ] **Step 1: Repoint the API's `/` route** in `api.py`

Change the `index()` handler to serve this demo:

```python
@app.get("/")
def index():
    html_path = os.path.join(detector.BASE, "VoiceGuard_LiveDemo (2).html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return PlainTextResponse("VoiceGuard_LiveDemo (2).html not found in " + detector.BASE, status_code=404)
```

- [ ] **Step 2: Add an API-key field + persistence** (near the top of the page's controls)

```html
<input id="apiKey" type="password" placeholder="API key (vg_...)" style="width:16rem"
       oninput="localStorage.setItem('vg_key', this.value)">
```
```javascript
// on load: restore saved key
document.getElementById('apiKey').value = localStorage.getItem('vg_key') || '';
function authHeaders() {
  const k = (document.getElementById('apiKey').value || '').trim();
  return k ? { 'Authorization': 'Bearer ' + k } : {};
}
```

- [ ] **Step 3: Replace the synchronous submit with submit→poll**

Replace the existing `fetch(SERVER+'/detect', ...)` call with:

```javascript
async function submitAndPoll(form) {
  // form: a FormData with the audio file under key 'file'
  const sub = await fetch(SERVER + '/detect', { method: 'POST', headers: authHeaders(), body: form });
  if (sub.status === 401) { throw new Error('Unauthorized — enter a valid API key'); }
  if (sub.status !== 202) { throw new Error('Submit failed: HTTP ' + sub.status); }
  const { job_id } = await sub.json();
  for (let i = 0; i < 120; i++) {                     // poll up to ~2 min
    await new Promise(r => setTimeout(r, 1000));
    const jr = await fetch(SERVER + '/jobs/' + job_id, { headers: authHeaders() });
    if (!jr.ok) { throw new Error('Poll failed: HTTP ' + jr.status); }
    const job = await jr.json();
    if (job.status === 'done')  { return job.result; }
    if (job.status === 'error') { throw new Error('Detection error: ' + (job.error || 'unknown')); }
  }
  throw new Error('Timed out waiting for result');
}
```
Wire the existing submit handler to call `const result = await submitAndPoll(form);` then
`renderResult(result);` (existing render + the new blocks below).

- [ ] **Step 4: Add render blocks for the new signals**

Add containers to the results area:
```html
<div id="ex-confidence"></div>
<img id="ex-heatmap" alt="Grad-CAM heatmap" style="max-width:100%;display:none">
<div id="ex-shap"></div>
<div id="ex-segments"></div>
<div id="ex-prosody"></div>
<div id="ex-audit" style="font:12px monospace;color:#888"></div>
```
```javascript
function renderExplainability(r) {
  document.getElementById('ex-confidence').textContent =
    'Confidence: ' + (r.confidence != null ? (r.confidence * 100).toFixed(0) + '%' : 'n/a');

  const img = document.getElementById('ex-heatmap');
  if (r.heatmap && r.heatmap.png_base64) {
    img.src = 'data:image/png;base64,' + r.heatmap.png_base64;
    img.style.display = 'block';
  } else { img.style.display = 'none'; }

  const shapEl = document.getElementById('ex-shap');
  if (r.shap) {
    shapEl.innerHTML = '<b>Model contributions (SHAP):</b> ' +
      ['aasist', 'wav2vec', 'rawnet'].map(k =>
        `${k}: ${r.shap[k] >= 0 ? '+' : ''}${r.shap[k].toFixed(2)}`).join(' &nbsp; ');
  } else { shapEl.textContent = 'SHAP: n/a (resolved at screener stage)'; }

  const seg = (r.flagged_segments || [])
    .map(s => `${s.start_sec.toFixed(1)}-${s.end_sec.toFixed(1)}s (${s.verdict})`).join(', ');
  document.getElementById('ex-segments').innerHTML =
    '<b>Flagged segments:</b> ' + (seg || 'none');

  const obs = (r.explanation && r.explanation.observations) || [];
  document.getElementById('ex-prosody').innerHTML =
    '<b>Acoustic notes:</b><ul>' + obs.map(o => `<li>${o}</li>`).join('') + '</ul>';

  document.getElementById('ex-audit').textContent = 'audit_id: ' + (r.audit_id || '');
}
```
Call `renderExplainability(result)` from the submit handler after the existing render. The
`flagged_segments` entries have exactly these keys: `start_sec`, `end_sec`, `score`, `verdict`
(confirmed in `detector.merge_chunks_to_segments`).

- [ ] **Step 5: Manual verification (browser)**

1. Start the stack: `python auth.py create --client me` (copy key), then `python -m uvicorn api:app --port 7860` and `python worker.py`.
2. Open `http://localhost:7860/`, paste the key, upload a clip.
3. Confirm: verdict + confidence show, the heatmap image renders, SHAP bar shows (for an escalated clip), flagged segments + acoustic notes list, and `audit_id` appears. Confirm a wrong/blank key shows the "Unauthorized" message.

- [ ] **Step 6: Report DONE (do not commit)**

Report DONE with a note of what was verified in the browser. Controller snapshots; user commits.

---

## Notes for the executor

- **Golden regression is the guardrail** — if `tests/test_golden.py` ever changes `verdict`/`score`, stop: explainability must be additive.
- `gradcam` and `detect()`'s explainability block run in the worker; matplotlib uses the Agg backend (no display needed).
- Do not commit — the controller snapshots each task; the user commits.
- `explain_out/` (Task 4) is throwaway output — flag it for `.gitignore` but don't track it.

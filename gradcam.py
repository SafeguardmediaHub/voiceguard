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
    try:
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
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        plt.close(fig)                 # always release the figure, even if savefig raises


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

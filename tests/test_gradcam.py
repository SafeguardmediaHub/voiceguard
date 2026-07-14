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
    # forward + full-backward hooks removed. Note: on this installed torch version (2.12.1),
    # register_full_backward_hook stores its handle in _backward_hooks (the older
    # _full_backward_hooks attribute the brief assumed no longer exists on this torch build;
    # verified empirically that _backward_hooks is populated on register and emptied on .remove()).
    assert len(detector.lcnn.block4._forward_hooks) == 0
    assert len(detector.lcnn.block4._backward_hooks) == 0
    assert detector.lcnn.training is False

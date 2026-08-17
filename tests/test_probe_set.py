# tests/test_probe_set.py
"""The shipped health-gate probe set (REMEDIATION_PLAN D9).

Weight-free: this only inspects files, so it runs in the CI-fast tier. The gate's
behaviour against the set is covered in tests/test_submodel_health.py.
"""
import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_DIR = os.path.join(REPO, "tests", "probe_clips")
MANIFEST = os.path.join(PROBE_DIR, "manifest.json")


def _manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def test_probe_set_exists():
    assert os.path.isdir(PROBE_DIR), (
        "the shipped probe set is missing; the promotion health gate cannot certify "
        "inside the image without it. Rebuild: python scripts/build_probe_set.py")
    assert os.path.exists(MANIFEST)


def test_every_clip_is_present_and_unmodified():
    for entry in _manifest()["clips"]:
        path = os.path.join(PROBE_DIR, entry["file"])
        assert os.path.exists(path), f"probe clip missing: {entry['file']}"
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == entry["sha256"], (
            f"{entry['file']} changed since the probe set was built; the gate's "
            "baseline is only reproducible against a fixed set")


def test_probe_set_spans_both_classes():
    """A real-only probe set silently breaks the collapse gate: a correct model
    confidently says 'real' to every real clip, so its spread collapses toward zero
    and a healthy model is reported COLLAPSED. Measured on this exact set before the
    fake half was added: LCNN 0.0076, RawNet3 0.0001, both healthy."""
    labels = {e.get("label") for e in _manifest()["clips"]}
    assert "real" in labels and "fake" in labels, (
        f"probe set must span both classes, got {labels}")


def test_licence_is_recorded_for_the_redistributed_clips():
    """Everything copied into this directory is baked into an image that gets
    pushed to a registry, so its licence has to be on the record."""
    m = _manifest()
    assert m.get("licence") == "CC0-1.0"
    assert m.get("corpus")
    copied = [e for e in m["clips"] if not e["file"].startswith("..")]
    assert copied, "expected the CC0 clips to be copied into the probe dir"
    assert all(e["label"] == "real" for e in copied)


def test_fake_half_is_referenced_not_copied():
    """The fake clips are reused from tests/golden_clips, which the image already
    ships for the golden regression. Referencing them keeps the probe set from
    adding any new redistributed audio."""
    referenced = [e for e in _manifest()["clips"] if e["file"].startswith("..")]
    assert referenced, "fake half should be referenced from golden_clips"
    for entry in referenced:
        assert entry["label"] == "fake"
        assert os.path.exists(os.path.join(PROBE_DIR, entry["file"]))


def test_probe_set_is_small_enough_to_ship():
    total = sum(os.path.getsize(os.path.join(PROBE_DIR, f))
                for f in os.listdir(PROBE_DIR) if f.endswith(".mp3"))
    assert total < 5 * 1024 * 1024, f"probe set is {total/1024/1024:.1f} MB; keep it small"

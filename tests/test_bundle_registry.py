import os, json, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bundle_registry as br


def _make_bundle(dirpath, version="vtest", corrupt=None, missing=None):
    """Create a fake bundle dir of 7 tiny files + a written manifest.
    corrupt: filename whose bytes are changed AFTER the manifest is written.
    missing: filename to delete AFTER the manifest is written."""
    os.makedirs(dirpath, exist_ok=True)
    for name in br.BUNDLE_FILES:
        with open(os.path.join(dirpath, name), "wb") as f:
            f.write(b"dummy-" + name.encode())
    br.write_manifest(dirpath, {
        "version": version,
        "metrics": {"val_eer": 0.13},
        "preprocessing": {"ensemble_peak_norm": True, "lcnn_peak_norm": False},
    })
    if corrupt:
        with open(os.path.join(dirpath, corrupt), "ab") as f:
            f.write(b"tampered")
    if missing:
        os.remove(os.path.join(dirpath, missing))
    return dirpath


def test_write_and_read_manifest_roundtrip(tmp_path):
    d = _make_bundle(str(tmp_path / "vtest"))
    m = br.read_manifest(d)
    assert m["version"] == "vtest"
    assert set(m["files"].keys()) == br.BUNDLE_FILES
    assert all("sha256" in v for v in m["files"].values())


def test_validate_bundle_accepts_intact(tmp_path):
    d = _make_bundle(str(tmp_path / "vtest"))
    br.validate_bundle(d)  # must not raise


def test_validate_bundle_rejects_missing_file(tmp_path):
    d = _make_bundle(str(tmp_path / "vtest"), missing="lcnn.pt")
    with pytest.raises(br.BundleError):
        br.validate_bundle(d)


def test_validate_bundle_rejects_tampered_file(tmp_path):
    d = _make_bundle(str(tmp_path / "vtest"), corrupt="xgb.json")
    with pytest.raises(br.BundleError):
        br.validate_bundle(d)


def test_register_and_get(tmp_path):
    store = str(tmp_path / "store")
    d = _make_bundle(str(tmp_path / "v1"), version="v1")
    reg = br.Registry(store_dir=store)
    v = reg.register_bundle(d)
    assert v == "v1"
    assert reg.get_bundle("v1")["version"] == "v1"
    assert [b["version"] for b in reg.list_bundles()] == ["v1"]
    assert reg.get_active() is None            # registered != active


def test_register_rejects_duplicate_version(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "a"), version="dup"))
    with pytest.raises(br.BundleError):
        reg.register_bundle(_make_bundle(str(tmp_path / "b"), version="dup"))


def test_register_rejects_incomplete_bundle(tmp_path):
    store = str(tmp_path / "store")
    d = _make_bundle(str(tmp_path / "bad"), version="bad", missing="rawnet.pt")
    reg = br.Registry(store_dir=store)
    with pytest.raises(br.BundleError):
        reg.register_bundle(d)


def test_verify_integrity_detects_tamper(tmp_path):
    store = str(tmp_path / "store")
    d = _make_bundle(str(tmp_path / "v1"), version="v1")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(d)
    assert reg.verify_integrity("v1") is True
    with open(os.path.join(d, "cal.json"), "ab") as f:   # tamper post-registration
        f.write(b"x")
    assert reg.verify_integrity("v1") is False


def test_promote_rollback_roundtrip(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "v1"), version="v1"))
    reg.register_bundle(_make_bundle(str(tmp_path / "v2"), version="v2"))
    reg.promote("v1", actor="test", reason="first")
    assert reg.get_active() == "v1"
    reg.promote("v2", actor="test", reason="upgrade")
    assert reg.get_active() == "v2"
    assert reg.rollback(actor="test", reason="regret") == "v1"
    assert reg.get_active() == "v1"
    assert reg.verify_active_chain() is True


def test_promote_refuses_unregistered(tmp_path):
    reg = br.Registry(store_dir=str(tmp_path / "store"))
    with pytest.raises(br.BundleError):
        reg.promote("ghost")


def test_promote_refuses_on_integrity_failure(tmp_path):
    store = str(tmp_path / "store")
    d = _make_bundle(str(tmp_path / "v1"), version="v1")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(d)
    with open(os.path.join(d, "aasist.pt"), "ab") as f:
        f.write(b"tamper")
    with pytest.raises(br.BundleError):
        reg.promote("v1")


def test_rollback_without_history_raises(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "v1"), version="v1"))
    reg.promote("v1")
    with pytest.raises(br.BundleError):   # only one entry, no prior
        reg.rollback()


def test_verify_active_chain_detects_tamper(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "v1"), version="v1"))
    reg.register_bundle(_make_bundle(str(tmp_path / "v2"), version="v2"))
    reg.promote("v1")
    reg.promote("v2")
    assert reg.verify_active_chain() is True
    # Edit a past entry's version WITHOUT recomputing its hash → the recomputed
    # entry_sha no longer matches, so the chain must report tampering.
    with open(reg.active_path) as f:
        log = json.load(f)
    log[0]["version"] = "hacked"
    with open(reg.active_path, "w") as f:
        json.dump(log, f, indent=2)
    assert reg.verify_active_chain() is False


# --- appended to tests/test_bundle_registry.py ---
import bundle_registry
from bundle_registry import Registry, BUNDLE_FILES, BundleError


def _make_pushable_bundle(store_dir, version="v9"):
    """Create a valid dummy bundle under store_dir/<version>, register + promote it."""
    bdir = os.path.join(store_dir, version)
    os.makedirs(bdir, exist_ok=True)
    for name in BUNDLE_FILES:
        with open(os.path.join(bdir, name), "wb") as f:
            f.write(f"dummy-{version}-{name}".encode())
    bundle_registry.write_manifest(bdir, {"version": version})
    reg = Registry(store_dir=store_dir)
    reg.register_bundle(bdir)
    reg.promote(version, actor="test", reason="seed")
    return reg


def test_push_then_pull_roundtrip(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.delenv("SPACES_PREFIX", raising=False)
    src_store = str(tmp_path / "src")
    reg = _make_pushable_bundle(src_store, "v9")
    reg.push_active(client=fake_s3)

    dst_store = str(tmp_path / "dst")
    reg2 = Registry(store_dir=dst_store)
    pulled = reg2.pull(client=fake_s3)
    assert pulled == "v9"
    assert reg2.get_active() == "v9"
    assert reg2.verify_integrity("v9") is True
    assert reg2.verify_active_chain() is True
    for name in BUNDLE_FILES:
        assert os.path.exists(os.path.join(dst_store, "v9", name))


def test_pull_rejects_tampered_file(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.delenv("SPACES_PREFIX", raising=False)
    src_store = str(tmp_path / "src")
    _make_pushable_bundle(src_store, "v9").push_active(client=fake_s3)

    # Flip one uploaded file's bytes in the fake store.
    key = "voiceguard/model_store/bundles/v9/xgb.json"
    fake_s3.store[("vg-bucket", key)] = b"TAMPERED"

    dst_store = str(tmp_path / "dst")
    reg2 = Registry(store_dir=dst_store)
    with pytest.raises(BundleError):
        reg2.pull(client=fake_s3)
    assert reg2.get_active() is None            # never activated a corrupt bundle


def test_pull_no_pointer_raises(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    reg = Registry(store_dir=str(tmp_path / "dst"))
    with pytest.raises(BundleError):
        reg.pull(client=fake_s3)                # empty remote, no version given

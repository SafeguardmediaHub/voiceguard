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


def test_integrity_problems_names_each_bad_file(tmp_path):
    store = str(tmp_path / "store")
    d = _make_bundle(str(tmp_path / "v1"), version="v1")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(d)
    assert reg.integrity_problems("v1") == []

    with open(os.path.join(d, "aasist.pt"), "ab") as f:
        f.write(b"tamper")
    os.remove(os.path.join(d, "lcnn.pt"))

    problems = reg.integrity_problems("v1")
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "aasist.pt" in joined and "sha256" in joined
    assert "lcnn.pt" in joined and "missing" in joined


def test_integrity_problems_distinguishes_unregistered_from_intact(tmp_path):
    """None (not registered) must not be confused with [] (registered and intact) —
    a caller treating both as falsy would skip verification on an unknown bundle."""
    reg = br.Registry(store_dir=str(tmp_path / "store"))
    assert reg.integrity_problems("ghost") is None
    assert reg.verify_integrity("ghost") is False


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
    # A named actor, so the failure is the unregistered-version check and not
    # _require_named_actor short-circuiting ahead of it.
    with pytest.raises(br.BundleError, match="unregistered"):
        reg.promote("ghost", actor="test")


def test_promote_refuses_on_integrity_failure(tmp_path):
    store = str(tmp_path / "store")
    d = _make_bundle(str(tmp_path / "v1"), version="v1")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(d)
    with open(os.path.join(d, "aasist.pt"), "ab") as f:
        f.write(b"tamper")
    with pytest.raises(br.BundleError, match="integrity"):
        reg.promote("v1", actor="test")


def test_rollback_without_history_raises(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "v1"), version="v1"))
    reg.promote("v1", actor="test")
    with pytest.raises(br.BundleError):   # only one entry, no prior
        reg.rollback(actor="test")


# ── H8: promotion/rollback require a named human approver ────────────────────

@pytest.mark.parametrize("actor", ["", "cli", "admin", "root", "unknown", "ab", "  "])
def test_promote_rejects_generic_actor(tmp_path, actor):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "v1"), version="v1"))
    with pytest.raises(br.BundleError, match="named human approver"):
        reg.promote("v1", actor=actor)
    assert reg.get_active() is None            # nothing was activated


def test_rollback_rejects_generic_actor(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "v1"), version="v1"))
    reg.register_bundle(_make_bundle(str(tmp_path / "v2"), version="v2"))
    reg.promote("v1", actor="michael.ologungbara")
    reg.promote("v2", actor="michael.ologungbara")
    with pytest.raises(br.BundleError, match="named human approver"):
        reg.rollback(actor="cli")
    assert reg.get_active() == "v2"            # rollback did not happen


def test_named_actor_lands_in_the_tamper_evident_chain(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "v1"), version="v1"))
    reg.promote("v1", actor="michael.ologungbara", reason="ship it")
    with open(reg.active_path) as f:
        log = json.load(f)
    assert log[-1]["actor"] == "michael.ologungbara"
    assert reg.verify_active_chain() is True


# ── M7: v9 can never become active again (its aasist.pt crashes detector.py) ──

def test_promote_refuses_blocked_version(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    blocked = next(iter(br.BLOCKED_VERSIONS))
    reg.register_bundle(_make_bundle(str(tmp_path / blocked), version=blocked))
    with pytest.raises(br.BundleError, match="blocked from activation"):
        reg.promote(blocked, actor="test")
    assert reg.get_active() is None


def test_rollback_refuses_into_a_blocked_version(tmp_path):
    """The dangerous path: v9 was active historically, so a plain rollback from
    its successor would land back on it and crash the service on restart."""
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    blocked = next(iter(br.BLOCKED_VERSIONS))
    reg.register_bundle(_make_bundle(str(tmp_path / blocked), version=blocked))
    reg.register_bundle(_make_bundle(str(tmp_path / "v9h"), version="v9h"))
    # Seed history directly: promote() itself refuses the blocked version.
    reg._append_active(blocked, prev_version=None, actor="migration", reason="historical")
    reg.promote("v9h", actor="test")
    with pytest.raises(br.BundleError, match="cannot roll back"):
        reg.rollback(actor="test")
    assert reg.get_active() == "v9h"


def test_verify_active_chain_detects_tamper(tmp_path):
    store = str(tmp_path / "store")
    reg = br.Registry(store_dir=store)
    reg.register_bundle(_make_bundle(str(tmp_path / "v1"), version="v1"))
    reg.register_bundle(_make_bundle(str(tmp_path / "v2"), version="v2"))
    reg.promote("v1", actor="test")
    reg.promote("v2", actor="test")
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


# Not "v9": that version is in BLOCKED_VERSIONS and can never be promoted, so
# using it as a throwaway fixture name made these tests fail on the guard rather
# than exercise push/pull.
PUSH_VERSION = "v9h"


def _make_pushable_bundle(store_dir, version=PUSH_VERSION):
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
    reg = _make_pushable_bundle(src_store, PUSH_VERSION)
    reg.push_active(client=fake_s3)

    dst_store = str(tmp_path / "dst")
    reg2 = Registry(store_dir=dst_store)
    pulled = reg2.pull(client=fake_s3)
    assert pulled == PUSH_VERSION
    assert reg2.get_active() == PUSH_VERSION
    assert reg2.verify_integrity(PUSH_VERSION) is True
    assert reg2.verify_active_chain() is True
    for name in BUNDLE_FILES:
        assert os.path.exists(os.path.join(dst_store, PUSH_VERSION, name))


def test_pull_rejects_tampered_file(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.delenv("SPACES_PREFIX", raising=False)
    src_store = str(tmp_path / "src")
    _make_pushable_bundle(src_store, PUSH_VERSION).push_active(client=fake_s3)

    # Flip one uploaded file's bytes in the fake store.
    key = f"voiceguard/model_store/bundles/{PUSH_VERSION}/xgb.json"
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

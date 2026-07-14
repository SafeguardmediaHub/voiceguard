import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEGUARD_AUTH_KEYS", str(tmp_path / "keys.json"))
    import auth
    return auth, str(tmp_path / "keys.json")


def test_create_and_verify(store):
    auth, _ = store
    key_id, plaintext = auth.create_key("Acme Bank")
    rec = auth.verify_key(plaintext)
    assert rec is not None
    assert rec["client"] == "Acme Bank"
    assert rec["key_id"] == key_id
    assert "key_sha256" not in rec           # hash never returned to callers


def test_wrong_and_empty_key_rejected(store):
    auth, _ = store
    auth.create_key("Acme")
    assert auth.verify_key("vg_not_a_real_key") is None
    assert auth.verify_key("") is None


def test_revoke(store):
    auth, _ = store
    key_id, plaintext = auth.create_key("Acme")
    assert auth.verify_key(plaintext) is not None
    assert auth.revoke_key(key_id) is True
    assert auth.verify_key(plaintext) is None   # revoked → rejected


def test_plaintext_never_stored(store):
    auth, path = store
    _, plaintext = auth.create_key("Acme")
    raw = open(path, encoding="utf-8").read()
    assert plaintext not in raw               # only the hash is on disk
    assert "key_sha256" in raw


def test_list_omits_hash(store):
    auth, _ = store
    auth.create_key("Acme")
    keys = auth.list_keys()
    assert len(keys) == 1 and "key_sha256" not in keys[0]

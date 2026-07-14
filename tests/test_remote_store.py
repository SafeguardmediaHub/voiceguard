import os
import sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import remote_store


def test_make_client_missing_var_raises(monkeypatch):
    for v in ("SPACES_KEY", "SPACES_SECRET", "SPACES_ENDPOINT", "SPACES_REGION", "SPACES_BUCKET"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(RuntimeError) as e:
        remote_store.make_client()
    assert "SPACES_" in str(e.value)


def test_bucket_prefix_defaults(monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.delenv("SPACES_PREFIX", raising=False)
    assert remote_store.bucket_prefix() == ("vg-bucket", "voiceguard/model_store")


def test_bucket_prefix_custom_strips_slash(monkeypatch):
    monkeypatch.setenv("SPACES_BUCKET", "vg-bucket")
    monkeypatch.setenv("SPACES_PREFIX", "custom/store/")
    assert remote_store.bucket_prefix() == ("vg-bucket", "custom/store")


def test_upload_download_roundtrip(fake_s3, tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"hello-bytes")
    remote_store.upload_file(fake_s3, "b", "k/a.bin", str(src))
    dst = tmp_path / "sub" / "a.bin"
    remote_store.download_file(fake_s3, "b", "k/a.bin", str(dst))
    assert dst.read_bytes() == b"hello-bytes"


def test_download_bytes_present_and_absent(fake_s3, tmp_path):
    src = tmp_path / "p.json"
    src.write_bytes(b'{"x":1}')
    remote_store.upload_file(fake_s3, "b", "k/p.json", str(src))
    assert remote_store.download_bytes(fake_s3, "b", "k/p.json") == b'{"x":1}'
    assert remote_store.download_bytes(fake_s3, "b", "missing/key") is None

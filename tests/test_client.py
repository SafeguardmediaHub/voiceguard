# tests/test_client.py
"""Backend-client tests. Uses an injected fake session — no network, no weights."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from voiceguard_client import VoiceGuardClient, VoiceGuardError


class FakeResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    """Replays a scripted list of responses and records the requests it received."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {method} {url}")
        return self._responses.pop(0)

    def post(self, url, **kw):
        return self._next("POST", url, **kw)

    def get(self, url, **kw):
        return self._next("GET", url, **kw)


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "sample.wav"
    p.write_bytes(b"RIFF0000WAVEfmt ")
    return str(p)


def test_submit_sends_bearer_key_and_returns_job_id(clip):
    s = FakeSession([FakeResponse(202, {"job_id": "j-1", "status": "queued",
                                        "status_url": "/jobs/j-1"})])
    c = VoiceGuardClient("https://voiceguard.internal:8443", "vg_secret", session=s)

    assert c.submit(clip) == "j-1"
    method, url, kw = s.calls[0]
    assert (method, url) == ("POST", "https://voiceguard.internal:8443/detect")
    assert kw["headers"]["Authorization"] == "Bearer vg_secret"


def test_wait_polls_until_done(clip):
    s = FakeSession([
        FakeResponse(200, {"job_id": "j-1", "status": "queued"}),
        FakeResponse(200, {"job_id": "j-1", "status": "running"}),
        FakeResponse(200, {"job_id": "j-1", "status": "done",
                           "result": {"verdict": "FAKE", "confidence": 0.91}}),
    ])
    c = VoiceGuardClient("https://voiceguard.internal:8443", "k", session=s)

    assert c.wait("j-1", interval=0)["verdict"] == "FAKE"
    assert len(s.calls) == 3


def test_wait_raises_on_job_error():
    s = FakeSession([FakeResponse(200, {"job_id": "j-1", "status": "error",
                                        "error": "decode failed"})])
    c = VoiceGuardClient("https://x", "k", session=s)

    with pytest.raises(VoiceGuardError, match="decode failed"):
        c.wait("j-1", interval=0)


def test_submit_honours_retry_after_on_429(clip, monkeypatch):
    slept = []
    monkeypatch.setattr("voiceguard_client.time.sleep", slept.append)
    s = FakeSession([
        FakeResponse(429, {"error": "Rate limit exceeded"}, headers={"Retry-After": "3"}),
        FakeResponse(202, {"job_id": "j-2", "status": "queued"}),
    ])
    c = VoiceGuardClient("https://x", "k", session=s)

    assert c.submit(clip) == "j-2"
    assert slept == [3.0]


def test_oversized_file_is_rejected_before_upload(tmp_path):
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (26 * 1024 * 1024))
    s = FakeSession([])                          # no request must be made
    c = VoiceGuardClient("https://x", "k", session=s)

    with pytest.raises(VoiceGuardError, match="25 MB"):
        c.submit(str(big))
    assert s.calls == []


def test_wait_times_out(monkeypatch):
    # started=0.0, then the deadline check after the first poll already exceeds 300s.
    ticks = iter([0.0, 400.0])
    monkeypatch.setattr("voiceguard_client.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("voiceguard_client.time.sleep", lambda _: None)
    s = FakeSession([FakeResponse(200, {"job_id": "j", "status": "running"})])
    c = VoiceGuardClient("https://x", "k", session=s)

    with pytest.raises(VoiceGuardError, match="timed out"):
        c.wait("j", timeout=300, interval=0)

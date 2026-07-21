# tests/test_api.py
import json, os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient

pytestmark = pytest.mark.weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_NAME = "fake_noizai_a4cd.mp3"
FAKE = os.path.join(REPO, "tests/golden_clips", FAKE_NAME)

# The expected verdict/score come from the golden manifest — the same source
# test_golden.py pins against — rather than being hardcoded here. Hardcoded
# copies go stale silently: this assertion sat at LIKELY_FAKE/0.8167 while the
# manifest and the detector both said AUTO_FAKE/0.9235, and nothing caught it
# because the weights CI tier never got far enough to run.
with open(os.path.join(REPO, "tests/golden_manifest.json"), encoding="utf-8") as _f:
    _GOLDEN = json.load(_f)
GOLDEN_FAKE = _GOLDEN["clips"][FAKE_NAME]
SCORE_TOL = _GOLDEN.get("score_tol", 1e-3)


@pytest.fixture(scope="module")
def auth_key(tmp_path_factory):
    d = tmp_path_factory.mktemp("c2")
    os.environ["VOICEGUARD_AUTH_KEYS"] = str(d / "keys.json")
    os.environ["VOICEGUARD_JOBS_DB"] = str(d / "jobs.db")
    os.environ["VOICEGUARD_JOBS_INPUT"] = str(d / "jobs_input")
    import auth
    _, key = auth.create_key("test-client")
    return key


@pytest.fixture(scope="module")
def other_key(auth_key):
    import auth
    _, key = auth.create_key("other-client")
    return key


@pytest.fixture(scope="module")
def client(auth_key):
    import api
    with TestClient(api.app) as c:
        yield c


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def test_ping_open(client):
    r = client.get("/ping")                       # no key — health endpoint is open
    assert r.status_code == 200 and r.json()["version"] == "V9"


def test_detect_requires_key(client):
    with open(FAKE, "rb") as f:
        r = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")})
    assert r.status_code == 401


def test_detect_bad_key(client):
    with open(FAKE, "rb") as f:
        r = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")},
                        headers=_hdr("vg_wrong"))
    assert r.status_code == 401


def test_detect_valid_key_returns_202(client, auth_key):
    with open(FAKE, "rb") as f:
        r = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")},
                        headers=_hdr(auth_key))
    assert r.status_code == 202
    j = r.json()
    assert j["status"] == "queued" and j["job_id"].startswith("j_")
    assert j["status_url"] == f"/jobs/{j['job_id']}"


def test_jobs_requires_key(client):
    assert client.get("/jobs/j_deadbeef").status_code == 401


def test_jobs_wrong_owner_404(client, auth_key, other_key):
    with open(FAKE, "rb") as f:
        job_id = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")},
                             headers=_hdr(auth_key)).json()["job_id"]
    assert client.get(f"/jobs/{job_id}", headers=_hdr(other_key)).status_code == 404   # not owner
    assert client.get(f"/jobs/{job_id}", headers=_hdr(auth_key)).status_code == 200    # owner


def test_detect_async_e2e(client, auth_key):
    with open(FAKE, "rb") as f:
        job_id = client.post("/detect", files={"file": ("fake.mp3", f, "audio/mpeg")},
                             headers=_hdr(auth_key)).json()["job_id"]
    # queued before the worker runs
    assert client.get(f"/jobs/{job_id}", headers=_hdr(auth_key)).json()["status"] == "queued"
    # drive the worker (FIFO), draining any jobs earlier tests left queued in the
    # shared module-scoped DB, until THIS job reaches a terminal state.
    import worker
    for _ in range(20):
        r = client.get(f"/jobs/{job_id}", headers=_hdr(auth_key))
        if r.json()["status"] in ("done", "error"):
            break
        assert worker.process_one() is True   # a job was available to process
    r = client.get(f"/jobs/{job_id}", headers=_hdr(auth_key))
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "done"
    assert j["result"]["verdict"] == GOLDEN_FAKE["verdict"]
    assert abs(j["result"]["score"] - GOLDEN_FAKE["score"]) < SCORE_TOL


def test_drift_requires_key(client, auth_key):
    assert client.get("/drift").status_code == 401
    assert client.get("/drift", headers=_hdr(auth_key)).status_code == 200


def test_detect_missing_file_400(client, auth_key):
    r = client.post("/detect", headers=_hdr(auth_key))     # valid key, no file
    assert r.status_code == 400
    assert "No file provided" in r.json().get("error", "")


def test_detect_oversize_413(client, auth_key, monkeypatch):
    monkeypatch.setenv("VOICEGUARD_MAX_UPLOAD_MB", "1")     # 1 MB limit for this test
    blob = b"\0" * 1_200_000                                # ~1.2 MB > 1 MB
    r = client.post("/detect", files={"file": ("big.wav", blob, "audio/wav")},
                    headers=_hdr(auth_key))
    assert r.status_code == 413
    assert r.json().get("max_mb") == 1


def test_cors_is_not_wildcard_by_default(client):
    r = client.get("/ping", headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-origin") != "*"

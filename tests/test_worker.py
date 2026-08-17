# tests/test_worker.py
import json, os, sys, shutil
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = "fake_noizai_a4cd.mp3"


def _golden(fname):
    """Expected detect() output for a golden clip, from the single source of truth.
    Hardcoding it here let this test go stale against the deployed bundle."""
    with open(os.path.join(REPO, "tests/golden_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["clips"][fname], manifest.get("score_tol", 1e-3)


def test_process_one_scores_a_real_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEGUARD_JOBS_DB", str(tmp_path / "jobs.db"))
    import jobs, worker
    exp, tol = _golden(CLIP)
    # copy a golden fake clip to a temp input the worker will consume + delete
    inp = str(tmp_path / "in.mp3")
    shutil.copy2(os.path.join(REPO, "tests/golden_clips", CLIP), inp)
    jid = jobs.enqueue("Acme", "k_1", inp)

    assert worker.process_one() is True
    j = jobs.get(jid)
    assert j["status"] == "done"
    assert j["result"]["verdict"] == exp["verdict"]
    assert abs(j["result"]["score"] - exp["score"]) < tol
    assert not os.path.exists(inp)          # worker deleted the input
    assert worker.process_one() is False    # queue now empty

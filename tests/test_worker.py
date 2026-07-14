# tests/test_worker.py
import os, sys, shutil
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_process_one_scores_a_real_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEGUARD_JOBS_DB", str(tmp_path / "jobs.db"))
    import jobs, worker
    # copy a golden fake clip to a temp input the worker will consume + delete
    inp = str(tmp_path / "in.mp3")
    shutil.copy2(os.path.join(REPO, "tests/golden_clips/fake_noizai_a4cd.mp3"), inp)
    jid = jobs.enqueue("Acme", "k_1", inp)

    assert worker.process_one() is True
    j = jobs.get(jid)
    assert j["status"] == "done"
    assert j["result"]["verdict"] == "LIKELY_FAKE"
    assert abs(j["result"]["score"] - 0.8167) < 1e-3
    assert not os.path.exists(inp)          # worker deleted the input
    assert worker.process_one() is False    # queue now empty

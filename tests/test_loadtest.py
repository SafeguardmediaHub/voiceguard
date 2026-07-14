import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import loadtest


def test_percentile_nearest_rank():
    vals = list(range(1, 101))            # 1..100
    assert loadtest.percentile(vals, 50) == 50
    assert loadtest.percentile(vals, 95) == 95
    assert loadtest.percentile(vals, 99) == 99
    assert loadtest.percentile([42.0], 95) == 42.0
    assert math.isnan(loadtest.percentile([], 95))


def test_evaluate_pass():
    r = loadtest.evaluate([10.0] * 100, elapsed_s=5.0, p95_ms=200.0, min_rps=1.67)
    assert r["passed"] is True
    assert r["p95"] == 10.0 and r["rps"] == 20.0 and r["n"] == 100


def test_evaluate_fail_slow_p95():
    # 10% of requests at 500ms pushes p95 over the 200ms gate
    r = loadtest.evaluate([10.0] * 90 + [500.0] * 10, elapsed_s=5.0, p95_ms=200.0, min_rps=1.67)
    assert r["passed"] is False


def test_evaluate_fail_low_rps():
    r = loadtest.evaluate([10.0] * 3, elapsed_s=10.0, p95_ms=200.0, min_rps=1.67)  # 0.3 rps
    assert r["passed"] is False

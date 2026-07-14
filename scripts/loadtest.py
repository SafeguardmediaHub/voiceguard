#!/usr/bin/env python3
"""loadtest.py — stdlib load generator for VoiceGuard POST /detect (Phase 7 / C4a).

Fires N concurrent submissions against a running API, measures submission latency,
and exits 0 only if p95 < threshold and achieved RPS >= target. No dependencies.

  python scripts/loadtest.py --url http://localhost:7860 --key vg_... \
      --clip tests/golden_clips/fake_noizai_a4cd.mp3
"""
import sys, os, time, math, uuid, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor


def percentile(sorted_values, p):
    if not sorted_values:
        return float("nan")
    rank = max(1, math.ceil(p / 100.0 * len(sorted_values)))   # nearest-rank
    return sorted_values[rank - 1]


def evaluate(latencies_ms, elapsed_s, p95_ms, min_rps):
    s = sorted(latencies_ms)
    n = len(s)
    rps = n / elapsed_s if elapsed_s > 0 else 0.0
    p95 = percentile(s, 95)
    return {
        "n": n,
        "p50": percentile(s, 50),
        "p95": p95,
        "p99": percentile(s, 99),
        "rps": rps,
        "passed": (n > 0 and p95 < p95_ms and rps >= min_rps),
    }


def _multipart(clip_bytes, filename):
    boundary = "----vg" + uuid.uuid4().hex
    head = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n").encode()
    body = head + clip_bytes + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def submit_once(url, key, clip_bytes, filename):
    body, ctype = _multipart(clip_bytes, filename)
    req = urllib.request.Request(url.rstrip("/") + "/detect", data=body, method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", ctype)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 202:
            raise RuntimeError(f"expected 202, got {resp.status}")
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def run_load(url, key, clip, n, concurrency):
    with open(clip, "rb") as f:
        clip_bytes = f.read()
    filename = os.path.basename(clip)
    latencies, errors = [], []

    def one(_):
        try:
            return ("ok", submit_once(url, key, clip_bytes, filename))
        except Exception as e:
            return ("err", str(e))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for kind, val in ex.map(one, range(n)):
            (latencies if kind == "ok" else errors).append(val)
    return latencies, errors, time.perf_counter() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description="VoiceGuard /detect load test")
    p.add_argument("--url", default="http://localhost:7860")
    p.add_argument("--key", required=True)
    p.add_argument("--clip", default="tests/golden_clips/fake_noizai_a4cd.mp3")
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--p95-ms", type=float, default=200.0)
    p.add_argument("--min-rps", type=float, default=100 / 60.0)
    args = p.parse_args(argv)

    latencies, errors, elapsed = run_load(args.url, args.key, args.clip,
                                          args.requests, args.concurrency)
    stats = evaluate(latencies, elapsed, args.p95_ms, args.min_rps)
    ok = stats["passed"] and not errors
    print(f"  requests={args.requests} ok={stats['n']} errors={len(errors)} elapsed={elapsed:.1f}s")
    print(f"  submission latency ms:  p50={stats['p50']:.1f}  p95={stats['p95']:.1f}  p99={stats['p99']:.1f}")
    print(f"  achieved RPS={stats['rps']:.1f}  (>= {args.min_rps:.2f} required)")
    print(f"  GATE (p95 < {args.p95_ms:.0f}ms and RPS >= {args.min_rps:.2f}): {'PASS' if ok else 'FAIL'}")
    if errors:
        print(f"  first error: {errors[0]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

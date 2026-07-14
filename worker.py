#!/usr/bin/env python3
"""worker.py — VoiceGuard async detection worker (Phase 7 / C2).

Polls the SQLite job queue, runs detector.detect on each job's input, writes the
result back, and deletes the input file. Run alongside the API:
    python worker.py
"""
import os, time, logging
import jobs
import detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("worker")


def process_one():
    """Claim and process one job. Returns True if a job was processed, else False."""
    job = jobs.claim_next()
    if job is None:
        return False
    job_id, path = job["job_id"], job["input_path"]
    log.info(f"processing {job_id} ({path})")
    try:
        result = detector.detect(path)
        jobs.complete(job_id, result)
        log.info(f"done {job_id}: {result.get('verdict')}")
    except Exception as e:
        jobs.fail(job_id, str(e))
        log.error(f"failed {job_id}: {e}")
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return True


def run_forever(poll_interval=0.5):
    log.info(f"worker started (active bundle {detector.ACTIVE_VERSION}); polling every {poll_interval}s")
    while True:
        try:
            processed = process_one()
        except Exception as e:
            # A DB error (e.g. 'database is locked' under load) must not kill the
            # worker — log and keep polling. requeue_stale on next start recovers.
            log.error(f"process_one crashed, continuing: {e}")
            processed = False
        if not processed:
            time.sleep(poll_interval)


if __name__ == "__main__":
    n = jobs.requeue_stale()
    if n:
        log.info(f"requeued {n} stale running job(s)")
    run_forever()

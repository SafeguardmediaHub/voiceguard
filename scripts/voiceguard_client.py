#!/usr/bin/env python3
"""voiceguard_client.py — reference client for the backend (design §11).

VoiceGuard is async: POST /detect returns 202 with a job_id, then you poll
GET /jobs/{job_id} until it is done or error.

    from voiceguard_client import VoiceGuardClient
    vg = VoiceGuardClient("https://voiceguard.internal:8443", os.environ["VOICEGUARD_API_KEY"])
    print(vg.detect("clip.wav")["verdict"])

TLS: the service uses Caddy's internal CA. Either install its root on this host
(docs/RUNBOOK-deploy.md §4) or pass verify="/path/to/voiceguard-root.crt".
"""
import os
import time

MAX_UPLOAD_MB = 25          # must match VOICEGUARD_MAX_UPLOAD_MB and deploy/Caddyfile
TERMINAL = ("done", "error")


class VoiceGuardError(RuntimeError):
    """Any non-recoverable failure: rejected upload, job error, or timeout."""


class VoiceGuardClient:
    def __init__(self, base_url, api_key, session=None, verify=True, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verify = verify
        self.timeout = timeout
        if session is None:
            import requests                     # lazy: tests inject a fake session
            session = requests.Session()
        self.session = session

    @property
    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def submit(self, path, max_retries=3):
        """Upload one audio file. Returns the job_id. Honours 429 Retry-After."""
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            # Fail here rather than burning an upload the server will 413 anyway.
            # Well over the cap the proxy resets the connection mid-body instead of
            # returning a clean 413, so this pre-check is what keeps the failure legible.
            raise VoiceGuardError(f"{path} is {size_mb:.1f} MB, over the {MAX_UPLOAD_MB} MB limit")

        for attempt in range(max_retries):
            with open(path, "rb") as fh:
                resp = self.session.post(
                    f"{self.base_url}/detect",
                    headers=self._headers,
                    files={"file": (os.path.basename(path), fh, "application/octet-stream")},
                    verify=self.verify,
                    timeout=self.timeout,
                )
            if resp.status_code == 429:
                # The server's rate limiter also flags probing-shaped traffic; backing
                # off as instructed is what keeps a client out of that heuristic.
                delay = float(resp.headers.get("Retry-After", 5))
                if attempt == max_retries - 1:
                    raise VoiceGuardError(f"rate limited after {max_retries} attempts")
                time.sleep(delay)
                continue
            if resp.status_code != 202:
                raise VoiceGuardError(f"submit failed ({resp.status_code}): {resp.json()}")
            return resp.json()["job_id"]

        raise VoiceGuardError("submit exhausted retries")

    def wait(self, job_id, timeout=300, interval=2.0):
        """Poll until the job finishes. Returns the result dict, raises on error."""
        started = time.monotonic()
        while True:
            resp = self.session.get(f"{self.base_url}/jobs/{job_id}",
                                    headers=self._headers, verify=self.verify,
                                    timeout=self.timeout)
            if resp.status_code != 200:
                raise VoiceGuardError(f"poll failed ({resp.status_code}): {resp.json()}")
            body = resp.json()
            status = body["status"]
            if status == "done":
                return body["result"]
            if status == "error":
                raise VoiceGuardError(f"job {job_id} failed: {body.get('error')}")
            if time.monotonic() - started > timeout:
                raise VoiceGuardError(f"job {job_id} timed out after {timeout}s (last status: {status})")
            time.sleep(interval)

    def detect(self, path, timeout=300, interval=2.0):
        """submit + wait. Returns the verdict dict."""
        return self.wait(self.submit(path), timeout=timeout, interval=interval)


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Submit one clip to VoiceGuard and print the verdict")
    p.add_argument("path")
    p.add_argument("--url", default=os.environ.get("VOICEGUARD_URL", "https://voiceguard.internal:8443"))
    p.add_argument("--key", default=os.environ.get("VOICEGUARD_API_KEY"))
    p.add_argument("--cacert", default=os.environ.get("VOICEGUARD_CACERT"))
    a = p.parse_args()
    if not a.key:
        raise SystemExit("set --key or $VOICEGUARD_API_KEY")
    client = VoiceGuardClient(a.url, a.key, verify=a.cacert or True)
    print(json.dumps(client.detect(a.path), indent=2))

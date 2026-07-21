# tests/test_docker_context.py
"""Guard: nothing sensitive may enter the Docker build context.

Dockerfile does `COPY . .`, so anything not excluded by .dockerignore is baked
into a layer that gets pushed to DigitalOcean Container Registry. This test is a
regression fence around that — it needs no weights, so it runs in the fast tier.
"""
import fnmatch
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Top-level names that must never be copied into the image.
FORBIDDEN = [
    "auth_keys.json",     # SHA-256 hashes of live client API keys
    "jobs.db",            # real job history, ~7 MB
    "jobs.db-wal",
    "jobs.db-shm",
    ".env",
    ".env.production",
    "jobs_input",         # transient uploads
    "AI_Audio_Detector_Phase_Plan.docx",
    ".pytest_cache",
    "voice_guard 0ffline",  # nested duplicate copy of the whole project
]

# Files the image genuinely needs — a too-broad exclusion would break the build.
REQUIRED = [
    "api.py",
    "worker.py",
    "detector.py",
    "requirements.txt",
    "VoiceGuard_LiveDemo (2).html",   # served by GET / (api.py:108)
    "model_store/ACTIVE.json",
    "model_store/registry.jsonl",
]


def _rules():
    """(pattern, is_negation) pairs from .dockerignore, comments and blanks dropped."""
    out = []
    with open(os.path.join(REPO, ".dockerignore"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("!"):
                out.append((line[1:].rstrip("/"), True))
            else:
                out.append((line.rstrip("/"), False))
    return out


def _excluded(path, rules):
    """Docker semantics: last matching rule wins; a directory pattern covers its children."""
    verdict = False
    for pattern, negated in rules:
        if fnmatch.fnmatch(path, pattern) or path.startswith(pattern + "/"):
            verdict = not negated
    return verdict


def test_sensitive_paths_are_excluded():
    rules = _rules()
    leaked = [p for p in FORBIDDEN if not _excluded(p, rules)]
    assert leaked == [], f"these would be baked into the image: {leaked}"


def test_required_paths_are_not_excluded():
    rules = _rules()
    dropped = [p for p in REQUIRED if _excluded(p, rules)]
    assert dropped == [], f".dockerignore is too broad, these are needed: {dropped}"

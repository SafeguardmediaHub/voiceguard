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
    "governance",         # tamper-evident audit log: runtime state, lives on /data
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
    # Without the probe set the promotion health gate cannot certify inside the
    # image, and --skip-health becomes the only way to promote (D9).
    "tests/probe_clips/manifest.json",
    "tests/probe_clips/cv_ha_26699728.mp3",
    "tests/golden_clips/fake_noizai_a4cd.mp3",   # the probe set's fake half
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


def test_deploy_config_never_sets_the_device_override():
    """detector.DEVICE honours $VOICEGUARD_DEVICE, defaulting to cpu.

    The override exists only so the H1 evaluation can run on a GPU box; production
    is a CPU droplet. If a deploy file ever set it, the serving path would silently
    change device — and latency, throughput and the cascade's threshold behaviour
    with it — without anyone editing detector.py. Fence it here.
    """
    offenders = []
    for rel in ("Dockerfile", "docker-compose.yml", "docker-compose.prod.yml",
                ".env.example", "deploy/entrypoint.sh"):
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                if "VOICEGUARD_DEVICE" in line and not line.strip().startswith("#"):
                    offenders.append(f"{rel}:{n}: {line.strip()}")
    assert offenders == [], (
        "deploy config sets VOICEGUARD_DEVICE; production must stay on the "
        f"default cpu path:\n  " + "\n  ".join(offenders))


def test_api_entrypoint_does_not_preload_pytorch_before_fork():
    """The FastAPI lifespan runs detector.startup_check(), including inference.

    Gunicorn --preload imports the PyTorch model in the master before forking;
    the worker can then deadlock in native thread-pool initialization.  Keep
    application loading in the workers themselves.
    """
    path = os.path.join(REPO, "deploy", "docker-entrypoint.sh")
    with open(path, encoding="utf-8") as f:
        entrypoint = f.read()
    assert "--preload" not in entrypoint, (
        "Gunicorn --preload can deadlock VoiceGuard's PyTorch startup check "
        "after workers fork"
    )


def _dockerfile_env():
    """DRIFT_OUTPUT_DIR=... style assignments from the Dockerfile's ENV lines."""
    env = {}
    with open(os.path.join(REPO, "Dockerfile"), encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip("\\").strip()
            if line.startswith("ENV "):
                line = line[4:].strip()
            elif "=" not in line or line.startswith("#"):
                continue
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    env[k] = v
    return env


def test_drift_output_dir_is_on_the_persistent_volume():
    """api.py READS drift artifacts from $DRIFT_OUTPUT_DIR; drift_monitor_3.py WRITES
    them there. Both default to a path inside the image (`<repo>/output` and a
    relative `output`), which in production means two different ephemeral dirs:
    /drift would answer empty forever, and every report would be lost on redeploy.
    Only an explicit /data path makes the reader and the writer meet on the volume.
    """
    value = _dockerfile_env().get("DRIFT_OUTPUT_DIR")
    assert value is not None, (
        "Dockerfile does not set DRIFT_OUTPUT_DIR, so drift_monitor_3.py writes to "
        "an ephemeral /app/output that api.py's /drift endpoint never reads.")
    assert value.startswith("/data/"), (
        f"DRIFT_OUTPUT_DIR={value} is not under /data — only /data is a mounted "
        "volume (docker-compose.prod.yml: vg-data:/data), so reports would not "
        "survive a redeploy.")


def test_sensitive_paths_are_excluded():
    rules = _rules()
    leaked = [p for p in FORBIDDEN if not _excluded(p, rules)]
    assert leaked == [], f"these would be baked into the image: {leaked}"


def test_required_paths_are_not_excluded():
    rules = _rules()
    dropped = [p for p in REQUIRED if _excluded(p, rules)]
    assert dropped == [], f".dockerignore is too broad, these are needed: {dropped}"

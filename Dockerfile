# VoiceGuard — API + worker image (self-contained handoff build).
# The active model bundle (v9h) is baked in, so the image runs fully offline with no
# DigitalOcean Spaces credentials. If SPACES_BUCKET is left unset the entrypoint skips
# the remote pull and uses the baked-in bundle (see docker-entrypoint.sh).
# Python 3.13 to match the env requirements.txt was frozen from (audioop-lts,
# numpy 2.4, torch 2.12 all need 3.13).
FROM python:3.13-slim

# System dependency: ffmpeg (detector shells out to it for audio decode).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first, so this layer caches across code changes.
COPY requirements.txt .
# Production is a CPU droplet — detector.DEVICE defaults to cpu and no deploy file may
# set $VOICEGUARD_DEVICE (fenced by tests/test_docker_context.py). But PyPI's linux
# torch wheels are the CUDA build: torch is 532 MB and pulls nvidia-cublas / cudnn /
# nccl / nvshmem / cusparselt, triton and cuda-toolkit — several GB of layer that this
# image can never execute. Install the +cpu builds of the same pinned versions first;
# --no-deps leaves every transitive pin to requirements.txt, which then sees torch==X
# as satisfied by X+cpu (PEP 440 ignores the local label) and downloads no torch.
RUN pins=$(grep -E '^torch(audio|vision|codec)?==' requirements.txt) \
 && echo "$pins" \
 && pip install --no-cache-dir --no-deps \
      --index-url https://download.pytorch.org/whl/cpu $pins \
 && pip install --no-cache-dir -r requirements.txt

# Bake external model weights into the image so runtime is offline + deterministic.
ENV HF_HOME=/opt/hf \
    AUDIOSEAL_CACHE_DIR=/opt/audioseal
RUN python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')"
RUN python -c "from audioseal import AudioSeal; AudioSeal.load_detector('audioseal_detector_16bits')"

# Bake the active model bundle (v9h) as its own layer — large + stable, so it caches
# across code changes. Only ACTIVE.json, registry.jsonl and model_store/v9h are in the
# build context (see .dockerignore); the flat models/ dir and inactive bundles are not.
COPY model_store /app/model_store

# Application code (models/, data/, scratch excluded via .dockerignore).
COPY . .

ENV VOICEGUARD_FFMPEG=ffmpeg \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VOICEGUARD_MODEL_STORE=/app/model_store \
    VOICEGUARD_JOBS_DB=/data/jobs.db \
    VOICEGUARD_JOBS_INPUT=/data/jobs_input \
    VOICEGUARD_AUTH_KEYS=/data/auth_keys.json \
    VOICEGUARD_GOVERNANCE_DIR=/data/governance \
    DRIFT_OUTPUT_DIR=/data/drift \
    PORT=7860 \
    WORKERS=3

EXPOSE 7860
COPY deploy/docker-entrypoint.sh /usr/local/bin/entrypoint.sh
# Build contexts copied from Windows can carry CRLF endings.  Normalize the
# entrypoint inside the Linux image so its /usr/bin/env bash shebang is valid.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]

# VoiceGuard — API + worker image.
# Models are NOT baked in (too large / product IP) — they are pulled from a DigitalOcean
# Spaces bucket at runtime via `bundle_registry.py pull --active` (see docker-entrypoint.sh).
FROM python:3.12-slim

# System dependency: ffmpeg (detector shells out to it for audio decode).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first, so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the wav2vec2 base weights into the image so runtime is offline + deterministic.
ENV HF_HOME=/opt/hf
RUN python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')"

# Application code (models/, data/, scratch excluded via .dockerignore).
COPY . .

ENV VOICEGUARD_FFMPEG=ffmpeg \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VOICEGUARD_MODEL_STORE=/app/model_store \
    VOICEGUARD_JOBS_DB=/data/jobs.db \
    VOICEGUARD_JOBS_INPUT=/data/jobs_input \
    VOICEGUARD_AUTH_KEYS=/data/auth_keys.json \
    PORT=7860 \
    WORKERS=3

EXPOSE 7860
COPY deploy/docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]

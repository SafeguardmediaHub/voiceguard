#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$VOICEGUARD_JOBS_INPUT" "$VOICEGUARD_MODEL_STORE" "$(dirname "$VOICEGUARD_JOBS_DB")" \
         "$VOICEGUARD_GOVERNANCE_DIR" "${DRIFT_OUTPUT_DIR:-/data/drift}"

# Fetch the active model bundle from DigitalOcean Spaces on first start (needs SPACES_* env).
if [ -n "${SPACES_BUCKET:-}" ] && [ ! -f "$VOICEGUARD_MODEL_STORE/ACTIVE.json" ]; then
  echo "[entrypoint] pulling active model bundle from Spaces..."
  python bundle_registry.py pull --active
fi

role="${1:-api}"
case "$role" in
  api)
    exec gunicorn -k uvicorn.workers.UvicornWorker --preload \
         -w "${WORKERS:-3}" -b "0.0.0.0:${PORT:-7860}" --timeout 120 api:app ;;
  worker)
    exec python worker.py ;;
  *)
    exec "$@" ;;   # anything else runs verbatim (e.g. `bundle_registry.py active`)
esac

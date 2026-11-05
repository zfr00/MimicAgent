#!/usr/bin/env bash
set -euo pipefail
KG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${KG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -m uvicorn tools.kg_backend:app \
    --host "${KG_HOST:-127.0.0.1}" --port "${KG_PORT:-10087}" --workers "${KG_WORKERS:-1}" "$@"

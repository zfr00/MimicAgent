#!/usr/bin/env bash
set -euo pipefail
ARIS_TRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ARIS_TRAIN_DIR}:${ARIS_TRAIN_DIR}/slime:${ARIS_TRAIN_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -m aris.launch "$@"

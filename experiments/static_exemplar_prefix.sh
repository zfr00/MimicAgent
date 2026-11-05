#!/usr/bin/env bash
set -euo pipefail
EXPERIMENTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXPERIMENTS_DIR}/run_experiment.sh" static_exemplar_prefix "$@"

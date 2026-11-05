#!/usr/bin/env bash
set -euo pipefail
SCEMQA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export EVAL_DATA="${SCEMQA_DATA:-${SCEMQA_ROOT}/data/ScEMQA/ScEMQA_EN.jsonl}"
exec bash "${SCEMQA_ROOT}/evaluation/run_eval.sh" \
    --images-root "${SCEMQA_ROOT}/data/ScEMQA/images" \
    --output "${SCEMQA_OUTPUT:-${SCEMQA_ROOT}/evaluation/output/ScEMQA/results.jsonl}" \
    --runs 1 "$@"

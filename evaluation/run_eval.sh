#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd -- "${EVAL_DIR}/.." && pwd)"
export PYTHONPATH="${EVAL_ROOT}:${EVAL_ROOT}/train${PYTHONPATH:+:${PYTHONPATH}}"
EVAL_ARGS=(--output "${EVAL_OUTPUT:-${EVAL_DIR}/output/results.jsonl}")
EVAL_HAS_INPUT=false
for argument in "$@"; do
    if [[ "$argument" == "--input" || "$argument" == --input=* ]]; then
        EVAL_HAS_INPUT=true
    fi
done
if [[ "$EVAL_HAS_INPUT" == false ]]; then
    EVAL_ARGS+=(--input "${EVAL_DATA:-${EVAL_ROOT}/data/bench/aris_eval.jsonl}")
fi
exec "${PYTHON:-python3}" "${EVAL_DIR}/evaluate.py" "${EVAL_ARGS[@]}" "$@"

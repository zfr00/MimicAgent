#!/usr/bin/env bash
set -euo pipefail
EXPERIMENTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_ROOT="$(cd -- "${EXPERIMENTS_DIR}/.." && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash experiments/run_experiment.sh EXPERIMENT {train|eval} [launcher arguments]" >&2
    exit 2
fi
EXPERIMENT_NAME="$1"
EXPERIMENT_PHASE="$2"
shift 2
case "${EXPERIMENT_NAME}" in
    wo_tools|prefix)
        if [[ "${EXPERIMENT_PHASE}" != eval ]]; then
            echo "${EXPERIMENT_NAME} is an evaluation baseline; use eval." >&2
            exit 2
        fi
        ;;
    wo_exemplar_recall|static_exemplar_prefix|wo_analogy_reward) ;;
    *) echo "Unknown experiment: ${EXPERIMENT_NAME}" >&2; exit 2 ;;
esac

case "${EXPERIMENT_PHASE}" in
    train)
        exec bash "${EXPERIMENTS_ROOT}/train/run_aris.sh" \
            --output "${EXPERIMENTS_ROOT}/train/outputs/${EXPERIMENT_NAME}" \
            "$@" --experiment "${EXPERIMENT_NAME}"
        ;;
    eval)
        exec bash "${EXPERIMENTS_ROOT}/evaluation/run_eval.sh" \
            --output "${EXPERIMENTS_ROOT}/evaluation/output/${EXPERIMENT_NAME}/results.jsonl" \
            "$@" --experiment "${EXPERIMENT_NAME}"
        ;;
    *) echo "Expected train or eval, got: ${EXPERIMENT_PHASE}" >&2; exit 2 ;;
esac

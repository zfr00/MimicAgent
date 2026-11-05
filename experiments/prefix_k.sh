#!/usr/bin/env bash
# Evaluate K=1..4 static prefixes on the SciExplore English split.
set -euo pipefail
PREFIX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX_OUTPUT_DIR="${PREFIX_K_OUTPUT_DIR:-${PREFIX_ROOT}/evaluation/output/prefix_k}"
PREFIX_DATA="${PREFIX_K_DATA:-${PREFIX_ROOT}/data/bench/EN_240.jsonl}"
PREFIX_ARGS=()
PREFIX_HAS_INPUT=false
for argument in "$@"; do
    case "$argument" in
        --input|--input=*) PREFIX_HAS_INPUT=true ;;
        --output|--output=*|--prefix-k|--prefix-k=*|--experiment|--experiment=*)
            echo 'Use PREFIX_K_OUTPUT_DIR to set the sweep output directory; K and experiment are set by this script.' >&2
            exit 2 ;;
    esac
done
if [[ "$PREFIX_HAS_INPUT" == false ]]; then
    PREFIX_ARGS+=(--input "$PREFIX_DATA")
fi
for k in 1 2 3 4; do
    bash "${PREFIX_ROOT}/evaluation/run_eval.sh" "${PREFIX_ARGS[@]}" "$@" \
        --experiment prefix --prefix-k "$k" \
        --output "${PREFIX_OUTPUT_DIR}/k${k}/results.jsonl"
done

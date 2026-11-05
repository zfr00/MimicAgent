#!/usr/bin/env bash
set -euo pipefail
ARIS_TRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Retain the installation entry point; ARIS support is included in the bundled sources.
exec "${PYTHON:-python3}" -B - "${ARIS_TRAIN_DIR}/slime" <<'PY'
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
checks = (
    ("slime/backends/megatron_utils/data.py", "log_rollout_data",
     {"metadata", "source_names", "micro_batch_indices"}),
    ("slime/backends/megatron_utils/model.py", "train_one_step",
     {"metadata", "advantages", "returns"}),
    ("slime/ray/rollout.py", "_split_train_data_by_dp",
     {"metadata", "round_number", "sample_indices"}),
)
for relative, function_name, required in checks:
    path = root / relative
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        sys.exit(f"Cannot check ARIS metadata support in {path}: {error}")
    functions = (
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    lists = (
        node for function in functions for node in ast.walk(function)
        if isinstance(node, (ast.List, ast.Tuple))
    )
    if not any(
        required <= {item.value for item in node.elts if isinstance(item, ast.Constant)
                     and isinstance(item.value, str)}
        for node in lists
    ):
        sys.exit(f"Missing ARIS metadata support in {path}: use the bundled train/slime sources")
print("Bundled slime ARIS metadata support verified.")
PY

#!/usr/bin/env bash
set -euo pipefail
ARIS_TRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARIS_PATCH="${ARIS_TRAIN_DIR}/patches/slime-aris.patch"
if patch -d "${ARIS_TRAIN_DIR}/slime" -p1 --reverse --dry-run --force --silent --fuzz=0 < "$ARIS_PATCH" >/dev/null 2>&1; then
    echo "ARIS metadata patch is already applied."
else
    patch -d "${ARIS_TRAIN_DIR}/slime" -p1 --forward --dry-run --batch --fuzz=0 < "$ARIS_PATCH"
    patch -d "${ARIS_TRAIN_DIR}/slime" -p1 --forward --batch --fuzz=0 --no-backup-if-mismatch < "$ARIS_PATCH"
    echo "Applied ARIS metadata patch."
fi

#!/usr/bin/env bash
# Diagnostics wheel from a tree snapshot. Usage: build-wheel.sh <tree> <workdir> <label>
set -euo pipefail
TREE="$1"; WORK="$2"; LABEL="$3"
mkdir -p "$WORK"
cp /tmp/fi-testwork/mlx-0.32.2-1f8e74e3f12f31365464a6867c6579f0e9b29d85.tar.gz "$WORK/" 2>/dev/null || true
export MLX_OMARCHY_WORK_DIR="$WORK"
export MLX_OMARCHY_SOURCE_COMMIT="4e116e54.$LABEL"
export CMAKE_BUILD_PARALLEL_LEVEL=12
cd "$TREE"
rm -rf "$WORK/venv-build"
cp -a /tmp/fi-work/venv-build "$WORK/venv-build" 2>/dev/null || true
./scripts/build-wheel.sh --diagnostics
ls -la "$TREE/dist"
echo WHEEL_OK

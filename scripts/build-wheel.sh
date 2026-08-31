#!/usr/bin/env bash
# Build the mlx-omarchy wheel end to end.
# This is the U5 packaging gate: scripts/build-wheel.sh
#
# Prepares the pinned upstream tree with the Omarchy-only backend, builds the
# python bindings, and writes exactly one wheel into dist/ at the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${MLX_OMARCHY_WORK_DIR:-$ROOT/.work}"
VENV_DIR="$WORK_DIR/venv-build"
DIST_DIR="$ROOT/dist"

echo "== prepare upstream tree =="
"$ROOT/scripts/prepare-mlx.sh"

echo "== build venv ($VENV_DIR) =="
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
venv_python="$VENV_DIR/bin/python"

# Offline-first: keep system packages when they satisfy the pyproject build
# requirements (setuptools>=80, typing_extensions, cmake>=3.25); pip install
# only what is missing.
if ! "$venv_python" - <<'EOF'
import setuptools, typing_extensions
assert int(setuptools.__version__.split(".")[0]) >= 80
EOF
then
  "$venv_python" -m pip install 'setuptools>=80' typing_extensions
fi

cmake_version="$(cmake --version 2>/dev/null | sed -n '1s/^cmake version //p' || true)"
if [[ -z "$cmake_version" || "$(printf '%s\n' 3.25 "$cmake_version" | sort -V | head -n1)" != "3.25" ]]; then
  "$venv_python" -m pip install 'cmake>=3.25'
fi

echo "== wheel build =="
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# setup.py appends CMAKE_ARGS to its cmake invocation; it splits the value on
# spaces, so keep each -D flag space separated.
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF"
export CMAKE_BUILD_PARALLEL_LEVEL=16
export PATH="$VENV_DIR/bin:$PATH"

"$venv_python" -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$DIST_DIR" "$WORK_DIR/mlx"

shopt -s nullglob
wheels=("$DIST_DIR"/mlx_omarchy-*.whl)
shopt -u nullglob
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "expected exactly one mlx_omarchy wheel in $DIST_DIR, found ${#wheels[@]}" >&2
  exit 1
fi
wheel="${wheels[0]}"

echo "== receipt =="
echo "[receipt] wheel: $wheel"
echo "[receipt] size: $(stat -c '%s' "$wheel") bytes"
echo "[receipt] sha256: $(sha256sum "$wheel" | cut -d' ' -f1)"

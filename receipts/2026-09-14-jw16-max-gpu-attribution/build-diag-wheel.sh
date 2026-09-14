#!/usr/bin/env bash
# Build a profiling-enabled (diag) mlx-omarchy wheel from the pinned b41e2b74
# worktree on jw16, into a private work dir and private dist dir. Never
# touches ~/src/mlx-omarchy .work or dist.
set -euo pipefail
ROOT=/var/tmp/mlx-omarchy-prof-b41e2b74
export MLX_OMARCHY_WORK_DIR="$ROOT/.work"
VENV="$MLX_OMARCHY_WORK_DIR/venv-build"
DIST="$ROOT/dist-diag"

echo "== prepare =="
"$ROOT/scripts/prepare-mlx.sh"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi
py="$VENV/bin/python"
"$py" - <<'PY'
import setuptools, typing_extensions
assert int(setuptools.__version__.split(".")[0]) >= 80
print("build deps ok", setuptools.__version__)
PY

rm -rf "$DIST"; mkdir -p "$DIST"
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DMLX_OMARCHY_GPU_PROFILING=ON -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas -DFETCHCONTENT_SOURCE_DIR_JSON=/var/tmp/mlx-omarchy-prof-b41e2b74/.deps-seed/json-src -DFETCHCONTENT_SOURCE_DIR_GGUFLIB=/var/tmp/mlx-omarchy-prof-b41e2b74/.deps-seed/gguflib-src -DFETCHCONTENT_SOURCE_DIR_FMT=/var/tmp/mlx-omarchy-prof-b41e2b74/.deps-seed/fmt-src -DFETCHCONTENT_SOURCE_DIR_NANOBIND=/var/tmp/mlx-omarchy-prof-b41e2b74/.deps-seed/nanobind-src"
export CMAKE_BUILD_PARALLEL_LEVEL=10
export PATH="$VENV/bin:$PATH"
echo "== wheel =="
"$py" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$DIST" "$MLX_OMARCHY_WORK_DIR/mlx"
w=$(echo "$DIST"/mlx_omarchy-*.whl)
echo "[receipt] wheel: $w"
sha256sum "$w"

#!/usr/bin/env bash
# Build a release (profiling OFF) mlx-omarchy wheel from the
# /var/tmp/qmm-tilem worktree on jw16, into a private work dir and
# private dist dir. Never touches ~/src/mlx-omarchy .work or dist.
#
# The two deviations from scripts/build-wheel.sh are the ones
# receipts/2026-09-14-jw16-max-gpu-attribution.md already documented for
# this host: Arch puts cblas.h under /usr/include/openblas, and
# prepare-mlx.sh re-runs FetchContent on every build so the downloads are
# served from a local seed. Neither reaches overlay/mlx/backend/omarchy/.
set -euo pipefail
ROOT=/var/tmp/qmm-tilem
export MLX_OMARCHY_WORK_DIR="$ROOT/.work"
VENV="$MLX_OMARCHY_WORK_DIR/venv-build"
DIST="$ROOT/dist-ab"
SEED="$ROOT/.deps-seed"

echo "== identity =="
git -C "$ROOT" log --oneline -1
git -C "$ROOT" status --short --untracked-files=no

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
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas -DFETCHCONTENT_SOURCE_DIR_JSON=$SEED/json-src -DFETCHCONTENT_SOURCE_DIR_GGUFLIB=$SEED/gguflib-src -DFETCHCONTENT_SOURCE_DIR_FMT=$SEED/fmt-src -DFETCHCONTENT_SOURCE_DIR_NANOBIND=$SEED/nanobind-src"
export CMAKE_BUILD_PARALLEL_LEVEL=10
export PATH="$VENV/bin:$PATH"
echo "== wheel =="
"$py" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$DIST" "$MLX_OMARCHY_WORK_DIR/mlx"
w=$(echo "$DIST"/mlx_omarchy-*.whl)
echo "[receipt] wheel: $w"
sha256sum "$w"
echo "== shader gate =="
# The three coopmat SPIR-V blobs must all be in the build, and the
# 32-row one must be byte-identical to the shipped shader compiled
# without -DTILE_ROWS.
sdir="$MLX_OMARCHY_WORK_DIR"/mlx/build/*/mlx.core/mlx/backend/omarchy/shaders
ls -l $sdir/qmm_coopmat*.spv || true
glslc -O --target-env=vulkan1.3 \
  "$ROOT/overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp" \
  -o /tmp/qmm-tilem-ref32.spv
sha256sum /tmp/qmm-tilem-ref32.spv $sdir/qmm_coopmat_f16.spv

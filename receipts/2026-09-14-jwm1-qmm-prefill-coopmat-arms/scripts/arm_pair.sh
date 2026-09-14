#!/usr/bin/env bash
# arm_pair.sh <tag> <shader-file> <primitives-file>
# Build a candidate wheel with both qmm_coopmat.comp and primitives.cpp
# replaced, in the 2026-09-13 profile-enabled b41e2b74 tree.
set -euo pipefail
BASE=/var/tmp/mlx-omarchy-profile-enabled-b41e2b74
SRC=/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74
WORK=/var/tmp/qmm-coop-arms
TAG=${1:?tag}
SHADER=${2:?shader file}
PRIM=${3:?primitives file}
PY="$SRC/.work/venv-build/bin/python"
OMA=$BASE/mlx/mlx/backend/omarchy
mkdir -p "$WORK/dist-$TAG"
cp "$SHADER" "$OMA/shaders/qmm_coopmat.comp"
cp "$PRIM" "$OMA/primitives.cpp"
sha256sum "$SHADER" "$PRIM"
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_OMARCHY_GPU_PROFILING=ON"
export CMAKE_BUILD_PARALLEL_LEVEL=4
export DEV_RELEASE=1
export MLX_OMARCHY_LOCAL_VERSION="diag.$TAG"
export MLX_OMARCHY_SOURCE_COMMIT=b41e2b74c330f910b24cab0e7516e306527858f0
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
echo "[$(date -Iseconds)] pip wheel $TAG"
"$PY" -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$WORK/dist-$TAG" "$BASE/mlx" > "$WORK/build-$TAG.log" 2>&1
echo "[$(date -Iseconds)] wheel done"
shopt -s nullglob
wheels=("$WORK/dist-$TAG"/mlx_omarchy-*.whl)
echo "wheel_count=${#wheels[@]}"
for w in "${wheels[@]}"; do
  echo "[receipt] wheel: $w"
  sha256sum "$w"
done

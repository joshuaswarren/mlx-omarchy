#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
test "$(git rev-parse --short=7 HEAD)" = fb63766
git apply -p3 /tmp/rms-reduction.patch
git add overlay/mlx/backend/omarchy/shaders/fast_norm.comp
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: match subgroup-first RMS reduction'
mkdir -p /tmp/mlx-fused-sdpa-wheel
mv dist/mlx_omarchy-*.whl /tmp/mlx-fused-sdpa-wheel/
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-rms-reduction-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}" > /tmp/mlx-rms-reduction-install.log 2>&1
.venv-accept/bin/python /tmp/mlx-long-operation-oracle.py /tmp/mlx-rms-reduction-long-operations
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-rms-reduction-matrix

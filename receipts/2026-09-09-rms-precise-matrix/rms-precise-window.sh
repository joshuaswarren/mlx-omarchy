#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
test "$(git rev-parse --short=7 HEAD)" = da9e7f6
cp /tmp/rms-precise.comp overlay/mlx/backend/omarchy/shaders/fast_norm.comp
git add overlay/mlx/backend/omarchy/shaders/fast_norm.comp
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: refine RMS reciprocal square root'
mkdir -p /tmp/mlx-rms-precise-wheel
mv dist/mlx_omarchy-*.whl /tmp/mlx-rms-precise-wheel/
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-rms-precise-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}" > /tmp/mlx-rms-precise-install.log 2>&1
.venv-accept/bin/python /tmp/mlx-fixed-rms.py /tmp/mlx-native-long-norm /tmp/mlx-precise-fixed-rms
.venv-accept/bin/python /tmp/mlx-long-operation-oracle.py /tmp/mlx-rms-precise-long-operations
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-rms-precise-matrix
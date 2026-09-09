#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
test "$(git rev-parse --short=7 HEAD)" = 01c5ef3
cp /tmp/rms-exact.comp overlay/mlx/backend/omarchy/shaders/fast_norm.comp
git add overlay/mlx/backend/omarchy/shaders/fast_norm.comp
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: preserve RMS reciprocal correction arithmetic'
mkdir -p /tmp/mlx-rms-exact-wheel
mv dist/mlx_omarchy-*.whl /tmp/mlx-rms-exact-wheel/
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-rms-exact-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}"
.venv-accept/bin/python /tmp/mlx-fixed-q4-rms.py /tmp/mlx-native-q4-long-operations /tmp/mlx-exact-fixed-q4-rms
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-rms-exact-matrix

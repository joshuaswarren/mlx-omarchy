#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
test "$(git rev-parse --short=7 HEAD)" = 2845dbc
cp /tmp/q4-reduce.comp overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp
git add overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: match native half-precision split reduction'
mkdir -p /tmp/mlx-q4-reduce-wheel
mv dist/mlx_omarchy-*.whl /tmp/mlx-q4-reduce-wheel/
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-q4-reduce-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}" > /tmp/mlx-q4-reduce-install.log 2>&1
.venv-accept/bin/python /tmp/mlx-fixed-q4-projection.py /tmp/mlx-native-q4-long-operations /tmp/mlx-reduce-fixed-q4
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-q4-reduce-matrix

#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
test "$(git rev-parse --short=7 HEAD)" = 8fd73d0
mkdir -p /tmp/mlx-fma-diagnostic-wheel
mv dist/mlx_omarchy-*.whl /tmp/mlx-fma-diagnostic-wheel/
restore() {
    .venv-accept/bin/python -m pip install --no-deps --force-reinstall /tmp/mlx-fma-diagnostic-wheel/mlx_omarchy-0.32.2.dev202609090935+8fd73d0-cp314-cp314-linux_aarch64.whl
    git show 8fd73d0:overlay/mlx/backend/omarchy/shaders/fast_norm.comp > overlay/mlx/backend/omarchy/shaders/fast_norm.comp
    git add overlay/mlx/backend/omarchy/shaders/fast_norm.comp
    git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: remove RMS diagnostic instrumentation'
}
trap restore EXIT
cp /tmp/fma-diagnostic.comp overlay/mlx/backend/omarchy/shaders/fast_norm.comp
git add overlay/mlx/backend/omarchy/shaders/fast_norm.comp
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: capture RMS intermediate values only'
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-fma-diagnostic-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}"
.venv-accept/bin/python /tmp/mlx-fixed-q4-rms.py /tmp/mlx-native-q4-long-operations /tmp/mlx-fma-fixed-q4-rms
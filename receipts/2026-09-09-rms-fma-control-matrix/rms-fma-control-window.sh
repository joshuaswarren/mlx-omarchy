#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
test "$(git rev-parse --short=7 HEAD)" = b5578df
restore() {
    .venv-accept/bin/python -m pip install --no-deps --force-reinstall /tmp/mlx-fma-diagnostic-wheel/mlx_omarchy-0.32.2.dev202609090935+8fd73d0-cp314-cp314-linux_aarch64.whl
    git show b5578df:overlay/mlx/backend/omarchy/embed_spirv.cmake > overlay/mlx/backend/omarchy/embed_spirv.cmake
    git rm overlay/mlx/backend/omarchy/preserve-rms-fma.py
    git add overlay/mlx/backend/omarchy/embed_spirv.cmake
    git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: remove FMA contraction control'
}
trap restore EXIT
cp /tmp/rms-live-embed.cmake overlay/mlx/backend/omarchy/embed_spirv.cmake
cp /tmp/preserve-rms-fma.py overlay/mlx/backend/omarchy/preserve-rms-fma.py
git add overlay/mlx/backend/omarchy/embed_spirv.cmake overlay/mlx/backend/omarchy/preserve-rms-fma.py
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: constrain forward RMS FMA contraction in SPIR-V'
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-rms-fma-control-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}"
AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true .venv-accept/bin/python /tmp/mlx-rms-nir-capture.py > /tmp/mlx-rms-fma-control-nir.log 2>&1
.venv-accept/bin/python /tmp/mlx-fixed-q4-rms.py /tmp/mlx-native-q4-long-operations /tmp/mlx-fma-control-fixed-q4-rms
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-rms-fma-control-matrix

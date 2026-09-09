#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
test "$(git rev-parse --short=7 HEAD)" = 08486e1
restore() {
    .venv-accept/bin/python -m pip install --no-deps --force-reinstall /tmp/mlx-fma-diagnostic-wheel/mlx_omarchy-0.32.2.dev202609090935+8fd73d0-cp314-cp314-linux_aarch64.whl
    for file in embed_spirv.cmake primitives.cpp shaders/fast_rope.comp; do
        git show "08486e1:overlay/mlx/backend/omarchy/$file" > "overlay/mlx/backend/omarchy/$file"
    done
    git rm overlay/mlx/backend/omarchy/preserve-rms-fma.py
    git add overlay/mlx/backend/omarchy
    git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: remove native rotary arithmetic control'
}
trap restore EXIT
cp /tmp/rms-live-embed.cmake overlay/mlx/backend/omarchy/embed_spirv.cmake
cp /tmp/preserve-rms-fma.py overlay/mlx/backend/omarchy/preserve-rms-fma.py
cp /tmp/live-rope-primitives.cpp overlay/mlx/backend/omarchy/primitives.cpp
cp /tmp/live-fast-rope.comp overlay/mlx/backend/omarchy/shaders/fast_rope.comp
git add overlay/mlx/backend/omarchy
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: match native rotary frequency and fused rotation'
mkdir /tmp/mlx-rope-control-old-dist
mv dist/mlx_omarchy-*.whl /tmp/mlx-rope-control-old-dist/
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-rope-native-control-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}"
.venv-accept/bin/python /tmp/mlx-rope-basis.py /tmp/mlx-corrected-rope-basis
.venv-accept/bin/python /tmp/mlx-q4-long-operation-oracle.py /tmp/mlx-rope-q4-long-operations
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-rope-native-control-matrix

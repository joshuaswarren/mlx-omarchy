#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
test "$(git rev-parse --short=7 HEAD)" = d535a25
restore() {
    .venv-accept/bin/python -m pip install --no-deps --force-reinstall /tmp/mlx-fma-diagnostic-wheel/mlx_omarchy-0.32.2.dev202609090935+8fd73d0-cp314-cp314-linux_aarch64.whl
    git restore --source=d535a25 --staged --worktree overlay
    git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: remove native prefill control after acceptance window'
}
trap restore EXIT
tar -xf /tmp/mlx-native-prefill-overlay.tar
 git add overlay
 git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: native-order cooperative prefill attention'
mkdir /tmp/mlx-native-prefill-old-dist
mv dist/mlx_omarchy-*.whl /tmp/mlx-native-prefill-old-dist/
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-native-prefill-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}"
export VK_DRIVER_FILES=/tmp/mesa-native-trig-icd.json
export MESA_SHADER_CACHE_DISABLE=true
.venv-accept/bin/python /tmp/mlx-fixed-attention.py /tmp/mlx-native-q4-long-operations /tmp/mlx-native-prefill-fixed-attention
.venv-accept/bin/python /tmp/assert-native-prefill.py
.venv-accept/bin/python /tmp/mlx-q4-long-operation-oracle.py /tmp/mlx-native-prefill-q4-operations
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-native-prefill-matrix

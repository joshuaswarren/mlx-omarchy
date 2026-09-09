#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
git switch --detach e646882
git apply -p3 /tmp/sdpa-decode.patch
git add overlay
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: fuse native-style single-query attention'
mkdir -p /tmp/mlx-rope-exp2-wheel
mv dist/mlx_omarchy-*.whl /tmp/mlx-rope-exp2-wheel/
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-fused-sdpa-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}" > /tmp/mlx-fused-sdpa-install.log 2>&1
.venv-accept/bin/python /tmp/mlx-decode-attention-check.py --expect-fused
.venv-accept/bin/python /tmp/mlx-fixed-attention.py /tmp/mlx-native-attention-oracle /tmp/mlx-fused-fixed-attention
.venv-accept/bin/python /tmp/mlx-gemv-coefficients.py /tmp/mlx-corrected-gemv-coefficients
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-fused-sdpa-matrix

#!/usr/bin/env bash
set -euo pipefail
base="$HOME/src/mlx-wide-gemv-screen"
root="$HOME/src/mlx-gemv-unroll-screen"
test ! -e "$root"
git clone --quiet --shared "$base" "$root"
cd "$root"
git apply /tmp/gemv-unroll.patch
git add overlay
git -c user.name='Performance experiment' -c user.email='experiment@localhost' commit -m 'experiment: four-step sequential accumulation for BF16 GEMV'
unset HK_PERF MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/gemv-unroll-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps -r "$base/receipts/2026-09-08-dense-final/requirements-baseline.txt" "${wheels[0]}" mlx-lm==0.31.3 > /tmp/gemv-unroll-install.log 2>&1
.venv-accept/bin/python /tmp/gemv-unroll-pairs.py "$root"

#!/usr/bin/env bash
set -euo pipefail
root="$HOME/src/mlx-rms-round-screen"
test ! -e "$root"
git clone --quiet https://github.com/joshuaswarren/mlx-omarchy.git "$root"
cd "$root"
git checkout --detach 83956814
git apply /tmp/mlx-rms-round.patch
git add overlay/mlx/backend/omarchy/shaders/fast_norm.comp
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'experiment: match native RMSNorm intermediate storage rounding'
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-rms-round-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps -r "$HOME/src/mlx-wide-gemv-screen/receipts/2026-09-08-dense-final/requirements-baseline.txt" "${wheels[0]}" mlx-lm==0.31.3 > /tmp/mlx-rms-round-install.log 2>&1
.venv-accept/bin/python /tmp/mlx-operation-oracle.py /tmp/mlx-rms-round-operation-oracle

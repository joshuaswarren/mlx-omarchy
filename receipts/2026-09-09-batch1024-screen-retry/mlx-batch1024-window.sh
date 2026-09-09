#!/usr/bin/env bash
set -euo pipefail
root="$HOME/src/mlx-batch1024-screen"
test -d "$root/.git"
cd "$root"
git checkout --detach 3106bdd8
git apply --unidiff-zero /tmp/mlx-batch1024.patch
git add overlay/mlx/backend/omarchy/encoder.h
git -c user.name='Performance experiment' -c user.email='experiment@localhost' commit -m 'experiment: measure 1024-node batches with existing byte cap'
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-batch1024-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps -r "$HOME/src/mlx-wide-gemv-screen/receipts/2026-09-08-dense-final/requirements-baseline.txt" "${wheels[0]}" mlx-lm==0.31.3 > /tmp/mlx-batch1024-install.log 2>&1
.venv-accept/bin/python /tmp/mlx-batch1024-pairs.py "$root"

#!/usr/bin/env bash
set -euo pipefail
base="$HOME/src/mlx-gemv-native-all-screen"
root="$HOME/src/mlx-gemv-native-guard-fresh"
test ! -e "$root"
git clone --quiet https://github.com/joshuaswarren/mlx-omarchy.git "$root"
cd "$root"
git apply /tmp/gemv-native.patch
git apply --unidiff-zero /tmp/gemv-native-all.patch
git apply --unidiff-zero /tmp/gemv-native-guard.patch
git add overlay
git -c user.name='Performance experiment' -c user.email='experiment@localhost' commit -m 'experiment: guard native GEMV subgroup requirements'
unset HK_PERF MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/gemv-native-guard-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps -r "$base/receipts/2026-09-08-dense-final/requirements-baseline.txt" "${wheels[0]}" mlx-lm==0.31.3 > /tmp/gemv-native-guard-install.log 2>&1
.venv-accept/bin/python /tmp/gemv-native-guard-pairs.py "$root"

#!/usr/bin/env bash
set -euo pipefail
base="$HOME/src/mlx-rope-drain-770ae465"
root="$HOME/src/mlx-timestamp-split-probe"
test ! -e "$root"
git clone --quiet --shared "$base" "$root"
cd "$root"
git apply /tmp/timestamp-split.patch
git add overlay/mlx/backend/omarchy/gpu_profiler.h
git -c user.name='Diagnostic experiment' -c user.email='diagnostic@localhost' commit -m 'experiment: isolate dispatch start timestamp with execution barrier'
unset HK_PERF MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE
CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh --diagnostics > /tmp/timestamp-split-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
python3.14 -m venv .venv-profile
.venv-profile/bin/python -m pip install --no-deps -r "$base/receipts/2026-09-08-dense-final/requirements-baseline.txt" "${wheels[0]}" > /tmp/timestamp-split-install.log 2>&1
MLX_OMARCHY_GPU_PROFILE=/tmp/dense-wall-split.jsonl .venv-profile/bin/python /tmp/dense-wall-probe.py /tmp/dense-wall-split.json
printf 'TIMESTAMP_SPLIT_PROBE_DONE\n'

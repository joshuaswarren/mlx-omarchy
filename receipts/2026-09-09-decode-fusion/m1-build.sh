#!/usr/bin/env bash
# Build one wheel of a checkout on the M1 and file it by kind.
#   m1-build.sh CHECKOUT release|diagnostics
# Release wheels stay in dist/ (paired-legs.py wants exactly one there);
# diagnostics wheels move to wheels/diag/ so the two never share dist/.
# Run under `flock -w 900 /tmp/m1-gpu.lock timeout 3600`.
set -euo pipefail
cd "$1"
kind="$2"
commit="$(git rev-parse --short=7 HEAD)"
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES
rm -rf .work/mlx/build .work/build-wheel 2>/dev/null || true
start=$(date +%s)
if [[ "$kind" == diagnostics ]]; then
  DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh --diagnostics > "build-$kind-$commit.log" 2>&1
  mkdir -p wheels/diag
  mv dist/mlx_omarchy-*.whl wheels/diag/
  wheel=(wheels/diag/mlx_omarchy-*+diag."$commit"-*.whl)
else
  DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > "build-$kind-$commit.log" 2>&1
  wheel=(dist/mlx_omarchy-*+"$commit"-*.whl)
fi
test "${#wheel[@]}" -eq 1
echo "BUILT $kind $commit $(( $(date +%s) - start ))s $(sha256sum "${wheel[0]}")"

#!/bin/sh
set -eu
ROOT=/home/joshuawarren/src/mlx-SubmissionGaps
BUILD=$ROOT/.work/build-submission-tests
OUT=${1:-/tmp/submission-gaps-tests}
mkdir -p "$OUT"

exec timeout 2400 flock -w 2400 /tmp/m1-gpu.lock /bin/sh -c '
  set -eu
  cd "'$ROOT'"
  scripts/prepare-mlx.sh
  cmake -S .work/mlx -B "'$BUILD'" \
    -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
  cmake --build "'$BUILD'" --target \
    omarchy_copy_offset_tests omarchy_runtime_tests omarchy_fast_ops_tests -j4

  run_driver() {
    label=$1
    driver=$2
    allow=$3
    export VK_DRIVER_FILES=$driver MLX_OMARCHY_ALLOW_NON_APPLE=$allow
    "'$BUILD'/tests/omarchy/omarchy_copy_offset_tests" > "'$OUT'/$label-copy.log" 2>&1
    "'$BUILD'/tests/omarchy/omarchy_runtime_tests" > "'$OUT'/$label-runtime.log" 2>&1
    "'$BUILD'/tests/omarchy/omarchy_fast_ops_tests" \
      --test-case="fused rope*" > "'$OUT'/$label-rope.log" 2>&1
  }

  run_driver llvmpipe /usr/share/vulkan/icd.d/lvp_icd.aarch64.json 1
  run_driver m1-fork /tmp/asahi_coopmat_icd.json 0
  run_driver m1-stock /home/joshuawarren/stock-mesa/stock-icd.json 0

  for log in "'$OUT'"/*.log; do
    printf "=== %s ===\n" "$(basename "$log")"
    tail -n 8 "$log"
  done
'
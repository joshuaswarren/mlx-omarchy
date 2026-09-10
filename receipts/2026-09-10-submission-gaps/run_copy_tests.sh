#!/bin/sh
set -eu
ROOT=/home/joshuawarren/src/mlx-SubmissionGaps
BUILD=$ROOT/.work/build-submission-tests
cd "$ROOT"
scripts/prepare-mlx.sh
cmake -S .work/mlx -B "$BUILD" \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build "$BUILD" --target omarchy_copy_offset_tests omarchy_runtime_tests -j4
VK_DRIVER_FILES=${1:-/usr/share/vulkan/icd.d/lvp_icd.aarch64.json} \
MLX_OMARCHY_ALLOW_NON_APPLE=${2:-1} \
  "$BUILD/tests/omarchy/omarchy_copy_offset_tests" \
  --test-case="back-to-back scalar fills stay ordered in one submission"

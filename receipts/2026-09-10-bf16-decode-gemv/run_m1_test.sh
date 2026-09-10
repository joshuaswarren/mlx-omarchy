#!/usr/bin/env bash
set -euo pipefail

scripts/prepare-mlx.sh
cmake -S .work/mlx -B .work/build-bf16-tests -G Ninja \
  -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=OFF \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build .work/build-bf16-tests -j4 --target omarchy_matmul_family_tests
ulimit -c 0
case_name="dense bf16 matmul matches host on every row across coopmat shapes"
timeout 600 .work/build-bf16-tests/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="$case_name" --out=receipt-bf16/m1-bf16-fork.log
VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
  timeout 600 .work/build-bf16-tests/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="$case_name" --out=receipt-bf16/m1-bf16-stock.log

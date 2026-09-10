#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
out=receipts/2026-09-10-decode-bound-3/tests
mkdir -p "$out"
cmake -S .work/mlx -B .work/build-decode-bound-3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF > "$out/m1-build.log" 2>&1
cmake --build .work/build-decode-bound-3 --target omarchy_runtime_tests -j4 >> "$out/m1-build.log" 2>&1

identity='import json,os; import mlx.core as mx; i=mx.device_info(); print(json.dumps({"cooperative_matrix_f32_8":i.get("cooperative_matrix_f32_8"),"device_name":i.get("device_name"),"mlx_version":mx.__version__,"VK_DRIVER_FILES":os.environ.get("VK_DRIVER_FILES")},sort_keys=True))'
filter='dependency-gated barriers keep hazard chains correct'

env -u VK_DRIVER_FILES .work/venv-release/bin/python -c "$identity" > "$out/m1-fork.log"
env -u VK_DRIVER_FILES .work/build-decode-bound-3/tests/omarchy/omarchy_runtime_tests --test-case="$filter" >> "$out/m1-fork.log" 2>&1
env -u VK_DRIVER_FILES MLX_OMARCHY_GATED_BARRIERS=1 .work/build-decode-bound-3/tests/omarchy/omarchy_runtime_tests --test-case="$filter" >> "$out/m1-fork.log" 2>&1

stock=/home/joshuawarren/stock-mesa/stock-icd.json
VK_DRIVER_FILES=$stock .work/venv-release/bin/python -c "$identity" > "$out/m1-stock.log"
VK_DRIVER_FILES=$stock .work/build-decode-bound-3/tests/omarchy/omarchy_runtime_tests --test-case="$filter" >> "$out/m1-stock.log" 2>&1
VK_DRIVER_FILES=$stock MLX_OMARCHY_GATED_BARRIERS=1 .work/build-decode-bound-3/tests/omarchy/omarchy_runtime_tests --test-case="$filter" >> "$out/m1-stock.log" 2>&1

cat "$out/m1-fork.log" "$out/m1-stock.log"

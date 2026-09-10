#!/usr/bin/env bash
# CPU-only prep on jwm1-linux. Run BEFORE taking the GPU lock; safe to
# overlap a GPU-bound neighbor block (niced, -j4).
set -euo pipefail
cd ~/src/mlx-requal
mkdir -p receipt-requal/wheels/candidate receipt-requal/logs
git rev-parse HEAD > receipt-requal/commit.txt

# 1. candidate wheel — plain stamped release wheel (DEV_RELEASE=1 stamps the
# commit; NO --diagnostics: the canonical baseline wheel b6d662a8 is a
# non-diagnostics build and the paired cells must share build mode)
nice -n 10 env DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  scripts/build-wheel.sh 2>&1 | tee receipt-requal/logs/build-wheel.log
cand_wheel=$(ls -t dist/mlx_omarchy-*.whl | head -1)
cp "$cand_wheel" receipt-requal/wheels/candidate/
sha256sum "$cand_wheel" | tee receipt-requal/logs/candidate-wheel.sha256

# 2. baseline wheel = committed canonical 12-matrix wheel (b6d662a8)
base_wheel="$HOME/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl"
echo "98821134f4bcf306ab1d64d6885ddf3d3e8e932311283bbd11f97aecf6a8e439  $base_wheel" | sha256sum -c -

# 3. venvs (fresh, one per wheel)
rm -rf .venv-requal-base .venv-requal-cand
python3.14 -m venv .venv-requal-base
.venv-requal-base/bin/pip install --quiet "$base_wheel" mlx-lm==0.31.3
python3.14 -m venv .venv-requal-cand
.venv-requal-cand/bin/pip install --quiet "$cand_wheel" mlx-lm==0.31.3

# 4. suite binaries (same flags as prior M1 receipts; CPU-only build)
cmake -S .work/mlx -B .work/build-requal -G Ninja \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  > receipt-requal/logs/cmake-suite.log 2>&1
nice -n 10 cmake --build .work/build-requal -j4 --target \
  omarchy_matmul_family_tests omarchy_runtime_tests mlx-omarchy-info \
  2>&1 | tail -3 | tee receipt-requal/logs/build-suite.log
echo BUILD-OK

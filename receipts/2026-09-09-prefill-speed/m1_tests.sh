#!/usr/bin/env bash
# Build and run the targeted omarchy C++ suites on the M1 for one commit
# (run under the GPU lock). Usage: m1_tests.sh <commit>
set -euo pipefail
COMMIT="$1"
ROOT="$HOME/src/mlx-PrefillSpeed"
cd "$ROOT"
git fetch -q origin wave/PrefillSpeed
git checkout -q "$COMMIT"
SHORT="$(git rev-parse --short=7 HEAD)"
OUT="receipts/2026-09-09-prefill-speed/screens/$SHORT/tests"
mkdir -p "$OUT"
hostname; date -u +%FT%TZ; echo "commit $SHORT"
scripts/prepare-mlx.sh > "$OUT/prepare.log" 2>&1
cmake -S .work/mlx -B .work/build-tests -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_BUILD_TYPE=Release \
  > "$OUT/configure.log" 2>&1
timeout 3000 cmake --build .work/build-tests -j4 --target omarchy_matmul_family_tests \
  omarchy_primitive_tests omarchy_fused_chain_tests omarchy_fast_ops_tests > "$OUT/build.log" 2>&1
run() {
  local bin="$1"; shift
  echo "== $bin $*"
  timeout 1500 ".work/build-tests/tests/omarchy/$bin" "$@" > "$OUT/$bin.log" 2>&1 || true
  tail -3 "$OUT/$bin.log"
}
run omarchy_matmul_family_tests -tc="register-blocked*,qmm coopmat*,qmm tile*,dense*,qmm_vec*"
run omarchy_primitive_tests -tc="four-wide*,f16 causal*,*attention*,*softmax*,*Matmul*,*matmul*,*elementwise*,*broadcast*"
run omarchy_fused_chain_tests
run omarchy_fast_ops_tests
date -u +%FT%TZ

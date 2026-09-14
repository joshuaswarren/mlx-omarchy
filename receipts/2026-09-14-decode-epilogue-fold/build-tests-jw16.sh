#!/usr/bin/env bash
# Build the two C++ batteries that gate the fold (fused_chain, kv_ops)
# from the cand checkout on jw16 with its glslc, in a build dir separate
# from the wheel's cmake cache, so the bit-exact assertions run on the
# real Honeykrisp compiler on both laptops.
set -euo pipefail
ROOT=/var/tmp/DecodeEpilogueFold
SEED=/var/tmp/qmm-tilem/.deps-seed
TREE="$ROOT/cand"
WORK="$TREE/.work"
BUILD="$ROOT/build-tests"
git -C "$TREE" log --oneline -1
rm -rf "$BUILD"
cmake -S "$WORK/mlx" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DBLAS_INCLUDE_DIRS=/usr/include/openblas \
  -DLAPACK_INCLUDE_DIRS=/usr/include/openblas \
  -DFETCHCONTENT_SOURCE_DIR_JSON=$SEED/json-src \
  -DFETCHCONTENT_SOURCE_DIR_GGUFLIB=$SEED/gguflib-src \
  -DFETCHCONTENT_SOURCE_DIR_FMT=$SEED/fmt-src \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND=$SEED/nanobind-src >/dev/null
ninja -C "$BUILD" -j "${JOBS:-8}" omarchy_fused_chain_tests omarchy_kv_ops_tests 2>&1 | grep -E "error|FAILED" || true
ls -l "$BUILD"/tests/omarchy/omarchy_fused_chain_tests "$BUILD"/tests/omarchy/omarchy_kv_ops_tests
sha256sum "$BUILD"/tests/omarchy/omarchy_fused_chain_tests "$BUILD"/tests/omarchy/omarchy_kv_ops_tests

#!/usr/bin/env bash
# Build + run the subgroup-vs-tree microbenchmark on jwm1.
#
# Reads PROTOCOL.md for the full rationale. This script exists so the
# BenchQueueM1 liveness procedure gets one concrete command and one
# concrete log path.
set -eu

# Always run from the worktree root so the relative shader paths
# resolve. The bench prints stderr from glslc/glslangValidator if
# either step fails; do not silence it.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.." || exit 1
WORKTREE="$(pwd)"

# 1. Ensure the staged MLX is fresh; the overlay tree holds the kernels
#    we measure, not the staged ones. (Skipped if .work/mlx already
#    present from the prior session.)
if [[ ! -d .work/mlx ]]; then
  bash scripts/prepare-mlx.sh
fi

# 2. Build the harness. Pinned to glslc compilation for the shaders;
#    the program will fall back to glslangValidator if glslc is not
#    available, but the receipt requires glslc per repo hard rule.
g++ -std=c++17 -O2 -o /tmp/subgroup-bench tools/subgroup-bench/bench.cpp

# 3. Run. Both modes are quick — the program exits with the verdict
#    on stdout as NDJSON. We tee to a log so the receipt can cite
#    exact bytes.
LOG=/tmp/subgroup-bench.log
{
  echo "=== bench ==="
  /tmp/subgroup-bench
  echo "=== bench --quick ==="
  /tmp/subgroup-bench --quick
} 2>&1 | tee "$LOG"

echo "BENCH_DONE log=$LOG"

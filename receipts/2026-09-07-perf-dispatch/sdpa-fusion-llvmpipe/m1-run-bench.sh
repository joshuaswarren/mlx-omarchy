#!/usr/bin/env bash
# Run the focused test then the microbench on the M1 under the GPU lock.
# Usage: m1-run-bench.sh <label> [qlen]
set -uo pipefail
LABEL="${1:-run}"; QLEN="${2:-1}"
H=joshuawarren@100.84.184.102
R=/home/joshuawarren/benchq/Fusion2
ssh -o BatchMode=yes $H "mkdir -p $R/logs && cd $R/build && flock /home/joshuawarren/benchq/gpu.lock timeout 600 ./tests/omarchy/omarchy_sdpa_decode_tests > $R/logs/test-$LABEL.log 2>&1; echo test_rc=\$?; tail -22 $R/logs/test-$LABEL.log; MLX_OMARCHY_SDPA_BENCH_QLEN=$QLEN MLX_OMARCHY_SDPA_BENCH_REPS=48 flock /home/joshuawarren/benchq/gpu.lock timeout 900 ./tests/omarchy/omarchy_sdpa_decode_bench 45 280 390 1024 4096 > $R/logs/bench-$LABEL.jsonl 2>&1; echo bench_rc=\$?; cat $R/logs/bench-$LABEL.jsonl"

#!/usr/bin/env bash
# Repeat the short-context bench several times for variance. Usage: m1-bench-short.sh <label>
set -uo pipefail
LABEL="${1:-short}"
H=joshuawarren@100.84.184.102
R=/home/joshuawarren/benchq/Fusion2
ssh -o BatchMode=yes $H "cd $R/build && for i in 1 2 3; do MLX_OMARCHY_SDPA_BENCH_REPS=200 flock /home/joshuawarren/benchq/gpu.lock timeout 600 ./tests/omarchy/omarchy_sdpa_decode_bench 45 128; done > $R/logs/bench-$LABEL.jsonl 2>&1; echo rc=\$?; cat $R/logs/bench-$LABEL.jsonl"

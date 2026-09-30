#!/bin/bash
# Jw16LevelBatch W1: level-batch projection microbench. Runs INSIDE a
# gpuwin window (it must not touch the mutex or the service itself):
#   bash /var/tmp/appbar/gpuwin.sh 'bash /var/tmp/lb1/run.sh'
set -euo pipefail
OUT=/var/tmp/lb1
mkdir -p "$OUT"
cd "$OUT"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
echo "host=$(hostname) stamp=$stamp boot_id=$(cat /proc/sys/kernel/random/boot_id)"
g++ -std=c++17 -O2 -o "$OUT/level-batch-bench" "$OUT/bench.cpp" -ldl
sha256sum "$OUT/bench.cpp" "$OUT/level-batch-bench"
"$OUT/level-batch-bench" 2>&1 | tee "$OUT/w1-$stamp.ndjson"
echo "W1-DONE stamp=$stamp"

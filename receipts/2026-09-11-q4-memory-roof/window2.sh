#!/usr/bin/env bash
# Window 2 on jwm1 for the Q4 memory-roof receipt: corrected-wall roof legs
# (valid copy peak after the 65536-grid fix) plus the wall-anchored
# candidate screen (base vs unroll/loadfirst/wg128 in the widedep
# environment). ONE top-level flock acquisition (fd 9); nothing nested;
# no driver swap; no production source change; never merges.
# Usage: bash receipts/2026-09-11-q4-memory-roof/window2.sh <bench-dir>
#   <bench-dir>: checkout root holding tools/q4-bw-bench (isolated clone).
set -uo pipefail
BENCH_DIR="${1:?usage: window2.sh <bench-dir>}"
R="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCH_DIR"
mkdir -p "$R"

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock"
flock -w 900 9 || exit 97
echo "$(date -Is) lock acquired"

commit="$(git rev-parse --short=7 HEAD)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
fail=0
echo "window2 commit=$commit host=$(hostname) stamp=$stamp"

echo "== phase 1: roof legs (corrected peaks) =="
bash tools/q4-bw-bench/run-roof-m1.sh > "$R/window2-roof.log" 2>&1 || fail=1
tail -3 "$R/window2-roof.log"

echo "== phase 2: gap candidate screen, two legs =="
until [ "$(python3 -c 'import os;print(int(os.getloadavg()[0]*100))')" -lt 100 ]; do sleep 5; done
g++ -std=c++17 -O2 -o /tmp/q4-bw-w2 tools/q4-bw-bench/bench.cpp || fail=1
/tmp/q4-bw-w2 --gap 2>&1 | tee "$R/window2-gap-a-$stamp.ndjson" || fail=1
sleep 10
/tmp/q4-bw-w2 --gap 2>&1 | tee "$R/window2-gap-b-$stamp.ndjson" || fail=1

echo "== phase 3: AGX shader dumps (register cost per side) =="
AGX_MESA_DEBUG=shaders /tmp/q4-bw-w2 --quick > /dev/null 2> "$R/window2-agx-dump.log" || true
grep -icE "register" "$R/window2-agx-dump.log" || true

echo "$(date -Is) window2 complete fail=$fail"
exit "$fail"

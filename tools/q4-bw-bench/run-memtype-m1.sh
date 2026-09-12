#!/usr/bin/env bash
# Memory-type A/B on the Q4 roof instrument (M1 / honeykrisp).
#
# Question: does the memory type a buffer is allocated from change
# achieved read bandwidth? Honeykrisp on the M1 exposes ONE heap
# (DEVICE_LOCAL, unified) and TWO types, both
# DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT (type 0 adds HOST_CACHED).
# There is no non-host-visible DEVICE_LOCAL type, so the arms are:
#   memtype 0 = what the production allocator picks today
#               (HOST_CACHED variant),
#   memtype 1 = the same heap without HOST_CACHED,
# plus one default-allocator leg proving the allocator pick == type 0.
#
# Run under ONE top-level flock:
#   flock -w 7200 /tmp/m1-gpu.lock timeout 1800 \
#     bash tools/q4-bw-bench/run-memtype-m1.sh
# The binary must already be built outside the lock:
#   g++ -std=c++17 -O2 -o /tmp/q4-bw-memtype tools/q4-bw-bench/bench.cpp
#
# Wall-anchored timing only; no production source change; interleaved
# a/b legs inside one window to beat the ~5% session noise floor.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/q4-bw-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || cat tools/q4-bw-bench/out/TREECOMMIT 2>/dev/null || echo not-git)"
host="$(hostname)"
log="tools/q4-bw-bench/out/memtype-session-$stamp.txt"
echo "host=$host commit=$commit stamp=$stamp" | tee "$log"
echo "-- binary provenance --" | tee -a "$log"
sha256sum /tmp/q4-bw-memtype | tee -a "$log"
sha256sum tools/q4-bw-bench/shaders/*.comp | tee -a "$log"
# The frozen base copy must equal the production shader (same anchor as
# the roof receipt).
diff -q tools/q4-bw-bench/shaders/qmm_vec_base.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
    | tee -a "$log" || {
  echo "FATAL: qmm_vec_base.comp drifted from the production shader" \
      | tee -a "$log"
  exit 1
}

quiet_gate() {
  until [ "$(python3 -c 'import os;print(int(os.getloadavg()[0]*100))')" -lt 100 ]; do
    sleep 5
  done
}

# Sample loadavg every 10 s in the background for contamination forensics.
(
  while true; do
    echo "$(date -u +%H:%M:%SZ) load1=$(cut -d' ' -f1 /proc/loadavg)" \
        >> "tools/q4-bw-bench/out/memtype-loadavg-$stamp.log"
    sleep 10
  done
) &
loader_pid=$!
trap 'kill $loader_pid 2>/dev/null || true' EXIT

# Sanity leg: the allocator's own pick (no override) must equal memtype 0.
quiet_gate
/tmp/q4-bw-memtype --roof --dumpmem 2>&1 \
    | tee "tools/q4-bw-bench/out/memtype-def-$stamp.ndjson"
sleep 10
# Interleaved A/B, three reps each side.
for rep in 1 2 3; do
  quiet_gate
  /tmp/q4-bw-memtype --roof --dumpmem --memtype 0 2>&1 \
      | tee "tools/q4-bw-bench/out/memtype-a$rep-$stamp.ndjson"
  sleep 10
  quiet_gate
  /tmp/q4-bw-memtype --roof --dumpmem --memtype 1 2>&1 \
      | tee "tools/q4-bw-bench/out/memtype-b$rep-$stamp.ndjson"
  [ "$rep" = 3 ] || sleep 10
done
echo "RUN_MEMTYPE_M1_DONE" | tee -a "$log"

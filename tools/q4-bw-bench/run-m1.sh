#!/usr/bin/env bash
# Q4 decode GEMV bench, M1 window. Run under
#   flock -w 2400 /tmp/m1-gpu.lock timeout 1800 bash tools/q4-bw-bench/run-m1.sh
# from the checkout root. Produces NDJSON logs under tools/q4-bw-bench/out/.
#
# Sides: base = frozen PRE-change production shader (commit ccc25c0f,
# sha256 pinned below); nativemap = the native qmv thread mapping that
# must equal the production shader in this checkout; nm32 = the same
# mapping packed four tiles per 256-thread workgroup (occupancy axis).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/q4-bw-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || echo not-git)"
host="$(hostname)"

echo "host=$host commit=$commit stamp=$stamp" | tee "tools/q4-bw-bench/out/session-$stamp.txt"
echo "-- shader provenance --" | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"
sha256sum tools/q4-bw-bench/shaders/*.comp | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"
# base is the pinned pre-change kernel; nativemap must equal the
# production shader so the measurement is about the shipped kernel.
echo "56aff1d36697331982b88eedbf1bcce74fee2a2e3e683c1a7edb12b14abc2a65  tools/q4-bw-bench/shaders/qmm_vec_base.comp" \
  | sha256sum -c - | tee -a "tools/q4-bw-bench/out/session-$stamp.txt" || {
  echo "FATAL: qmm_vec_base.comp is not the pinned pre-change shader" \
      | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"
  exit 1
}
diff -q tools/q4-bw-bench/shaders/qmm_vec_cand_nativemap.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
    | tee -a "tools/q4-bw-bench/out/session-$stamp.txt" || {
  echo "FATAL: qmm_vec_cand_nativemap.comp drifted from the production shader" \
      | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"
  exit 1
}

g++ -std=c++17 -O2 -o /tmp/q4-bw-bench tools/q4-bw-bench/bench.cpp

# Two independent legs for repeatability; each leg bit-compares
# nativemap and nm32 against base on identical input buffers, then times
# base/nativemap/nm32 interleaved per rep. --occ adds the grid-scaling
# occupancy sweep (starvation test) on padded buffers.
/tmp/q4-bw-bench --occ 2>&1 | tee "tools/q4-bw-bench/out/bench-a-$stamp.ndjson"
/tmp/q4-bw-bench --occ --quick 2>&1 | tee "tools/q4-bw-bench/out/bench-b-$stamp.ndjson"

echo "RUN_M1_DONE" | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"

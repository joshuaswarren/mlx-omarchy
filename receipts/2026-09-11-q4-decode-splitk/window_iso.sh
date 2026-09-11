#!/usr/bin/env bash
# Iso-bench GPU window for the Q4 split-K receipt: the production-shader
# bandwidth bench (default mode: per-shape isolated + bit/ulp compare;
# gap mode: in-model interleaving arms) on frozen copies of the current
# production shader. ONE top-level flock; quiet-gated; driver verified
# in-window. Run on jwm1 from the worktree root:
#   bash receipts/2026-09-11-q4-decode-splitk/window_iso.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="receipts/2026-09-11-q4-decode-splitk/m1-logs"
mkdir -p "$OUT"
echo "== iso window start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
la1=$(cut -d' ' -f1 /proc/loadavg)
echo "loadavg before window: $la1"
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 1.0 else 1)" "$la1"; then
  echo "QUIET_GATE_FAILED loadavg=$la1"
  exit 2
fi
exec 9>/tmp/m1-gpu.lock
if ! flock -w 600 9; then
  echo "FLOCK_TIMEOUT after 600s"
  exit 3
fi
echo "flock acquired $(date -u +%H:%M:%SZ)"
pacman -Q mesa-honeykrisp-omarchy linux-asahi || true
commit="$(git rev-parse --short=7 HEAD)"
echo "commit=$commit"
echo "-- shader provenance --"
sha256sum tools/q4-bw-bench/shaders/qmm_vec_base.comp \
    tools/q4-bw-bench/shaders/qmm_vec_splitk.comp \
    tools/q4-bw-bench/shaders/qmm_vec_splitk_reduce.comp
diff -q tools/q4-bw-bench/shaders/qmm_vec_base.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
  || { echo "FATAL: frozen base copy drifted"; exit 1; }
diff -q tools/q4-bw-bench/shaders/qmm_vec_splitk.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
  || { echo "FATAL: splitk copy drifted"; exit 1; }
g++ -std=c++17 -O2 -o /tmp/q4-bw-sk tools/q4-bw-bench/bench.cpp
until [ "$(python3 -c 'import os;print(int(os.getloadavg()[0]*100))')" -lt 100 ]; do sleep 5; done
echo "== default mode leg A =="
/tmp/q4-bw-sk 2>&1 | tee "$OUT/iso-a-$stamp.ndjson"
sleep 5
echo "== default mode leg B =="
/tmp/q4-bw-sk 2>&1 | tee "$OUT/iso-b-$stamp.ndjson"
sleep 5
echo "== gap mode leg A =="
/tmp/q4-bw-sk --gap 2>&1 | tee "$OUT/gap-a-$stamp.ndjson"
sleep 5
echo "== gap mode leg B =="
/tmp/q4-bw-sk --gap 2>&1 | tee "$OUT/gap-b-$stamp.ndjson"
echo "flock release $(date -u +%H:%M:%SZ)"
flock -u 9
echo "== iso window end $(date -u +%Y-%m-%dT%H:%M:%SZ) =="

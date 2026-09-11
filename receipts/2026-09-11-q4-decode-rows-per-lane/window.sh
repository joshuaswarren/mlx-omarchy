#!/usr/bin/env bash
# Rows-per-lane screen window on jwm1 (receipts/2026-09-11-q4-decode-rows-per-lane).
# ONE top-level flock acquisition (fd 9); nothing nested; no driver swap;
# no production source change in this window; never merges.
# Usage: bash receipts/2026-09-11-q4-decode-rows-per-lane/window.sh
set -uo pipefail
R="$(cd "$(dirname "$0")" && pwd)"
cd ~/src/mlx-Q4RowsPerLane
mkdir -p "$R"
exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock"
flock -w 900 9 || exit 97
echo "$(date -Is) lock acquired"

commit="$(git rev-parse --short=7 HEAD)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
fail=0
echo "window commit=$commit host=$(hostname) stamp=$stamp"

echo "== phase 0: provenance and quiet gate =="
pacman -Q mesa-honeykrisp-omarchy || fail=1
# The base copy must equal the production shader for the measurement to
# be about the shipped kernel.
diff -q tools/q4-bw-bench/shaders/qmm_vec_base.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp || {
  echo "FATAL: qmm_vec_base.comp drifted from the production shader"
  exit 1
}
until [ "$(python3 -c 'import os;print(int(os.getloadavg()[0]*100))')" -lt 100 ]; do
  sleep 5
done
g++ -std=c++17 -O2 -o /tmp/q4-bw-rpl tools/q4-bw-bench/bench.cpp || fail=1

echo "== phase 1: default (subgroup) eq + isolated, two legs =="
# Bit-identity on the M1 prints first in each leg ("k":"eq"); the isolated
# per-shape wall timings follow. No timing is quotable unless every eq row
# in every leg is zero.
/tmp/q4-bw-rpl 2>&1 | tee "$R/rpl-a-$stamp.ndjson" || fail=1
sleep 10
/tmp/q4-bw-rpl 2>&1 | tee "$R/rpl-b-$stamp.ndjson" || fail=1

echo "== phase 2: gap candidate screen (in-model widedep), two legs =="
sleep 10
/tmp/q4-bw-rpl --gap 2>&1 | tee "$R/rpl-gap-a-$stamp.ndjson" || fail=1
sleep 10
/tmp/q4-bw-rpl --gap 2>&1 | tee "$R/rpl-gap-b-$stamp.ndjson" || fail=1

echo "== phase 3: AGX shaderdb register stats, shader cache disabled =="
MESA_SHADER_CACHE_DISABLE=true AGX_MESA_DEBUG=shaderdb \
    /tmp/q4-bw-rpl --quick > /dev/null 2> "$R/rpl-shaderdb-$stamp.log" || true
grep -c "gprs" "$R/rpl-shaderdb-$stamp.log" || true

echo "$(date -Is) window complete fail=$fail"
exit "$fail"

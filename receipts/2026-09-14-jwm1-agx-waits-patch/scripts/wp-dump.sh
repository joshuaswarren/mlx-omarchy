#!/usr/bin/env bash
# wp-dump.sh <arm> <outdir>
# Capture, under the GPU lock, for one ICD arm:
#   <outdir>/<arm>.dump.log    AGX shader dump (final packed ISA)
#   <outdir>/<arm>.firing.log  AGXWAITS_DEBUG=1 region-pass trace
#   <outdir>/<arm>.driver.json which libvulkan_asahi.so actually mapped
# Separate runs so the ISA dump is not interleaved with the trace.
set -euo pipefail

ARM=${1:?arm: base|patched}
OUT=${2:?outdir}
D=/home/joshuawarren/benchq/waitspatch
BENCH=/home/joshuawarren/benchq/qmm-coop-bench
PY="$D/py-$ARM"

test -x "$PY" || { echo "no such arm wrapper: $PY"; exit 1; }
mkdir -p "$OUT"
cd "$BENCH"

echo "== [$ARM] driver binding proof"
timeout -k 10s 300s flock -w 60 /tmp/m1-gpu.lock \
  "$PY" /tmp/whichdriver.py > "$OUT/$ARM.driver.json" 2>&1 \
  && echo "   exactly one driver mapped" \
  || { echo "   FAILED - driver binding not unique"; cat "$OUT/$ARM.driver.json"; exit 1; }
grep -E '"|sha256' "$OUT/$ARM.driver.json" | head -8

echo "== [$ARM] ISA dump"
timeout -k 10s 900s flock -w 60 /tmp/m1-gpu.lock env \
  AGX_MESA_DEBUG=shaders \
  "$PY" qmm_dump_probe.py > "$OUT/$ARM.dump.log" 2>&1
echo "   lines=$(wc -l < "$OUT/$ARM.dump.log") simd_matrix_fmadd32=$(grep -c simd_matrix_fmadd32 "$OUT/$ARM.dump.log" || true)"
grep -E 'median_ms|DUMP-PROBE-DONE' "$OUT/$ARM.dump.log" || true

echo "== [$ARM] region-pass firing trace"
timeout -k 10s 900s flock -w 60 /tmp/m1-gpu.lock env \
  AGXWAITS_DEBUG=1 \
  "$PY" qmm_dump_probe.py > "$OUT/$ARM.firing.log" 2>&1
echo "   AGXWAITS lines=$(grep -c '^AGXWAITS:' "$OUT/$ARM.firing.log" || true)"
for k in ACCEPT else-arrival close carry drain 'reject shape' 'reject target' 'overflow-drain'; do
  printf '   %-16s %s\n' "$k" "$(grep -c "AGXWAITS: .*$k" "$OUT/$ARM.firing.log" || true)"
done
grep -E 'median_ms|DUMP-PROBE-DONE' "$OUT/$ARM.firing.log" || true

echo "== [$ARM] lock after: $(fuser /tmp/m1-gpu.lock >/dev/null 2>&1 && echo held || echo free)"

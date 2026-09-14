#!/usr/bin/env bash
# wp-prefill.sh <rounds> <outdir>
# Interleaved 1053-token end-to-end prefill A/B across the two ICD arms.
# run_profile.sh takes and releases /tmp/m1-gpu.lock per run, so the arms
# alternate round by round rather than running in two blocks: that is what
# controls for machine drift across the sequence.
set -euo pipefail

ROUNDS=${1:?rounds}
OUT=${2:?outdir}
D=/home/joshuawarren/benchq/waitspatch
RP=/home/joshuawarren/benchq/qmmpad/run_profile.sh

mkdir -p "$OUT"

for r in $(seq 1 "$ROUNDS"); do
  for arm in base patched; do
    echo "===== round $r arm $arm"
    bash "$RP" "$D/py-$arm" "$OUT/$arm-r$r" "waits-$arm-r$r" \
      > "$OUT/$arm-r$r.log" 2>&1 \
      && echo "  ok" || { echo "  FAILED rc=$?"; tail -25 "$OUT/$arm-r$r.log"; }
    grep -E "prefill|tok/s|GPU busy|QmmPrefillCoopmatF16" \
      "$OUT/$arm-r$r.log" 2>/dev/null | head -8 || true
  done
done

echo "== lock after: $(fuser /tmp/m1-gpu.lock >/dev/null 2>&1 && echo held || echo free)"

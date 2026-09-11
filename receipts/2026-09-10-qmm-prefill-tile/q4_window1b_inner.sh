#!/bin/sh
# Q4 1K prefill window 1b (inner): short follow-up after the probe import
# crash. Wheel and venv from window 1 are intact; this only reruns the
# schedule screen and the attribution probes under the lock.
set -eu
ulimit -c 0
OUT=$HOME/benchq/q4prefill-window1
cd "$OUT"
echo "[window1b] $(date) resume probes; wheel + venv from window 1"
for arm in 0 1 2 3; do
  MLX_OMARCHY_QMM_COOP_TILE=$arm AGX_SIMDMAT=1 \
    "$HOME/venv-q4tile/bin/python" -u q4_prefill_probe.py qmm \
    > "$OUT/probe-qmm-arm$arm.log" 2>&1
  echo "[window1b] arm $arm:"
  grep -E "^\{" "$OUT/probe-qmm-arm$arm.log"
done
echo "[window1b] attribution probes (attn, norms, lmhead)"
MLX_OMARCHY_QMM_COOP_TILE=0 AGX_SIMDMAT=1 \
  "$HOME/venv-q4tile/bin/python" -u q4_prefill_probe.py attn norms lmhead \
  > "$OUT/probe-attrib.log" 2>&1
grep -E "^\{" "$OUT/probe-attrib.log"
echo "[window1b] DONE"

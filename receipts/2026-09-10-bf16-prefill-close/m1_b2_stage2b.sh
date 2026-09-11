#!/usr/bin/env bash
# Window B2 stage 2b: re-run the attribution probes with
# MLX_DISABLE_COMPILE=1 (stage 2 phase B omitted it; the Honeykrisp
# backend refuses compiled bf16 tapes by design). Capped flock -w 3600.
set -uo pipefail
cd ~/src/mlx-Bf16PrefillClose
R=receipts/2026-09-10-bf16-prefill-close
[[ -d .venv-prefill-base && -d .venv-prefill-cand ]] || {
  echo "FATAL: venvs missing"; exit 3; }

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 3600s)"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded 3600s"; exit 4; }
echo "$(date -Is) lock acquired"
{
  echo "== stage2b: attribution probes, MLX_DISABLE_COMPILE=1 =="
  env MLX_DISABLE_COMPILE=1 .venv-prefill-base/bin/python \
    "$R/attribution_probe.py" --reps 9 \
    --out "$R/m1-logs/attribution-base.ndjson" \
    > "$R/m1-logs/attribution-base.log" 2>&1
  echo "attribution_base rc=$?"
  env MLX_DISABLE_COMPILE=1 .venv-prefill-cand/bin/python \
    "$R/attribution_probe.py" --reps 9 \
    --out "$R/m1-logs/attribution-cand.ndjson" \
    > "$R/m1-logs/attribution-cand.log" 2>&1
  echo "attribution_cand rc=$?"
  echo "STAGE2B-OK"
} >> "$R/window-b2-stage2.log" 2>&1
rc=$?
echo "$(date -Is) stage 2b complete rc=$rc (lock released)"
exit $rc

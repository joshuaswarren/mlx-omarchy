#!/usr/bin/env bash
# Window 1 on jwm1-linux: component attribution + gate A/B on the BASE
# (a5b8c4ab) wheel, no source change in the wheel. Held under a single
# top-level /tmp/m1-gpu.lock flock with a capped wait. Runs:
#   1. attribution_components.py  - component-isolated wall-clock probes
#      (f32 composition components, bf16 composition components, sdpa whole
#      under both gates in-process, projection/lm_head/norm/rope rates,
#      host launch floor)
#   2. oracle_f64.py              - f64 oracle: token streams both gates,
#      numpy f64 truth, attention-block ULP study
#   3. bench_matrix A/B           - digest legs with the gate off, then on
#      (env on the same wheel; fork driver)
# Builds are NOT done here; this window only runs the GPU work.
set -uo pipefail
cd ~/src/mlx-bf16-prefill-attn
R=receipts/2026-09-11-bf16-prefill-attention
PY=.venv-attn-base/bin/python
WHEEL=dist/mlx_omarchy-0.32.2.dev202609111512+a5b8c4a-cp314-cp314-linux_aarch64.whl
[[ -x $PY ]] || { echo "FATAL: base venv missing"; exit 3; }
[[ -f $WHEEL ]] || { echo "FATAL: base wheel missing"; exit 3; }
mkdir -p "$R/m1-logs" "$R/matrix/attn-gate-off-fork" "$R/matrix/attn-gate-on-fork"

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 7200s)"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded 7200s"; exit 4; }
echo "$(date -Is) lock acquired"
{
  echo "== quiet gate =="
  pgrep -fa 'bench_decode|bench_matrix|mlx_lm' | grep -v $$ || true

  echo "== phase A: component attribution (base wheel, gate default) =="
  timeout 1500 $PY "$R/attribution_components.py" --reps 9 \
    --out "$R/m1-logs/attribution-base.ndjson" \
    > "$R/m1-logs/attribution-base.log" 2>&1
  echo "attribution rc=$?"

  echo "== phase B: f64 oracle =="
  timeout 3600 $PY "$R/oracle_f64.py" \
    --out "$R/m1-logs/oracle.ndjson" \
    > "$R/m1-logs/oracle.log" 2>&1
  echo "oracle rc=$?"

  echo "== phase C: digest A/B (gate off = shipped composition) =="
  mkdir -p "$R/matrix"
  env -u MLX_OMARCHY_SDPA_BF16_FAST MLX_DISABLE_COMPILE=1 timeout 1800 \
    scripts/bench_matrix.py --mode run \
    --python "$PY" --wheel "$WHEEL" \
    --host-label "jwm1-attn-gate-off-fork" --timeout 900 \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --out "$R/matrix/attn-gate-off-fork/matrix.json" \
    > "$R/matrix/attn-gate-off-fork.log" 2>&1
  echo "gate-off matrix rc=$? verified=$(grep -c 'verified=match' "$R/matrix/attn-gate-off-fork.log" || true)"

  echo "== phase D: digest A/B (gate on = bf16 composition) =="
  env MLX_OMARCHY_SDPA_BF16_FAST=1 MLX_DISABLE_COMPILE=1 timeout 1800 \
    scripts/bench_matrix.py --mode run \
    --python "$PY" --wheel "$WHEEL" \
    --host-label "jwm1-attn-gate-on-fork" --timeout 900 \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --out "$R/matrix/attn-gate-on-fork/matrix.json" \
    > "$R/matrix/attn-gate-on-fork.log" 2>&1
  echo "gate-on matrix rc=$? verified=$(grep -c 'verified=match' "$R/matrix/attn-gate-on-fork.log" || true)"

  echo "WINDOW1-DONE"
} > "$R/window1.log" 2>&1
rc=$?
echo "$(date -Is) window 1 complete rc=$rc (lock released)"
exit $rc

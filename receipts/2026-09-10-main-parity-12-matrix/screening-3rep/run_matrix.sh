#!/usr/bin/env bash
# Canonical 12-leg parity matrix on current main (b6d662a8), jwm1.
# 3 paired repetitions, alternating fork (installed honeykrisp-omarchy,
# coopmat-capable) and stock Mesa 26.1.7 (private ICD). Each run takes
# /tmp/m1-gpu.lock with a bounded wait and a bounded timeout.
set -uo pipefail
ROOT=/home/joshuawarren/src/mlx-main-b6d662a8
OUT=/tmp/parity12
PY="$ROOT/.work/venv-run/bin/python"
cd "$ROOT"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
echo "wheel: $WHEEL"
echo "wheel sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"

for rep in 1 2 3; do
  for drv in fork stock; do
    if [ "$drv" = fork ]; then
      ICD=(); LABEL="jwm1 honeykrisp-fork-26.3.0-devel main-b6d662a rep${rep}"
    else
      ICD=(env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json)
      LABEL="jwm1 stock-mesa-26.1.7 main-b6d662a rep${rep}"
    fi
    echo "=== run rep${rep} ${drv} $(date -u +%H:%M:%S) ==="
    flock -w 2400 /tmp/m1-gpu.lock timeout 1500 \
      "${ICD[@]}" "$PY" scripts/bench_matrix.py --mode run \
      --python "$PY" \
      --wheel "$WHEEL" \
      --host-label "$LABEL" \
      --timeout 900 \
      --out "$OUT/${drv}-rep${rep}.json" \
      > "$OUT/${drv}-rep${rep}.log" 2>&1
    rc=$?
    echo "rep${rep} ${drv} exit=${rc}"
    grep -E "^  (measured|failed|skipped)|decode [0-9]" "$OUT/${drv}-rep${rep}.log" | tail -6
  done
done
echo ALL_RUNS_DONE

#!/usr/bin/env bash
# Canonical closing 12-leg parity matrix on current main (711327ce), jwm1.
# One top-level flock on /tmp/m1-gpu.lock capped at two hours (held by the
# caller). Warmup matrix per driver run and discarded, then 12 measured
# repetitions per driver, fork/stock alternating, fresh bench_matrix
# process each. Fork = installed honeykrisp-omarchy (coopmat-capable),
# stock = Mesa 26.1.7 via private ICD.
set -uo pipefail
ROOT=/home/joshuawarren/src/mlx-main-711327ce
OUT=/tmp/parity12-20260911
PY="$ROOT/.work/venv-run/bin/python"
mkdir -p "$OUT"
cd "$ROOT"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
echo "wheel: $WHEEL"
echo "wheel sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"

run_one() {  # drv label outfile
  local drv="$1" label="$2" out="$3"
  local -a ICD=()
  if [ "$drv" != fork ]; then
    ICD=(env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json)
  fi
  echo "=== run $out $(date -u +%H:%M:%S) ==="
  timeout 1500 "${ICD[@]}" "$PY" scripts/bench_matrix.py --mode run \
    --python "$PY" \
    --wheel "$WHEEL" \
    --host-label "$label" \
    --timeout 900 \
    --out "$OUT/$out.json" \
    > "$OUT/$out.log" 2>&1
  local rc=$?
  echo "$out exit=${rc}"
  grep -E "^  (measured|failed|skipped)" "$OUT/$out.log" | tail -6
}

run_one fork  "jwm1 honeykrisp-fork-26.3.0-devel main-711327ce warmup-discarded" fork-warmup
run_one stock "jwm1 stock-mesa-26.1.7 main-711327ce warmup-discarded" stock-warmup
for rep in 01 02 03 04 05 06 07 08 09 10 11 12; do
  run_one fork  "jwm1 honeykrisp-fork-26.3.0-devel main-711327ce rep${rep}" "fork-rep${rep}"
  run_one stock "jwm1 stock-mesa-26.1.7 main-711327ce rep${rep}" "stock-rep${rep}"
done
echo ALL_RUNS_DONE

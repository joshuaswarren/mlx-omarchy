#!/usr/bin/env bash
# Strand-1 window on jwm1-linux: bf16-alpha-fix proofs and digest gates.
# Produces:
#   1. alpha tree fam/rt suites on the M1 (fork driver)
#   2. fast_ops alpha tree: full run. The 8 f16 sdpa cases are EXPECTED to
#      throw on the M1 (pre-existing on main; llvmpipe passes them); the
#      alpha regression test itself must PASS via the coopmat route.
#   3. fail-proof: fast-trap (gate relaxed, pre-fix shader) must FAIL the
#      alpha regression test; fast-alpha must PASS it.
#   4. f64 probe: ULP stats for the scaled bf16 matmul on both routes.
#   5. digest gates: fork + stock x 3 reps, alpha wheel - all six canonical
#      Q4 digests and all BF16 pins must hold (alpha==1 bit-identity).
set -uo pipefail
cd ~/src/mlx-omarchy-alpha-m1
R=receipts/2026-09-11-bf16-alpha-fix
mkdir -p "$R/m1-logs" "$R/matrix"
PY=.venv-alpha/bin/python
WHEEL=$(cat s1-logs/alpha-wheel-path.txt)
[[ -x /tmp/fam-alpha && -x /tmp/fast-alpha && -x /tmp/rt-alpha ]] || {
  echo "FATAL: alpha suite binaries missing"; exit 3; }
[[ -x /tmp/fast-trap && -x /tmp/probe-alpha ]] || {
  echo "FATAL: trap/probe binaries missing"; exit 3; }
[[ -f $WHEEL ]] || { echo "FATAL: alpha wheel missing"; exit 3; }

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 7200s)"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded 7200s"; exit 4; }
echo "$(date -Is) lock acquired"
driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)
echo "driver: $driver_pkg"
[[ $driver_pkg == "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1" ]] || {
  echo "FATAL: fork driver is not the pinned honeykrisp build"; exit 5; }
{
  echo "== phase 1: f64 probe (fast, isolated first) =="
  /tmp/probe-alpha 2>&1 | tee "$R/m1-logs/probe.log" | grep -E "PROBE|Status" || true

  echo "== phase 2: fail-proof (regression test case only) =="
  echo "-- trap (pre-fix shader + relaxed gate): expect FAILURE"
  /tmp/fast-trap --test-case="*scores scale through MatmulBF16Coopmat*" \
    --out="$R/m1-logs/fastops-trap-alpha-test.log" 2>&1 | tail -3
  echo "trap rc=$?"
  grep -E "ERROR|require_close|failed" "$R/m1-logs/fastops-trap-alpha-test.log" | head -4 || true
  echo "-- fixed: expect SUCCESS"
  /tmp/fast-alpha --test-case="*scores scale through MatmulBF16Coopmat*" \
    --out="$R/m1-logs/fastops-alpha-alpha-test.log" 2>&1 | tail -3
  echo "alpha rc=$?"

  echo "== phase 3: alpha suites =="
  timeout 1800 /tmp/fam-alpha --out="$R/m1-logs/m1-fork-family-alpha.log" 2>&1 | tail -2
  echo "family_alpha rc=$?"
  grep -q 'row-mismatch' "$R/m1-logs/m1-fork-family-alpha.log" && {
    echo "FATAL: alpha family suite row mismatch"; exit 3; }
  timeout 1800 /tmp/fast-alpha --out="$R/m1-logs/m1-fork-fastops-alpha.log" 2>&1 | tail -4
  echo "fastops_alpha rc=$? (f16 sdpa cases expected to throw: pre-existing)"
  timeout 900 /tmp/rt-alpha --out="$R/m1-logs/m1-fork-runtime-alpha.log" 2>&1 | tail -2
  echo "runtime_alpha rc=$?"
  grep -qE 'row-mismatch|FAILED' "$R/m1-logs/m1-fork-runtime-alpha.log" && {
    echo "FATAL: alpha runtime suite failure"; exit 3; }

  echo "== phase 4: digest gates (alpha wheel) =="
  run_matrix() {  # driver rep
    local driver=$1 rep=$2
    local run="$R/matrix/${rep}-${driver}-alpha"
    mkdir -p "$run"
    if [[ $driver == stock ]]; then
      env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
        MLX_DISABLE_COMPILE=1 timeout 1800 ~/src/mlx-bf16-prefill-attn/scripts/bench_matrix.py --mode run \
        --python "$PY" --wheel "$WHEEL" \
        --host-label "jwm1-${rep}-${driver}-alpha" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run.log" 2>&1
    else
      env MLX_DISABLE_COMPILE=1 timeout 1800 ~/src/mlx-bf16-prefill-attn/scripts/bench_matrix.py --mode run \
        --python "$PY" --wheel "$WHEEL" \
        --host-label "jwm1-${rep}-${driver}-alpha" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run.log" 2>&1
    fi
    local ok
    ok=$(grep -c 'verified=match' "$run.log" || true)
    echo "$rep-$driver-alpha: verified=match x${ok}"
  }
  run_matrix fork warmup  # discarded
  for rep in r1 r2 r3; do
    run_matrix fork "$rep"
    run_matrix stock "$rep"
  done
  echo DIGESTS-OK
} > "$R/window_s1.log" 2>&1
rc=$?
echo "$(date -Is) S1 window complete rc=$rc (lock released)"
exit $rc

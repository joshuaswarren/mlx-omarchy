#!/usr/bin/env bash
# Window 2 on jwm1-linux: candidate wheel (default-flip + alpha coopmat)
# suites, paired matrix, and candidate attribution. Single top-level
# /tmp/m1-gpu.lock flock, capped wait; the wheel + suite binary builds
# happened OUTSIDE the lock (m1_build_cand.sh, and the base wheel was
# rebuilt from a5b8c4ab after the cand build wiped dist/).
#
# Decision inputs this window produces:
#   - fam/fast_ops/rt suite binaries from the candidate tree run clean
#     (abort gate; fast_ops carries the coopmat alpha regression test)
#   - paired prefill matrix: warmup + 3 reps x {fork,stock} x {base,cand}
#   - digest gates per cell, evaluated post-hoc by make_verdict.py
#   - attribution on the base wheel (runtimes fixed) and the cand wheel
#     (--default-is-fast so "gate off" really means f32 composition there)
#   - f64 oracle on BOTH wheels (the cand wheel's bf16fast rows ride the
#     fixed alpha coopmat kernel; the base wheel's ride the non-coopmat
#     fallback, so both sides of the decision are measured where they run)
set -uo pipefail
cd ~/src/mlx-bf16-prefill-attn
R=receipts/2026-09-11-bf16-prefill-attention
PYB=.venv-attn-base/bin/python
PYC=.venv-attn-cand/bin/python
WHEELB=$(cat "$R/m1-logs/base-wheel-path.txt" 2>/dev/null)
WHEELC=$(cat "$R/m1-logs/cand-wheel-path.txt" 2>/dev/null)
[[ -x $PYC ]] || { echo "FATAL: cand venv missing"; exit 3; }
[[ -f $WHEELB && -f $WHEELC ]] || { echo "FATAL: wheels missing"; exit 3; }
[[ -x /tmp/fam-attn && -x /tmp/fast-attn && -x /tmp/rt-attn ]] || {
  echo "FATAL: cand suite binaries missing"; exit 3; }
mkdir -p "$R/m1-logs" "$R/matrix"

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 7200s)"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded 7200s"; exit 4; }
echo "$(date -Is) lock acquired"
driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)
echo "driver: $driver_pkg"
[[ $driver_pkg == "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1" ]] || {
  echo "FATAL: fork driver is not the pinned honeykrisp build"; exit 5; }
{
  echo "== phase A: cand suite binaries (abort gate) =="
  timeout 1800 /tmp/fam-attn --out="$R/m1-logs/attn-fork-family-cand.log" 2>&1 | tail -2
  echo "family_cand rc=$?"
  grep -q 'row-mismatch' "$R/m1-logs/attn-fork-family-cand.log" && {
    echo "FATAL: cand family suite row mismatch"; exit 3; }
  timeout 900 /tmp/fast-attn --out="$R/m1-logs/attn-fork-fastops-cand.log" 2>&1 | tail -2
  echo "fastops_cand rc=$?"
  grep -qE 'FAILED|Status: FAILED' "$R/m1-logs/attn-fork-fastops-cand.log" && {
    echo "FATAL: cand fast_ops suite failure"; exit 3; }
  timeout 900 /tmp/rt-attn --out="$R/m1-logs/attn-fork-runtime-cand.log" 2>&1 | tail -2
  echo "runtime_cand rc=$?"
  grep -qE 'row-mismatch|FAILED' "$R/m1-logs/attn-fork-runtime-cand.log" && {
    echo "FATAL: cand runtime suite failure"; exit 3; }
  echo "== phase B: cooldown =="
  sleep 120

  echo "== phase C: paired matrix =="
  run_matrix() {  # driver cell wheel label
    local driver=$1 cell=$2 wheel=$3 label=$4
    local run="$R/matrix/$label"
    mkdir -p "$run"
    if [[ $driver == stock ]]; then
      env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
        MLX_DISABLE_COMPILE=1 timeout 1800 scripts/bench_matrix.py --mode run \
        --python ".venv-attn-$cell/bin/python" --wheel "$wheel" \
        --host-label "jwm1-$label" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run.log" 2>&1
    else
      env MLX_DISABLE_COMPILE=1 timeout 1800 scripts/bench_matrix.py --mode run \
        --python ".venv-attn-$cell/bin/python" --wheel "$wheel" \
        --host-label "jwm1-$label" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run.log" 2>&1
    fi
    local ok
    ok=$(grep -c 'verified=match' "$run.log" || true)
    echo "$label: verified=match x${ok} json=$(python3 -c "import json,sys;print('ok' if json.load(open('$run/matrix.json')).get('legs') else 'MISSING')" 2>/dev/null || echo MISSING)"
  }
  run_matrix fork base "$WHEELB" warmup   # discarded
  for rep in 1 2 3; do
    for driver in fork stock; do
      run_matrix "$driver" base "$WHEELB" "r${rep}-${driver}-base"
      run_matrix "$driver" cand "$WHEELC" "r${rep}-${driver}-cand"
    done
  done
  echo MATRIX-OK

  echo "== phase D: base attribution (probe fixed, lm_head included) =="
  timeout 1500 $PYB "$R/attribution_components.py" --reps 9 --tag base \
    --out "$R/m1-logs/attribution-base.ndjson" \
    > "$R/m1-logs/attribution-base.log" 2>&1
  echo "attribution_base rc=$?"

  echo "== phase E: f64 oracle on both wheels =="
  timeout 3600 $PYB "$R/oracle_f64.py" --tag base \
    --out "$R/m1-logs/oracle-base.ndjson" \
    > "$R/m1-logs/oracle-base.log" 2>&1
  echo "oracle_base rc=$?"
  timeout 3600 $PYC "$R/oracle_f64.py" --tag cand --default-is-fast \
    --out "$R/m1-logs/oracle-cand.ndjson" \
    > "$R/m1-logs/oracle-cand.log" 2>&1
  echo "oracle_cand rc=$?"

  echo "== phase F: cand attribution (--default-is-fast) =="
  timeout 1500 $PYC "$R/attribution_components.py" --reps 9 --tag cand \
    --default-is-fast \
    --out "$R/m1-logs/attribution-cand.ndjson" \
    > "$R/m1-logs/attribution-cand.log" 2>&1
  echo "attribution_cand rc=$?"

  echo WINDOW2-DONE
} > "$R/window2.log" 2>&1
rc=$?
echo "$(date -Is) window 2 complete rc=$rc (lock released)"
exit $rc

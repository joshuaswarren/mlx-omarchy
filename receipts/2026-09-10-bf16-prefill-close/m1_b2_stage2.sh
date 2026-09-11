#!/usr/bin/env bash
# Window B2 stage 2 on jwm1-linux (split per Main's re-sequencing).
# Re-acquires /tmp/m1-gpu.lock with a CAPPED wait after
# Q4GemvNativeMapping and PythonHostParity have taken their windows, then:
# family+runtime suites (base and cand binaries from stage 1) as abort
# gates, attribution probes, cooldown, paired matrix (warmup + 3 reps x
# {fork,stock} x {base,cand}) with digest gates evaluated post-hoc by
# summarize_prefill.py. Never holds a lock across a build; all builds
# happened in stage 1. Wait cap: flock -w 7200.
set -uo pipefail
cd ~/src/mlx-Bf16PrefillClose
R=receipts/2026-09-10-bf16-prefill-close
cand_wheel=$(cat "$R/m1-logs/cand-wheel-path.txt")
base_wheel="$HOME/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl"
[[ -x /tmp/fam-base && -x /tmp/fam-cand && -x /tmp/rt-base && -x /tmp/rt-cand ]] || {
  echo "FATAL: stage 1 suite binaries missing"; exit 3; }
[[ -d .venv-prefill-base && -d .venv-prefill-cand ]] || {
  echo "FATAL: stage 1 venvs missing"; exit 3; }

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 7200s)"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded 7200s"; exit 4; }
echo "$(date -Is) lock acquired"
{
  echo "== stage2 phase A: family + runtime suites, fork driver =="
  timeout 1800 /tmp/fam-base --out="$R/m1-logs/m1-fork-family-base.log" 2>&1 | tail -2
  echo "family_base rc=$?"
  if grep -q 'row-mismatch' "$R/m1-logs/m1-fork-family-base.log"; then
    echo "FATAL: base family suite row mismatch; environment broken"; exit 3
  fi
  timeout 900 /tmp/rt-base --out="$R/m1-logs/m1-fork-runtime-base.log" 2>&1 | tail -2
  echo "runtime_base rc=$?"
  timeout 1800 /tmp/fam-cand --out="$R/m1-logs/m1-fork-family-cand.log" 2>&1 | tail -2
  echo "family_cand rc=$?"
  if grep -q 'row-mismatch' "$R/m1-logs/m1-fork-family-cand.log"; then
    echo "FATAL: cand family suite row mismatch; k_base fix insufficient"; exit 3
  fi
  timeout 900 /tmp/rt-cand --out="$R/m1-logs/m1-fork-runtime-cand.log" 2>&1 | tail -2
  echo "runtime_cand rc=$?"

  echo "== stage2 phase B: attribution probes (fork driver, base and cand) =="
  .venv-prefill-base/bin/python "$R/attribution_probe.py" --reps 9 \
    --out "$R/m1-logs/attribution-base.ndjson" \
    > "$R/m1-logs/attribution-base.log" 2>&1
  echo "attribution_base rc=$?"
  .venv-prefill-cand/bin/python "$R/attribution_probe.py" --reps 9 \
    --out "$R/m1-logs/attribution-cand.ndjson" \
    > "$R/m1-logs/attribution-cand.log" 2>&1
  echo "attribution_cand rc=$?"

  echo "== stage2 phase C: cooldown before matrix =="
  sleep 180

  # llvmpipe fallback screening runs on the local x86_64 box (this M1
  # carries no llvmpipe ICD); its tile-path legs are recorded there.
  run_matrix() {  # driver cell wheel label
    local driver=$1 cell=$2 wheel=$3 label=$4
    local run="$R/matrix/$label"
    mkdir -p "$run"
    if [[ $driver == stock ]]; then
      env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
        MLX_DISABLE_COMPILE=1 timeout 1800 scripts/bench_matrix.py --mode run \
        --python ".venv-prefill-$cell/bin/python" --wheel "$wheel" \
        --host-label "jwm1-$label" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run/matrix.log" 2>&1
    else
      env MLX_DISABLE_COMPILE=1 timeout 1800 scripts/bench_matrix.py --mode run \
        --python ".venv-prefill-$cell/bin/python" --wheel "$wheel" \
        --host-label "jwm1-$label" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run/matrix.log" 2>&1
    fi
    local ok
    ok=$(grep -c 'verified=match' "$run/matrix.log" || true)
    echo "$label: verified=match x${ok}"
  }
  run_matrix fork base "$base_wheel" warmup   # discarded
  for rep in 1 2 3; do
    for driver in fork stock; do
      run_matrix "$driver" base "$base_wheel" "r${rep}-${driver}-base"
      run_matrix "$driver" cand "$cand_wheel" "r${rep}-${driver}-cand"
    done
  done
  echo MATRIX-OK
} > "$R/window-b2-stage2.log" 2>&1
rc=$?
echo "$(date -Is) stage 2 complete rc=$rc (lock released)"
exit $rc

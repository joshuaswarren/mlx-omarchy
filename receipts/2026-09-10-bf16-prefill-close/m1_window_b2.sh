#!/usr/bin/env bash
# Window B2 on jwm1-linux: FIXED candidate wheel (pack_pair k_base fix)
# + dual suite binaries (base shader and cand shader built separately),
# probe flags sweep, attribution probes, paired matrix with digest gates.
# Replaces window B (killed: its candidate wheel shipped the broken
# scalar-path staging; see window-b-broken-run1.log).
# ONE top-level flock held for the whole window (builds included, per the
# jwm1 fleet rule that builds never overlap a neighbor's measurement
# window). Usage:
#   ssh jwm1 'cd ~/src/mlx-Bf16PrefillClose && nohup env CAND_COMMIT=<sha> \
#     bash receipts/2026-09-10-bf16-prefill-close/m1_window_b2.sh \
#     > /tmp/bf16-window-b2-launch.log 2>&1 &'
# Env: CAND_COMMIT=<7+ char commit sha> (stamps the wheel).
set -uo pipefail
cd ~/src/mlx-Bf16PrefillClose
R=receipts/2026-09-10-bf16-prefill-close
mkdir -p "$R/m1-logs" "$R/matrix" "$R/wheels/candidate"
base_wheel="$HOME/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl"
echo "98821134f4bcf306ab1d64d6885ddf3d3e8e932311283bbd11f97aecf6a8e439  $base_wheel" \
  | sha256sum -c - || { echo "FATAL: base wheel mismatch"; exit 1; }
exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock"
flock -w 28800 9
echo "$(date -Is) lock acquired"
{
  echo "== phase 0: driver capability gate =="
  g++ -std=c++17 -O2 -o /tmp/bf16-prefill-bench tools/bf16-prefill-bench/bench.cpp \
    || { echo "FATAL: probe build"; exit 3; }
  /tmp/bf16-prefill-bench --tiny 2>&1 | grep '"k":"dev"' \
    | tee "$R/m1-logs/driver-gate-b2.txt"
  grep -q '"coopmat":true' "$R/m1-logs/driver-gate-b2.txt" || {
    echo "FATAL: default driver lacks cooperative matrix; bailing."
    exit 3
  }

  echo "== phase 1: candidate wheel build (niced, -j4) =="
  echo "${CAND_COMMIT:-unknown}" | tee "$R/m1-logs/cand-commit.txt"
  nice -n 10 env DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 \
    MLX_OMARCHY_SOURCE_COMMIT="${CAND_COMMIT:-local}" \
    scripts/build-wheel.sh 2>&1 | tail -5 | tee "$R/m1-logs/build-wheel-cand2.log"
  cand_wheel=$(ls -t dist/mlx_omarchy-*.whl | head -1)
  cp "$cand_wheel" "$R/wheels/candidate/"
  sha256sum "$cand_wheel" | tee "$R/m1-logs/candidate-wheel.sha256"

  echo "== phase 2: fresh venvs =="
  rm -rf .venv-prefill-base .venv-prefill-cand
  python3.14 -m venv .venv-prefill-base
  .venv-prefill-base/bin/pip install --quiet "$base_wheel" mlx-lm==0.31.3
  python3.14 -m venv .venv-prefill-cand
  .venv-prefill-cand/bin/pip install --quiet "$cand_wheel" mlx-lm==0.31.3

  echo "== phase 3: suite binaries, base AND cand shader built separately =="
  SHAD=.work/mlx/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp
  BASE_SHAD="$HOME/src/mlx-main-b6d662a8/overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp"
  [[ -f "$BASE_SHAD" ]] || { echo "FATAL: base shader source missing"; exit 3; }
  sha256sum "$SHAD" "$BASE_SHAD" | tee "$R/m1-logs/shader-hashes-b2.txt"
  cmp -s "$SHAD" "$BASE_SHAD" && {
    echo "FATAL: cand shader == base shader; sync failed"; exit 3; }
  cp "$SHAD" /tmp/cand-bf16.comp.keep
  build_suites() {
    nice -n 10 cmake --build .work/build-prefill -j4 --target \
      omarchy_matmul_family_tests omarchy_runtime_tests mlx-omarchy-info \
      2>&1 | tail -2
  }
  cp "$BASE_SHAD" "$SHAD"
  build_suites | tee "$R/m1-logs/build-suite-base.log"
  cp .work/build-prefill/tests/omarchy/omarchy_matmul_family_tests /tmp/fam-base
  cp .work/build-prefill/tests/omarchy/omarchy_runtime_tests /tmp/rt-base
  cp /tmp/cand-bf16.comp.keep "$SHAD"
  build_suites | tee "$R/m1-logs/build-suite-cand.log"
  cp .work/build-prefill/tests/omarchy/omarchy_matmul_family_tests /tmp/fam-cand
  cp .work/build-prefill/tests/omarchy/omarchy_runtime_tests /tmp/rt-cand
  cmp -s /tmp/fam-base /tmp/fam-cand && {
    echo "FATAL: base and cand suite binaries identical"; exit 3; }
  sha256sum /tmp/fam-base /tmp/fam-cand /tmp/rt-base /tmp/rt-cand \
    | tee "$R/m1-logs/suite-binaries.sha256"

  echo "== phase 4: family + runtime suites, fork driver (base then cand) =="
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

  echo "== phase 5: probe flags sweep gate (fork driver) =="
  /tmp/bf16-prefill-bench > "$R/m1-logs/probe-flags-sweep.ndjson" \
    2> "$R/m1-logs/probe-flags-sweep.err"
  python3 - "$R/m1-logs/probe-flags-sweep.ndjson" <<'PY'
import json, sys
bad = 0; n = 0; flags = set()
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("k") == "correct" and "cand_vs_base_mismatch" in r:
        n += 1; flags.add(r["flags"])
        if r["cand_vs_base_mismatch"] or r["dead_rows"]:
            bad += 1; print("BAD", r)
print(f"probe cells={n} flags={sorted(flags)} bad={bad}")
sys.exit(0 if (bad == 0 and n >= 144 and flags == {1, 0, 5, 4}) else 1)
PY
  [[ $? -eq 0 ]] || { echo "FATAL: probe flags sweep failed"; exit 3; }

  echo "== phase 6: attribution probes (fork driver, base and cand) =="
  .venv-prefill-base/bin/python "$R/attribution_probe.py" --reps 9 \
    --out "$R/m1-logs/attribution-base.ndjson" \
    > "$R/m1-logs/attribution-base.log" 2>&1
  echo "attribution_base rc=$?"
  .venv-prefill-cand/bin/python "$R/attribution_probe.py" --reps 9 \
    --out "$R/m1-logs/attribution-cand.ndjson" \
    > "$R/m1-logs/attribution-cand.log" 2>&1
  echo "attribution_cand rc=$?"

  echo "== phase 7: cooldown before matrix =="
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
} > "$R/window-b2.log" 2>&1
rc=$?
echo "$(date -Is) window complete rc=$rc"
exit $rc

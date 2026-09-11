#!/bin/sh
# CoopmatIlpChains window B (inner): paired fork/stock prefill matrix
# for the ILP scheduling change. Runs under ONE top-level flock
# /tmp/m1-gpu.lock. Args: CAND_SHA BASE_SHA.
# Wheels: candidate = CAND_SHA (wave/CoopmatIlpChains tip), base =
# BASE_SHA (origin/main) built in the same window from the same clone,
# so the paired cells isolate the kernel change only.
set -eu
ulimit -c 0
CAND_SHA=$1
BASE_SHA=$2
REPO=$HOME/src/mlx-omarchy-cfbench
OUT=$HOME/benchq/qmm-coop-bench
VROOT=$OUT/venvs
mkdir -p "$OUT" "$VROOT"
cd "$REPO"

build_wheel_at() {  # sha label
  git checkout --quiet --detach "$1"
  bash scripts/prepare-mlx.sh > "$OUT/prepare-$2.log" 2>&1
  sh scripts/build-wheel.sh > "$OUT/wheel-$2.log" 2>&1
  cp dist/mlx_omarchy-*.whl "$OUT/wheel-$2.whl"
  basename "$OUT/wheel-$2.whl"
}

git fetch --quiet origin main wave/CoopmatIlpChains
test "$(git rev-parse origin/wave/CoopmatIlpChains)" = "$CAND_SHA" || { echo "SHA-MISMATCH cand"; exit 8; }
test "$(git rev-parse origin/main)" = "$BASE_SHA" || { echo "SHA-MISMATCH base"; exit 8; }
echo "[cficB] candidate=$(build_wheel_at "$CAND_SHA" cand)"
echo "[cficB] base=$(build_wheel_at "$BASE_SHA" base)"

make_venv() {  # cell wheel
  rm -rf "$VROOT/$1"
  python3 -m venv "$VROOT/$1"
  "$VROOT/$1/bin/pip" install -q --no-cache-dir "$OUT/wheel-$2.whl"
  "$VROOT/$1/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  local lib="$VROOT/$1/lib/python3.14/site-packages/mlx/lib/libmlx.so"
  local member loaded
  member=$(unzip -p "$OUT/wheel-$2.whl" mlx/lib/libmlx.so | sha256sum | cut -d' ' -f1)
  loaded=$(sha256sum "$lib" | cut -d' ' -f1)
  echo "[gate] $1 member=$member loaded=$loaded"
  [ "$member" = "$loaded" ] || { echo "FATAL payload mismatch $1"; exit 8; }
}
make_venv base base
make_venv cand cand

run_cell() {  # driver cell python label
  local driver=$1 cell=$2 py=$3 label=$4
  local run="$OUT/matrix/$label"
  mkdir -p "$run"
  if [ "$driver" = stock ]; then
    env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json MLX_DISABLE_COMPILE=1 \
      timeout 2400 scripts/bench_matrix.py --mode run \
      --python "$py" --wheel "$OUT/wheel-$cell.whl" \
      --host-label "jwm1-$label" --timeout 900 \
      --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
      --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
      --out "$run/matrix.json" > "$run/matrix.log" 2>&1
  else
    env MLX_DISABLE_COMPILE=1 \
      timeout 2400 scripts/bench_matrix.py --mode run \
      --python "$py" --wheel "$OUT/wheel-$cell.whl" \
      --host-label "jwm1-$label" --timeout 900 \
      --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
      --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
      --out "$run/matrix.json" > "$run/matrix.log" 2>&1
  fi
  local ok
  ok=$(grep -c 'verified=match' "$run/matrix.log" || true)
  echo "[cficB] $label: verified=match x${ok}"
}

# Discarded warmup, then 3 reps x {fork,stock} x {base,cand}.
run_cell fork base "$VROOT/base/bin/python" warmup
for rep in 1 2 3; do
  for driver in fork stock; do
    run_cell "$driver" base "$VROOT/base/bin/python" "r${rep}-${driver}-base"
    run_cell "$driver" cand "$VROOT/cand/bin/python" "r${rep}-${driver}-cand"
  done
done
echo "[cficB] MATRIX-OK"
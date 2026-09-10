#!/usr/bin/env bash
set -euo pipefail

root=${1:-$HOME/src/mlx-PrefillClose}
label=$2
icd=${3:-}
scratch_home=$HOME/fakehome-PrefillClose
out="$root/receipts/2026-09-10-prefill-close/batching/$label"
base_wheel=$(printf '%s\n' "$root"/receipts/2026-09-10-prefill-close/wheels/main/*.whl)
cand_wheel=$(printf '%s\n' "$root"/receipts/2026-09-10-prefill-close/wheels/gated/*.whl)
mkdir -p "$out"

run_matrix() {
  local side=$1 pair=$2 python=$3 wheel=$4
  local -a env_args=(HOME="$scratch_home" MLX_DISABLE_COMPILE=1 AGX_SIMDMAT=1)
  if [[ -n "$icd" ]]; then
    env_args+=(VK_DRIVER_FILES="$icd")
  fi
  env "${env_args[@]}" flock -w 1800 /tmp/m1-gpu.lock timeout 900 \
    "$python" "$root/scripts/bench_matrix.py" --mode run --python "$python" \
    --host-label jwm1-linux --wheel "$wheel" \
    --select short-decode-32 --select long-decode-128 \
    --select longctx-1024-decode-32 --timeout 600 \
    --out "$out/pair-${pair}-${side}.json" \
    > "$out/pair-${pair}-${side}.log" 2>&1
}

run_matrix base 1 "$root/.venv-base/bin/python" "$base_wheel"
run_matrix cand 1 "$root/.venv-cand/bin/python" "$cand_wheel"
run_matrix cand 2 "$root/.venv-cand/bin/python" "$cand_wheel"
run_matrix base 2 "$root/.venv-base/bin/python" "$base_wheel"
run_matrix base 3 "$root/.venv-base/bin/python" "$base_wheel"
run_matrix cand 3 "$root/.venv-cand/bin/python" "$cand_wheel"

"$root/.venv-cand/bin/python" \
  "$root/receipts/2026-09-10-prefill-close/summarize_pairs.py" "$out"

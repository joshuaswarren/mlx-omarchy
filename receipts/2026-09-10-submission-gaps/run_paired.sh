#!/usr/bin/env bash
set -euo pipefail

root=/home/joshuawarren/src/mlx-SubmissionGaps
out=${1:-/tmp/submission-gaps-paired-fork}
icd=${2-}
base_wheel=$(printf '%s\n' "$root"/.baseline-current/dist/*.whl)
cand_wheel=$(printf '%s\n' "$root"/dist/*.whl)
mkdir -p "$out"
exec timeout 1800 flock -w 1800 /tmp/m1-gpu.lock /bin/bash -c '
  set -euo pipefail
  run_matrix() {
    local side=$1 pair=$2 python=$3 wheel=$4 simdmat=1
    if [[ -n "'$icd'" ]]; then
      export VK_DRIVER_FILES="'$icd'"
      simdmat=0
    else
      unset VK_DRIVER_FILES
    fi
    env HOME=/home/joshuawarren HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
      MESA_SHADER_CACHE_DISABLE=true AGX_SIMDMAT="$simdmat" \
      timeout 900 "$python" "'$root'/receipts/2026-09-10-submission-gaps/bench_with_identity.py" \
      --mode run --python "$python" --host-label jwm1-linux --wheel "$wheel" \
      --select short-decode-32 --select long-decode-128 \
      --select longctx-1024-decode-32 --timeout 600 \
      --out "'$out'/$pair-$side.json" > "'$out'/$pair-$side.log" 2>&1
  }
  run_matrix base warmup "'$root'/.venv-current-main/bin/python" "'$base_wheel'"
  run_matrix cand warmup "'$root'/.venv-candidate/bin/python" "'$cand_wheel'"
  run_matrix base pair-1 "'$root'/.venv-current-main/bin/python" "'$base_wheel'"
  run_matrix cand pair-1 "'$root'/.venv-candidate/bin/python" "'$cand_wheel'"
  run_matrix cand pair-2 "'$root'/.venv-candidate/bin/python" "'$cand_wheel'"
  run_matrix base pair-2 "'$root'/.venv-current-main/bin/python" "'$base_wheel'"
  run_matrix base pair-3 "'$root'/.venv-current-main/bin/python" "'$base_wheel'"
  run_matrix cand pair-3 "'$root'/.venv-candidate/bin/python" "'$cand_wheel'"
'

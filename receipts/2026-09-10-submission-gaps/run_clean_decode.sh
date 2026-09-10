#!/usr/bin/env bash
set -euo pipefail
root=/home/joshuawarren/src/mlx-SubmissionGaps
out=${1:-/tmp/submission-gaps-clean-decode}
base_wheel=$(printf '%s\n' "$root"/.baseline-main/dist/*.whl)
cand_wheel=$(printf '%s\n' "$root"/dist/*.whl)
mkdir -p "$out"

exec timeout 1800 flock -w 2400 /tmp/m1-gpu.lock /bin/bash -c '
  set -euo pipefail
  run_one() {
    local side=$1 run=$2 python=$3 wheel=$4
    env -u VK_DRIVER_FILES HOME=/home/joshuawarren HF_HUB_OFFLINE=1 \
      MLX_DISABLE_COMPILE=1 MESA_SHADER_CACHE_DISABLE=true AGX_SIMDMAT=1 \
      timeout 900 "$python" \
      "'$root'/receipts/2026-09-10-submission-gaps/bench_with_identity.py" \
      --mode run --python "$python" --host-label jwm1-linux --wheel "$wheel" \
      --select longctx-1024-decode-32 --timeout 600 \
      --out "'$out'/$run-$side.json" > "'$out'/$run-$side.log" 2>&1
  }
  run_one base warmup "'$root'/.venv-baseline/bin/python" "'$base_wheel'"
  run_one cand warmup "'$root'/.venv-candidate/bin/python" "'$cand_wheel'"
  run_one base pair-1 "'$root'/.venv-baseline/bin/python" "'$base_wheel'"
  run_one cand pair-1 "'$root'/.venv-candidate/bin/python" "'$cand_wheel'"
  run_one cand pair-2 "'$root'/.venv-candidate/bin/python" "'$cand_wheel'"
  run_one base pair-2 "'$root'/.venv-baseline/bin/python" "'$base_wheel'"
  run_one base pair-3 "'$root'/.venv-baseline/bin/python" "'$base_wheel'"
  run_one cand pair-3 "'$root'/.venv-candidate/bin/python" "'$cand_wheel'"
'
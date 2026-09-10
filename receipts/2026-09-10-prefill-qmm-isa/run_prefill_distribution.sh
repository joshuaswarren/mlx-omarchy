#!/bin/sh
set -eu
ROOT=/home/joshuawarren/src/mlx-GemvNativeForm
PY=/home/joshuawarren/src/mlx-PrefillQmmIsa/.venv-accept/bin/python
OUT=/tmp/prefill-qmm-isa-distribution-ec61b29
mkdir -p "$OUT/m262" "$OUT/m1053"
exec flock -w 2400 /tmp/m1-gpu.lock timeout 1800 env \
  VK_ICD_FILENAMES=/tmp/asahi_coopmat_icd.json \
  AGX_SIMDMAT=1 HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
  MESA_SHADER_CACHE_DISABLE=true \
  /bin/sh -c '
    for spec in m262:/tmp/prompt-262.txt m1053:/tmp/prompt-1053.txt; do
      label=${spec%%:*}; prompt=${spec#*:}
      MLX_OMARCHY_GPU_PROFILE="'$OUT'/$label/profile.jsonl" \
        "'$PY'" "'$ROOT'/scripts/profile_generate.py" \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --prompt "$(cat "$prompt")" --max-tokens 2 --temp 0 --seed 0 \
        --markers "'$OUT'/$label/markers.jsonl" \
        > "'$OUT'/$label/run.log" 2>&1
    done
  '

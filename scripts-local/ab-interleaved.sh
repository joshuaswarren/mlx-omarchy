#!/usr/bin/env bash
# Interleaved 3-round A/B: base env vs default env, one lock hold.
# Usage: ab-interleaved.sh <root> <model> <tree> <venv-src>
set -euo pipefail
ROOT=${1:?}; MODEL=${2:?}; TREE=${3:?}; VENV_SRC=${4:-}
WHEEL=$(echo "$ROOT"/dist/mlx_omarchy-*.whl)
VENV="$ROOT/venv-run"
py="$VENV/bin/python"

rm -f "$ROOT"/ab-{base,n16}.jsonl "$ROOT"/ab-e2e.jsonl
for round in 1 2 3; do
  env MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 MLX_OMARCHY_QMM_COOPMAT_N16_WG_PER_CORE=0 \
    flock -w 60 /tmp/m1-gpu.lock "$py" "$ROOT/probe.py" >> "$ROOT/ab-base.jsonl" 2>&1
  env MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 \
    flock -w 60 /tmp/m1-gpu.lock "$py" "$ROOT/probe.py" >> "$ROOT/ab-n16.jsonl" 2>&1
  echo "round $round probes done"
done
echo "== interleaved e2e ctx262 x3 =="
for round in 1 2 3; do
  for mode in base n16; do
    if [[ $mode == base ]]; then E=MLX_OMARCHY_QMM_COOPMAT_N16_WG_PER_CORE=0; else E=""; fi
    out=$(env MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 $E \
      flock -w 60 /tmp/m1-gpu.lock "$py" "$TREE/scripts/bench_decode.py" \
        --model "$MODEL" --prompt "$(cat /var/tmp/prompts-ctx262.txt)" --tokens 32 \
        --temp 0 --seed 0 --warmup-tokens 4 --wheel "$WHEEL" 2>&1 | \
      grep -o '"prefill_tps": [0-9.]*\|ids_sha256_16": "[0-9a-f]*' | tr '\n' ' ')
    echo "round$round $mode $out" | tee -a "$ROOT/ab-e2e.jsonl"
  done
done
echo "== done =="

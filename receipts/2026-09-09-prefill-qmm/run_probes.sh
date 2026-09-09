#!/usr/bin/env bash
# Usage: run_probes.sh CHECKOUT LABEL [qmm|dense|both]
set -euo pipefail
ROOT="$1"
LABEL="$2"
MODE="${3:-both}"
OUT="$ROOT/receipts/2026-09-09-prefill-qmm/probes/$LABEL"
PY="$ROOT/.venv-accept/bin/python"
mkdir -p "$OUT"
run_mode() {
  local mode="$1"
  for shape in hidden-hidden hidden-kv hidden-intermediate intermediate-hidden; do
    case "$shape" in
      hidden-hidden) k=896; n=896 ;;
      hidden-kv) k=896; n=128 ;;
      hidden-intermediate) k=896; n=4864 ;;
      intermediate-hidden) k=4864; n=896 ;;
    esac
    for m in 262 1053; do
      stem="$mode-$shape-m$m"
      MLX_OMARCHY_GPU_PROFILE="$OUT/$stem.profile.jsonl" \
        "$PY" "$ROOT/receipts/2026-09-09-prefill-qmm/kernel_probe.py" \
        "$mode" "$shape" "$m" > "$OUT/$stem.run.json"
      "$PY" "$ROOT/receipts/2026-09-09-prefill-qmm/analyze_probe.py" \
        "$OUT/$stem.profile.jsonl" \
        --compute-h "$ROOT/overlay/mlx/backend/omarchy/compute.h" \
        --mode "$mode" --m "$m" --k "$k" --n "$n" \
        > "$OUT/$stem.json"
    done
  done
}
case "$MODE" in
  qmm|dense) run_mode "$MODE" ;;
  both) run_mode qmm; run_mode dense ;;
  *) echo "mode must be qmm, dense, or both" >&2; exit 2 ;;
esac

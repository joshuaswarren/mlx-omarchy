#!/usr/bin/env bash
set -euo pipefail
root="$1"
label="$2"
wheel="$3"
out="$root/receipts/2026-09-10-prefill-close/profiles/$label"
py="$root/.venv-prof/bin/python"
mkdir -p "$out"
"$py" "$root/scripts/mlx_provenance.py" --expect-wheel "$wheel" > "$out/provenance.txt"
for shape in hidden-hidden hidden-kv hidden-intermediate intermediate-hidden; do
  case "$shape" in
    hidden-hidden) k=896; n=896 ;;
    hidden-kv) k=896; n=128 ;;
    hidden-intermediate) k=896; n=4864 ;;
    intermediate-hidden) k=4864; n=896 ;;
  esac
  for m in 262 1053; do
    stem="qmm-$shape-m$m"
    target=QmmTileRbF16
    if ((m >= 1024)); then target=QmmTileRbPreciseF16; fi
    MLX_OMARCHY_GPU_PROFILE="$out/$stem.profile.jsonl" \
      "$py" "$root/receipts/2026-09-09-prefill-qmm/kernel_probe.py" \
      qmm "$shape" "$m" > "$out/$stem.run.json"
    "$py" "$root/receipts/2026-09-10-prefill-close/analyze_qmm_profile.py" \
      "$out/$stem.profile.jsonl" \
      --compute-h "$root/overlay/mlx/backend/omarchy/compute.h" \
      --target "$target" --m "$m" --k "$k" --n "$n" > "$out/$stem.json"
  done
done

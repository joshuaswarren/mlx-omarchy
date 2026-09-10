#!/usr/bin/env bash
set -euo pipefail

label="$1"
root="$HOME/src/mlx-prefill-qmm-native"
out="$root/receipts/2026-09-10-prefill-qmm-native/profiles/$label"
py="$root/.venv-accept/bin/python"
mkdir -p "$out"
cd "$root"
unset VK_DRIVER_FILES VK_ICD_FILENAMES
export AGX_SIMDMAT=1
{
  hostname
  git rev-parse HEAD
  sha256sum dist/mlx_omarchy-*.whl
  "$py" - <<'PY'
import mlx.core as mx
print(f"mlx_version={mx.__version__}")
print(f"cooperative_matrix_f32_8={mx.device_info()['cooperative_matrix_f32_8']}")
print(mx.device_info())
PY
} | tee "$out/provenance.txt"

for shape in hidden-hidden hidden-kv hidden-intermediate intermediate-hidden; do
  case "$shape" in
    hidden-hidden) k=896; n=896 ;;
    hidden-kv) k=896; n=128 ;;
    hidden-intermediate) k=896; n=4864 ;;
    intermediate-hidden) k=4864; n=896 ;;
  esac
  for m in 262 1053; do
    stem="$shape-m$m"
    MLX_OMARCHY_GPU_PROFILE="$out/$stem.profile.jsonl" timeout 600 \
      "$py" "$root/receipts/2026-09-10-prefill-qmm-native/kernel_probe.py" \
      "$shape" "$m" > "$out/$stem.run.json"
    "$py" "$root/receipts/2026-09-10-prefill-qmm-native/analyze_probe.py" \
      "$out/$stem.profile.jsonl" \
      --compute-h "$root/overlay/mlx/backend/omarchy/compute.h" \
      --m "$m" --k "$k" --n "$n" > "$out/$stem.json"
    cat "$out/$stem.run.json"
    cat "$out/$stem.json"
  done
done

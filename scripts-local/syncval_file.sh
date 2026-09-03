#!/bin/sh
# Sync-val retry with the documented settings-file mechanism.
set -eu
cat > "$HOME/benchq/vk_layer_settings.txt" <<EOF
khronos_validation.syncval = true
khronos_validation.validate_core = true
khronos_validation.check_image_support = false
EOF
V="$HOME/venv-bqm1-gate/bin/python"
Q4="$HOME/cached-model-marker"
LOG="$HOME/benchq/logs/syncval-file.log"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
P="What is the capital of France? Answer in one word."
env -u MLX_DISABLE_COMPILE \
  MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  VK_LAYER_SETTINGS_PATH="$HOME/benchq/vk_layer_settings.txt" \
  "$V" -m mlx_lm generate \
  --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
  > "$LOG" 2>&1 || echo "rc=$?"
echo "== generated =="
sed -n "2,3p" "$LOG"
echo "== validation messages =="
grep -c -i -E "SYNC-HAZARD|HAZARD|VUID|Validation" "$LOG" || true
grep -i -E "SYNC-HAZARD|HAZARD|VUID" "$LOG" | head -15
echo "== abort line =="
grep -E "Cos argument|Sigmoid" "$LOG" | head -3

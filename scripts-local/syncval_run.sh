#!/bin/sh
# Corrupting case under Vulkan sync-val, then GPU-assisted validation.
set -eu
cd ~/src/mlx-omarchy-bqm1-build
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
VENV="$HOME/venv-bqm1-gate"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir dist/mlx_omarchy-*.whl
  "$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q --no-cache-dir -r "$REQS"
fi
WHEEL=$(ls dist/mlx_omarchy-*.whl)
MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
LOADED=$(sha256sum "$LIB" | cut -d" " -f1)
echo "[gate] wheel=$(basename "$WHEEL")"
echo "[gate] member=$MEMBER"
echo "[gate] loaded=$LOADED"
[ "$MEMBER" = "$LOADED" ] || { echo "FATAL: payload mismatch" >&2; exit 8; }
"$VENV/bin/python" -c "import mlx.core as mx; print('gate', mx.__version__)"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
P="What is the capital of France? Answer in one word."
run_case() {
  TAG=$1
  shift
  LOG="$HOME/benchq/logs/syncval-$TAG.log"
  echo "=== $TAG ==="
  env -u MLX_DISABLE_COMPILE \
    MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
    "$@" \
    "$VENV/bin/python" -m mlx_lm generate \
    --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
    > "$LOG" 2>&1 || echo "rc=$?"
  echo "rc recorded; generated text line:"
  sed -n "2,3p" "$LOG"
  echo "hazard/validation lines:"
  grep -c -i -E "hazard|SYNCVAL|validation" "$LOG" || true
  grep -i -E "hazard|SYNCVAL" "$LOG" | head -12
}
run_case baseline
run_case syncval \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  khronos_validation.syncval=true
run_case gpuav \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  khronos_validation.gpuav=true

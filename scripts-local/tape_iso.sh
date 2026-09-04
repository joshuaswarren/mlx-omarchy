#!/bin/sh
# Tape-layer isolation (TapeLayerIsolation protocol) at 7c25feb+.
set -eu
cd ~/src/mlx-omarchy-bqm1-build
git fetch --quiet origin main
git checkout --quiet --detach origin/main
echo "testing commit $(git rev-parse --short HEAD)"
OUTDIR="$HOME/benchq/wheels/tapeiso"
if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
  sh scripts/build-wheel.sh 2>&1 | grep -E "source commit|sha256"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
fi
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
VENV="$HOME/venv-bqm1-tapeiso"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir "$OUTDIR"/*.whl
  "$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q --no-cache-dir -r "$REQS"
fi
WHEEL=$(ls "$OUTDIR"/*.whl)
MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
LOADED=$(sha256sum "$LIB" | cut -d" " -f1)
echo "[gate] loaded=$LOADED member=$MEMBER"
[ "$MEMBER" = "$LOADED" ] || { echo "FATAL: payload mismatch" >&2; exit 8; }
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
run_step() {
  TAG=$1
  shift
  LOG="$HOME/benchq/logs/tapeiso-$TAG.log"
  echo "=== STEP $TAG ==="
  env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
    MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 "$@" \
    "$VENV/bin/python" scripts/differential_compile.py --mode realpath \
    --model "$MODEL" --prompt "What is the capital of France?" --steps 32 \
    --out-dir "$HOME/benchq/tapeiso-$TAG" > "$LOG" 2>&1
  RC=$?
  echo "STEP $TAG rc=$RC"
  grep -i -E "switch|TAPE_" "$LOG" | head -2
  grep -i -E "diverg|abort|Refused|RuntimeError|bitwise" "$LOG" | head -4
  tail -2 "$LOG"
}
run_step 0-baseline
run_step 1-per-node MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1
run_step 2-full-barriers MLX_OMARCHY_TAPE_FULL_BARRIERS=1
run_step 3-no-reuse MLX_OMARCHY_TAPE_NO_REUSE=1

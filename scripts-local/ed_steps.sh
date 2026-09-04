#!/bin/sh
# TapeLayerIsolation steps E and D at 4c44e19: 5 native runs each.
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
git fetch --quiet origin 4c44e19 2>/dev/null || git fetch --quiet origin main
git checkout --quiet --detach 4c44e19 2>/dev/null || git checkout --quiet --detach origin/main
echo "testing commit $(git rev-parse --short HEAD)"
OUTDIR="$HOME/benchq/wheels/ed-steps"
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
  sh scripts/build-wheel.sh 2>&1 | grep -E "source commit|sha256:"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
fi
VENV="$HOME/venv-bqm1-edsteps"
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
"$VENV/bin/python" -c "import mlx.core as mx; print('mx.__version__', mx.__version__)"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
run_step() {
  TAG=$1
  shift
  GONE=0
  ABORT=0
  ARMED=0
  i=1
  while [ "$i" -le 5 ]; do
    LOG="$HOME/benchq/logs/ed-$TAG.run$i.log"
    if env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
        MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 "$@" \
        "$V" -m mlx_lm generate \
        --model "$Q4" --prompt "$(cat "$HOME/benchq/prompt-france.txt")" \
        --max-tokens 32 --temp 0 --seed 0 > "$LOG" 2>&1 \
        && grep -q "Paris" "$LOG"; then
      GONE=$((GONE + 1))
    else
      ABORT=$((ABORT + 1))
      grep -E "Cos argument" "$LOG" | head -1
    fi
    grep -q -E "NO_BUFFER_CACHE active|sync every|SYNC_EVERY|drain" "$LOG" && ARMED=$((ARMED + 1))
    i=$((i + 1))
  done
  echo "STEP $TAG: gone=$GONE abort=$ABORT armed_marker_runs=$ARMED"
}
run_step E MLX_OMARCHY_NO_BUFFER_CACHE=1
run_step D MLX_OMARCHY_TAPE_SYNC_EVERY=1

#!/bin/sh
# FusedRope legs: BASELINE (main tip) vs FUSED (3c32cce), diag wheels.
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
git fetch --quiet origin main
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
P="What is the capital of France? Answer in one word."

build_side() {
  SIDE=$1
  REF=$2
  cd "$REPO"
  git checkout --quiet --detach "$(git rev-parse "$REF")"
  echo "== $SIDE at $(git rev-parse --short HEAD) =="
  OUTDIR="$HOME/benchq/wheels/rope-$SIDE"
  if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
    sh scripts/build-wheel.sh --diagnostics 2>&1 | grep -E "source commit|sha256:"
    mkdir -p "$OUTDIR"
    cp dist/mlx_omarchy-*.whl "$OUTDIR"/
  fi
  VENV="$HOME/venv-bqm1-rope-$SIDE"
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
  echo "[gate] side=$SIDE member=$MEMBER loaded=$LOADED"
  [ "$MEMBER" = "$LOADED" ] || { echo "FATAL: $SIDE payload mismatch" >&2; exit 8; }
}

build_side baseline origin/main
build_side fused rope-provisional-bqm1

echo "== TOK/S legs (bench_decode, pinned 64, 4 runs discard first) =="
for SIDE in baseline fused; do
  VENV="$HOME/venv-bqm1-rope-$SIDE"
  WHEEL=$(ls "$HOME/benchq/wheels/rope-$SIDE"/*.whl)
  r=1
  while [ "$r" -le 4 ]; do
    LOG="$HOME/benchq/logs/rope-tok-$SIDE.run$r.log"
    if [ "$r" = 1 ]; then
      MLX_DISABLE_COMPILE=1 "$VENV/bin/python" "$REPO/scripts/bench_decode.py" \
        --model "$Q4" --prompt "$P" --tokens 64 --wheel "$WHEEL" > "$LOG" 2>&1
      echo "tok $SIDE run$r (discarded): $(grep -c 'decode' "$LOG") lines"
    else
      MLX_DISABLE_COMPILE=1 "$VENV/bin/python" "$REPO/scripts/bench_decode.py" \
        --model "$Q4" --prompt "$P" --tokens 64 --wheel "$WHEEL" > "$LOG" 2>&1
      echo "tok $SIDE run$r: $(grep 'decode ' "$LOG" | head -1)"
    fi
    r=$((r + 1))
  done
done

echo "== SUBMISSION legs (fragmentation_probe, 48 tokens, 4 runs discard first) =="
for SIDE in baseline fused; do
  VENV="$HOME/venv-bqm1-rope-$SIDE/bin/python"
  r=1
  while [ "$r" -le 4 ]; do
    LOG="$HOME/benchq/logs/rope-frag-$SIDE.run$r.log"
    D="$HOME/benchq/rope-frag-$SIDE.run$r"
    mkdir -p "$D"
    if [ "$r" = 1 ]; then
      MLX_DISABLE_COMPILE=1 "$VENV" "$REPO/scripts/fragmentation_probe.py" \
        --model "$Q4" --prompt "$P" --max-tokens 48 --seed 0 \
        --events "$D/events.jsonl" --markers "$D/markers.jsonl" > "$LOG" 2>&1
      echo "frag $SIDE run$r (discarded)"
    else
      MLX_DISABLE_COMPILE=1 "$VENV" "$REPO/scripts/fragmentation_probe.py" \
        --model "$Q4" --prompt "$P" --max-tokens 48 --seed 0 \
        --events "$D/events.jsonl" --markers "$D/markers.jsonl" > "$LOG" 2>&1
      echo "frag $SIDE run$r: $(grep -i -E "submissions|evals|dispatches|tokens" "$LOG" | tail -3 | tr '\n' ' ')"
    fi
    r=$((r + 1))
  done
done
echo "ROPE RUNS DONE $(date '+%F %T')"

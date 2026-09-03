#!/bin/sh
# README 8-core column re-measure: current-main wheel, pinned-length decode.
# usage: column_measure.sh <full-sha>
set -eu
SHA=$1
cd ~/src/mlx-omarchy-bqm1-build
OUTDIR="$HOME/benchq/wheels/sha-column"
if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
  git fetch --quiet origin "$SHA" || true
  git checkout --quiet --detach "$SHA"
  echo "== build $SHA =="
  sh scripts/build-wheel.sh 2>&1 | grep "\[receipt\]"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
fi
VENV="$HOME/venv-bqm1-column"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q "$OUTDIR"/*.whl
  "$VENV/bin/pip" install -q --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q -r ~/benchq/support-reqs.txt
fi
"$VENV/bin/python" -c "import mlx.core as mx; print('column', mx.__version__)"
echo "== settle =="
sleep 45
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
P="What is the capital of France? Answer in one word."
echo "START $(date '+%F %T') load: $(uptime)"
MLX_DISABLE_COMPILE=1 "$VENV/bin/python" "$HOME/src/mlx-omarchy-bqm1-build/scripts/bench_decode.py" \
  --model "$Q4" --prompt "$P" --tokens 64 2>&1 | tee ~/benchq/logs/column-q4.log
MLX_DISABLE_COMPILE=1 "$VENV/bin/python" "$HOME/src/mlx-omarchy-bqm1-build/scripts/bench_decode.py" \
  --model "$BF16" --prompt "$P" --tokens 64 2>&1 | tee ~/benchq/logs/column-bf16.log
echo "END $(date '+%F %T') load: $(uptime)"

#!/bin/sh
# Build (if needed) one commit's wheel on jwm1 and run the decode legs.
# usage: sha_measure.sh <full-sha> <tag> [on|off]   on = compile enabled
set -eu
SHA=$1
TAG=$2
CMODE=${3:-off}
cd ~/src/mlx-omarchy-bqm1-build
OUTDIR="$HOME/benchq/wheels/sha-$TAG"
if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
  git fetch --quiet origin "$SHA" || true
  git checkout --quiet --detach "$SHA"
  echo "== build $SHA ($TAG) =="
  sh scripts/build-wheel.sh 2>&1 | grep "\[receipt\]"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
else
  echo "== skip build $SHA ($TAG), wheel present =="
fi
VENV="$HOME/venv-bqm1-$TAG"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q "$OUTDIR"/*.whl
  "$VENV/bin/pip" install -q --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q -r ~/benchq/support-reqs.txt
fi
"$VENV/bin/python" -c "import mlx.core as mx; print('$TAG', mx.__version__)"
echo "== settle =="
sleep 45
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
PROMPT=~/benchq/prompt-france.txt
echo "START $(date '+%F %T') compile=$CMODE load: $(uptime)"
i=1
while [ "$i" -le 5 ]; do
  if [ "$CMODE" = on ]; then
    env -u MLX_DISABLE_COMPILE "$VENV/bin/python" -m mlx_lm generate \
      --model "$BF16" --prompt "$(cat "$PROMPT")" \
      --max-tokens 32 --temp 0 --seed 0 > ~/benchq/logs/$TAG-BF16.run$i.log 2>&1
    env -u MLX_DISABLE_COMPILE "$VENV/bin/python" -m mlx_lm generate \
      --model "$Q4" --prompt "$(cat "$PROMPT")" \
      --max-tokens 32 --temp 0 --seed 0 > ~/benchq/logs/$TAG-Q4.run$i.log 2>&1
  else
    MLX_DISABLE_COMPILE=1 "$VENV/bin/python" -m mlx_lm generate \
      --model "$BF16" --prompt "$(cat "$PROMPT")" \
      --max-tokens 32 --temp 0 --seed 0 > ~/benchq/logs/$TAG-BF16.run$i.log 2>&1
    MLX_DISABLE_COMPILE=1 "$VENV/bin/python" -m mlx_lm generate \
      --model "$Q4" --prompt "$(cat "$PROMPT")" \
      --max-tokens 32 --temp 0 --seed 0 > ~/benchq/logs/$TAG-Q4.run$i.log 2>&1
  fi
  echo "done $TAG run$i"
  i=$((i + 1))
done
echo "END $(date '+%F %T') load: $(uptime)"
python3 ~/benchq/parse_legs.py \
  'Prompt: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  'Generation: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  "$TAG-BF16" "$TAG-Q4"

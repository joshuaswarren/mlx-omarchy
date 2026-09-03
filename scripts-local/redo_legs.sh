#!/bin/sh
# Redo queue: re-measure every leg whose venv was poisoned by the
# support-reqs direct-URL pin. Wheels are reused (their hashes were
# recorded at build time); venvs are rebuilt clean and gated on the
# loaded libmlx.so matching the wheel member.
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"

provision() {
  TAG=$1
  WHEELDIR=$2
  VENV="$HOME/venv-$TAG"
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir "$WHEELDIR"/*.whl
  "$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q --no-cache-dir -r "$REQS"
  WHEEL=$(ls "$WHEELDIR"/*.whl)
  MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
  LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
  LOADED=$(sha256sum "$LIB" | cut -d" " -f1)
  if [ "$MEMBER" != "$LOADED" ]; then
    echo "FATAL $TAG: loaded libmlx $LOADED != wheel member $MEMBER" >&2
    exit 8
  fi
  echo "[gate] $TAG OK wheel=$(basename "$WHEEL") libmlx=$LOADED"
}

provision r3fd "$HOME/benchq/wheels/pollab-base"
provision r4ea "$HOME/benchq/wheels/pollab-fix"
provision rff4 "$HOME/benchq/wheels/sha-DORD"
provision rcol "$HOME/benchq/wheels/sha-column"

BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
PROMPT="$HOME/benchq/prompt-france.txt"

echo "START $(date '+%F %T') load: $(uptime)"
for PAIR in "r3fd DBASE" "r4ea DFIX" "rff4 DORD"; do
  set -- $PAIR
  V=$1
  for P in BF16 Q4; do
    if [ "$P" = BF16 ]; then M="$BF16"; else M="$Q4"; fi
    sh ~/benchq/run_leg.sh ~/venv-$V/bin/python "$M" "$PROMPT" - 5 "R$2-$P"
  done
done
echo "MID $(date '+%F %T')"
python3 ~/benchq/parse_legs.py \
  'Prompt: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  'Generation: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  RDBASE-BF16 RDBASE-Q4 RDFIX-BF16 RDFIX-Q4 RDORD-BF16 RDORD-Q4

echo "== column re-measure (clean) =="
Q="$HOME/models/Qwen2.5-0.5B-Instruct-4bit-mlx"
B="$HOME/models/Qwen2.5-0.5B-Instruct-bf16-mlx"
MLX_DISABLE_COMPILE=1 ~/venv-rcol/bin/python "$REPO/scripts/bench_decode.py" \
  --model "$Q4" --prompt "What is the capital of France? Answer in one word." --tokens 64 \
  > ~/benchq/logs/column-clean-q4.log 2>&1
MLX_DISABLE_COMPILE=1 ~/venv-rcol/bin/python "$REPO/scripts/bench_decode.py" \
  --model "$BF16" --prompt "What is the capital of France? Answer in one word." --tokens 64 \
  > ~/benchq/logs/column-clean-bf16.log 2>&1
echo "--- 4-bit ---"; cat ~/benchq/logs/column-clean-q4.log
echo "--- bf16 ---"; cat ~/benchq/logs/column-clean-bf16.log
echo "END $(date '+%F %T') load: $(uptime)"

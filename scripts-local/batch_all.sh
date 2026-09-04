#!/bin/sh
# Combined queue: (1) batching before/after 7c25feb -> 7c3d6b4 (eager tok/s
# + submissions/token), (2) TapeLayerIsolation tree at 7c3d6b4,
# (3) compile-ON abort acceptance. Provenance gate on every venv.
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
git fetch --quiet origin main
git checkout --quiet --detach 7c3d6b4 2>/dev/null || { git fetch --quiet origin 7c3d6b4 2>/dev/null || true; git checkout --quiet --detach FETCH_HEAD; }
AFTER_COMMIT=$(git rev-parse --short HEAD)
echo "== AFTER commit: $AFTER_COMMIT =="

REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"

gate() {
  VENV=$1
  WHEELDIR=$2
  WHEEL=$(ls "$WHEELDIR"/*.whl)
  MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
  LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
  LOADED=$(sha256sum "$LIB" | cut -d" " -f1)
  VER=$("$VENV/bin/pip" show mlx-omarchy | grep ^Version | cut -d" " -f2)
  echo "[gate] venv=$VENV ver=$VER libmlx=$LOADED"
  [ "$MEMBER" = "$LOADED" ] || { echo "FATAL: $VENV payload mismatch (member $MEMBER)" >&2; exit 8; }
}

AFTERDIR="$HOME/benchq/wheels/batch-after"
if ! ls "$AFTERDIR"/*.whl >/dev/null 2>&1; then
  sh scripts/build-wheel.sh 2>&1 | grep -E "source commit|sha256:"
  mkdir -p "$AFTERDIR"
  cp dist/mlx_omarchy-*.whl "$AFTERDIR"/
fi
if [ ! -x "$HOME/venv-bqm1-after/bin/python" ]; then
  python3 -m venv "$HOME/venv-bqm1-after"
  "$HOME/venv-bqm1-after/bin/pip" install -q --no-cache-dir "$AFTERDIR"/*.whl
  "$HOME/venv-bqm1-after/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$HOME/venv-bqm1-after/bin/pip" install -q --no-cache-dir -r "$REQS"
fi
gate "$HOME/venv-bqm1-after" "$AFTERDIR"
gate "$HOME/venv-bqm1-tapeiso" "$HOME/benchq/wheels/tapeiso"

BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
P="What is the capital of France?"

echo "== PHASE 1: eager legs before/after =="
for SIDE in tapeiso after; do
  TAG=BEFORE
  [ "$SIDE" = after ] && TAG=AFTER
  for PR in BF16 Q4; do
    if [ "$PR" = BF16 ]; then M="$BF16"; else M="$Q4"; fi
    sh ~/benchq/run_leg.sh "$HOME/venv-bqm1-$SIDE/bin/python" "$M" \
      "$HOME/benchq/prompt-france.txt" - 5 "BAT-$TAG-$PR"
  done
done
python3 ~/benchq/parse_legs.py \
  'Prompt: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  'Generation: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  BAT-BEFORE-BF16 BAT-BEFORE-Q4 BAT-AFTER-BF16 BAT-AFTER-Q4

echo "== PHASE 3: compile-ON abort acceptance (0/20 + Paris) =="
PASS=0
ABORTED=0
i=1
while [ "$i" -le 20 ]; do
  LOG="$HOME/benchq/logs/batch-abort.run$i.log"
  if env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
      "$HOME/venv-bqm1-after/bin/python" -m mlx_lm generate \
      --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
      > "$LOG" 2>&1 && grep -q "Paris" "$LOG"; then
    PASS=$((PASS + 1)); echo "run$i PASS"
  else
    ABORTED=1
    echo "run$i FAIL - abort at run $i; verbatim:"
    grep -E "Cos argument|RuntimeError" "$LOG" | head -2
    echo "ABORT_PROTOCOL: FAIL at run $i (pass=$PASS)"
    break
  fi
  i=$((i + 1))
done
[ "$ABORTED" = "0" ] && echo "ABORT_PROTOCOL: $PASS/20 pass, 0 aborts"

echo "== PHASE 2 prep: diagnostics wheel + profiles =="
DIAGDIR="$HOME/benchq/wheels/batch-diag"
if ! ls "$DIAGDIR"/*.whl >/dev/null 2>&1; then
  sh scripts/build-wheel.sh --diagnostics 2>&1 | grep -E "source commit|sha256:"
  mkdir -p "$DIAGDIR"
  cp dist/mlx_omarchy-*.whl "$DIAGDIR"/
fi
DVENV="$HOME/venv-bqm1-batchdiag"
if [ ! -x "$DVENV/bin/python" ]; then
  python3 -m venv "$DVENV"
  "$DVENV/bin/pip" install -q --no-cache-dir "$DIAGDIR"/*.whl
  "$DVENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$DVENV/bin/pip" install -q --no-cache-dir -r "$REQS"
fi
DMEMBER=$(unzip -p "$DIAGDIR"/*.whl mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
DLOADED=$(sha256sum "$DVENV/lib/python3.14/site-packages/mlx/lib/libmlx.so" | cut -d" " -f1)
echo "[gate] diag loaded=$DLOADED member=$DMEMBER"
[ "$DMEMBER" = "$DLOADED" ] || { echo "FATAL: diag payload mismatch" >&2; exit 8; }
PC="Count from 1 to 100."
for PR in bf16 q4; do
  if [ "$PR" = bf16 ]; then M="$BF16"; else M="$Q4"; fi
  sleep 20
  env -u MLX_DISABLE_COMPILE MLX_OMARCHY_GPU_PROFILE="$HOME/benchq/batch-p-$PR.jsonl" \
    "$DVENV/bin/python" "$REPO/scripts/profile_generate.py" \
    --model "$M" --prompt "$PC" --max-tokens 96 --temp 0 --seed 0 \
    --markers "$HOME/benchq/batch-m-$PR.jsonl" 2>&1 | tail -2
done
nohup sh "$HOME/benchq/batch_analyze.sh" >/dev/null 2>&1 &

echo "== PHASE 2: TapeLayerIsolation tree at 7c3d6b4 =="
run_tree() {
  TAG=$1
  shift
  LOG="$HOME/benchq/logs/tapeiso3-$TAG.log"
  RC=0
  env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
    MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 "$@" \
    "$HOME/venv-bqm1-after/bin/python" scripts/differential_compile.py --mode realpath \
    --model "$Q4" --prompt "What is the capital of France?" --steps 32 \
    --out-dir "$HOME/benchq/tapeiso3-$TAG" > "$LOG" 2>&1 || RC=$?
  echo "=== TREE $TAG rc=$RC ==="
  grep -i -E "TAPE_|switch" "$LOG" | head -2
  grep -i -E "diverg|Cos argument|Sigmoid|RuntimeError|bitwise" "$LOG" | head -3
}
run_tree 0-baseline
run_tree 1-per-node MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1
run_tree 2-full-barriers MLX_OMARCHY_TAPE_FULL_BARRIERS=1
run_tree 3-no-reuse MLX_OMARCHY_TAPE_NO_REUSE=1
echo "ALL DONE $(date '+%F %T')"

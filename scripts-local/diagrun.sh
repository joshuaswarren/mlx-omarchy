#!/bin/sh
# Submission-count profiling on a clean diagnostics venv (current main).
set -eu
cd ~/src/mlx-omarchy-bqm1-build
git fetch --quiet origin main || true
git checkout --quiet --detach origin/main
REPO="$HOME/src/mlx-omarchy-bqm1-build"
OUTDIR="$HOME/benchq/wheels/diag-local"
ls "$OUTDIR"/*.whl >/dev/null 2>&1
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
VENV="$HOME/venv-bqm1-diagclean"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir "$OUTDIR"/*.whl
  "$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q --no-cache-dir -r "$REQS"
fi
"$VENV/bin/python" -c "import mlx.core as mx; print('diagclean', mx.__version__)"
SO=$(find "$VENV/lib" -path "*mlx/lib*" -name "libmlx.so" | head -1)
HITS=$(strings "$SO" | grep -c MLX_OMARCHY_GPU_PROFILE || true)
echo "profiling literals in libmlx.so: $HITS"
if [ "$HITS" -lt 1 ]; then
  echo "FATAL: profiling harness not in installed payload; aborting" >&2
  exit 9
fi
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
# Long greedy generation: no EOS suppression in this path, so the prompt
# must produce >= 64 tokens by itself.
P="Count from 1 to 100."
run_one() {
  TAG=$1
  MODEL=$2
  sleep 20
  echo "=== $TAG $(date '+%T') ==="
  MLX_DISABLE_COMPILE=1 MLX_OMARCHY_GPU_PROFILE="$HOME/benchq/diag-p-$TAG.jsonl" \
    "$VENV/bin/python" "$REPO/scripts/profile_generate.py" \
    --model "$MODEL" --prompt "$P" --max-tokens 96 --temp 0 --seed 0 \
    --markers "$HOME/benchq/diag-m-$TAG.jsonl" 2>&1 | tail -2
  python3 "$REPO/scripts/profile_analyze.py" \
    "$HOME/benchq/diag-p-$TAG.jsonl" \
    --compute-h "$REPO/.work/mlx/mlx/backend/omarchy/compute.h" \
    --markers "$HOME/benchq/diag-m-$TAG.jsonl" \
    > "$HOME/benchq/diag-analyze-$TAG.txt" 2>&1
  grep -i -E "submissions|dispatches|decode|prefill|busy|wall|submit|wait|token" \
    "$HOME/benchq/diag-analyze-$TAG.txt" | head -40
}
run_one q4 "$Q4"
run_one bf16 "$BF16"

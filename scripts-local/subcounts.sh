#!/bin/sh
# Task A: submissions/token + dispatches/token + time breakdown, eager mode,
# profiling wheel v0.3.3-diag.1, pinned 64-token generation.
set -eu
cd "$HOME/benchq/wheels"
mkdir -p diag
if ! ls "$HOME"/benchq/wheels/diag/*.whl >/dev/null 2>&1; then
  curl -sL --fail -o diag/mlx_omarchy-0.32.2.dev202609031348+diag-cp314-cp314-linux_aarch64.whl \
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
VENV="$HOME/venv-bqm1-diag5"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir diag/*.whl
  "$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q --no-cache-dir -r "$REQS"
fi
  "$VENV/bin/pip" install -q -r ~/benchq/support-reqs.txt
fi
"$VENV/bin/python" -c "import mlx.core as mx; print('diag', mx.__version__)"
SO=$(find "$VENV/lib" -name "core.cpython-*-aarch64-linux-gnu.so" | head -1)
printf "profiling harness present: "
strings "$SO" | grep -c "MLX_OMARCHY_GPU_PROFILE" || true
REPO="$HOME/src/mlx-omarchy-bqm1-build"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
P="What is the capital of France? Answer in one word."
run_one() {
  TAG=$1
  MODEL=$2
  echo "=== $TAG ==="
  sleep 30
  MLX_DISABLE_COMPILE=1 MLX_OMARCHY_GPU_PROFILE="$HOME/benchq/diag-p-$TAG.jsonl" \
    "$VENV/bin/python" "$REPO/scripts/profile_generate.py" \
    --model "$MODEL" --prompt "$P" --max-tokens 64 --temp 0 --seed 0 \
    --markers "$HOME/benchq/diag-m-$TAG.jsonl" 2>&1 | tail -3
  "$VENV/bin/python" "$REPO/scripts/profile_analyze.py" \
    "$HOME/benchq/diag-p-$TAG.jsonl" \
    --compute-h "$REPO/.work/mlx/mlx/backend/omarchy/compute.h" \
    --markers "$HOME/benchq/diag-m-$TAG.jsonl" \
    > "$HOME/benchq/diag-analyze-$TAG.txt" 2>&1 || \
  python3 "$REPO/scripts/profile_analyze.py" \
    "$HOME/benchq/diag-p-$TAG.jsonl" \
    --markers "$HOME/benchq/diag-m-$TAG.jsonl" \
    > "$HOME/benchq/diag-analyze-$TAG.txt" 2>&1
  grep -i -E "submission|dispatch|token|busy|wall|phase|submit|wait" "$HOME/benchq/diag-analyze-$TAG.txt" | head -30
}
run_one q4 "$Q4"
run_one bf16 "$BF16"

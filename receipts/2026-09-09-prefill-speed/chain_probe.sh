#!/usr/bin/env bash
# Build one commit's release wheel and dump the fused chain programs the
# model dispatches during a 262-token prefill (run under the GPU lock).
set -euo pipefail
COMMIT="$1"
ROOT="$HOME/src/mlx-PrefillSpeed"
cd "$ROOT"
git fetch -q origin wave/PrefillSpeed && git checkout -q "$COMMIT"
unset MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH
rm -rf dist
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 timeout 3600 scripts/build-wheel.sh > /tmp/chain-probe-build.log 2>&1
w=(dist/mlx_omarchy-*.whl)
.venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${w[0]}"
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3)
prompt=$(cat /tmp/profiles-main-keep/main-21d6a56/prompt-long.txt)
for compile in 1 0; do
  echo "== MLX_DISABLE_COMPILE=$compile"
  MLX_DISABLE_COMPILE=$compile MLX_OMARCHY_CHAIN_DUMP=1 .venv-accept/bin/python scripts/bench_decode.py \
    --model "$MODEL" --prompt "$prompt" --tokens 4 --temp 0 --seed 0 --warmup-tokens 0 2>&1 | grep -E "^\[chain\]" | sort | uniq -c | head -20
done

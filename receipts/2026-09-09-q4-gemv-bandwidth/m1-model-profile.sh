#!/usr/bin/env bash
# In-model GPU profile of the Q4 decode: install the checkout's
# diagnostics wheel into its .venv-accept, generate 32 tokens from the
# pinned Qwen2.5-0.5B-Instruct-4bit with MLX_OMARCHY_GPU_PROFILE, and
# analyze the trace (scripts/profile_analyze.py).
#   m1-model-profile.sh CHECKOUT OUT_DIR
# Run under `flock -w 900 /tmp/m1-gpu.lock timeout 3600`.
set -euo pipefail
cd "$1"
out="$2"
mkdir -p "$out"
commit="$(git rev-parse --short=7 HEAD)"
wheel=(wheels/diag/mlx_omarchy-*+diag."$commit"-*.whl)
test "${#wheel[@]}" -eq 1
sha256sum "${wheel[0]}" > "$out/wheel.sha256"
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES
export MESA_SHADER_CACHE_DISABLE=true HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
.venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${wheel[0]}"
version="$(.venv-accept/bin/python -c 'import mlx.core as mx; print(mx.__version__)')"
case "$version" in *"+diag.$commit") ;; *) echo "wheel stamp $version != diag.$commit" >&2; exit 1;; esac
echo "$version" > "$out/version.txt"
model="$(.venv-accept/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen2.5-0.5B-Instruct-4bit', revision='a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3', local_files_only=True))")"
MLX_OMARCHY_GPU_PROFILE="$out/4bit.jsonl" .venv-accept/bin/python scripts/profile_generate.py \
  --model "$model" --prompt 'Explain how a computer executes a program, step by step, in detail.' \
  --max-tokens 32 --markers "$out/4bit-markers.jsonl" > "$out/4bit.log" 2>&1
.venv-accept/bin/python scripts/profile_analyze.py "$out/4bit.jsonl" --markers "$out/4bit-markers.jsonl" \
  --compute-h overlay/mlx/backend/omarchy/compute.h > "$out/4bit-analysis.txt" 2>&1
grep -m1 "QmmVecQ4" "$out/4bit-analysis.txt"
echo PROFILE_DONE

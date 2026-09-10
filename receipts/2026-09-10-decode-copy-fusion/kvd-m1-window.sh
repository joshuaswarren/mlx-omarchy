#!/usr/bin/env bash
# DecodeCopyFusion GPU window on jwm1. ONE flock acquisition, never
# nested, never unlinked. Phases: (1) five test batteries, (2) dispatch
# census stock(KV_DIRECT=0) then fork(default) with the diagnostics
# wheel, (3) paired fork/stock decode legs, 3 pairs, alternating order,
# six canonical Q4 digests hard-gated with one retry on mismatch.
#   kvd-m1-window.sh
# CPU prep (kvd-m1-setup.sh) must be complete first.
set -uo pipefail
cand=~/src/mlx-KvDirect
out=$cand/receipts/2026-09-10-decode-copy-fusion/m1
mkdir -p "$out"
exec 9>/tmp/m1-gpu.lock
echo "waiting for GPU lock at $(date -u +%FT%TZ)"
flock -w 3600 9 || { echo "GPU lock not acquired within 1h"; exit 1; }
echo "GPU lock acquired at $(date -u +%FT%TZ)"
cd "$cand"
commit="$(git rev-parse --short=7 HEAD)"
echo "$commit" > "$out/source-commit.txt"

clean_env() {
  unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES MLX_OMARCHY_KV_DIRECT
  export MESA_SHADER_CACHE_DISABLE=true HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
}

# ---- phase 1: test batteries -------------------------------------------
clean_env
for suite in omarchy_kv_ops_tests omarchy_runtime_tests omarchy_fast_ops_tests omarchy_fast_regression_tests omarchy_fused_chain_tests; do
  if timeout 3600 .work/build-accept/tests/omarchy/"$suite" > "$out/$suite.log" 2>&1; then
    printf 'SUITE_PASS=%s\n' "$suite" | tee -a "$out/suites.txt"
  else
    printf 'SUITE_FAIL=%s\n' "$suite" | tee -a "$out/suites.txt"
  fi
  tail -2 "$out/$suite.log"
done

# ---- phase 2: dispatch census, stock then fork --------------------------
clean_env
wheel_diag=(wheels/diag/mlx_omarchy-*+diag."$commit"-*.whl)
test "${#wheel_diag[@]}" -eq 1
sha256sum "${wheel_diag[0]}" > "$out/wheel-diag.sha256"
.venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${wheel_diag[0]}"
version="$(.venv-accept/bin/python -c 'import mlx.core as mx; print(mx.__version__)')"
case "$version" in *"+diag.$commit") ;; *) echo "diag wheel stamp $version != diag.$commit" >&2; exit 1;; esac
echo "$version" > "$out/wheel-diag-version.txt"
model="$(.venv-accept/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen2.5-0.5B-Instruct-4bit', revision='a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3', local_files_only=True))")"
for arm in stock fork; do
  if [ "$arm" = stock ]; then export MLX_OMARCHY_KV_DIRECT=0; else unset MLX_OMARCHY_KV_DIRECT; fi
  MLX_OMARCHY_GPU_PROFILE="$out/census-$arm.jsonl" .venv-accept/bin/python scripts/profile_generate.py \
    --model "$model" --prompt 'What is the capital of France?' \
    --max-tokens 32 --markers "$out/census-$arm-markers.jsonl" > "$out/census-$arm.log" 2>&1
  unset MLX_OMARCHY_KV_DIRECT
  .venv-accept/bin/python scripts/profile_analyze.py "$out/census-$arm.jsonl" \
    --markers "$out/census-$arm-markers.jsonl" \
    --compute-h overlay/mlx/backend/omarchy/compute.h > "$out/census-$arm-analysis.txt" 2>&1
  grep -E "SliceUpdatePairF16|dispatches" "$out/census-$arm-analysis.txt" | head -5
done

# ---- phase 3: paired fork/stock legs ------------------------------------
clean_env
wheel_rel=(dist/mlx_omarchy-*+"$commit"-*.whl)
test "${#wheel_rel[@]}" -eq 1
sha256sum "${wheel_rel[0]}" > "$out/wheel-release.sha256"
.venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${wheel_rel[0]}"
rm -rf "$out/legs"
.venv-accept/bin/python scripts/kvd-paired.py "$cand" .venv-accept/bin/python "${wheel_rel[0]}" "$out/legs" 3 2>&1 | tee "$out/legs.log"
echo "window end: $(date -u +%FT%TZ)"
echo ALL_DONE

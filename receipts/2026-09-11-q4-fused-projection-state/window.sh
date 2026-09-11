#!/usr/bin/env bash
# Q4 fused-projection state window: the authoritative per-token dispatch
# census on current main at the three Q4 legs, plus one fusion-off
# control census at the short leg. No timing legs, no code change under
# test - this window only measures.
#   flock -w 1200 /tmp/m1-gpu.lock timeout 1200 bash window.sh CHECKOUT OUT_DIR
set -euo pipefail
checkout="$1"; out="$2"
commit="$(git rev-parse --short=7 HEAD)"
wheel=(dist/mlx_omarchy-*+diag."$commit"-*.whl)
test "${#wheel[@]}" -eq 1
mkdir -p "$out"
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION \
      MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES MLX_OMARCHY_FUSED_GEMV
export MESA_SHADER_CACHE_DISABLE=true HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
sha256sum "${wheel[0]}" > "$out/wheel.sha256"
.venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${wheel[0]}"
version="$(.venv-accept/bin/python -c 'import mlx.core as mx; print(mx.__version__)')"
case "$version" in *"+diag.$commit") ;; *) echo "wheel stamp $version != diag.$commit" >&2; exit 1;; esac
echo "$version" > "$out/version.txt"
git rev-parse HEAD > "$out/commit.txt"

REC="receipts/2026-09-11-q4-fused-projection-state"

# Three-leg census, fusion at its default (on).
bash -c "timeout 240 .venv-accept/bin/python $REC/census.py '$out' short short --max-tokens 16"
bash -c "timeout 240 .venv-accept/bin/python $REC/census.py '$out' long long --max-tokens 16"
bash -c "timeout 240 .venv-accept/bin/python $REC/census.py '$out' ctx1024 ctx1024 --max-tokens 16"
# Fusion-off control at the short leg: the before/after dispatch count.
bash -c "timeout 240 .venv-accept/bin/python $REC/census.py '$out' short-off short --fused-gemv 0 --max-tokens 16"
echo WINDOW_CENSUS_DONE

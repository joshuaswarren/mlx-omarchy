#!/usr/bin/env bash
# Release-side GPU-gap probe window: three instrument modes on one
# binary (off = release path, noiso = timestamps without the profiler's
# isolation barrier, iso = shipped diag instrument), short leg.
#   flock -w 1200 /tmp/m1-gpu.lock timeout 900 bash window-gap.sh CHECKOUT OUT_DIR
set -euo pipefail
checkout="$1"; out="$2"
cd "$checkout"
commit="$(git rev-parse --short=7 HEAD)"
wheel=(dist/mlx_omarchy-*+diag."$commit"-*.whl)
test "${#wheel[@]}" -eq 1
mkdir -p "$out"
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION \
      MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES \
      MLX_OMARCHY_PROFILE_NOISOBAR
export MESA_SHADER_CACHE_DISABLE=true HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
sha256sum "${wheel[0]}" | tee "$out/wheel.sha256"
.venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${wheel[0]}"
version="$(.venv-accept/bin/python -c 'import mlx.core as mx; print(mx.__version__)')"
case "$version" in *"+diag.$commit") ;; *) echo "wheel stamp $version != diag.$commit" >&2; exit 1;; esac
echo "$version" | tee "$out/version.txt"
git rev-parse HEAD > "$out/commit.txt"
REC="receipts/2026-09-11-q4-fused-projection-state"
bash -c "timeout 260 .venv-accept/bin/python $REC/gap-probe.py '$out' short"
echo WINDOW_GAP_DONE

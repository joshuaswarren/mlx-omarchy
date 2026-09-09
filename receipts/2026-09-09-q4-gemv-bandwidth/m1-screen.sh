#!/usr/bin/env bash
# Kernel bandwidth screen on the M1: install a diagnostics wheel into the
# checkout's .venv-accept, run kernel_bandwidth.py for the named
# variants, and dump Honeykrisp shader statistics for the default kernel.
#   m1-screen.sh CHECKOUT OUT_DIR VARIANT [VARIANT ...]
# Run under `flock -w 900 /tmp/m1-gpu.lock timeout 3600`.
set -euo pipefail
cd "$1"
out="$2"; shift 2
commit="$(git rev-parse --short=7 HEAD)"
wheel=(wheels/diag/mlx_omarchy-*+diag."$commit"-*.whl)
test "${#wheel[@]}" -eq 1
mkdir -p "$out"
sha256sum "${wheel[0]}" > "$out/wheel.sha256"
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES
export MESA_SHADER_CACHE_DISABLE=true
.venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${wheel[0]}"
version="$(.venv-accept/bin/python -c 'import mlx.core as mx; print(mx.__version__)')"
case "$version" in *"+diag.$commit") ;; *) echo "wheel stamp $version != diag.$commit" >&2; exit 1;; esac
echo "$version" > "$out/version.txt"
args=()
for v in "$@"; do args+=(--variant "$v"); done
.venv-accept/bin/python /tmp/gb-kernel_bandwidth.py run .venv-accept/bin/python "$out" . --reps 20 "${args[@]}" 2>&1 | tee "$out/screen.log"
# Compiler statistics and disassembly of the f16 Q4 kernel at K=N=896.
for v in "$@"; do
  env_v=()
  [[ "$v" != default ]] && env_v=(MLX_OMARCHY_Q4_GEMV_SCREEN="$v")
  env "${env_v[@]}" AGX_MESA_DEBUG=shaders,shaderdb .venv-accept/bin/python - > "$out/shader-$v.txt" 2>&1 <<'EOF'
import mlx.core as mx
import numpy as np
rng = np.random.default_rng(1)
K, N = 896, 896
x = mx.array(rng.standard_normal((1, K)).astype(np.float32)).astype(mx.float16)
w = mx.array(rng.integers(0, 2**32, size=(N, K // 8), dtype=np.uint64).astype(np.uint32))
s = mx.array(rng.uniform(-1, 1, size=(N, K // 64)).astype(np.float32)).astype(mx.float16)
b = mx.array(rng.uniform(-1, 1, size=(N, K // 64)).astype(np.float32)).astype(mx.float16)
mx.eval(x, w, s, b)
mx.eval(mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4))
print('version', mx.__version__)
EOF
done
echo SCREEN_DONE

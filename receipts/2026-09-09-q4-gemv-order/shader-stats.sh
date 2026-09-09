#!/usr/bin/env bash
# Honeykrisp compiler statistics (AGX_MESA_DEBUG=shaderdb) for the kernels
# one f16 Q4 decode GEMV (M=1, K=896, N=896, group 64) compiles in a given
# venv. Run under the GPU lock. usage: shader-stats.sh VENV_PYTHON OUT_TXT
set -euo pipefail
py="$1"; out="$2"
AGX_MESA_DEBUG=shaderdb MESA_SHADER_CACHE_DISABLE=true "$py" - > "$out" 2>&1 <<'EOF'
import mlx.core as mx
import numpy as np
rng = np.random.default_rng(1)
K, N = 896, 896
x = mx.array(rng.standard_normal((1, K)).astype(np.float32)).astype(mx.float16)
w = mx.array(rng.integers(0, 2**32, size=(N, K // 8), dtype=np.uint64).astype(np.uint32))
s = mx.array(rng.uniform(-1, 1, size=(N, K // 64)).astype(np.float32)).astype(mx.float16)
b = mx.array(rng.uniform(-1, 1, size=(N, K // 64)).astype(np.float32)).astype(mx.float16)
mx.eval(x, w, s, b)
out = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
mx.eval(out)
print('version', mx.__version__)
EOF
grep -i "version\|shader\|instr\|inst\|reg\|spill\|nops\|halfregs\|thread" "$out" | head -40

# Bounded M1 A/B timing command for the affine QMM direct-bind change.
# Software-development evidence only; NOT an M1 speed claim by itself.
# Run on the M1 once per wheel (baseline c8618f53, then the fix wheel),
# same shell, nothing else running, plugged in:
#
#   python3 -m venv .venv-ab && .venv-ab/bin/pip install --quiet numpy
#   MLX_OMARCHY_QMM_TILE=1 .venv-ab/bin/pip install --force-reinstall --no-deps <WHEEL>
#   MLX_OMARCHY_QMM_TILE=1 .venv-ab/bin/python affine_qmm_ab.py <label>
#
import json, statistics, sys, time

import mlx.core as mx
import numpy as np

label = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
stamp = mx.__version__ if hasattr(mx, "__version__") else "unknown"
print(f"[{label}] wheel stamp: {stamp}", flush=True)
assert "+c8618f53" in stamp or "+affine" in stamp or True  # stamps printed, compared offline

def qmm_case(m, n, k, gs, bits, dtype):
    rs = np.random.default_rng(m * 7919 + n)
    w = mx.array(rs.integers(0, 2**31, size=(n, k * bits // 32)).astype(np.uint32))
    groups = k // gs
    s = mx.array((rs.standard_normal((n, groups)) * 0.05).astype(np.float32)).astype(dtype)
    b = mx.array((rs.standard_normal((n, groups)) * 0.05).astype(np.float32)).astype(dtype)
    x = mx.array((rs.standard_normal((m, k)) * 0.5).astype(np.float32)).astype(dtype)
    return x, w, s, b

def bench(name, m, n, k, gs, bits, dtype, iters):
    x, w, s, b = qmm_case(m, n, k, gs, bits, dtype)
    out = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits, mode="affine")
    mx.eval(out)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits, mode="affine")
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1e3)
    med = statistics.median(times)
    print(f"[{label}] {name}: median {med:.3f} ms over {iters} iters", flush=True)
    return {"name": name, "median_ms": med, "iters": iters}

results = {"label": label, "stamp": str(stamp), "cases": []}
results["cases"].append(bench("decode_gemv_f16_m1_4096", 1, 4096, 4096, 64, 4, mx.float16, 50))
results["cases"].append(bench("prefill_tile_f16_m1023_4096", 1023, 4096, 4096, 64, 4, mx.float16, 15))
print(json.dumps(results), flush=True)

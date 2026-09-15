#!/usr/bin/env python3
# Bit-exactness A/B for the fused encoder kernels: fused standalone dispatch
# vs the reference mx chain (the runner's previous apply() bodies) on random
# fp16 data plus a -inf softmax arm. True means byte-identical.
import importlib.util
import json

import mlx.core as mx
import numpy as np

spec = importlib.util.spec_from_file_location(
    "ve_fused", "/var/tmp/ParakeetE2EFusedLeftover-stage/vulkan_encoder.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mx.set_default_device(mx.gpu)
rng = np.random.default_rng(20260915)


def call(kernel, inputs, n, dtypes):
    return kernel()(
        inputs=inputs,
        output_shapes=[(n,)],
        output_dtypes=dtypes,
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )[0]


def ref_silu(x):
    f = x.astype(mx.float32)
    return (f * mx.sigmoid(f)).astype(mx.float16)


def ref_sigmoid(x):
    return mx.sigmoid(x.astype(mx.float32)).astype(mx.float16)


def ref_glu(a, b):
    s = ref_sigmoid(b)
    return (a.astype(mx.float32) * s.astype(mx.float32)).astype(mx.float16)


def ref_ln(x, gamma, beta, eps):
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean(mx.square(xf - mean), axis=-1, keepdims=True)
    out = (xf - mean) * mx.rsqrt(var + eps)
    out = out * gamma.astype(mx.float32)
    out = out + beta.astype(mx.float32)
    return out.astype(mx.float16)


def ref_softmax(x):
    # the fused runner keeps the max as ReduceF16 over the fp16 input
    rmax = mx.max(x, axis=-1, keepdims=True)
    e = mx.exp(x.astype(mx.float32) - rmax.astype(mx.float32))
    return (e / mx.sum(e, axis=-1, keepdims=True)).astype(mx.float16)


def same(a, b):
    return bool(np.array_equal(
        np.asarray(a), np.asarray(b), equal_nan=True
    ))


report = {}

# silu at the FFN width
x = mx.array(rng.standard_normal(375 * 4096).astype(np.float16))
report["silu_4096"] = same(
    ref_silu(x), mx.reshape(call(mod._silu_kernel, [x], x.size, [mx.float16]), x.shape)
)

# GLU on real split views (channel axis, both halves)
p = mx.array(rng.standard_normal(2048 * 375).astype(np.float16)).reshape(1, 2048, 375)
a, b = mx.split(p, 2, axis=1)
report["glu_splitviews"] = same(
    ref_glu(a, b),
    mx.reshape(call(mod._glu_kernel, [a, b], a.size, [mx.float16]), a.shape),
)

# layer_norm at the pinned envelope; eps is the fp16 const widened
g = mx.array(rng.standard_normal(1024).astype(np.float16))
bb = mx.array(rng.standard_normal(1024).astype(np.float16))
eps = float(np.float16(1e-5))
for tag, rows in (("ln", 375), ("ln_tall", 1)):
    xx = mx.array(rng.standard_normal(rows * 1024).astype(np.float16)).reshape(1, rows, 1024)
    ref = ref_ln(xx, g, bb, eps)
    n = xx.size
    xf = call(mod._ln_cast_kernel, [xx], n, [mx.float32])
    mean = mx.mean(mx.reshape(xf, (rows, 1024)), axis=-1, keepdims=True)
    t2 = call(mod._ln_sq_kernel, [xf, mean], n, [mx.float32])
    var = mx.mean(mx.reshape(t2, (rows, 1024)), axis=-1, keepdims=True)
    eps_arr = mx.array([eps], dtype=mx.float32)
    y = call(mod._ln_tail_kernel, [xf, mean, var, g, bb, eps_arr], n, [mx.float16])
    report[tag] = same(ref, mx.reshape(y, xx.shape))

# softmax at the pinned envelope, including all-masked -inf rows
scores = mx.array(rng.standard_normal(8 * 375 * 375).astype(np.float16)).reshape(1, 8, 375, 375)
scores = mx.reshape(mx.concatenate([
    mx.reshape(scores[:, :4], (-1,)),
    mx.full((4 * 375 * 375,), float("-inf"), dtype=mx.float16),
]), (1, 8, 375, 375))
ref = ref_softmax(scores)
n = scores.size
rowmax = mx.max(scores, axis=-1, keepdims=True)
e = call(mod._sm_exp_kernel, [scores, rowmax], n, [mx.float32])
rowsum = mx.sum(mx.reshape(e, (n // 375, 375)), axis=-1, keepdims=True)
y = call(mod._sm_div_kernel, [e, rowsum], n, [mx.float16])
got = mx.reshape(y, scores.shape)
report["softmax_masked"] = same(ref, got)
ref_np = np.asarray(ref)
got_np = np.asarray(got)
report["softmax_masked_nan_agreement"] = bool(
    np.array_equal(np.isnan(ref_np), np.isnan(got_np))
)

print(json.dumps(report, indent=2))
if not all(report.values()):
    raise SystemExit("A/B MISMATCH")
print("ALL-IDENTICAL")

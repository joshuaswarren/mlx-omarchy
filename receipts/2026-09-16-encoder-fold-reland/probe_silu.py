"""Isolate the bias+silu lincheck mismatch: (a) tail-only via 1-block zero
bias, (b) exact-fp32 numpy model of the silu tail, (c) unrounded-sum elision
model. Usage: probe_silu.py <vulkan_encoder.py>
"""
import importlib.util
import sys

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.gpu)

spec = importlib.util.spec_from_file_location("folded", sys.argv[1])
folded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folded)


def u16(a):
    return np.asarray(a.view(mx.uint16)).astype(np.uint16)


rng = np.random.default_rng(0)
rows, k, n_out = 375, 1024, 1024
blocks = k // 16
x = mx.array(rng.standard_normal((rows, k)).astype(np.float16))
w = mx.array(rng.standard_normal((n_out, k)).astype(np.float16))
bias = mx.array(rng.standard_normal((n_out,)).astype(np.float16))
lhs = mx.reshape(x, (rows, k))
rhs = mx.reshape(w, (n_out, k))
p16 = folded._linear_f16_coopmat_kernel()(
    inputs=[lhs, rhs],
    output_shapes=[(blocks, rows, n_out)],
    output_dtypes=[mx.float16],
    grid=((n_out + 31) // 32 * 32, (rows + 31) // 32, blocks),
    threadgroup=(32, 1, 1),
    stream=mx.gpu,
)[0]
mx.eval(p16)

chain = folded._leftover_chain_kernel()(
    inputs=[p16], output_shapes=[(rows, n_out)], output_dtypes=[mx.float16],
    grid=(rows * n_out, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu)[0]
b = (mx.reshape(chain, (1, rows, n_out)).astype(mx.float32)
     + bias.astype(mx.float32)).astype(mx.float16)
mx.eval(b)

# (a) tail-only: fused silu kernel, 1-block partials = b, zero bias
zb = mx.zeros((n_out,), mx.float16)
tail_in = mx.reshape(b, (1, rows, n_out))
s_tail = folded._leftover_chain_bias_silu_kernel()(
    inputs=[tail_in, zb], output_shapes=[(rows, n_out)],
    output_dtypes=[mx.float16], grid=(rows * (n_out // 2), 1, 1),
    threadgroup=(256, 1, 1), stream=mx.gpu)[0]
mx.eval(s_tail)

# stock silu on b
n = b.size
s_stock = folded._silu_kernel()(
    inputs=[mx.reshape(b, (n,))], output_shapes=[(n,)],
    output_dtypes=[mx.float16], grid=(n, 1, 1), threadgroup=(256, 1, 1),
    stream=mx.gpu)[0]
mx.eval(s_stock)

a = u16(mx.reshape(s_tail, (1, rows, n_out)))
c = u16(mx.reshape(s_stock, (1, rows, n_out)))
print("tail-only (1 block, zero bias) vs stock silu: mismatch =",
      int((a != c).sum()), "/", a.size)

# (b) exact fp32 numpy model of the tail on the same w
wnp = np.asarray(b).astype(np.float32).ravel()
sig = (np.float32(1.0) / (np.float32(1.0) + np.exp(-wnp))).astype(np.float32)
ref = (wnp * sig).astype(np.float16)
ss = np.asarray(s_stock).astype(np.float16).ravel()
st = np.asarray(s_tail).astype(np.float16).ravel()
print("stock vs numpy-f32 model:  mismatch =", int((ss != ref).sum()), "/", ss.size)
print("tail  vs numpy-f32 model:  mismatch =", int((st != ref).sum()), "/", st.size)

# (c) unrounded-sum elision model: silu applied to the f32 (unrounded) sum
chainnp = np.asarray(chain).astype(np.float32).reshape(1, rows, n_out)
v32 = chainnp + np.asarray(bias).astype(np.float32).reshape(1, n_out)
sig2 = (np.float32(1.0) / (np.float32(1.0) + np.exp(-v32))).astype(np.float32)
elid = (v32 * sig2).astype(np.float16).ravel()
print("tail  vs unrounded-elision model: mismatch =",
      int((st != elid).sum()), "/", st.size)
print("stock vs unrounded-elision model: mismatch =",
      int((ss != elid).sum()), "/", ss.size)

# ulp histogram of tail vs stock where they differ
d = (st.astype(np.uint16).astype(np.int32) - ss.astype(np.uint16).astype(np.int32))
nz = d[d != 0]
if nz.size:
    vals, cnts = np.unique(nz, return_counts=True)
    print("delta histogram (top 8):",
          list(zip(vals.tolist(), cnts.tolist()))[:8])

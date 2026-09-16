"""Confirm elided per-step fp16 rounding in the fused silu kernel's chain,
and test fix variants. Usage: probe2.py <vulkan_encoder.py>
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


def chain_stock(partials, rows, n_out):
    return folded._leftover_chain_kernel()(
        inputs=[partials], output_shapes=[(rows, n_out)],
        output_dtypes=[mx.float16], grid=(rows * n_out, 1, 1),
        threadgroup=(256, 1, 1), stream=mx.gpu)[0]


rng = np.random.default_rng(0)
rows, k, n_out = 375, 1024, 1024
blocks = k // 16
x = mx.array(rng.standard_normal((rows, k)).astype(np.float16))
w = mx.array(rng.standard_normal((n_out, k)).astype(np.float16))
bias = mx.array(rng.standard_normal((n_out,)).astype(np.float16))
p16 = folded._linear_f16_coopmat_kernel()(
    inputs=[mx.reshape(x, (rows, k)), mx.reshape(w, (n_out, k))],
    output_shapes=[(blocks, rows, n_out)],
    output_dtypes=[mx.float16],
    grid=((n_out + 31) // 32 * 32, (rows + 31) // 32, blocks),
    threadgroup=(32, 1, 1), stream=mx.gpu,
)[0]
mx.eval(p16)
b = mx.reshape(chain_stock(p16, rows, n_out), (1, rows, n_out))
b = (b.astype(mx.float32) + bias.astype(mx.float32)).astype(mx.float16)
mx.eval(b)
n = b.size
s_stock = folded._silu_kernel()(
    inputs=[mx.reshape(b, (n,))], output_shapes=[(n,)],
    output_dtypes=[mx.float16], grid=(n, 1, 1), threadgroup=(256, 1, 1),
    stream=mx.gpu)[0]
mx.eval(s_stock)

# mechanism check: numpy model with per-step rounding elided (f32 acc)
pnp = np.asarray(p16).astype(np.float32)
acc = pnp[0].copy()
for bi in range(1, blocks):
    acc += pnp[bi]
w_elid = acc + np.asarray(bias).astype(np.float32)  # f32 acc, no mid rounding
sig = (np.float32(1.0) / (np.float32(1.0) + np.exp(-w_elid))).astype(np.float32)
s_elid = (w_elid * sig).astype(np.float16).ravel()
ss = np.asarray(s_stock).astype(np.float16).ravel()
print("stock vs elided-acc numpy model:", int((ss != s_elid).sum()), "/", ss.size)

# V1: volatile-forced per-step rounding
V1 = mx.fast.metal_kernel(
    name="probe_chain_bias_silu_volatile",
    input_names=["partials", "bias"], output_names=["reduced"],
    source="""
        uint pairs = partials_shape[2] / 2u;
        uint words_row = partials_shape[1] * pairs;
        uint blocks = partials_shape[0];
        uint index = thread_position_in_grid.x;
        uint row = index / pairs;
        uint pair = index - row * pairs;
        uint word = row * pairs + pair;
        uint col = word * 2u;
        volatile half a0 = half(partials[col]);
        volatile half a1 = half(partials[col + 1u]);
        for (uint block = 1u; block < blocks; ++block) {
            uint base = col + block * words_row * 2u;
            a0 = a0 + half(partials[base]);
            a1 = a1 + half(partials[base + 1u]);
        }
        half w0 = half(float(a0) + float(bias[pair * 2u]));
        half w1 = half(float(a1) + float(bias[pair * 2u + 1u]));
        float s0 = float(w0) * (1.0 / (1.0 + exp(-float(w0))));
        float s1 = float(w1) * (1.0 / (1.0 + exp(-float(w1))));
        reduced[col] = half(s0);
        reduced[col + 1u] = half(s1);
    """,
    compile_options={"math_mode": "safe"},
)

# V2: round through explicit half lvalue via helper returning half
V2 = mx.fast.metal_kernel(
    name="probe_chain_bias_silu_bits",
    input_names=["partials", "bias"], output_names=["reduced"],
    source="""
        uint pairs = partials_shape[2] / 2u;
        uint words_row = partials_shape[1] * pairs;
        uint blocks = partials_shape[0];
        uint index = thread_position_in_grid.x;
        uint row = index / pairs;
        uint pair = index - row * pairs;
        uint word = row * pairs + pair;
        uint col = word * 2u;
        uint16_t b0 = as_type<uint16_t>(half(partials[col]));
        uint16_t b1 = as_type<uint16_t>(half(partials[col + 1u]));
        for (uint block = 1u; block < blocks; ++block) {
            uint base = col + block * words_row * 2u;
            half t0 = half(partials[base]);
            half t1 = half(partials[base + 1u]);
            b0 = as_type<uint16_t>(as_type<half>(b0) + t0);
            b1 = as_type<uint16_t>(as_type<half>(b1) + t1);
        }
        half w0 = half(float(as_type<half>(b0)) + float(bias[pair * 2u]));
        half w1 = half(float(as_type<half>(b1)) + float(bias[pair * 2u + 1u]));
        float s0 = float(w0) * (1.0 / (1.0 + exp(-float(w0))));
        float s1 = float(w1) * (1.0 / (1.0 + exp(-float(w1))));
        reduced[col] = half(s0);
        reduced[col + 1u] = half(s1);
    """,
    compile_options={"math_mode": "safe"},
)

grid = (rows * (n_out // 2), 1, 1)
for name, kern in (("V1-volatile", V1), ("V2-bits", V2)):
    try:
        out = kern(inputs=[p16, bias], output_shapes=[(rows, n_out)],
                     output_dtypes=[mx.float16], grid=grid,
                     threadgroup=(256, 1, 1), stream=mx.gpu)[0]
        mx.eval(out)
        mm = int((u16(out) != u16(s_stock.reshape(1, rows, n_out))).sum())
        print(f"{name} vs stock: mismatch = {mm} / {rows * n_out}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}")

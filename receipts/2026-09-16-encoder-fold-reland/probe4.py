"""Force the w0 fp16 round in the fused chain+bias+silu kernel.
V3: ushort bit round-trip. V5: global write-read-back. V4: threadgroup stage.
Each vs stock on (375,1024,1024). Usage: probe4.py <vulkan_encoder.py>
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


HEAD = """
        uint pairs = partials_shape[2] / 2u;
        uint words_row = partials_shape[1] * pairs;
        uint blocks = partials_shape[0];
        uint index = thread_position_in_grid.x;
        uint row = index / pairs;
        uint pair = index - row * pairs;
        uint word = row * pairs + pair;
        uint col = word * 2u;
        half a0 = half(partials[col]);
        half a1 = half(partials[col + 1u]);
        for (uint block = 1u; block < blocks; ++block) {
            uint base = col + block * words_row * 2u;
            a0 = a0 + half(partials[base]);
            a1 = a1 + half(partials[base + 1u]);
        }
"""
TAIL0 = """
        float s0 = float(w0) * (1.0 / (1.0 + exp(-float(w0))));
        float s1 = float(w1) * (1.0 / (1.0 + exp(-float(w1))));
        reduced[col] = half(s0);
        reduced[col + 1u] = half(s1);
"""


def mk(name, body):
    return mx.fast.metal_kernel(
        name=name,
        input_names=["partials", "bias"],
        output_names=["reduced"],
        source=HEAD + body + TAIL0,
        compile_options={"math_mode": "safe"},
    )


V3 = mk("probe_v3_ushort", """
        half w0 = as_type<half>(as_type<ushort>(half(float(a0) + float(bias[pair * 2u]))));
        half w1 = as_type<half>(as_type<ushort>(half(float(a1) + float(bias[pair * 2u + 1u]))));
""")

V5 = mk("probe_v5_memround", """
        reduced[col] = half(float(a0) + float(bias[pair * 2u]));
        reduced[col + 1u] = half(float(a1) + float(bias[pair * 2u + 1u]));
        half w0 = reduced[col];
        half w1 = reduced[col + 1u];
""")

V4 = mk("probe_v4_tg", """
        threadgroup half tg[2];
        tg[0] = half(float(a0) + float(bias[pair * 2u]));
        tg[1] = half(float(a1) + float(bias[pair * 2u + 1u]));
        half w0 = tg[0];
        half w1 = tg[1];
""")

rng = np.random.default_rng(0)
rows, k, n_out = 375, 1024, 1024
blocks = k // 16
x = mx.array(rng.standard_normal((rows, k)).astype(np.float16))
w = mx.array(rng.standard_normal((n_out, k)).astype(np.float16))
bias = mx.array(rng.standard_normal((n_out,)).astype(np.float16))
p16 = folded._linear_f16_coopmat_kernel()(
    inputs=[mx.reshape(x, (rows, k)), mx.reshape(w, (n_out, k))],
    output_shapes=[(blocks, rows, n_out)], output_dtypes=[mx.float16],
    grid=((n_out + 31) // 32 * 32, (rows + 31) // 32, blocks),
    threadgroup=(32, 1, 1), stream=mx.gpu)[0]
mx.eval(p16)
chain = folded._leftover_chain_kernel()(
    inputs=[p16], output_shapes=[(rows, n_out)], output_dtypes=[mx.float16],
    grid=(rows * n_out, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu)[0]
b = (mx.reshape(chain, (1, rows, n_out)).astype(mx.float32)
     + bias.astype(mx.float32)).astype(mx.float16)
mx.eval(b)
n = b.size
s_stock = folded._silu_kernel()(
    inputs=[mx.reshape(b, (n,))], output_shapes=[(n,)],
    output_dtypes=[mx.float16], grid=(n, 1, 1), threadgroup=(256, 1, 1),
    stream=mx.gpu)[0]
mx.eval(s_stock)
ref = u16(s_stock.reshape(1, rows, n_out))

grid = (rows * (n_out // 2), 1, 1)
for name, kern in (("V3-ushort", V3), ("V5-memround", V5), ("V4-tg", V4)):
    try:
        out = kern(inputs=[p16, bias], output_shapes=[(rows, n_out)],
                   output_dtypes=[mx.float16], grid=grid,
                   threadgroup=(256, 1, 1), stream=mx.gpu)[0]
        mx.eval(out)
        mm = int((u16(out) != ref).sum())
        print(f"{name} vs stock: mismatch = {mm} / {rows * n_out}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}")

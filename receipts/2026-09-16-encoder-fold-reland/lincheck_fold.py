"""Gate 1 for the fold reland: bit-exactness of the fused chain+bias(+silu)
epilogue vs the CURRENT stock dispatch chain (chain kernel + astype-f32 bias
add + silu kernel) on the coopmat fp16-partials path, on the four exact
program shapes. u16-view comparison; ALL-EXACT or MISMATCH-FOUND.

Usage: lincheck_fold.py <vulkan_encoder.py>
"""
import importlib.util
import sys

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.gpu)

spec = importlib.util.spec_from_file_location("folded", sys.argv[1])
folded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folded)


def coopmat_partials(lhs, rhs):
    rows, k = lhs.shape
    n_out = rhs.shape[0]
    blocks = k // 16
    return folded._linear_f16_coopmat_kernel()(
        inputs=[lhs, rhs],
        output_shapes=[(blocks, rows, n_out)],
        output_dtypes=[mx.float16],
        grid=((n_out + 31) // 32 * 32, (rows + 31) // 32, blocks),
        threadgroup=(32, 1, 1),
        stream=mx.gpu,
    )[0]


def stock(partials, bias, rows, n_out, with_silu):
    out = folded._leftover_chain_kernel()(
        inputs=[partials],
        output_shapes=[(rows, n_out)],
        output_dtypes=[mx.float16],
        grid=(rows * n_out, 1, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )[0]
    out = mx.reshape(out, (1, rows, n_out))
    out = (out.astype(mx.float32) + bias.astype(mx.float32)).astype(mx.float16)
    if with_silu:
        n = out.size
        s = folded._silu_kernel()(
            inputs=[mx.reshape(out, (n,))],
            output_shapes=[(n,)],
            output_dtypes=[mx.float16],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
            stream=mx.gpu,
        )[0]
        out = mx.reshape(s, (1, rows, n_out))
    return out


def fused(partials, bias, rows, n_out, with_silu):
    kern = (
        folded._leftover_chain_bias_silu_kernel
        if with_silu
        else folded._leftover_chain_bias_kernel
    )
    out = kern()(
        inputs=[partials, bias],
        output_shapes=[(rows, n_out)],
        output_dtypes=[mx.float16],
        grid=(rows * (n_out // 2), 1, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )[0]
    return mx.reshape(out, (1, rows, n_out))


def u16(a):
    return np.asarray(a.view(mx.uint16)).astype(np.uint16)


rng = np.random.default_rng(0)
shapes = [(375, 512, 640), (375, 1024, 1024), (375, 1024, 4096), (375, 4096, 1024)]
ok = True
for (rows, k, n_out) in shapes:
    x = mx.array(rng.standard_normal((rows, k)).astype(np.float16))
    w = mx.array(rng.standard_normal((n_out, k)).astype(np.float16))
    bias = mx.array(rng.standard_normal((n_out,)).astype(np.float16))
    p16 = coopmat_partials(x, w)
    mx.eval(p16)
    for with_silu in (False, True):
        s = stock(p16, bias, rows, n_out, with_silu)
        f = fused(p16, bias, rows, n_out, with_silu)
        mx.eval(s)
        mx.eval(f)
        same = np.array_equal(u16(s), u16(f))
        n_mism = int((u16(s) != u16(f)).sum())
        name = "bias+silu" if with_silu else "bias"
        print(f"({rows},{k},{n_out}) {name}: u16_equal={same} (mismatch={n_mism}/{s.size})")
        ok = ok and same
print("ALL-EXACT" if ok else "MISMATCH-FOUND")

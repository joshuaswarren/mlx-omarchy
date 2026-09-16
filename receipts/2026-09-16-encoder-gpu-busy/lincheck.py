"""Bit-exactness: new f16 coopmat linear path vs old f32 batched path."""
import sys
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.gpu)
sys.path.insert(0, "/var/tmp/enc-gpubusy")
import importlib.util
spec = importlib.util.spec_from_file_location("cut", "/var/tmp/enc-gpubusy/vulkan_encoder_cut.py")
cut = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cut)

def old_partials(x, w):
    rows = int(np.prod(x.shape[:-1]))
    k = x.shape[-1]
    blocks = k // 16
    xb = mx.transpose(mx.reshape(x, (rows, blocks, 16)), (1, 0, 2)).astype(mx.float32)
    wb = mx.transpose(mx.reshape(w, (w.shape[0], blocks, 16)), (1, 2, 0)).astype(mx.float32)
    return xb @ wb  # f32 [blocks, rows, N]

def old_chain(partials_f32, rows, n_out):
    return cut._leftover_chain_kernel()(
        inputs=[partials_f32],
        output_shapes=[(rows, n_out)],
        output_dtypes=[mx.float16],
        grid=(rows * n_out, 1, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )[0]

def new_linear_partials(lhs, rhs):
    rows, k = lhs.shape
    n_out = rhs.shape[0]
    blocks = k // 16
    return cut._linear_f16_coopmat_kernel()(
        inputs=[lhs, rhs],
        output_shapes=[(blocks, rows, n_out)],
        output_dtypes=[mx.float16],
        grid=((n_out + 31) // 32 * 32, (rows + 31) // 32, blocks),
        threadgroup=(32, 1, 1),
        stream=mx.gpu,
    )[0]

def u16(a):
    return np.asarray(a.view(mx.uint16)).astype(np.uint16)

rng = np.random.default_rng(0)
shapes = [(375, 512, 640), (375, 1024, 1024), (375, 1024, 4096), (375, 4096, 1024)]
ok = True
for (rows, k, n_out) in shapes:
    blocks = k // 16
    x = mx.array(rng.standard_normal((rows, k)).astype(np.float16))
    w = mx.array(rng.standard_normal((n_out, k)).astype(np.float16))
    bias = mx.array(rng.standard_normal((n_out,)).astype(np.float16))
    # new path
    p16 = new_linear_partials(x, w)
    mx.eval(p16)
    chain_new = cut._leftover_chain_kernel()(
        inputs=[p16], output_shapes=[(rows, n_out)], output_dtypes=[mx.float16],
        grid=(rows * n_out, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu)[0]
    out_new = mx.reshape(chain_new, (1, rows, n_out))
    out_new = (out_new.astype(mx.float32) + bias.astype(mx.float32)).astype(mx.float16)
    mx.eval(out_new)
    # old path
    p32 = old_partials(x, w)
    mx.eval(p32)
    p32_as_f16 = p32.astype(mx.float16)  # reference: RNE(f32 partial)
    mx.eval(p32_as_f16)
    chain_old = old_chain(p32, rows, n_out)
    out_old = mx.reshape(chain_old, (1, rows, n_out))
    out_old = (out_old.astype(mx.float32) + bias.astype(mx.float32)).astype(mx.float16)
    mx.eval(out_old)
    # compare
    same_partials = np.array_equal(u16(p16), u16(p32_as_f16))
    same_chain = np.array_equal(u16(chain_new), u16(chain_old))
    same_out = np.array_equal(u16(out_new), u16(out_old))
    n_mism_p = int((u16(p16) != u16(p32_as_f16)).sum())
    n_mism_c = int((u16(chain_new) != u16(chain_old)).sum())
    print(f"({rows},{k},{n_out}): partials_u16_equal={same_partials} (mismatch={n_mism_p}/{p16.size}) "
          f"chain_u16_equal={same_chain} (mismatch={n_mism_c}) out_equal={same_out}")
    ok = ok and same_partials and same_chain and same_out
print("ALL-EXACT" if ok else "MISMATCH-FOUND")

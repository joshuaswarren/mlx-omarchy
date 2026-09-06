# Throwaway A/B probe (recovery session 2026-09-06): per-eval
# vkCmdCopyBuffer delta over affine QuantizedMatmul variants, plus
# output byte hashes for cross-build parity. Usage:
#   python3 affine_qmm_probe.py <out.json> [--odd-offset-negative]
import ctypes, hashlib, json, os, sys

import mlx.core as mx

def find_libmlx():
    import mlx.core
    so = mlx.core.__file__
    assert so and so.endswith(".so"), so
    return so

class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "gpu_primitive_dispatches", "vk_submissions", "vk_buffer_copies",
        "vk_buffer_fills", "vk_compute_dispatches", "omarchy_finalize_calls",
        "commit_calls_with_work", "commit_calls_noop")]

lib = ctypes.CDLL(find_libmlx())
lib.mlx_omarchy_trace_snapshot.argtypes = [ctypes.POINTER(Snap)]
lib.mlx_omarchy_trace_snapshot.restype = None

def counters():
    s = Snap()
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return {n: getattr(s, n) for n, _ in Snap._fields_}

def rng(shape, key, dtype=mx.float32):
    import numpy as np
    rs = np.random.default_rng(key)
    return mx.array(rs.standard_normal(shape).astype(np.float32) * 0.5).astype(dtype)

def run_variant(name, m, n, k, gs, bits, dtype, transpose, batched, tile_env):
    os.environ["MLX_OMARCHY_QMM_TILE"] = tile_env
    import numpy as np
    rs = np.random.default_rng(hash((name)) % (2**32))
    groups = k // gs
    if transpose:
        w_words = mx.array(rs.integers(0, 2**31, size=(n, k * bits // 32)).astype(np.uint32))
        param_shape = (n, groups)
    else:
        w_words = mx.array(rs.integers(0, 2**31, size=(k, n * bits // 32)).astype(np.uint32))
        param_shape = (k, n // gs)
    scales = (mx.array(rs.standard_normal(param_shape).astype(np.float32) * 0.05)).astype(dtype)
    biases = (mx.array(rs.standard_normal(param_shape).astype(np.float32) * 0.05)).astype(dtype)
    x = rng((m, k), 1234, dtype)
    before = counters()
    out = mx.quantized_matmul(x, w_words, scales, biases, transpose=transpose,
                              group_size=gs, bits=bits, mode="affine")
    mx.eval(out)
    mx.eval(mx.zeros((1,)))  # drain batch
    after = counters()
    return {
        "name": name,
        "copies_delta": after["vk_buffer_copies"] - before["vk_buffer_copies"],
        "dispatch_delta": after["vk_compute_dispatches"] - before["vk_compute_dispatches"],
        "out_sha256": hashlib.sha256(np.asarray(out.astype(mx.float32)).tobytes()).hexdigest(),
    }

def nonaffine_probe():
    import numpy as np
    rs = np.random.default_rng(7)
    n, k, gs = 8, 64, 32
    w = mx.array(rs.integers(0, 2**32, size=(n, k // 8), dtype=np.uint32))
    s = mx.array(np.ones((n, k // gs), np.uint8))
    x = rng((4, k), 99)
    before = counters()
    out = mx.quantized_matmul(x, w, s, transpose=True, group_size=gs, bits=4, mode="mxfp4")
    mx.eval(out)
    mx.eval(mx.zeros((1,)))
    after = counters()
    return {"name": "nonaffine_mxfp4", "copies_delta": after["vk_buffer_copies"] - before["vk_buffer_copies"]}

def odd_offset_case():
    # f16 scales/biases at odd element offsets: named refusal before the
    # direct-bind change, computed value after.
    import numpy as np
    rs = np.random.default_rng(5)
    m, n, k, gs, bits = 1, 20, 128, 64, 4
    groups = k // gs
    w_words = mx.array(rs.integers(0, 2**31, size=(n, k * bits // 32)).astype(np.uint32))
    s_pad = mx.array(np.concatenate([[0.0], rs.standard_normal(n * groups) * 0.05]).astype(np.float16))
    b_pad = mx.array(np.concatenate([[0.0, 0.0, 0.0], rs.standard_normal(n * groups) * 0.05]).astype(np.float16))
    x = rng((m, k), 77, mx.float16)
    s_view = mx.reshape(s_pad[1:], (n, groups))
    b_view = mx.reshape(b_pad[3:], (n, groups))
    out = mx.quantized_matmul(x, w_words, s_view, b_view, transpose=True,
                              group_size=gs, bits=bits, mode="affine")
    mx.eval(out)
    ref = mx.quantized_matmul(x, w_words,
                              mx.reshape(mx.array(np.asarray(s_pad)[1:].copy()), (n, groups)),
                              mx.reshape(mx.array(np.asarray(b_pad)[3:].copy()), (n, groups)),
                              transpose=True, group_size=gs, bits=bits, mode="affine")
    mx.eval(ref)
    return hashlib.sha256(np.asarray(out.astype(mx.float32)).tobytes()).hexdigest(),            hashlib.sha256(np.asarray(ref.astype(mx.float32)).tobytes()).hexdigest()

def main():
    print("quantized_matmul doc:", (mx.quantized_matmul.__doc__ or "")[:400], flush=True)
    out_path = sys.argv[1]
    cases = []
    cases.append(run_variant("vec_f32_m1_T", 1, 37, 128, 64, 4, mx.float32, True, False, "1"))
    cases.append(run_variant("tile_f32_m7_T", 7, 37, 128, 64, 4, mx.float32, True, False, "1"))
    cases.append(run_variant("scalar_f32_m7_T", 7, 37, 128, 64, 4, mx.float32, True, False, "0"))
    cases.append(run_variant("vec_f16_m1_T", 1, 37, 128, 64, 4, mx.float16, True, False, "1"))
    cases.append(run_variant("tile_f16_m7_NT", 7, 32, 128, 32, 4, mx.float16, False, False, "1"))
    cases.append(run_variant("tile_f32_batched", 7, 32, 128, 32, 8, mx.float32, True, False, "1"))
    cases.append(run_variant("bits6_m7_T", 7, 40, 192, 32, 6, mx.float32, True, False, "1"))
    cases.append(nonaffine_probe())
    result = {"cases": cases}
    if "--odd-offset-negative" in sys.argv:
        try:
            h, href = odd_offset_case()
            result["odd_offset"] = {"out": h, "ref": href, "match": h == href}
        except Exception as e:
            result["odd_offset"] = {"refused": str(e)[:300]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

main()

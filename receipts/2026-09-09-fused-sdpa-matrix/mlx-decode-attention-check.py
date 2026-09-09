import ctypes
import importlib.metadata
import json
import sys
sys.path.insert(0, '/tmp/mlx-native-oracle-scripts')
import mlx.core as mx
import numpy as np
from mlx_provenance import installed_provenance

provenance = installed_provenance(dist_name='mlx' if sys.platform == 'darwin' else 'mlx-omarchy')
assert provenance['verified'] == 'match'
print(json.dumps({'provenance':provenance}), flush=True)
expect_fused = '--expect-fused' in sys.argv
if expect_fused:
    library = ctypes.CDLL(str(importlib.metadata.distribution('mlx-omarchy').locate_file('mlx/lib/libmlx.so')))
    snapshot = library.mlx_omarchy_trace_snapshot
    snapshot.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    snapshot.restype = None

def dispatch_count():
    if not expect_fused:
        return 0
    values = (ctypes.c_uint64 * 8)()
    snapshot(values)
    return values[4]

rng = np.random.default_rng(4217)
for dtype, tolerance in ((mx.float32, 3e-6), (mx.float16, 0.0002), (mx.bfloat16, 0.002)):
    for length, reverse in ((1, False), (44, False), (1053, False), (44, True)):
        q = mx.array(rng.normal(0, 0.3, (2, 1, 6, 64)).astype(np.float32)).astype(dtype).transpose(0, 2, 1, 3)
        k = mx.array(rng.normal(0, 0.3, (2, length*2, 2, 64)).astype(np.float32)).astype(dtype)[:, ::2].transpose(0, 2, 1, 3)
        v = mx.array(rng.normal(0, 0.3, (2, length*2, 2, 64)).astype(np.float32)).astype(dtype)[:, ::2].transpose(0, 2, 1, 3)
        if reverse:
            k, v = k[:, ::-1], v[:, ::-1]
        mx.eval(q, k, v)
        qn, kn, vn = [np.array(x.astype(mx.float32)).astype(np.float64) for x in (q, k, v)]
        kn, vn = [np.repeat(x, 3, axis=1) for x in (kn, vn)]
        scores = np.einsum('bhqd,bhnd->bhqn', qn, kn) * 0.125
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        expected = probabilities @ vn
        before = dispatch_count()
        result = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
        try:
            mx.eval(result)
        except RuntimeError as error:
            if not (expect_fused and reverse):
                raise
            assert "ScaledDotProductAttention batch stride" in str(error), str(error)
            print(json.dumps({"dtype": str(dtype), "reversed_heads": True, "status": "named_rejection", "error": str(error)}), flush=True)
            continue
        dispatches = dispatch_count() - before
        if expect_fused and not reverse:
            assert dispatches == 1, (str(dtype), length, dispatches)
        causal = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask='causal')
        mx.eval(causal)
        observed = np.array(result.astype(mx.float32))
        np.testing.assert_allclose(observed, expected, atol=tolerance, rtol=0)
        np.testing.assert_array_equal(observed, np.array(causal.astype(mx.float32)))
        print(json.dumps({'dtype':str(dtype),'kv_length':length,'compute_dispatches':dispatches if expect_fused else None,'shape':list(result.shape),'max_abs_error':float(np.max(np.abs(observed-expected))),'status':'pass'}), flush=True)

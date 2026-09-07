import json
import os
import mlx.core as mx
import numpy as np

mx.set_default_device(mx.gpu)
rng = np.random.default_rng(731)
results = []
for dtype, precision, minimum_exp in [(mx.float16, 10, -24), (mx.bfloat16, 7, -133), (mx.float32, 23, -149)]:
    for m, n, k in [(1, 17, 64), (1, 37, 896), (1, 9, 4864), (32, 32, 128), (33, 17, 64)]:
        x = mx.array(rng.uniform(-1, 1, (m, k)), dtype=dtype)
        q = mx.array(rng.integers(0, 2**32, (n, k // 8), dtype=np.uint32))
        scales = mx.array(rng.uniform(0.01, 0.2, (n, k // 64)), dtype=dtype)
        biases = mx.array(rng.uniform(-1, 0, (n, k // 64)), dtype=dtype)
        mx.eval(x, q, scales, biases)
        mx.synchronize()
        x_host = np.array(x.astype(mx.float32)).astype(np.float64)
        words = np.array(q).astype(np.uint32)
        unpacked = ((words[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(n, k)
        weight = unpacked.astype(np.float64) * np.repeat(np.array(scales.astype(mx.float32)).astype(np.float64), 64, axis=-1)
        weight += np.repeat(np.array(biases.astype(mx.float32)).astype(np.float64), 64, axis=-1)
        expected = x_host @ weight.T
        gamma = (k + 3) * np.finfo(np.float32).eps / (1 - (k + 3) * np.finfo(np.float32).eps)
        ulp = np.exp2(np.maximum(minimum_exp, np.floor(np.log2(np.maximum(np.abs(expected), np.finfo(np.float64).tiny))) - precision))
        bound = np.abs(x_host) @ np.abs(weight).T * gamma + ulp
        outputs = []
        for defaults in (False, True):
            for key in ('MLX_OMARCHY_QMM_TILE_RB', 'MLX_OMARCHY_QMM_VEC_Q4_WORD'):
                if defaults:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = '0'
            value = mx.quantized_matmul(x, q, scales, biases, transpose=True, group_size=64, bits=4)
            mx.eval(value)
            mx.synchronize()
            actual = np.array(value.astype(mx.float32)).astype(np.float64)
            assert np.isfinite(actual).all(), (str(dtype), m, n, k, defaults, 'nonfinite')
            assert np.all(np.abs(actual - expected) <= bound), (str(dtype), m, n, k, defaults, 'host mismatch')
            outputs.append(actual)
        assert np.all(np.abs(outputs[0] - outputs[1]) <= 2 * bound), (str(dtype), m, n, k, 'baseline mismatch')
        results.append({'dtype': str(dtype), 'shape': [m, n, k], 'host_max_error': float(np.max(np.abs(outputs[1] - expected))), 'host_max_bound': float(np.max(bound)), 'baseline_max_difference': float(np.max(np.abs(outputs[0] - outputs[1])))})
print(json.dumps({'device': str(mx.default_device()), 'cases': results, 'passed': len(results)}, indent=2))

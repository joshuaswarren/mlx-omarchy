import json
import numpy as np
import mlx.core as mx

rng = np.random.default_rng(19)
results = []
for n, k in [(4, 65), (7, 127), (31, 128), (33, 129), (128, 896), (896, 4864), (4097, 257), (151936, 65)]:
    x = rng.integers(-8, 9, size=(1, k)).astype(np.float32) / 8
    w = rng.integers(-8, 9, size=(n, k)).astype(np.float32) / 8
    a = mx.array(x).astype(mx.bfloat16)
    b = mx.array(w).astype(mx.bfloat16)
    actual = np.array((a @ b.T).astype(mx.float32))
    reference = x @ w.T
    bits = reference.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    np.testing.assert_array_equal(actual, rounded)
    results.append({'n': n, 'k': k, 'exact_dyadic_reference': True})
print(json.dumps({'mlx_file': mx.__file__, 'checks': results, 'scope': 'Indexing, tail and reduction checks on exact dyadic inputs; not native model-logit parity.'}))

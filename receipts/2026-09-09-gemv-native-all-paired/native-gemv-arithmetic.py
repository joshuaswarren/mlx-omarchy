import json
import numpy as np
import mlx.core as mx


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def oracle(x, w, fused):
    n, k = w.shape
    partial = np.zeros((n, 32), dtype=np.float32)
    for block in range(0, k, 128):
        for item in range(4):
            indices = block + np.arange(32) * 4 + item
            valid = indices < k
            indices = indices[valid]
            if fused:
                partial[:, valid] = (partial[:, valid].astype(np.float64) + w[:, indices].astype(np.float64) * x[indices].astype(np.float64)).astype(np.float32)
            else:
                partial[:, valid] += w[:, indices] * x[indices]
    for step in (16, 8, 4, 2, 1):
        partial[:, :step] += partial[:, step:2 * step]
    return bf16(partial[:, 0])


rng = np.random.default_rng(71)
rows = []
for n, k in [(7, 65), (33, 129), (128, 896), (896, 4864), (4097, 257)]:
    x = bf16(rng.normal(size=k).astype(np.float32))
    w = bf16(rng.normal(size=(n, k)).astype(np.float32))
    actual = np.array((mx.array(x[None]).astype(mx.bfloat16) @ mx.array(w).astype(mx.bfloat16).T).astype(mx.float32))[0]
    fused = oracle(x, w, True)
    unfused = oracle(x, w, False)
    rows.append({'n': n, 'k': k, 'fused_mismatches': int(np.count_nonzero(actual != fused)), 'unfused_mismatches': int(np.count_nonzero(actual != unfused)), 'max_abs_error_fused': float(np.max(np.abs(actual - fused)))})
print(json.dumps({'checks': rows, 'scope': 'CPU emulation of the candidate partial-sum topology, fused and unfused arithmetic. Not native-runtime logit capture.'}))

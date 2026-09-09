import sys
from pathlib import Path

import numpy as np

source = Path(sys.argv[1])
q, k, v = [np.load(source / f'call0-sdpa-{key}.npy')[0] for key in ('q', 'k', 'v')]
expected = np.load(source / 'call0-sdpa-output.npy')[0]
k = np.repeat(k, q.shape[0] // k.shape[0], axis=0)
v = np.repeat(v, q.shape[0] // v.shape[0], axis=0)

def fma(a, b, c):
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)

scores = np.zeros((q.shape[0], q.shape[1], k.shape[1]), np.float32)
for dim in range(q.shape[-1]):
    scores = fma(q[:, :, dim, None], k[:, None, :, dim], scores)
scores *= np.float32(np.float32(0.125) * np.float32(np.log2(np.e)))
mask = np.arange(k.shape[1])[None, :] <= np.arange(q.shape[1])[:, None]
scores = np.where(mask, scores, -np.inf)
maximum = np.full(q.shape[:2], -np.inf, np.float32)
denominator = np.zeros(q.shape[:2], np.float32)
output = np.zeros_like(q)
exp_inputs = []
exp_reference = np.load(sys.argv[2]) if len(sys.argv) > 2 else None
exp_offset = 0
def exp2(value):
    global exp_offset
    exp_inputs.append(value.ravel())
    if exp_reference is None:
        return np.exp2(value.astype(np.float64)).astype(np.float32)
    result = exp_reference[exp_offset:exp_offset + value.size].reshape(value.shape)
    exp_offset += value.size
    return result
for start in range(0, k.shape[1], 32):
    stop = min(start + 32, k.shape[1])
    block = scores[:, :, start:stop]
    new_max = np.maximum(maximum, block.max(axis=-1))
    probabilities = exp2(block - new_max[..., None])
    factor = exp2(maximum - new_max)
    maximum = new_max
    padded = np.pad(probabilities, ((0, 0), (0, 0), (0, 32 - probabilities.shape[-1])))
    partial = padded.reshape(*q.shape[:2], 4, 8)
    partial = partial[..., 0::2] + partial[..., 1::2]
    partial = partial[..., 0::2] + partial[..., 1::2]
    partial = partial[..., 0] + partial[..., 1]
    total = np.zeros(q.shape[:2], np.float32)
    for frag in range(4):
        total += partial[..., frag]
    denominator = fma(denominator, factor, total)
    output *= factor[..., None]
    for index in range(stop - start):
        output = fma(probabilities[:, :, index, None], v[:, None, start + index, :], output)
np.save("/tmp/mlx-attention-exp-inputs.npy", np.concatenate(exp_inputs))
np.savez("/tmp/mlx-attention-predivide.npz", numerator=output, denominator=denominator)
output = (output / denominator[..., None]).astype(np.float16).astype(np.float32)
assert np.isfinite(output).all()
np.save('/tmp/mlx-attention-online-cpu.npy', output)
print('different', np.count_nonzero(output != expected), 'max_error', np.max(np.abs(output - expected)))

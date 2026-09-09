import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

source, out = map(Path, sys.argv[1:])
out.mkdir()
arrays = [np.load(source / f'call0-sdpa-{key}.npy') for key in ('q', 'k', 'v')]
expected = np.load(source / 'call0-sdpa-output.npy')
for dtype, label in ((mx.float16, 'f16'), (mx.float32, 'f32')):
    q, k, v = [mx.array(a).astype(dtype) for a in arrays]
    value = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask='causal').astype(mx.float16)
    mx.eval(value)
    actual = np.array(value.astype(mx.float32))
    np.save(out / f'{label}.npy', actual)
    print(json.dumps({'dtype': label, 'different': int(np.count_nonzero(actual != expected)), 'max_error': float(np.max(np.abs(actual - expected))), 'version': mx.__version__}), flush=True)

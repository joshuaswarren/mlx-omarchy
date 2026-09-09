import os
import sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
import mlx.core as mx
import numpy as np
from mlx_lm import load
out = Path(sys.argv[1])
out.mkdir()
model, _ = load(str(Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'))
for name in ('q_proj', 'k_proj', 'v_proj'):
    layer = getattr(model.model.layers[0].self_attn, name)
    x = mx.eye(896, dtype=layer.scales.dtype)
    effective = mx.quantized_matmul(x, layer.weight, layer.scales, layer.biases, transpose=True, group_size=64, bits=4).T
    expected = mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=64, bits=4)
    mx.eval(effective, expected)
    a = np.array(effective.astype(mx.float32))
    b = np.array(expected.astype(mx.float32))
    np.save(out / f'{name}-effective.npy',a)
    np.save(out / f'{name}-dequantized.npy',b)
    print(name, np.count_nonzero(a!=b), float(np.max(np.abs(a-b))),flush=True)

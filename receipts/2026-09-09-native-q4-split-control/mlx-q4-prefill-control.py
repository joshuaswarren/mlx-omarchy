import json
import os
import sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['MLX_DISABLE_COMPILE'] = '1'
sys.path.insert(0, '/tmp/mlx-native-oracle-scripts')
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_provenance import installed_provenance
inputs, out = map(Path, sys.argv[1:])
out.mkdir()
provenance = installed_provenance(dist_name='mlx' if sys.platform == 'darwin' else 'mlx-omarchy')
assert provenance['verified'] == 'match'
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
model, _ = load(str(Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'))
for name in ('q_proj', 'k_proj', 'v_proj'):
    layer = getattr(model.model.layers[0].self_attn, name)
    x = mx.array(np.load(inputs / f'call0-attn-{name}-input.npy')).astype(layer.scales.dtype)
    qmm = mx.quantized_matmul(x, layer.weight, layer.scales, layer.biases, transpose=True, group_size=64, bits=4)
    weight = mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=64, bits=4)
    rounded = x @ weight.T
    wide = mx.dequantize(layer.weight, layer.scales.astype(mx.float32), layer.biases.astype(mx.float32), group_size=64, bits=4)
    unrounded = (x.astype(mx.float32) @ wide.T).astype(x.dtype)
    mx.eval(qmm, rounded, unrounded)
    target = np.array(qmm.astype(mx.float32))
    split = max(1, min(512 // (((x.shape[-2] + 31) // 32) * ((layer.weight.shape[0] + 31) // 32)), x.shape[-1] // 64))
    while x.shape[-1] % (split * 64):
        split -= 1
    width = x.shape[-1] // split
    partials = mx.stack([x[..., i*width:(i+1)*width] @ weight[:, i*width:(i+1)*width].T for i in range(split)])
    split_result = mx.sum(partials, axis=0)
    for label, value in (("rounded", rounded), ("unrounded", unrounded), ("split", split_result)):
        actual = np.array(value.astype(mx.float32))
        np.save(out / f"{name}-{label}.npy", actual)
        print(json.dumps({'projection':name,'method':label,'different':int(np.count_nonzero(actual!=target)),'max_error':float(np.max(np.abs(actual-target)))}),flush=True)

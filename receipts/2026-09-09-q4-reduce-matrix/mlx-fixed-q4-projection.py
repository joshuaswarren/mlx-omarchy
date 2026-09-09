import json
import os
import sys
from pathlib import Path

os.environ['MLX_DISABLE_COMPILE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
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
model_path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'
model, _ = load(str(model_path))
for name in ('q_proj', 'k_proj', 'v_proj'):
    layer = getattr(model.model.layers[0].self_attn, name)
    for key in ('weight', 'scales', 'biases', 'bias'):
        value = getattr(layer, key)
        mx.eval(value)
        np.save(out / f'{name}-{key}.npy', np.array(value if key == 'weight' else value.astype(mx.float32)))
    for call in (0, 35):
        x = mx.array(np.load(inputs / f'call{call}-attn-{name}-input.npy')).astype(layer.scales.dtype)
        product = mx.quantized_matmul(x, layer.weight, layer.scales, layer.biases, transpose=True, group_size=layer.group_size, bits=layer.bits)
        result = layer(x)
        mx.eval(product, result)
        np.save(out / f'call{call}-{name}-product.npy', np.array(product.astype(mx.float32)))
        values = np.array(result.astype(mx.float32))
        np.save(out / f'call{call}-{name}-output.npy', values)
        expected = np.load(inputs / f'call{call}-attn-{name}-output.npy')
        print(json.dumps({'call':call,'projection':name,'dtype':str(x.dtype),'group_size':layer.group_size,'bits':layer.bits,'different':int(np.count_nonzero(values!=expected)),'max_error':float(np.max(np.abs(values-expected)))}), flush=True)

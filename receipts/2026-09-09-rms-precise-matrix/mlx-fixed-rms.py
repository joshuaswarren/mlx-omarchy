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
model_path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e'
model, _ = load(str(model_path))
norm = model.model.layers[0].input_layernorm
np.save(out / 'weight.npy', np.array(norm.weight.astype(mx.float32)))
for call in (0, 36):
    x = mx.array(np.load(inputs / f'call{call}-norm1-input.npy')).astype(mx.bfloat16)
    for label, value in [('weighted', norm(x)), ('unweighted', mx.fast.rms_norm(x, None, norm.eps)), ('float32', mx.fast.rms_norm(x.astype(mx.float32), None, norm.eps))]:
        mx.eval(value)
        result = np.array(value.astype(mx.float32))
        np.save(out / f'call{call}-{label}.npy', result)
        if label == 'weighted':
            expected = np.load(inputs / f'call{call}-norm1-output.npy')
            print(json.dumps({'call': call, 'different': int(np.count_nonzero(result != expected)), 'max_error': float(np.max(np.abs(result - expected)))}), flush=True)

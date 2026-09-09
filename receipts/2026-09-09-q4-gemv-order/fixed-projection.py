#!/usr/bin/env python3
"""Fixed-input screen: run every layer-0 projection of the pinned Q4 model on
the installed mlx-omarchy wheel with the captured native inputs and count
elements that differ from the captured native outputs.

usage: fixed-projection.py NATIVE_CAPTURE_DIR OUT_DIR [PROVENANCE_SCRIPTS_DIR]

call0 is the prefill call (M = prompt length, general Qmm path, unchanged by
this change); call35 is decode (M = 1, the qmm_vec Q4 word path under test).
"""
import json
import os
import sys
from pathlib import Path

os.environ['MLX_DISABLE_COMPILE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
inputs, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).resolve().parents[2] / 'scripts'))
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_provenance import installed_provenance

out.mkdir()
provenance = installed_provenance(dist_name='mlx' if sys.platform == 'darwin' else 'mlx-omarchy')
assert provenance['verified'] == 'match', provenance
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
model_path = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'
model, _ = load(str(model_path))
layer0 = model.model.layers[0]
targets = [('attn', 'q_proj'), ('attn', 'k_proj'), ('attn', 'v_proj'), ('attn', 'o_proj'),
           ('mlp', 'gate_proj'), ('mlp', 'up_proj'), ('mlp', 'down_proj')]
results = []
for label, name in targets:
    layer = getattr(layer0.self_attn if label == 'attn' else layer0.mlp, name)
    for call in (0, 35):
        x = mx.array(np.load(inputs / f'call{call}-{label}-{name}-input.npy')).astype(layer.scales.dtype)
        result = layer(x)
        mx.eval(result)
        values = np.array(result.astype(mx.float32))
        np.save(out / f'call{call}-{label}-{name}-output.npy', values)
        expected = np.load(inputs / f'call{call}-{label}-{name}-output.npy')
        assert values.shape == expected.shape, (values.shape, expected.shape)
        different = int(np.count_nonzero(values != expected))
        record = {'call': call, 'projection': name, 'm': int(x.shape[-2]), 'k': int(x.shape[-1]), 'n': int(values.shape[-1]),
                  'dtype': str(x.dtype), 'group_size': layer.group_size, 'bits': layer.bits,
                  'different': different, 'max_error': float(np.max(np.abs(values - expected)))}
        if different:
            idx = np.flatnonzero((values != expected).reshape(-1))[:8]
            record['first_mismatches'] = [{'index': int(i), 'device': float(values.reshape(-1)[i]), 'native': float(expected.reshape(-1)[i])} for i in idx]
        results.append(record)
        print(json.dumps(record), flush=True)
(out / 'results.json').write_text(json.dumps(results, indent=2))
decode_total = sum(r['different'] for r in results if r['call'] == 35)
prefill_total = sum(r['different'] for r in results if r['call'] == 0)
print(json.dumps({'decode_mismatches': decode_total, 'prefill_mismatches': prefill_total}), flush=True)

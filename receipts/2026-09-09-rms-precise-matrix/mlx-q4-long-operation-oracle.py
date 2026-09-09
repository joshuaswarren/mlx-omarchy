import json
import os
import sys
from pathlib import Path

os.environ['MLX_DISABLE_COMPILE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/tmp/mlx-native-oracle-scripts')
import bench_decode
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import qwen2
from mlx_provenance import installed_provenance

out = Path(sys.argv[1])
prompt = json.loads(Path('/tmp/mlx-long-manifest.json').read_text())['prompts']['long']['text']
out.mkdir()
prov = installed_provenance(dist_name='mlx' if sys.platform == 'darwin' else 'mlx-omarchy')
assert prov['verified'] == 'match'
(out / 'provenance.json').write_text(json.dumps(prov, indent=2))
model_call = qwen2.Model.__call__
block_call = qwen2.TransformerBlock.__call__
state = {'step': -1, 'layers': {}, 'targets': {}, 'capture_attention': False, 'rope_call': 0}

def save(name, value):
    mx.eval(value)
    np.save(out / name, np.array(value.astype(mx.float32)))

def model_trace(self, *args, **kwargs):
    state['step'] += 1
    state['layers'] = {id(layer): i for i, layer in enumerate(self.model.layers)}
    first = self.model.layers[0]
    state['first_attention'] = id(first.self_attn)
    state['targets'] = {id(first.input_layernorm): 'norm1', id(first.post_attention_layernorm): 'norm2'}
    for label, module in [('attn', first.self_attn), ('mlp', first.mlp)]:
        for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
            if hasattr(module, key):
                state['targets'][id(getattr(module, key))] = label + '-' + key
    result = model_call(self, *args, **kwargs)
    if state['step'] in (0, 35):
        save(f"step{state['step']}-logits.npy", result[:, -1:, :])
    return result

def block_trace(self, *args, **kwargs):
    capture = state['step'] in (0, 35) and state['layers'][id(self)] == 0
    name = f"step{state['step']}-layer{state['layers'][id(self)]:02d}"
    if capture:
        save(name + '-input.npy', args[0])
    result = block_call(self, *args, **kwargs)
    if capture:
        save(name + '-output.npy', result)
    return result

def report(prefill_ns, token_times, requested, ids=None, prompt_tokens=None, device=None):
    assert ids is not None and len(ids) == requested == 40
    data = {'ids': ids, 'digest': bench_decode.ids_digest(ids), 'prompt_tokens': prompt_tokens, 'device': device, 'purpose': 'Instrumented numerical trace, no performance claim'}
    (out / 'generation.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)

def wrap(original):
    def operation(self, *args, **kwargs):
        name = state['targets'].get(id(self))
        capture = name is not None and state['step'] in (0, 35)
        if capture:
            save(f"call{state['step']}-{name}-input.npy", args[0])
        result = original(self, *args, **kwargs)
        if capture:
            save(f"call{state['step']}-{name}-output.npy", result)
        return result
    return operation

attention_call = qwen2.Attention.__call__
rope_call = mx.fast.rope
sdpa_call = mx.fast.scaled_dot_product_attention

def attention_trace(self, *args, **kwargs):
    state['capture_attention'] = id(self) == state['first_attention'] and state['step'] in (0, 35)
    state['rope_call'] = 0
    try:
        return attention_call(self, *args, **kwargs)
    finally:
        state['capture_attention'] = False

def rope_trace(*args, **kwargs):
    result = rope_call(*args, **kwargs)
    if state['capture_attention']:
        name = f"call{state['step']}-rope{state['rope_call']}"
        save(name + '-input.npy', args[0])
        save(name + '-output.npy', result)
        state['rope_call'] += 1
    return result

def sdpa_trace(*args, **kwargs):
    result = sdpa_call(*args, **kwargs)
    if state['capture_attention']:
        for label, value in zip(('q', 'k', 'v'), args):
            save(f"call{state['step']}-sdpa-{label}.npy", value)
        save(f"call{state['step']}-sdpa-output.npy", result)
    return result

qwen2.Attention.__call__ = attention_trace
mx.fast.rope = rope_trace
mx.fast.scaled_dot_product_attention = sdpa_trace
nn.Linear.__call__ = wrap(nn.Linear.__call__)
nn.QuantizedLinear.__call__ = wrap(nn.QuantizedLinear.__call__)
nn.RMSNorm.__call__ = wrap(nn.RMSNorm.__call__)
qwen2.Model.__call__ = model_trace
qwen2.TransformerBlock.__call__ = block_trace
bench_decode.report = report
model = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'
sys.argv = ['bench_decode.py', '--model', str(model), '--prompt', prompt, '--tokens', '40', '--warmup-tokens', '0']
bench_decode.main()
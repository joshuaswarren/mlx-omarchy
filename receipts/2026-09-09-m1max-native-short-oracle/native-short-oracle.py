import json
import os
import sys
from pathlib import Path

os.environ['MLX_DISABLE_COMPILE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/tmp/mlx-native-oracle-scripts')
import bench_decode
import mlx.core as mx
import mlx_lm.sample_utils as samples
import numpy as np

out = Path('/tmp/mlx-native-short-oracle-aligned')
out.mkdir(exist_ok=True)
from mlx_provenance import installed_provenance
provenance = installed_provenance(dist_name='mlx')
assert provenance['verified'] == 'match' and provenance['mx_version'] == '0.32.2'
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
original_sampler = samples.make_sampler
original_report = bench_decode.report
for mode, pin in [('4bit', 'a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'), ('bf16', '56d07e766edd7159fbe12ed12d9cf114bf38bf1e')]:
    calls = []
    def make_sampler(*args, **kwargs):
        sampler = original_sampler(*args, **kwargs)
        def capture(logits):
            index = len(calls)
            calls.append(index)
            if index == 14:
                mx.eval(logits)
                np.save(out / f'{mode}-short-generated14-sampler-input.npy', np.array(logits.astype(mx.float32)))
            return sampler(logits)
        return capture
    def report(prefill_ns, token_times, requested, ids=None, prompt_tokens=None, device=None):
        assert ids is not None and len(ids) == requested == 32
        record = dict(model=mode, ids=ids, digest=bench_decode.ids_digest(ids), prompt_tokens=prompt_tokens, device=device, sampler_calls=len(calls), purpose='Numerical reference only; instrumented M1 Max is not same-chip performance evidence')
        (out / f'{mode}-short.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    samples.make_sampler = make_sampler
    bench_decode.report = report
    model = Path.home() / f'.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-{mode}/snapshots/{pin}'
    sys.argv = ['bench_decode.py', '--model', str(model), '--prompt', 'Hi', '--tokens', '32', '--warmup-tokens', '0']
    bench_decode.main()
samples.make_sampler = original_sampler
bench_decode.report = original_report

import json
import os
import sys
from pathlib import Path

os.environ['MLX_DISABLE_COMPILE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/tmp/mlx-native-oracle-scripts')
import bench_decode
from mlx_provenance import installed_provenance

out = Path('/tmp/mlx-native-long-oracle')
out.mkdir(exist_ok=True)
provenance = installed_provenance(dist_name='mlx')
assert provenance['verified'] == 'match' and provenance['mx_version'] == '0.32.2'
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
prompt = json.loads(Path('/tmp/mlx-long-manifest.json').read_text())['prompts']['long']['text']
for mode, pin in [('4bit', 'a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'), ('bf16', '56d07e766edd7159fbe12ed12d9cf114bf38bf1e')]:
    def report(prefill_ns, token_times, requested, ids=None, prompt_tokens=None, device=None):
        assert ids is not None and len(ids) == requested == 128
        assert prompt_tokens == 262
        record = dict(model=mode, ids=ids, digest=bench_decode.ids_digest(ids), prompt_tokens=prompt_tokens, device=device, purpose='Numerical reference only; M1 Max is not same-chip performance evidence')
        (out / f'{mode}-long.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    bench_decode.report = report
    model = Path.home() / f'.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-{mode}/snapshots/{pin}'
    sys.argv = ['bench_decode.py', '--model', str(model), '--prompt', prompt, '--tokens', '128', '--warmup-tokens', '4']
    bench_decode.main()

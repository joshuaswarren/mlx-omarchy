import json
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
out = root / 'receipts/2026-09-09-native-guard-profile'
py = root / '.venv-profile/bin/python'
models = [('4bit','a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'), ('bf16','56d07e766edd7159fbe12ed12d9cf114bf38bf1e')]
for name, revision in models:
    model = snapshot_download('mlx-community/Qwen2.5-0.5B-Instruct-' + name, revision=revision, local_files_only=True)
    trace = out / (name + '.jsonl')
    markers = out / (name + '-markers.jsonl')
    env = {k:v for k,v in os.environ.items() if not k.startswith('MLX_')}
    env.update(MLX_DISABLE_COMPILE='1', MLX_OMARCHY_GPU_PROFILE=str(trace), HF_HUB_OFFLINE='1')
    with (out / (name + '.log')).open('w') as log:
        subprocess.run([str(py), 'scripts/profile_generate.py', '--model', model, '--prompt', 'Explain how a computer executes a program, step by step, in detail.', '--max-tokens', '32', '--markers', str(markers)], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    events = [json.loads(s) for s in markers.read_text().splitlines()]
    assert trace.stat().st_size > 0
    with (out / (name + '-analysis.txt')).open('w') as log:
        subprocess.run([str(py), 'scripts/profile_analyze.py', str(trace), '--markers', str(markers), '--compute-h', 'overlay/mlx/backend/omarchy/compute.h'], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    print('PROFILE_CAPTURED',name,len(events),trace.stat().st_size,flush=True)

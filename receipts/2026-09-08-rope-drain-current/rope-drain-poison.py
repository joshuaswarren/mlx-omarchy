import json
import shutil
import subprocess
from pathlib import Path

root = Path.home() / 'src/mlx-rope-drain-770ae465'
out = root / 'receipts/2026-09-08-rope-drain-current/poison'
out.mkdir(exist_ok=False)
source = root / 'receipts/parity-baseline-20260908'
text = (source / 'run-baseline.py').read_text()
needle = 'env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1",'
assert text.count(needle) == 1
text = text.replace(needle, 'env.update(MLX_OMARCHY_POISON_FREED="1", HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1",')
text = text.replace('ROOT = HERE.parents[1]', 'ROOT = Path(' + repr(str(root)) + ')')
(out / 'run.py').write_text(text)
shutil.copyfile(source / 'capture-ids.py', out / 'capture-ids.py')
py = root / '.venv-accept/bin/python'
wheels = list((root / 'dist').glob('*.whl'))
assert len(wheels) == 1
subprocess.run([str(py), str(out / 'run.py'), str(py), str(wheels[0]), '1'], cwd=root, check=True, timeout=3600)
ids = lambda p: [json.loads(s)['ids'] for s in p.read_text().splitlines()]
assert ids(out / 'rep1.ids.jsonl') == ids(root / 'receipts/2026-09-08-rope-drain-current/pair-1-candidate/rep1.ids.jsonl')
print('POISON_FULL_ID_PASS', flush=True)

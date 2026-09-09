import json
import subprocess
from pathlib import Path

root = Path.home() / 'src/mlx-wide-gemv-screen'
out = root / 'receipts/2026-09-09-wide-layout-screen'
out.mkdir(exist_ok=False)
source = root / 'receipts/parity-baseline-20260908'
hook = r'''import mlx_lm.utils as _utils
_original_load = _utils.load

def _layout_load(*args, **kwargs):
    model, tokenizer = _original_load(*args, **kwargs)
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    updates = []
    for key, value in tree_flatten(model.parameters()):
        if value.ndim == 2 and value.dtype == mx.bfloat16 and value.shape[0] >= 4096:
            packed = mx.contiguous(value.T).T
            mx.eval(packed)
            assert mx.array_equal(value, packed).item(), key
            updates.append((key, packed))
    model.update(tree_unflatten(updates))
    import os, json
    with open(os.environ["MLX_PAIR_IDS"] + ".layout.jsonl", "a") as log:
        log.write(json.dumps({"checked_matrices": len(updates), "all_equal": True}) + "\n")
    return model, tokenizer

_utils.load = _layout_load
'''
(out / 'capture-ids.py').write_text(hook + (source / 'capture-ids.py').read_text())
text = (source / 'run-baseline.py').read_text().replace('ROOT = HERE.parents[1]', 'ROOT = Path(' + repr(str(root)) + ')')
(out / 'run.py').write_text(text)
py = root / '.venv-accept/bin/python'
wheels = list((root / 'dist').glob('*.whl'))
assert len(wheels) == 1
subprocess.run([str(py), str(out / 'run.py'), str(py), str(wheels[0]), '1'], cwd=root, check=True, timeout=3600)
ids = lambda p: [json.loads(s)['ids'] for s in p.read_text().splitlines()]
result = {'all_full_ids_equal': ids(out / 'rep1.ids.jsonl') == ids(root / 'receipts/2026-09-09-wide-gemv-accept/pair-1-candidate/rep1.ids.jsonl'), 'scope': 'Diagnostic load-time physical layout permutation only; every BF16 matrix checked equal elementwise; default backend unchanged.'}
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('LAYOUT_SCREEN_DONE', result, flush=True)

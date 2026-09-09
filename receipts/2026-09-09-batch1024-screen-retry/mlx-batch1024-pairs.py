import importlib.util
import json
import shutil
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = Path.home() / 'src/mlx-q4-direct-final'
source = root / 'receipts/parity-baseline-20260908'
spec = importlib.util.spec_from_file_location('runner', source / 'run-baseline.py')
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
setattr(runner, "ROOT", root)
out = root / 'receipts/2026-09-09-batch1024-screen-retry'
out.mkdir(exist_ok=True)
expected = [json.loads(s)['ids'] for s in (base / 'receipts/2026-09-09-q4-direct-final/pair-1-candidate/rep1.ids.jsonl').read_text().splitlines()]
legs = {}
for side, tree in [('baseline', base), ('candidate', root)]:
    wheels = list((tree / 'dist').glob('*.whl'))
    assert len(wheels) == 1
    legs[side] = (tree / '.venv-accept/bin/python', wheels[0])
runs = []
for rep in range(1, 3):
    pair = {}
    for side in (('baseline', 'candidate') if rep % 2 else ('candidate', 'baseline')):
        folder = out / f'pair-{rep}-{side}'
        folder.mkdir()
        shutil.copyfile(source / 'capture-ids.py', folder / 'capture-ids.py')
        setattr(runner, "HERE", folder)
        py, wheel = legs[side]
        sys.argv = ['run-baseline.py', str(py), str(wheel), '1']
        runner.main()
        ids = [json.loads(s)['ids'] for s in (folder / 'rep1.ids.jsonl').read_text().splitlines()]
        assert ids == expected
        print('LINUX_ID_EQUALITY', side, ids == expected, flush=True)
        pair[side] = {r['leg_id']: r for r in json.loads((folder / 'baseline-summary.json').read_text())['legs']}
    runs.append(pair)
    print('PAIRED_SCREEN_COMPLETE', rep, flush=True)
rows = []
for key in sorted(runs[0]['baseline']):
    row = {'leg_id': key}
    for metric in ('decode_tok_s_median', 'prefill_tok_s_median'):
        values = {side: [p[side][key][metric] for p in runs] for side in legs}
        medians = {side: statistics.median(v) for side, v in values.items()}
        row[metric] = {'samples': values, 'medians': medians, 'candidate_over_baseline': medians['candidate'] / medians['baseline']}
    rows.append(row)
(out / 'paired-summary.json').write_text(json.dumps({'paired_repetitions': 2, 'measured_legs': 24, 'all_full_generated_ids_equal': ids == expected, 'results': rows}, indent=2) + '\n')
print('BATCH1024_SCREEN_DONE', flush=True)
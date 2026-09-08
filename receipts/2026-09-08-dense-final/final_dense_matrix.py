import importlib.util
import json
import shutil
import statistics
import sys
from pathlib import Path

root, baseline, candidate_python, candidate_wheel = map(Path, sys.argv[1:])
source = root / 'receipts/parity-baseline-20260908'
spec = importlib.util.spec_from_file_location('matrix_runner', source / 'run-baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
base_wheels = list((baseline / 'dist-parity-release').glob('*.whl'))
assert len(base_wheels) == 1, base_wheels
legs = {
    'baseline': (baseline / '.venv-benchmark/bin/python', base_wheels[0]),
    'candidate': (candidate_python, candidate_wheel),
}
expected_ids = [json.loads(line)['ids'] for line in (baseline / 'receipts/parity-baseline-20260908/rep1.ids.jsonl').read_text().splitlines()]
assert len(expected_ids) == 6
output = root / 'receipts/2026-09-08-dense-final'
output.mkdir(parents=True, exist_ok=True)
runs = []
for rep in range(1, 6):
    pair = {}
    for side in (('baseline', 'candidate') if rep % 2 else ('candidate', 'baseline')):
        folder = output / f'pair-{rep}-{side}'
        folder.mkdir()
        shutil.copyfile(source / 'capture-ids.py', folder / 'capture-ids.py')
        runner.HERE = folder
        py, wheel = legs[side]
        sys.argv = ['run-baseline.py', str(py), str(wheel), '1']
        runner.main()
        ids = [json.loads(line)['ids'] for line in (folder / 'rep1.ids.jsonl').read_text().splitlines()]
        assert ids == expected_ids, (rep, side, 'full generated-ID mismatch')
        summary = json.loads((folder / 'baseline-summary.json').read_text())
        pair[side] = {row['leg_id']: row for row in summary['legs']}
    assert pair['baseline'].keys() == pair['candidate'].keys()
    runs.append(pair)
    print('PAIRED_FULL_ID_PASS', rep, flush=True)
rows = []
for key in sorted(runs[0]['baseline']):
    row = {'leg_id': key}
    for metric in ('decode_tok_s_median', 'prefill_tok_s_median'):
        values = {side: [pair[side][key][metric] for pair in runs] for side in legs}
        medians = {side: statistics.median(v) for side, v in values.items()}
        row[metric] = {'samples': values, 'medians': medians, 'candidate_over_baseline': medians['candidate'] / medians['baseline']}
    rows.append(row)
result = {'source_commit': 'c19e1ecd', 'paired_repetitions': 5, 'measured_legs': 60, 'all_full_generated_ids_equal': True, 'results': rows}
(output / 'paired-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)

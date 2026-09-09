#!/usr/bin/env python3
"""Paired A/B of the six 0.5B legs: alternate baseline and candidate trees
(each with its own .venv-accept and single dist/ wheel), assert the two
wheels carry the expected, different source stamps, and summarize medians
and candidate/baseline ratios.

usage: paired-legs.py CANDIDATE_ROOT BASELINE_ROOT OUT_DIR PAIRS CANDIDATE_COMMIT BASELINE_COMMIT
"""
import importlib.util
import json
import shutil
import statistics
import sys
from pathlib import Path

root, base, out = (Path(a).absolute() for a in sys.argv[1:4])
pairs = int(sys.argv[4])
stamps = {'candidate': sys.argv[5], 'baseline': sys.argv[6]}
assert stamps['candidate'] != stamps['baseline'], stamps
source = root / 'receipts/parity-baseline-20260908'
spec = importlib.util.spec_from_file_location('runner', source / 'run-baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
out.mkdir(exist_ok=True)
sides = {}
for side, tree in (('baseline', base), ('candidate', root)):
    wheels = list((tree / 'dist').glob('*.whl'))
    assert len(wheels) == 1, (side, wheels)
    assert f'+{stamps[side]}-' in wheels[0].name, (side, wheels[0].name, stamps[side])
    sides[side] = (tree / '.venv-accept/bin/python', wheels[0])
native = json.loads((root / 'receipts/native-baseline-2026-09-06/native-2026-09-06-nc1.json').read_text())
native_digest = {leg['leg_id']: leg['metrics']['generated_ids_sha256_16'] for leg in native['legs'] if leg.get('status') == 'measured'}
runs = []
for rep in range(1, pairs + 1):
    pair = {}
    for side in (('baseline', 'candidate') if rep % 2 else ('candidate', 'baseline')):
        folder = out / f'pair-{rep}-{side}'
        folder.mkdir()
        shutil.copyfile(source / 'capture-ids.py', folder / 'capture-ids.py')
        runner.HERE = folder
        py, wheel = sides[side]
        sys.argv = ['run-baseline.py', str(py), str(wheel), '1']
        runner.main()
        pair[side] = {leg['leg_id']: leg for leg in json.loads((folder / 'baseline-summary.json').read_text())['legs']}
        print('PAIR', rep, side, 'done', flush=True)
    runs.append(pair)
rows = []
for key in sorted(runs[0]['baseline']):
    row = {'leg_id': key, 'native_digest': native_digest.get(key)}
    for side in sides:
        digests = sorted({d for p in runs for d in p[side][key]['generated_ids_sha256_16']})
        row[f'{side}_digests'] = digests
        row[f'{side}_native_match'] = digests == [native_digest.get(key)]
    for metric in ('decode_tok_s_median', 'prefill_tok_s_median'):
        values = {side: [p[side][key][metric] for p in runs] for side in sides}
        medians = {side: statistics.median(v) for side, v in values.items()}
        row[metric] = {'samples': values, 'medians': medians, 'candidate_over_baseline': medians['candidate'] / medians['baseline']}
    rows.append(row)
summary = {'paired_repetitions': pairs, 'stamps': stamps,
           'wheels': {side: {'path': str(w), 'sha256': __import__('hashlib').sha256(w.read_bytes()).hexdigest()} for side, (_, w) in sides.items()},
           'results': rows}
(out / 'paired-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for row in rows:
    print(row['leg_id'], 'decode', row['decode_tok_s_median']['medians'], 'ratio %.3f' % row['decode_tok_s_median']['candidate_over_baseline'],
          'native_match', row['baseline_native_match'], row['candidate_native_match'], flush=True)
print('PAIRED_DONE', flush=True)

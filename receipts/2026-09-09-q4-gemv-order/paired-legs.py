#!/usr/bin/env python3
"""Paired A/B(/C) of the six 0.5B legs. Each side is a checkout with its
own .venv-accept and a single dist/ wheel; the wheel is installed into
that venv, its version stamp asserted against the expected commit, and
the sides alternate order every repetition. Summarizes medians, ratios
against the first side, and native digest matches.

usage: paired-legs.py RUNNER_ROOT OUT_DIR REPS NAME=TREE=STAMP [NAME=TREE=STAMP ...]
"""
import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

root, out = (Path(a).absolute() for a in sys.argv[1:3])
reps = int(sys.argv[3])
sides = {}
for spec in sys.argv[4:]:
    name, tree, stamp = spec.split('=')
    tree = Path(tree).absolute()
    wheels = list((tree / 'dist').glob('*.whl'))
    assert len(wheels) == 1, (name, wheels)
    assert f'+{stamp}-' in wheels[0].name, (name, wheels[0].name, stamp)
    py = tree / '.venv-accept/bin/python'
    subprocess.run([str(py), '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', str(wheels[0])], check=True)
    version = subprocess.run([str(py), '-c', 'import mlx.core as mx; print(mx.__version__)'], check=True, capture_output=True, text=True).stdout.strip()
    assert version.endswith('+' + stamp), (name, version, stamp)
    sides[name] = {'python': py, 'wheel': wheels[0], 'stamp': stamp, 'version': version,
                   'sha256': hashlib.sha256(wheels[0].read_bytes()).hexdigest()}
assert len({s['stamp'] for s in sides.values()}) == len(sides), 'sides must be different builds'
names = list(sides)
source = root / 'receipts/parity-baseline-20260908'
spec = importlib.util.spec_from_file_location('runner', source / 'run-baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
out.mkdir(exist_ok=True)
native = json.loads((root / 'receipts/native-baseline-2026-09-06/native-2026-09-06-nc1.json').read_text())
native_digest = {leg['leg_id']: leg['metrics']['generated_ids_sha256_16'] for leg in native['legs'] if leg.get('status') == 'measured'}
runs = []
for rep in range(1, reps + 1):
    order = names[rep - 1:] + names[:rep - 1]
    result = {}
    for name in order:
        folder = out / f'rep-{rep}-{name}'
        folder.mkdir()
        shutil.copyfile(source / 'capture-ids.py', folder / 'capture-ids.py')
        runner.HERE = folder
        sys.argv = ['run-baseline.py', str(sides[name]['python']), str(sides[name]['wheel']), '1']
        runner.main()
        result[name] = {leg['leg_id']: leg for leg in json.loads((folder / 'baseline-summary.json').read_text())['legs']}
        print('REP', rep, name, 'done', flush=True)
    runs.append(result)
rows = []
for key in sorted(runs[0][names[0]]):
    row = {'leg_id': key, 'native_digest': native_digest.get(key), 'sides': {}}
    for name in names:
        digests = sorted({d for r in runs for d in r[name][key]['generated_ids_sha256_16']})
        entry = {'digests': digests, 'native_match': digests == [native_digest.get(key)]}
        for metric in ('decode_tok_s_median', 'prefill_tok_s_median'):
            samples = [r[name][key][metric] for r in runs]
            entry[metric.replace('_median', '')] = {'samples': samples, 'median': statistics.median(samples)}
        row['sides'][name] = entry
    first = row['sides'][names[0]]
    for name in names[1:]:
        row['sides'][name]['decode_over_' + names[0]] = row['sides'][name]['decode_tok_s']['median'] / first['decode_tok_s']['median']
        row['sides'][name]['prefill_over_' + names[0]] = row['sides'][name]['prefill_tok_s']['median'] / first['prefill_tok_s']['median']
    rows.append(row)
summary = {'repetitions': reps, 'sides': {n: {k: str(v) for k, v in s.items()} for n, s in sides.items()}, 'results': rows}
(out / 'paired-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for row in rows:
    print(row['leg_id'], {n: (round(e['decode_tok_s']['median'], 2), e['native_match']) for n, e in row['sides'].items()}, flush=True)
print('PAIRED_DONE', flush=True)

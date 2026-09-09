#!/usr/bin/env python3
"""Run the six 0.5B bench legs once through receipts/parity-baseline-20260908/
run-baseline.py and compare each generated-ID digest against the native
M1 baseline.

usage: run-legs.py REPO_ROOT VENV_PYTHON WHEEL OUT_DIR
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

root, py, wheel, out = (Path(a).resolve() for a in sys.argv[1:5])
source = root / 'receipts/parity-baseline-20260908'
out.mkdir(exist_ok=True)
shutil.copyfile(source / 'capture-ids.py', out / 'capture-ids.py')
spec = importlib.util.spec_from_file_location('runner', source / 'run-baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
runner.HERE = out
sys.argv = ['run-baseline.py', str(py), str(wheel), '1']
runner.main()

native = json.loads((root / 'receipts/native-baseline-2026-09-06/native-2026-09-06-nc1.json').read_text())
native_digest = {}
for leg in native['legs']:
    if leg.get('status') == 'measured':
        native_digest[leg['leg_id']] = leg['metrics']['generated_ids_sha256_16']
summary = json.loads((out / 'baseline-summary.json').read_text())
for leg in summary['legs']:
    leg['native_digest'] = native_digest.get(leg['leg_id'])
    leg['native_match'] = leg['generated_ids_sha256_16'] == [leg['native_digest']]
summary['native_baseline'] = 'native-baseline-2026-09-06/native-2026-09-06-nc1.json'
summary['native_generation_matches'] = sum(1 for leg in summary['legs'] if leg['native_match'])
(out / 'baseline-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps([{k: leg[k] for k in ('leg_id', 'decode_tok_s_median', 'prefill_tok_s_median', 'generated_ids_sha256_16', 'native_digest', 'native_match')} for leg in summary['legs']], indent=1))

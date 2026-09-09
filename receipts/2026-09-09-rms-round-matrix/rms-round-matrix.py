import importlib.util
import shutil
import sys
from pathlib import Path

root = Path.home() / 'src/mlx-rms-round-screen'
source = root / 'receipts/parity-baseline-20260908'
out = root / 'receipts/2026-09-09-rms-round-matrix'
out.mkdir()
shutil.copyfile(source / 'capture-ids.py', out / 'capture-ids.py')
spec = importlib.util.spec_from_file_location('runner', source / 'run-baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
runner.HERE = out
wheels = list((root / 'dist').glob('*.whl'))
assert len(wheels) == 1
sys.argv = ['run-baseline.py', str(root / '.venv-accept/bin/python'), str(wheels[0]), '1']
runner.main()

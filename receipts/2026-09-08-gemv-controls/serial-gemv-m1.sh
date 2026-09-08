#!/usr/bin/env bash
set -euo pipefail
hostname
date -u +%FT%TZ
root="$HOME/src/mlx-serial-gemv-64dea25b"
base="$HOME/src/mlx-parity-baseline-20260908"
prior="$HOME/src/mlx-coalesced-gemv-01b28eb6"
test -x "$base/.venv-benchmark/bin/python"
test ! -e "$root"
git clone --quiet --shared "$prior" "$root"
git -C "$root" checkout --quiet --detach 64dea25b
cd "$root"
test "$(git rev-parse --short=8 HEAD)" = 64dea25b
test -z "$(git status --porcelain)"
printf 'SOURCE=%s\n' "$(git rev-parse HEAD)"
unset MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION HK_PERF
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/serial-gemv-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
wheel="$root/${wheels[0]}"
out="$root/receipts/2026-09-08-serial-gemv"
mkdir -p "$out"
cp "$prior/receipts/2026-09-08-coalesced-gemv/run-baseline.py" "$out/"
cp "$prior/receipts/2026-09-08-coalesced-gemv/capture-ids.py" "$out/"
sed -i 's/coalesced sequential GEMV/serial sequential GEMV/g;s/jwm1-linux-coalesced-gemv/jwm1-linux-serial-gemv/g' "$out/run-baseline.py"
"$base/.venv-benchmark/bin/python" -m pip freeze --exclude mlx --exclude mlx-omarchy --exclude mlx-lm > "$out/requirements-baseline.txt"
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps -r "$out/requirements-baseline.txt" "$wheel" mlx-lm==0.31.3 > /tmp/serial-gemv-install.log 2>&1
.venv-accept/bin/python "$out/run-baseline.py" "$root/.venv-accept/bin/python" "$wheel" 1
python3 - "$base" "$out" <<'PY'
import json
import sys
from pathlib import Path
base, out = map(Path, sys.argv[1:])
baseline = base / 'receipts/parity-baseline-20260908'
a = [json.loads(line) for line in (baseline / 'rep1.ids.jsonl').read_text().splitlines()]
b = [json.loads(line) for line in (out / 'rep1.ids.jsonl').read_text().splitlines()]
assert len(a) == len(b) == 6
old = {r['leg_id']:r for r in json.loads((baseline / 'baseline-summary.json').read_text())['legs']}
new = json.loads((out / 'baseline-summary.json').read_text())['legs']
rows = [{'leg_id':r['leg_id'], 'decode_ratio':r['decode_tok_s_median']/old[r['leg_id']]['decode_tok_s_median'], 'prefill_ratio':r['prefill_tok_s_median']/old[r['leg_id']]['prefill_tok_s_median']} for r in new]
result = {'all_ids_equal':all(x['ids']==y['ids'] for x,y in zip(a,b)), 'per_leg_ids_equal':[x['ids']==y['ids'] for x,y in zip(a,b)], 'ratios':rows, 'scope':'one candidate repetition versus the recorded five-repetition baseline; not final acceptance'}
(out / 'smoke-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
date -u +%FT%TZ
printf 'SERIAL_GEMV_SMOKE_DONE\n'

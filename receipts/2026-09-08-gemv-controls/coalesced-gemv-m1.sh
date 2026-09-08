#!/usr/bin/env bash
set -euo pipefail
hostname
date -u +%FT%TZ
root="$HOME/src/mlx-coalesced-gemv-01b28eb6"
base="$HOME/src/mlx-parity-baseline-20260908"
test -x "$base/.venv-benchmark/bin/python"
test ! -e "$root"
git clone --quiet --shared "$HOME/src/mlx-bf16-rope-20260908" "$root"
git -C "$root" fetch --quiet /tmp/coalesced-gemv.bundle refs/heads/perf/coalesced-serial-gemv
git -C "$root" checkout --quiet --detach FETCH_HEAD
cd "$root"
test "$(git rev-parse --short=8 HEAD)" = 01b28eb6
test -z "$(git status --porcelain)"
printf 'SOURCE=%s\n' "$(git rev-parse HEAD)"
unset MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION HK_PERF
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/coalesced-gemv-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
wheel="$root/${wheels[0]}"
out="$root/receipts/2026-09-08-coalesced-gemv"
"$base/.venv-benchmark/bin/python" -m pip freeze --exclude mlx --exclude mlx-omarchy --exclude mlx-lm > "$out/requirements-baseline.txt"
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps -r "$out/requirements-baseline.txt" "$wheel" mlx-lm==0.31.3 > /tmp/coalesced-gemv-install.log 2>&1
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
old = json.loads((baseline / 'baseline-summary.json').read_text())['legs']
new = json.loads((out / 'baseline-summary.json').read_text())['legs']
old = {row['leg_id']: row for row in old}
rows = []
for row in new:
    ref = old[row['leg_id']]
    rows.append({'leg_id': row['leg_id'], 'decode_ratio': row['decode_tok_s_median'] / ref['decode_tok_s_median'], 'prefill_ratio': row['prefill_tok_s_median'] / ref['prefill_tok_s_median']})
result = {'all_ids_equal': all(x['ids'] == y['ids'] for x, y in zip(a, b)), 'per_leg_ids_equal': [x['ids'] == y['ids'] for x, y in zip(a, b)], 'ratios': rows, 'scope': 'one candidate repetition versus the recorded five-repetition baseline; not final acceptance'}
(out / 'smoke-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
date -u +%FT%TZ
printf 'COALESCED_GEMV_SMOKE_DONE\n'

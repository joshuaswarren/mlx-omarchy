#!/usr/bin/env bash
set -euo pipefail
ssh jwm1-linux "python3 - <<'PY'
import json, os
me=os.getpid()
excluded=set()
p=me
while p>1 and p not in excluded:
    excluded.add(p)
    try:
        fields=open(f'/proc/{p}/stat').read().split()
        p=int(fields[3])
    except Exception:
        break
markers=(b'mlx-bf16-gemv-sourcefix-4f90815c/.work-sourcefix', b'bf16-gemv-sourcefix-continuation-20260913T085453Z', b'sourcefix-continuation.sh')
res=[]
for name in os.listdir('/proc'):
    if not name.isdigit() or int(name) in excluded: continue
    try: cmd=open(f'/proc/{name}/cmdline','rb').read().replace(b'\\0',b' ')
    except OSError: continue
    if any(m in cmd for m in markers): res.append({'pid':int(name),'cmd':cmd.decode(errors='replace')})
print(json.dumps({'excluded_ancestry':sorted(excluded),'residual_workers':res},sort_keys=True))
assert not res, res
PY
if flock -n /tmp/m1-gpu.lock true; then echo lock_free=true; else echo lock_free=false; exit 1; fi"

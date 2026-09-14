#!/bin/bash
set -euo pipefail
DIR=/var/tmp/jw16-ane-soak-20260913
cd "$DIR"
python3 device_health.py > pre-health.json
sudo -n python3 set_read.py > set-pre.json
python3 - <<'PY'
import json, sys
s = json.load(open("set-pre.json"))
if s["set0"]["ACTUAL"] != "0xf":
    print("ABORT SET0 not 0xf", s, file=sys.stderr)
    sys.exit(2)
print("SET0_PRE", json.dumps(s["set0"]))
PY
date --iso-8601=seconds > started_at
set +e
python3 run_100.py | tee soak-summary.json
rc=${PIPESTATUS[0]}
set -e
date --iso-8601=seconds > finished_at
python3 device_health.py > post-health.json
sudo -n python3 set_read.py > set-post.json
echo RUN_RC=$rc
echo SET_POST
cat set-post.json
python3 - <<'PY'
import hashlib, pathlib, collections
runs = pathlib.Path("runs")
hashes = collections.Counter()
for p in sorted(runs.glob("y_*.bin")):
    hashes[hashlib.sha256(p.read_bytes()).hexdigest()] += 1
print("Y_FILES", sum(hashes.values()))
print("Y_HASHES", dict(hashes))
PY
sha256sum /var/tmp/jw16-ane-first-exec/mlx-omarchy-ane-worker /var/tmp/jw16-ane-first-exec/libane.so
cat /proc/sys/kernel/random/boot_id
exit "$rc"

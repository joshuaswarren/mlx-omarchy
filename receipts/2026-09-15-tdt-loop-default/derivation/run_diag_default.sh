#!/bin/bash
# Native-arm gate: no flags, the loop must reproduce the native decision log.
set -e
exec 9>/tmp/m1-gpu.lock
flock -x -w 3600 9
mkdir -p /tmp/vulkan-tdt-142
cd /var/tmp/TdtLoopDefault
PYTHONPATH=/var/tmp/TdtLoopDefault/pkg:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages \
  ~/venv-agxgen/bin/python fused_diagnose.py > fused_diagnose_out.json 2>&1
echo DIAG_EXIT=$?
python3 - <<'PYEOF'
import json
d = json.load(open("fused_diagnose_out.json"))
print("decode_path:", d["free_decode"]["decode_path"])
print("first_divergence:", d["free_decode"]["first_divergence"])
PYEOF

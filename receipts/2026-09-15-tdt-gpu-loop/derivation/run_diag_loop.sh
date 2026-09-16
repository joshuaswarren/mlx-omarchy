#!/bin/bash
# Native-arm gate for the GPU-resident TDT loop (--tdt-loop).
set -e
exec 9>/tmp/m1-gpu.lock
flock -x -w 3600 9
mkdir -p /tmp/vulkan-tdt-142
cd /var/tmp/TdtGpuLoop
PYTHONPATH=/var/tmp/TdtGpuLoop/pkg:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages \
  ~/venv-agxgen/bin/python fused_diagnose.py --tdt-loop > fused_diagnose_out.json 2>&1
echo DIAG_EXIT=$?
python3 - <<'PYEOF'
import json
d = json.load(open("fused_diagnose_out.json"))
print("first_divergence:", d["free_decode"]["first_divergence"])
d142 = d.get("decision_142")
if d142:
    print("decision142 token:", d142["actual_token_id"], "logit delta:", d142["actual_minus_native_logit"])
PYEOF

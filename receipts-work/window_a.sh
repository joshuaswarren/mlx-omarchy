#!/usr/bin/env bash
# Window A: BF16 decode shape->kernel probe + attribution arms A.
set -euo pipefail
cd ~/src/mlx-Bf16DecodeAttribution
OUT=receipts-work/2026-09-11-bf16-decode-attribution
mkdir -p "$OUT/arms" "$OUT/probe"
RUNPY=$PWD/.work/venv-run-bf16dec/bin/python
ABLPY=$PWD/.work/venv-ablate-bf16dec/bin/python
CTRL=$PWD/.work/venv-bf16dec/bin/python
BASE=$HOME/src/mlx-bf16dec-base/dist/mlx_omarchy-0.32.2.dev202609111524+a5b8c4a-cp314-cp314-linux_aarch64.whl
ABLDIST=dist/mlx_omarchy-0.32.2.dev202609111520+d2ef0db-cp314-cp314-linux_aarch64.whl
export MLX_DISABLE_COMPILE=1
export HF_HUB_OFFLINE=1
ulimit -c 0

# ---- 1. shape->kernel probe: one process per decode matmul shape ----
for spec in "qproj 1 896 896" "kproj 1 896 128" "gate 1 896 4864" "down 1 4864 896" "lmhead 1 896 151936"; do
  set -- $spec
  name=$1; m=$2; k=$3; n=$4
  MLX_OMARCHY_GPU_PROFILE="$OUT/probe/$name.jsonl" \
  MLX_OMARCHY_GPU_PROFILE_LABEL="$name" \
  $CTRL - "$m" "$k" "$n" <<'PYEOF'
import sys
import mlx.core as mx
import numpy as np
m, k, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(0)
x = mx.array(rng.standard_normal((m, k)).astype(np.float16)).astype(mx.bfloat16)
w = mx.array(rng.standard_normal((n, k)).astype(np.float16)).astype(mx.bfloat16)
y = x @ w.transpose()
mx.eval(y)
print("probe", m, k, n, "ok")
PYEOF
done

# analyze: kernel enum -> name per probe file
python3 - <<'PYEOF'
import collections, glob, json, os, re
hdr_path = os.path.expanduser(
    "~/src/mlx-Bf16DecodeAttribution/overlay/mlx/backend/omarchy/compute.h")
names, inside = [], False
for line in open(hdr_path):
    if "enum class ComputeKernel" in line:
        inside = True
        continue
    if inside:
        if "}" in line:
            break
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*,?\s*$", line)
        if m and m.group(1) != "Count":
            names.append(m.group(1))
for f in sorted(glob.glob(os.path.expanduser(
        "~/src/mlx-Bf16DecodeAttribution/receipts-work/2026-09-11-bf16-decode-attribution/probe/*.jsonl"))):
    kernels = []
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("k") == "d":
            ke = d.get("ke")
            kernels.append(names[ke] if ke is not None and ke < len(names)
                           else "unk%s" % ke)
    print(os.path.basename(f), dict(collections.Counter(kernels)))
PYEOF

# ---- 2. attribution arms A ----
timeout 3300 python3 /tmp/run_arms_bf16.py \
  --python "$RUNPY" \
  --ablate-python "$ABLPY" \
  --wheel "$BASE" \
  --ablate-wheel "$ABLDIST" \
  --arms baseline gemv attn cast bf16fast \
  --out "$OUT/arms"
echo WINDOW_A_DONE

#!/usr/bin/env bash
# WrongValueSweep narrow-fix gate window: churn probe, conv/nn suites,
# six digests + pins, narrow-fix timing vs main.
set -uo pipefail
WT=$HOME/src/mlx-omarchy-wvs
R=$WT/receipts/2026-09-11-wrong-value-sweep
REQUAL=$HOME/src/mlx-omarchy-requal-20260911
MAIN_WHEEL=$REQUAL/dist/mlx_omarchy-0.32.2.dev202609112117+2a9add42-cp314-cp314-linux_aarch64.whl
BRANCH_WHEEL=$(ls -t $WT/dist/mlx_omarchy-*.whl | head -1)
echo "branch wheel: $(basename $BRANCH_WHEEL)"
cd "$WT"

driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)
echo "driver: $driver_pkg"

$WT/.work/venv-branch/bin/python -m pip install -q --force-reinstall --no-deps "$BRANCH_WHEEL"

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for lock"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -Is) LOCK ACQUIRED"
T0=$(date +%s)

echo "== churn probe (narrow fix) =="
$WT/.work/venv-branch/bin/python - <<'PYEOF'
import mlx.core as mx
import numpy as np
fails = 0
N = 40
for t in range(N):
    garb = [mx.random.normal((1 << 18,)) for _ in range(4)]
    mx.eval(garb)
    del garb
    a_np = np.random.rand(24).astype(np.float32)
    v_np = np.random.rand(4).astype(np.float32)
    got = np.array(mx.convolve(mx.array(a_np), mx.array(v_np), mode='same')).astype(np.float32)
    want = np.convolve(a_np, v_np, mode='same')
    if (~np.isclose(got, want, atol=1e-5)).any():
        fails += 1
print(f"CHURN-PROBE: fails {fails}/{N}")
PYEOF

for f in test_conv test_nn; do
  echo "== [py-m1-branch-narrow] $f =="
  MLX_ENABLE_TF32=0 timeout 900 "$WT/.work/venv-branch/bin/python" -m pytest \
    "$REQUAL/.work/mlx/python/tests/$f.py" -q --no-header -p no:cacheprovider \
    --junitxml="$R/py/$f-narrow.xml" > "$R/py/$f-narrow.log" 2>&1
  tail -1 "$R/py/$f-narrow.log"
done

run_matrix() {
  local label=$1 py=$2 wheel=$3
  local run="$R/matrix/$label"
  mkdir -p "$run"
  timeout 1500 "$HOME/src/mlx-bf16-prefill-attn/scripts/bench_matrix.py" --mode run \
    --python "$py" --wheel "$wheel" \
    --host-label "jwm1-$label" --timeout 900 \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --out "$run/matrix.json" > "$run/run.log" 2>&1
  echo "$label: measured=$(grep -c '"status": "measured"' "$run/matrix.json" 2>/dev/null || echo 0)"
}
run_matrix r2-branch "$WT/.work/venv-branch/bin/python" "$BRANCH_WHEEL"

"$WT/.work/venv-branch/bin/python" - "$R" <<'PYEOF'
import json, sys
from pathlib import Path
CANONICAL = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
    "qwen25-0.5b-bf16:long-decode-128": "8690dc83246b39f8",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
d = Path(sys.argv[1])
for label in ("r1-branch", "r2-branch"):
    mj = d / "matrix" / label / "matrix.json"
    if not mj.exists():
        print(label, "MISSING"); continue
    run = json.loads(mj.read_text()); held = True; rows = []
    for leg in run["legs"]:
        if leg["leg_id"] not in CANONICAL: continue
        if leg["status"] != "measured":
            held = False; rows.append((leg["leg_id"], "unmeasured", None)); continue
        got = leg["metrics"]["generated_ids_sha256_16"]
        ok = got == CANONICAL[leg["leg_id"]]
        held &= ok
        rows.append((leg["leg_id"], got, leg["metrics"].get("decode_tok_s")))
    print(label, "ALL-HELD" if held else "VIOLATION")
    for r in rows: print("  ", r)
PYEOF

echo "$(date -Is) narrow-gate window complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"

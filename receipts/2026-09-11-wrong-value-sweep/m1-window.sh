#!/usr/bin/env bash
# WrongValueSweep batched M1 window: cast pre/post, five-file suite,
# canonical digests + BF16 pins, branch-vs-main timing legs.
set -uo pipefail
WT=$HOME/src/mlx-omarchy-wvs
R=$WT/receipts/2026-09-11-wrong-value-sweep
REQUAL=$HOME/src/mlx-omarchy-requal-20260911
MAIN_WHEEL=$REQUAL/dist/mlx_omarchy-0.32.2.dev202609112117+2a9add42-cp314-cp314-linux_aarch64.whl
BRANCH_WHEEL=$(ls $WT/dist/mlx_omarchy-*.whl 2>/dev/null | head -1)
[[ -f "$BRANCH_WHEEL" ]] || { echo "FATAL: no branch wheel"; exit 3; }
cd "$WT"

driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || pacman -Q mesa 2>/dev/null || echo MISSING)
echo "driver: $driver_pkg" | tee "$R/driver.txt"

for v in main branch; do
  WHEEL=$MAIN_WHEEL; [[ $v == branch ]] && WHEEL=$BRANCH_WHEEL
  PY=$WT/.work/venv-$v/bin/python
  [[ -x $PY ]] || python3 -m venv "$WT/.work/venv-$v"
  $PY -c "import mlx_lm, numpy, pytest" 2>/dev/null || {
    $PY -m pip install -q numpy pytest "mlx_lm==0.31.3"; }
  $PY -m pip install -q --force-reinstall --no-deps "$WHEEL"
  echo "$v venv <- $(basename $WHEEL)"
done

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for lock"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -Is) LOCK ACQUIRED"
T0=$(date +%s)

for v in main branch; do
  echo "== cast $v ==" | tee -a "$R/cast-prepost.txt"
  $WT/.work/venv-$v/bin/python /tmp/wvs-cast.py 2>&1 | tee -a "$R/cast-prepost.txt"
done

for f in test_conv test_nn test_compile test_optimizers test_ops; do
  echo "== [py-m1-branch] $f =="
  MLX_ENABLE_TF32=0 timeout 900 "$WT/.work/venv-branch/bin/python" -m pytest \
    "$REQUAL/.work/mlx/python/tests/$f.py" -v --no-header -p no:cacheprovider \
    --junitxml="$R/py/$f.xml" > "$R/py/$f.log" 2>&1
  tail -1 "$R/py/$f.log"
done

run_matrix() { # label python wheel
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
run_matrix warmup-branch "$WT/.work/venv-branch/bin/python" "$BRANCH_WHEEL"
run_matrix warmup-main   "$WT/.work/venv-main/bin/python"   "$MAIN_WHEEL"
run_matrix r1-branch     "$WT/.work/venv-branch/bin/python" "$BRANCH_WHEEL"
run_matrix r1-main       "$WT/.work/venv-main/bin/python"   "$MAIN_WHEEL"

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
d = Path(sys.argv[1]); report = {}
for label in ("r1-branch", "r1-main"):
    mj = d / "matrix" / label / "matrix.json"
    if not mj.exists():
        report[label] = "MISSING"; continue
    run = json.loads(mj.read_text()); cell = {}; held = True
    for leg in run["legs"]:
        if leg["leg_id"] not in CANONICAL:
            continue
        if leg["status"] != "measured":
            cell[leg["leg_id"]] = leg["status"]; held = False; continue
        got = leg["metrics"]["generated_ids_sha256_16"]
        cell[leg["leg_id"]] = {"digest": got, "held": got == CANONICAL[leg["leg_id"]]}
        held &= (got == CANONICAL[leg["leg_id"]])
    report[label] = {"held": held, "legs": cell}
(d / "digest-gates.json").write_text(json.dumps(report, indent=1))
for label, r in report.items():
    if r == "MISSING":
        print(label, "MISSING")
    else:
        print(label, "ALL-HELD" if r["held"] else "VIOLATION")
PYEOF

echo "$(date -Is) window complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"

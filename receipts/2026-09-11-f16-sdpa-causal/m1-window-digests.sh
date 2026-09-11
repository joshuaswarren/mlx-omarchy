#!/usr/bin/env bash
# Digest-only window on jwm1: the phase-5 redo after mlx_lm was installed
# into the branch wheel venv. Fork driver, warmup + r1 + r2, canonical
# Q4 digests + BF16 pins.
set -uo pipefail
WT=$HOME/src/sdpa-causal-m1
R=$WT/receipts/2026-09-11-f16-sdpa-causal
cd "$WT"

driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)
[[ $driver_pkg == "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1" ]] || {
  echo "FATAL: fork driver is not the pinned honeykrisp build"; exit 5; }

WHEEL=$(ls "$WT"/dist/mlx_omarchy-*.whl 2>/dev/null | head -1)
PY_NEW=$WT/.work/venv-fix/bin/python
"$PY_NEW" -c "import mlx_lm" || { echo "FATAL: mlx_lm missing"; exit 3; }

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 7200s)"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded 7200s"; exit 4; }
echo "$(date -Is) lock acquired"

run_matrix() {  # rep
  local rep=$1
  local run="$R/matrix/${rep}-fork-causalfix"
  mkdir -p "$run"
  env MLX_DISABLE_COMPILE=1 timeout 1800 \
    "$HOME/src/mlx-bf16-prefill-attn/scripts/bench_matrix.py" --mode run \
    --python "$PY_NEW" --wheel "$WHEEL" \
    --host-label "jwm1-${rep}-fork-causalfix" --timeout 900 \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --out "$run/matrix.json" > "$R/matrix/${rep}-fork-causalfix.log" 2>&1
  echo "$rep: measured=$(grep -c '"status": "measured"' "$run/matrix.json" 2>/dev/null || echo 0)"
}
run_matrix warmup
for rep in r1 r2; do run_matrix "$rep"; done

"$PY_NEW" - "$R" <<'EOF'
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
report = {"cells": {}, "all_held": True}
for run_dir in sorted(d.glob("matrix/r*-fork-causalfix")):
    mj = run_dir / "matrix.json"
    if not mj.exists():
        continue
    label = run_dir.name
    run = json.loads(mj.read_text())
    cell = {}
    for leg in run["legs"]:
        if leg["leg_id"].split(":")[0] not in ("qwen25-0.5b-4bit", "qwen25-0.5b-bf16"):
            continue
        if leg["status"] != "measured":
            cell[leg["leg_id"]] = {"status": leg["status"]}
            report["all_held"] = False
            continue
        got = leg["metrics"]["generated_ids_sha256_16"]
        want = CANONICAL[leg["leg_id"]]
        held = got == want
        cell[leg["leg_id"]] = {"digest": got, "expected": want, "held": held}
        report["all_held"] &= held
    missing = [leg for leg in CANONICAL if leg not in cell]
    if missing:
        report["all_held"] = False
        cell["_missing"] = missing
    report["cells"][label] = cell
(d / "digest-gates.json").write_text(json.dumps(report, indent=1))
print("DIGESTS:", "ALL-HELD" if report["all_held"] else "VIOLATION")
EOF

echo "$(date -Is) digest window complete (lock released)"

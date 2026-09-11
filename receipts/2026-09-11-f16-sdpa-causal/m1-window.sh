#!/usr/bin/env bash
# F16 SDPA causal-ragged fix window on jwm1. One GPU-lock window, batched:
#   1. pre-fix repro: the 4 upstream test_sdpa causal-ragged failures + NaN
#      on the pre-fix 2a9add42 wheel (the requal pytest venv),
#   2. fixed-wheel venv from the branch dist wheel + provenance check,
#   3. upstream test_fast_sdpa.py with the fixed wheel: the 4 subtests pass,
#   4. C++ regression tests (built outside the lock),
#   5. canonical digest gates: fork driver, warmup + r1 + r2, all six Q4
#      digests and every BF16 pin must hold.
set -uo pipefail
WT=$HOME/src/sdpa-causal-m1
REQUAL=$HOME/src/mlx-omarchy-requal-20260911
R=$WT/receipts/2026-09-11-f16-sdpa-causal
mkdir -p "$R/m1-logs" "$R/matrix"
cd "$WT"

driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)
echo "driver: $driver_pkg"
[[ $driver_pkg == "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1" ]] || {
  echo "FATAL: fork driver is not the pinned honeykrisp build"; exit 5; }

WHEEL=$(ls "$WT"/dist/mlx_omarchy-*.whl 2>/dev/null | head -1)
[[ -f $WHEEL ]] || { echo "FATAL: branch wheel missing"; exit 3; }
[[ -x $WT/.work/build-cpp/tests/omarchy/omarchy_sdpa_causal_ragged_tests ]] || {
  echo "FATAL: cpp regression binaries missing"; exit 3; }

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 7200s)"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded 7200s"; exit 4; }
echo "$(date -Is) lock acquired"

PY_OLD=$REQUAL/receipts/.requal-py-stage/venv-pytest/bin/python
TESTS=$WT/.work/mlx/python/tests

echo "== phase 1: pre-fix repro (2a9add42 wheel) =="
MLX_ENABLE_TF32=0 timeout 900 "$PY_OLD" -m pytest "$TESTS/test_fast_sdpa.py" \
  -v --no-header -p no:cacheprovider > "$R/m1-logs/prefix-repro.log" 2>&1
grep -E "^_ TestFastSDPA" "$R/m1-logs/prefix-repro.log" | sed 's/ (dtype.*//' || true
echo "prefix assert failures: $(grep -c 'AssertionError' "$R/m1-logs/prefix-repro.log")"

echo "== phase 2: fixed-wheel venv =="
rm -rf "$WT/.work/venv-fix"
cp -a "$REQUAL/receipts/.requal-py-stage/venv-pytest" "$WT/.work/venv-fix" || exit 3
"$WT/.work/venv-fix/bin/python" -m pip install --force-reinstall --no-deps -q \
  "$WHEEL" || exit 3
"$WT/.work/venv-fix/bin/python" "$WT/scripts/mlx_provenance.py" \
  --expect-wheel "$WHEEL" > "$R/python-provenance.json" || exit 3
"$WT/.work/venv-fix/bin/python" - <<'EOF' || exit 3
import json
p = json.load(open("receipts/2026-09-11-f16-sdpa-causal/python-provenance.json"))
assert p["verified"] == "match", p
stamp = (p.get("dist_version") or "").partition("+")[2]
print("provenance:", p["dist_version"], "stamp:", stamp)
EOF
PY_NEW=$WT/.work/venv-fix/bin/python

echo "== phase 3: upstream test_fast_sdpa with fixed wheel =="
MLX_ENABLE_TF32=0 timeout 900 "$PY_NEW" -m pytest "$TESTS/test_fast_sdpa.py" \
  -v --no-header -p no:cacheprovider --junitxml="$R/sdpa-fixed.xml" \
  > "$R/m1-logs/fix-sdpa.log" 2>&1
tail -2 "$R/m1-logs/fix-sdpa.log"
echo "fixed failures: $(grep -cE '^_ TestFastSDPA.*FAILED|AssertionError' "$R/m1-logs/fix-sdpa.log")"
grep -E "qsl=127" "$R/m1-logs/fix-sdpa.log" | head -5 || true

echo "== phase 4: C++ regression tests =="
MLX_OMARCHY_ALLOW_NON_APPLE=1 "$WT/.work/build-cpp/tests/omarchy/omarchy_sdpa_causal_ragged_tests" \
  > "$R/m1-logs/cpp-causal-ragged.log" 2>&1
echo "cpp ragged rc=$?"
tail -2 "$R/m1-logs/cpp-causal-ragged.log"
MLX_OMARCHY_ALLOW_NON_APPLE=1 "$WT/.work/build-cpp/tests/omarchy/omarchy_fast_regression_tests" \
  > "$R/m1-logs/cpp-sdpa-norm.log" 2>&1
echo "cpp sdpa-norm rc=$?"
tail -2 "$R/m1-logs/cpp-sdpa-norm.log"

echo "== phase 5: canonical digest gates (fork driver) =="
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
    label = run_dir.name.split("/")[0]
    run = json.loads(mj.read_text())
    cell = {}
    for leg in run["legs"]:
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

echo "$(date -Is) window complete (lock released)"

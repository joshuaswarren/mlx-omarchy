#!/usr/bin/env bash
# Window B2 stage 1b on jwm1-linux: re-run ONLY the GPU phase of stage 1
# (driver gate + probe) with the corrected probe (flags=1 cells, 36-cell
# gate), reusing the wheel/venvs/suite binaries banked by stage 1.
# Capped flock -w 1800; releases immediately after the probe.
# Preserves the invalid flags-sweep evidence before overwriting.
set -uo pipefail
cd ~/src/mlx-Bf16PrefillClose
R=receipts/2026-09-10-bf16-prefill-close
[[ -d .venv-prefill-base && -d .venv-prefill-cand ]] || {
  echo "FATAL: stage 1 venvs missing"; exit 3; }
[[ -f "$R/m1-logs/candidate-wheel.sha256" ]] || {
  echo "FATAL: stage 1 wheel missing"; exit 3; }
[[ -x /tmp/fam-base && -x /tmp/fam-cand ]] || {
  echo "FATAL: stage 1 suite binaries missing"; exit 3; }
mv "$R/m1-logs/probe-flags-sweep.ndjson" "$R/m1-logs/probe-flags-sweep-invalid.ndjson" 2>/dev/null
mv "$R/window-b2-stage1.log" "$R/window-b2-stage1-attempt1.log" 2>/dev/null

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 1800s)"
flock -w 1800 9 || { echo "FATAL: lock wait exceeded 1800s"; exit 4; }
echo "$(date -Is) lock acquired"
{
  g++ -std=c++17 -O2 -o /tmp/bf16-prefill-bench tools/bf16-prefill-bench/bench.cpp \
    || { echo "FATAL: probe build"; exit 3; }
  /tmp/bf16-prefill-bench --tiny 2>&1 | grep '"k":"dev"' \
    | tee "$R/m1-logs/driver-gate-b2.txt"
  grep -q '"coopmat":true' "$R/m1-logs/driver-gate-b2.txt" || {
    echo "FATAL: default driver lacks cooperative matrix; bailing."
    exit 3
  }
  echo "== stage1b: probe gate (36 cells, flags=1) =="
  /tmp/bf16-prefill-bench > "$R/m1-logs/probe-flags-sweep.ndjson" \
    2> "$R/m1-logs/probe-flags-sweep.err"
  python3 - "$R/m1-logs/probe-flags-sweep.ndjson" <<'PY'
import json, sys
bad = 0; n = 0; flags = set()
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("k") == "correct" and "cand_vs_base_mismatch" in r:
        n += 1; flags.add(r["flags"] if "flags" in r else 1)
        if r["cand_vs_base_mismatch"] or r["dead_rows"]:
            bad += 1; print("BAD", r)
print(f"probe cells={n} bad={bad}")
sys.exit(0 if (bad == 0 and n >= 36) else 1)
PY
  [[ $? -eq 0 ]] || { echo "FATAL: probe gate failed"; exit 3; }
  echo "STAGE1-OK"
} >> "$R/window-b2-stage1.log" 2>&1
rc=$?
echo "$(date -Is) stage 1b complete rc=$rc (lock released)"
exit $rc

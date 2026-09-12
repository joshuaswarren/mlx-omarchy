#!/usr/bin/env bash
# PrefillNativeOrder window 1: kernel A/B probe (FMA route ladder vs
# cooperative-matrix route) plus a one-repetition six-leg digest screen
# on both drivers. Branch wave/PrefillNativeOrder 70f1096, bench wheel.
set -uo pipefail
ROOT="$HOME/src/mlx-omarchy-fma"
R="$ROOT/receipts/2026-09-12-prefill-native-order"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
PY="$ROOT/.work/venv-run/bin/python"
PROBE="$ROOT/benchq/qmm-fma/probe.py"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
COMMIT_SHORT=70f1096
mkdir -p "$R"

echo "=== window1 start $(date -u +%FT%TZ) pid $$"
exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED (window1 announced start)"
T0=$(date +%s)

{
  echo "driver-fork: $(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)"
  echo "driver-stock: $(pacman -Q mesa 2>/dev/null || echo MISSING)"
  echo "kernel: $(uname -r)"
  echo "machine: $(uname -m) $(uname -p)"
  echo "stock-icd-sha256: $(sha256sum "$STOCK_ICD" | cut -d' ' -f1)"
  echo "wheel: $(basename "$WHEEL")"
  echo "wheel-sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "source-commit: $(git -C "$ROOT" rev-parse HEAD)"
} > "$R/drivers-window1.txt"
cat "$R/drivers-window1.txt"

ok=0
for i in $(seq 1 60); do
  l=$(cut -d" " -f1 /proc/loadavg)
  if awk -v x="$l" 'BEGIN{exit !(x < 1.0)}'; then
    ok=$((ok+1)); [ "$ok" -ge 3 ] && break
  else
    ok=0
  fi
  echo "quiet-gate wait $i load=$l $(date -u +%FT%TZ)"
  sleep 20
done
echo "quiet gate done ok=$ok load=$(cut -d' ' -f1 /proc/loadavg)"

( while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 10; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

echo "=== driver-level coopmat construction probe $(date -u +%H:%M:%S) ==="
PB="$ROOT/benchq/qmm-fma"
gcc -O2 -o "$R/coopmat_probe" "$PB/coopmat_probe.c" -lvulkan
spirv-as --target-env vulkan1.3 "$PB/coopmat_sb_load.spvasm" -o "$R/sb_load.spv"
spirv-as --target-env vulkan1.3 "$PB/coopmat_private_load.spvasm" -o "$R/private_load.spv"
"$R/coopmat_probe" "$R/sb_load.spv" success > "$R/coopmat-probe-control.log" 2>&1
tail -3 "$R/coopmat-probe-control.log"
"$R/coopmat_probe" "$R/private_load.spv" failure > "$R/coopmat-probe-private.log" 2>&1
grep -E "SPIR-V|pipeline-creation|expect" "$R/coopmat-probe-private.log"

probe_run() {  # side dtype cfg tag
  local side="$1" dtype="$2" cfg="$3" tag="$4"
  local -a CFG=()
  [ "$cfg" != "-" ] && CFG=(--cfg "$cfg")
  echo "=== probe $tag $(date -u +%H:%M:%S) ==="
  timeout 600 "$PY" "$PROBE" --side "$side" --dtype "$dtype" \
    "${CFG[@]}" --warmup 4 --timed 30 \
    > "$R/probe-$tag.log" 2>&1
  local rc=$?
  echo "probe $tag exit=$rc"
  grep -E '"shape"' "$R/probe-$tag.log" || echo "probe $tag EMPTY"
}

probe_run fma    q4   - c0
probe_run fma    q4   1 c1
probe_run fma    q4   2 c2
probe_run coopmat q4  - coopmat
probe_run fma    bf16 - b0
probe_run fma    bf16 1 b1
probe_run coopmat bf16 - coopmat

"$PY" - "$R" <<'PYSHA'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
def cells(tag):
    out = {}
    for line in (d / f"probe-{tag}.log").read_text().splitlines():
        if line.startswith('{"side"'):
            r = json.loads(line)
            out[r["shape"]] = r["sha"]
    return out
groups = {
    "q4":  [("fma-cfg0", "c0"), ("fma-cfg1", "c1"), ("fma-cfg2", "c2"), ("coopmat", "coopmat")],
    "bf16": [("fma-cfg0", "b0"), ("fma-cfg1", "b1"), ("coopmat", "coopmat")],
}
report = {}
for dt, sides in groups.items():
    tables = {name: cells(tag) for name, tag in sides}
    ref = tables.get("coopmat") or tables["fma-cfg0"]
    for shape, want in ref.items():
        row = {"coopmat": want}
        for name, _ in sides:
            got = tables[name].get(shape)
            row[name] = got
            row[f"{name}-bits-equal"] = (got == want)
        report[f"{dt}:{shape}"] = row
        print(dt, shape, {k: v for k, v in row.items() if "bits-equal" in k or k == "coopmat"})
(d / "probe-bit-identity.json").write_text(json.dumps(report, indent=1) + "\n")
PYSHA

run_one() {  # drv label outfile
  local drv="$1" label="$2" out="$3"
  local -a ICD=()
  if [ "$drv" = stock ]; then
    ICD=(env VK_DRIVER_FILES="$STOCK_ICD")
  else
    ICD=(env)
  fi
  echo "=== matrix run $out $(date -u +%H:%M:%S) ==="
  timeout 1500 "${ICD[@]}" "$PY" scripts/bench_matrix.py --mode run \
    --python "$PY" \
    --wheel "$WHEEL" \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --host-label "$label" \
    --timeout 900 \
    --out "$R/$out.json" \
    > "$R/$out.log" 2>&1
  local rc=$?
  echo "$out exit=$rc $(date -u +%H:%M:%S)"
  grep -cE '"status": "measured"' "$R/$out.json" 2>/dev/null || true
}

HF_HUB_OFFLINE=1 run_one fork  "jwm1 honeykrisp-fork ${COMMIT_SHORT} screen" screen-fork
HF_HUB_OFFLINE=1 run_one stock "jwm1 stock-mesa ${COMMIT_SHORT} screen" screen-stock

"$PY" - "$R" <<'PYCHECK'
import json, sys
from pathlib import Path
Q4 = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
}
BF16 = {
    "fork": {
        "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
        "qwen25-0.5b-bf16:long-decode-128": "8690dc83246b39f8",
        "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
    },
    "stock": {
        "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
        "qwen25-0.5b-bf16:long-decode-128": "46108ad71157cb4d",
        "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
    },
}
d = Path(sys.argv[1])
failures = []
for drv in ("fork", "stock"):
    mj = d / f"screen-{drv}.json"
    if not mj.exists():
        failures.append(f"{drv}: MISSING"); continue
    run = json.loads(mj.read_text())
    for leg in run["legs"]:
        lid = leg["leg_id"]
        if leg["status"] != "measured":
            failures.append(f"{drv} {lid}: status={leg['status']}")
            continue
        got = leg["metrics"]["generated_ids_sha256_16"]
        want = Q4.get(lid) or BF16[drv][lid]
        ok = got == want
        print(f"{drv} {lid}: {got} want {want} {'HELD' if ok else 'MOVEMENT'}")
        if not ok:
            failures.append(f"{drv} {lid}: {got} != {want}")
print("SCREEN_DIGESTS_HELD" if not failures else "SCREEN_DIGEST_FAILURES")
(d / "screen-digests.txt").write_text("\n".join(failures) + "\n")
sys.exit(1 if failures else 0)
PYCHECK
rc=$?
[ $rc -ne 0 ] && echo "SCREEN DIGEST FAILURE rc=$rc"

kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ) window1 complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"

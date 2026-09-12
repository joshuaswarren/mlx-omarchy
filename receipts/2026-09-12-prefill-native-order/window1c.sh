#!/usr/bin/env bash
# PrefillNativeOrder window 1c: A/B probe ladder (probe fixed) plus the
# PRECISE_ACCUM digest screen - stock Q4 1K (the cell that moved under
# the fused expression) and all six fork legs, cfg 3, against pins.
set -uo pipefail
ROOT="$HOME/src/mlx-omarchy-fma"
R="$ROOT/receipts/2026-09-12-prefill-native-order"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
PY="$ROOT/.work/venv-run/bin/python"
PROBE="$ROOT/benchq/qmm-fma/probe.py"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
mkdir -p "$R"

echo "=== window1c start $(date -u +%FT%TZ) pid $$"
exec 9>/tmp/m1-gpu.lock
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
T0=$(date +%s)

{
  echo "driver-fork: $(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)"
  echo "wheel: $(basename "$WHEEL")"
  echo "wheel-sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "source-commit: $(git -C "$ROOT" rev-parse HEAD)"
} > "$R/drivers-window1c.txt"
cat "$R/drivers-window1c.txt"

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

probe_run fma     q4   3 c3
probe_run fma     q4   0 c0
probe_run fma     q4   1 c1
probe_run fma     q4   2 c2
probe_run coopmat q4   - coopmat
probe_run fma     bf16 0 b0
probe_run fma     bf16 1 b1
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
    "q4":  [("fma-precise-c3", "c3"), ("fma-cfg0", "c0"), ("fma-cfg1", "c1"),
            ("fma-cfg2", "c2"), ("coopmat", "coopmat")],
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
        flags = {k: v for k, v in row.items() if "bits-equal" in k}
        print(dt, shape, flags)
(d / "probe-bit-identity.json").write_text(json.dumps(report, indent=1) + "\n")
PYSHA

run_one() {  # drv label outfile wheel extra_env
  local drv="$1" label="$2" out="$3" wheel="$4"
  local -a ICD=()
  if [ "$drv" = stock ]; then
    ICD=(env VK_DRIVER_FILES="$STOCK_ICD" MLX_OMARCHY_QMM_FMA_CFG=3)
  else
    ICD=(env MLX_OMARCHY_QMM_FMA_CFG=3)
  fi
  echo "=== matrix run $out $(date -u +%H:%M:%S) ==="
  timeout 1500 "${ICD[@]}" "$PY" scripts/bench_matrix.py --mode run \
    --python "$PY" \
    --wheel "$wheel" \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --host-label "$label" \
    --timeout 900 \
    --out "$R/$out.json" \
    > "$R/$out.log" 2>&1
  echo "$out exit=$? $(date -u +%H:%M:%S)"
}

HF_HUB_OFFLINE=1 run_one stock "jwm1 stock-mesa precise-screen" precise-stock "$WHEEL"
HF_HUB_OFFLINE=1 run_one fork  "jwm1 honeykrisp-fork precise-screen" precise-fork "$WHEEL"

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
for drv in ("fork", "stock"):
    run = json.loads((d / f"precise-{drv}.json").read_text())
    for leg in run["legs"]:
        lid = leg["leg_id"]
        got = leg["metrics"]["generated_ids_sha256_16"]
        want = Q4.get(lid) or BF16[drv][lid]
        print(f"{drv} {lid}: {got} want {want} "
              f"{'HELD' if got == want else 'MOVEMENT'}")
PYCHECK

kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ) window1c complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"

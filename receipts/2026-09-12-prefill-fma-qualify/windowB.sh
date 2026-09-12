#!/usr/bin/env bash
# PrefillFmaQualify window B: qualify the GATED landing wheel (b1805b19,
# FMA reserved for non-coopmat drivers). Six-leg digest matrix candidate
# vs base, fork + stock, discarded warmup + 3 reps, interleaved; stock
# kernel probes on the gated wheel; digest + paired-perf assertions.
set -uo pipefail
ROOT="$HOME/src/mlx-omarchy-fmaq"
ROOT_BASE="$HOME/src/mlx-omarchy-fma-base"
R="$ROOT/receipts/2026-09-12-prefill-fma-qualify"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
BASE_WHEEL="$(ls "$ROOT_BASE"/dist/mlx_omarchy-*.whl | head -n1)"
PY="$ROOT/.work/venv-run/bin/python"
PY_BASE="$ROOT_BASE/.work/venv-run/bin/python"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
mkdir -p "$R"

echo "=== windowB start $(date -u +%FT%TZ) pid $$"
exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED (windowB announced start)"
T0=$(date +%s)

{
  echo "driver-fork: $(vulkaninfo --summary 2>/dev/null | grep driverInfo | head -1)"
  echo "driver-stock: $(VK_DRIVER_FILES="$STOCK_ICD" vulkaninfo --summary 2>/dev/null | grep driverInfo | head -1)"
  echo "kernel: $(uname -r)  machine: $(uname -m)"
  echo "chip: Apple M1 (G13G B1)  host-label: jwm1-linux (placeholder)"
  echo "wheel-gated: $(basename "$WHEEL")"
  echo "wheel-gated-sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "wheel-base: $(basename "$BASE_WHEEL")"
  echo "wheel-base-sha256: $(sha256sum "$BASE_WHEEL" | cut -d' ' -f1)"
  echo "source-commit-gated: $(git -C "$ROOT" rev-parse HEAD)"
  echo "source-commit-base: a0775c37"
} > "$R/drivers-windowB.txt"
cat "$R/drivers-windowB.txt"

ok=0
for i in $(seq 1 60); do
  l=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v x="$l" 'BEGIN{exit !(x < 1.0)}'; then
    ok=$((ok+1)); [ "$ok" -ge 3 ] && break
  else
    ok=0
  fi
  sleep 20
done
echo "quiet gate done ok=$ok load=$(cut -d' ' -f1 /proc/loadavg)"
( while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 10; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

run_one() {  # drv side rep
  local drv="$1" side="$2" rep="$3"
  local out="${side}-${drv}-${rep}"
  local wheel="$WHEEL" py="$PY"
  if [ "$side" = base ]; then wheel="$BASE_WHEEL"; py="$PY_BASE"; fi
  local -a ICD=()
  if [ "$drv" = stock ]; then ICD=(env VK_DRIVER_FILES="$STOCK_ICD"); else ICD=(env); fi
  echo "=== matrix run $out $(date -u +%H:%M:%S) ==="
  HF_HUB_OFFLINE=1 timeout 1500 "${ICD[@]}" "$PY" scripts/bench_matrix.py --mode run \
    --python "$py" \
    --wheel "$wheel" \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --host-label "jwm1 ${drv} gated-${side} ${rep}" \
    --timeout 900 \
    --out "$R/$out.json" \
    > "$R/$out.log" 2>&1
  echo "$out exit=$? $(date -u +%H:%M:%S)"
}
run_one fork  cand  warmup
run_one fork  base  warmup
run_one stock cand  warmup
run_one stock base  warmup
run_one stock base  r1
run_one stock cand  r1
run_one fork  base  r1
run_one fork  cand  r1
run_one fork  cand  r2
run_one fork  base  r2
run_one stock cand  r2
run_one stock base  r2
run_one stock base  r3
run_one stock cand  r3
run_one fork  base  r3
run_one fork  cand  r3

echo "=== kernel probes on gated wheel (stock only; fork cand==base by gate) $(date -u +%H:%M:%S) ==="
probe_run() {  # wheelset side dtype
  local ws="$1" side="$2" dtype="$3"
  local py="$PY"; [ "$ws" = base ] && py="$PY_BASE"
  local tag="probeB-stock-${ws}-${side}-${dtype}"
  {
    echo "# $tag $(date -u +%FT%TZ)"
    echo "# gated wheel: $(basename "$WHEEL") sha256 $(sha256sum "$WHEEL" | cut -d' ' -f1)"
    echo "# base wheel:  $(basename "$BASE_WHEEL") sha256 $(sha256sum "$BASE_WHEEL" | cut -d' ' -f1)"
  } > "$R/$tag.log"
  HF_HUB_OFFLINE=1 timeout 900 env VK_DRIVER_FILES="$STOCK_ICD" \
    "$py" "$ROOT/benchq/qmm-fma/probe.py" \
    --side "$side" --dtype "$dtype" --warmup 4 --timed 30 >> "$R/$tag.log" 2>&1
  echo "$tag exit=$?"
  grep -E '"gflops"' "$R/$tag.log" | tail -8
}
probe_run cand fma     q4
probe_run base coopmat q4
probe_run cand fma     bf16
probe_run base coopmat bf16

echo "=== digest + provenance assertions $(date -u +%FT%TZ)"
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
rows = []
for side in ("cand", "base"):
    for drv in ("fork", "stock"):
        for rep in ("warmup", "r1", "r2", "r3"):
            p = d / f"{side}-{drv}-{rep}.json"
            if not p.exists():
                failures.append(f"{side}-{drv}-{rep}: missing")
                continue
            run = json.loads(p.read_text())
            legs = {l["leg_id"]: l for l in run["legs"]}
            pins = dict(Q4); pins.update(BF16[drv])
            for lid, want in pins.items():
                leg = legs.get(lid)
                if leg is None or leg.get("status") != "measured":
                    failures.append(f"{side}-{drv}-{rep} {lid}: not measured")
                    continue
                got = leg["metrics"]["generated_ids_sha256_16"]
                prov = leg["metrics"].get("provenance_line", "")
                ok = got == want
                if not ok:
                    failures.append(f"{side}-{drv}-{rep} {lid}: {got} != {want}")
                if rep in ("r1", "r2", "r3"):
                    rows.append((f"{side}-{drv}-{rep}", lid, got, want, ok, prov[:72]))
for r in rows:
    print(("HELD " if r[4] else "MOVED"), r[0], r[1], r[2], "want", r[3])
    print("      prov:", r[5])
print(f"TOTAL {len(rows)} pinned cells, {len(failures)} failures")
for f in failures:
    print("FAIL:", f)
sys.exit(1 if failures else 0)
PYCHECK
rc=$?
[ $rc -ne 0 ] && echo "DIGEST ASSERTION FAILED rc=$rc"

echo "=== performance summary $(date -u +%FT%TZ)"
"$PY" - "$R" <<'PYPERF'
import json, statistics, sys
from pathlib import Path
NATIVE = {
    "qwen25-0.5b-4bit:short-decode-32":       (150.57, 294.1),
    "qwen25-0.5b-4bit:long-decode-128":       (146.77, 1213.0),
    "qwen25-0.5b-4bit:longctx-1024-decode-32":(140.38, 1840.9),
    "qwen25-0.5b-bf16:short-decode-32":       (56.43, 232.6),
    "qwen25-0.5b-bf16:long-decode-128":       (55.72, 1007.7),
    "qwen25-0.5b-bf16:longctx-1024-decode-32":(54.55, 1655.7),
}
d = Path(sys.argv[1])
out = {}
for drv in ("fork", "stock"):
    cells = {}
    for lid, (ndec, npre) in NATIVE.items():
        per = {}
        for kind in ("cand", "base"):
            dec, pre = [], []
            for rep in ("r1", "r2", "r3"):
                p = d / f"{kind}-{drv}-{rep}.json"
                if not p.exists():
                    continue
                run = json.loads(p.read_text())
                for leg in run["legs"]:
                    if leg["leg_id"] == lid and leg["status"] == "measured":
                        dec.append(leg["metrics"]["decode_tok_s"])
                        pre.append(leg["metrics"]["prefill_tok_s"])
            if len(dec) == 3 and len(pre) == 3:
                per[kind] = {
                    "decode_reps": dec, "prefill_reps": pre,
                    "decode_tok_s_median": round(statistics.median(dec), 2),
                    "prefill_tok_s_median": round(statistics.median(pre), 2),
                }
            else:
                per[kind] = {"error": f"reps dec={len(dec)} pre={len(pre)}"}
        cand, base = per["cand"], per["base"]
        cells[lid] = {
            "cand": cand, "base": base,
            "prefill_delta_pct": round(100.0 * (cand["prefill_tok_s_median"] - base["prefill_tok_s_median"]) / base["prefill_tok_s_median"], 2) if "prefill_tok_s_median" in cand and "prefill_tok_s_median" in base else None,
            "decode_delta_pct": round(100.0 * (cand["decode_tok_s_median"] - base["decode_tok_s_median"]) / base["decode_tok_s_median"], 2) if "decode_tok_s_median" in cand and "decode_tok_s_median" in base else None,
            "cand_prefill_fraction_of_native": round(cand.get("prefill_tok_s_median", 0) / npre, 4) if "prefill_tok_s_median" in cand else None,
            "base_prefill_fraction_of_native": round(base.get("prefill_tok_s_median", 0) / npre, 4) if "prefill_tok_s_median" in base else None,
        }
    out[drv] = cells
(d / "perf-fma-qualify-gated.json").write_text(json.dumps(out, indent=1) + "\n")
for drv, cells in out.items():
    print(f"--- {drv}")
    for lid, c in cells.items():
        print(f"{lid}: prefill cand {c['cand'].get('prefill_tok_s_median')} vs base {c['base'].get('prefill_tok_s_median')} tok/s -> {c['prefill_delta_pct']}%  (native fraction {c['base_prefill_fraction_of_native']} -> {c['cand_prefill_fraction_of_native']})")
        print(f"           decode cand {c['cand'].get('decode_tok_s_median')} vs base {c['base'].get('decode_tok_s_median')} tok/s -> {c['decode_delta_pct']}%")
PYPERF

kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ) windowB complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"

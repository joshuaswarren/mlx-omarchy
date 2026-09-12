#!/usr/bin/env bash
# AgxWaitBatching qualification window (receipts/2026-09-12-agx-wait-batching).
# CONDITIONALLY approved by root: run ONLY after host capacity/hazard gtests
# pass. One flock on /tmp/m1-gpu.lock, one process tree, quiet gate, unchanged
# wheel (a2e38c3) and pinned inputs. No arithmetic re-pins: any digest move is
# a rejection recorded as such.
#
# Sequence:
#   0. provenance capture (driver commits, ICD sha256s, wheel sha256, package)
#   1. quiet gate: 1-min loadavg < 1.0 three checks 20 s apart
#   2. control gates FIRST: dominant-cell digest on base driver and on patched
#      driver must equal the stock pin 5179630cd4a7c3f9; abort if not
#   3. kernel probe interleaved base/patched, 3 rounds
#   4. bench_matrix interleaved: discarded warmup per driver, then 3 rounds
#      patched/base, 6 canonical legs, pins asserted per run
#   5. digest + perf summaries written into the receipt dir
set -uo pipefail
R="$HOME/benchq/qmm-coop-bench"
PARITY="$HOME/src/mlx-omarchy-parity"
WHEEL="$(ls "$PARITY"/dist/mlx_omarchy-0.32.2.dev202609121038+a2e38c3-*.whl | head -n1)"
PY="$PARITY/.work/venv-run/bin/python"
ICD_BASE="$R/icd-waitbatch-base.json"
ICD_PATCHED="$R/icd-waitbatch-patched.json"
OUT="$R/qual"
PIN_DOMINANT="5179630cd4a7c3f9"
mkdir -p "$OUT"
echo "=== window start $(date -u +%FT%TZ) pid $$"
exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
T0=$(date +%s)

{
  echo "driver-patched-commit: $(git -C "$HOME/src/mesa-wt-waitbatch" rev-parse HEAD)"
  echo "driver-base-commit: $(git -C "$HOME/src/mesa-wt-waitbatch-base" rev-parse HEAD)"
  echo "icd-base-sha256: $(sha256sum "$ICD_BASE" | cut -d' ' -f1)"
  echo "icd-patched-sha256: $(sha256sum "$ICD_PATCHED" | cut -d' ' -f1)"
  echo "wheel: $(basename "$WHEEL")"
  echo "wheel-sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "source-commit: $(git -C "$PARITY" rev-parse HEAD)"
  echo "package-fork: $(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)"
  echo "kernel: $(uname -r)"
  echo "machine: $(uname -m) $(uname -p)"
  echo "window-script-sha256: $(sha256sum "$0" | cut -d' ' -f1)"
} > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"

# quiet gate
ok=0
for i in $(seq 1 270); do
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
[ "$ok" -ge 3 ] || { echo "FATAL: quiet gate never passed"; exit 4; }

probe() { # icd outfile
  local icd="$1" out="$2"
  echo "=== probe $out $(date -u +%H:%M:%S)"
  VK_DRIVER_FILES="$icd" AGX_SIMDMAT=1 \
    "$HOME/venv-qmmcoop/bin/python" "$R/qmm_coop_bench_probe.py" \
    > "$out" 2>&1
  local rc=$?
  echo "$out exit=$rc"
  return $rc
}

dominant_digest() { # probe-log -> dominant cell digest or EMPTY
  grep '"shape": "1053x896x9728"' "$1" \
    | grep -o '"f16_digest": "[0-9a-f]*"' | cut -d'"' -f4 | head -1
}

# control gates first
probe "$ICD_BASE" "$OUT/control-base.log"    || echo "WARN control-base rc"
probe "$ICD_PATCHED" "$OUT/control-patched.log" || echo "WARN control-patched rc"
B=$(dominant_digest "$OUT/control-base.log")
P=$(dominant_digest "$OUT/control-patched.log")
echo "control digests: base=$B patched=$P pin=$PIN_DOMINANT"
[ "$B" = "$PIN_DOMINANT" ] || { echo "FATAL: base control digest moved"; exit 4; }
[ "$P" = "$PIN_DOMINANT" ] || { echo "FATAL: patched control digest moved"; exit 4; }

# interleaved kernel probe, 3 rounds
for rep in 1 2 3; do
  probe "$ICD_BASE"    "$OUT/probe-base-r$rep.log"
  probe "$ICD_PATCHED" "$OUT/probe-patched-r$rep.log"
done

# bench_matrix interleaved: warmups discarded, then 3 rounds patched/base
run_matrix() { # icd label outfile
  local icd="$1" label="$2" out="$3"
  echo "=== matrix $out $(date -u +%H:%M:%S)"
  (cd "$PARITY" && VK_DRIVER_FILES="$icd" HF_HUB_OFFLINE=1 timeout 1500 \
    "$PY" scripts/bench_matrix.py --mode run \
    --python "$PY" \
    --wheel "$WHEEL" \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --host-label "$label" \
    --timeout 900 \
    --out "$OUT/$out.json" \
    > "$OUT/$out.log" 2>&1)
  echo "$out exit=$? $(date -u +%H:%M:%S)"
}
run_matrix "$ICD_PATCHED" "jwm1 waitbatch-patched warmup-discarded" patched-warmup
run_matrix "$ICD_BASE"    "jwm1 waitbatch-base warmup-discarded"    base-warmup
for rep in r1 r2 r3; do
  run_matrix "$ICD_PATCHED" "jwm1 waitbatch-patched $rep" "patched-$rep"
  run_matrix "$ICD_BASE"    "jwm1 waitbatch-base $rep"    "base-$rep"
done

# digest assertions: 6 legs x 3 reps x 2 drivers, fork pins on both
"$PY" - "$OUT" <<'PYCHECK'
import json, sys
from pathlib import Path
PINS = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
    "qwen25-0.5b-bf16:long-decode-128": "8690dc83246b39f8",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
d = Path(sys.argv[1])
report, failures = {}, []
for drv in ("patched", "base"):
    for rep in ("r1", "r2", "r3"):
        mj = d / f"{drv}-{rep}.json"
        if not mj.exists():
            failures.append(f"{drv}-{rep}: MISSING"); continue
        run = json.loads(mj.read_text())
        legs = {}
        for leg in run["legs"]:
            lid = leg["leg_id"]
            if leg["status"] != "measured":
                failures.append(f"{drv}-{rep} {lid}: status={leg['status']}")
                continue
            got = leg["metrics"]["generated_ids_sha256_16"]
            want = PINS.get(lid)
            ok = (got == want)
            legs[lid] = {"digest": got, "expected": want, "held": ok}
            if not ok:
                failures.append(f"{drv}-{rep} {lid}: got {got} want {want} (MOVEMENT)")
        report[f"{drv}-{rep}"] = legs
(d / "digest-matrix.json").write_text(json.dumps(report, indent=1) + "\n")
for f in failures:
    print("FAIL", f)
print("ALL_DIGESTS_HELD" if not failures else "DIGEST_FAILURES")
PYCHECK
rc=$?
[ $rc -ne 0 ] && echo "DIGEST ASSERTION FAILED rc=$rc"

# performance summary: medians of 3, patched vs base
"$PY" - "$OUT" <<'PYPERF'
import json, statistics, sys
from pathlib import Path
NATIVE = {
    "qwen25-0.5b-4bit:short-decode-32":        (150.57, 294.1),
    "qwen25-0.5b-4bit:long-decode-128":        (146.77, 1213.0),
    "qwen25-0.5b-4bit:longctx-1024-decode-32": (140.38, 1840.9),
    "qwen25-0.5b-bf16:short-decode-32":        (56.43, 232.6),
    "qwen25-0.5b-bf16:long-decode-128":        (55.72, 1007.7),
    "qwen25-0.5b-bf16:longctx-1024-decode-32": (54.55, 1655.7),
}
d = Path(sys.argv[1])
out = {}
for drv in ("patched", "base"):
    cells = {}
    for lid, (ndec, npre) in NATIVE.items():
        dec, pre = [], []
        for rep in ("r1", "r2", "r3"):
            run = json.loads((d / f"{drv}-{rep}.json").read_text())
            for leg in run["legs"]:
                if leg["leg_id"] == lid and leg["status"] == "measured":
                    dec.append(leg["metrics"]["decode_tok_s"])
                    pre.append(leg["metrics"]["prefill_tok_s"])
        if len(dec) == 3 and len(pre) == 3:
            mdec, mpre = statistics.median(dec), statistics.median(pre)
            cells[lid] = {
                "decode_tok_s_median": round(mdec, 2),
                "prefill_tok_s_median": round(mpre, 2),
                "decode_fraction": round(mdec / ndec, 4),
                "prefill_fraction": round(mpre / npre, 4),
                "decode_reps": dec, "prefill_reps": pre,
            }
        else:
            cells[lid] = {"error": f"reps dec={len(dec)} pre={len(pre)}"}
    out[drv] = cells
(d / "perf-fractions.json").write_text(json.dumps(out, indent=1) + "\n")
for drv in ("patched", "base"):
    for lid, c in out[drv].items():
        if "error" in c:
            print(drv, lid, c["error"])
        else:
            print(f"{drv} {lid}: dec {c['decode_fraction']} pre {c['prefill_fraction']}")
PYPERF

# kernel probe summary
python3 - "$OUT" <<'PYK'
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
cells = {}
for drv in ("base", "patched"):
    for rep in ("1", "2", "3"):
        for line in open(d / f"probe-{drv}-r{rep}.log"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict) and "shape" in r:
                cells.setdefault(r["shape"], {}).setdefault(drv, {})[rep] = r
out = {}
for shape, per in cells.items():
    row = {}
    for drv in ("base", "patched"):
        ms = [v["median_ms"] for v in per.get(drv, {}).values()]
        tf = [v["tflops"] for v in per.get(drv, {}).values()]
        dg = {v["f16_digest"] for v in per.get(drv, {}).values()}
        if ms:
            row[drv] = {"median_ms": round(statistics.median(ms), 4),
                        "median_tflops": round(statistics.median(tf), 1),
                        "digests": sorted(dg)}
    if "base" in row and "patched" in row:
        row["delta_pct_ms"] = round(
            100.0 * (row["patched"]["median_ms"] - row["base"]["median_ms"])
            / row["base"]["median_ms"], 2)
    out[shape] = row
(d / "probe-summary.json").write_text(json.dumps(out, indent=1) + "\n")
for shape, row in out.items():
    b, p = row.get("base"), row.get("patched")
    if b and p:
        print(f"{shape}: base {b['median_tflops']} patched {p['median_tflops']} "
              f"GFLOP/s ({row['delta_pct_ms']:+.1f}% ms) "
              f"digests {'OK' if b['digests'] == p['digests'] else 'DIFF'}")
PYK
echo "$(date -u +%FT%TZ) window complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"

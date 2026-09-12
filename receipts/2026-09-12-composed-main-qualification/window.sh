#!/usr/bin/env bash
# Composed-main qualification window (receipts/2026-09-12-composed-main-qualification).
# One tree: origin/main ab08be8b. One top-level flock on /tmp/m1-gpu.lock.
# Inside the window, in order:
#   1. driver identity capture
#   2. canonical digest + performance matrix: fork and stock, discarded
#      warmup per driver, then 3 measured repetitions per driver,
#      alternating drivers, fresh bench_matrix process each run
#   3. digest assertions against the committed pins (fork and stock)
#   4. standing M1 battery: every suite under overlay/tests/omarchy/
#      named in AGENTS.md plus the capability-sim profile matrix
# Wheel and test binaries must already be built (CPU work, outside lock).
set -uo pipefail
ROOT="$HOME/src/mlx-omarchy-composed"
R="$ROOT/receipts/2026-09-12-composed-main-qualification"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
PY="$ROOT/.work/venv-run/bin/python"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
COMMIT_SHORT=ab08be8
mkdir -p "$R"

echo "=== wrapper start $(date -u +%FT%TZ) pid $$"

exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED (window announced start)"
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
} > "$R/drivers.txt"
cat "$R/drivers.txt"

# quiet gate: 1-min loadavg < 1.0 on three checks 20 s apart
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
echo "quiet gate done ok=$ok load=$(cut -d" " -f1 /proc/loadavg)"

( while :; do echo "$(date +%s) $(cut -d" " -f1-3 /proc/loadavg)"; sleep 10; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

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

HF_HUB_OFFLINE=1 run_one fork  "jwm1 honeykrisp-fork ab08be8b warmup-discarded" fork-warmup
HF_HUB_OFFLINE=1 run_one stock "jwm1 stock-mesa ab08be8b warmup-discarded" stock-warmup
for rep in r1 r2 r3; do
  HF_HUB_OFFLINE=1 run_one fork  "jwm1 honeykrisp-fork ab08be8b $rep" "fork-$rep"
  HF_HUB_OFFLINE=1 run_one stock "jwm1 stock-mesa ab08be8b $rep" "stock-$rep"
done

echo "=== digest + provenance assertions $(date -u +%FT%TZ)"
"$PY" - "$R" <<'PYCHECK'
import json, sys
from pathlib import Path
# Committed pins: Q4 identical on both drivers; BF16 per driver
# (receipts/2026-09-11-bf16-decode-fused-land/README.md).
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
report = {}
for drv in ("fork", "stock"):
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
            prov = leg["metrics"]["provenance_line"]
            want_for_drv = Q4.get(lid) or BF16[drv][lid]
            ok = got == want_for_drv
            stamped = COMMIT_SHORT in prov
            legs[lid] = {"digest": got, "expected": want_for_drv,
                         "held": ok, "provenance_stamped": stamped}
            if not ok:
                failures.append(f"{drv}-{rep} {lid}: got {got} want {want_for_drv} (MOVEMENT)")
            if not stamped:
                failures.append(f"{drv}-{rep} {lid}: provenance missing {COMMIT_SHORT}: {prov}")
        report[f"{drv}-{rep}"] = legs
(d / "digest-matrix.json").write_text(json.dumps(report, indent=1) + "\n")
for f in failures:
    print("FAIL", f)
print("ALL_DIGESTS_HELD" if not failures else "DIGEST_FAILURES")
sys.exit(1 if failures else 0)
PYCHECK
rc=$?
[ $rc -ne 0 ] && echo "DIGEST ASSERTION FAILED rc=$rc"

echo "=== performance summary $(date -u +%FT%TZ)"
"$PY" - "$R" <<'PYPERF'
import json, statistics, sys
from pathlib import Path
NATIVE = {  # receipts/native-baseline-2026-09-06/native-2026-09-06-summary.json
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
for drv in ("fork", "stock"):
    for lid, c in out[drv].items():
        if "error" in c:
            print(drv, lid, c["error"])
        else:
            print(f"{drv} {lid}: dec {c['decode_fraction']} pre {c['prefill_fraction']}")
PYPERF

echo "=== standing M1 battery $(date -u +%FT%TZ)"
BAT="$R/battery"; mkdir -p "$BAT"
SUITES="omarchy_runtime_tests omarchy_primitive_tests omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_kv_ops_tests omarchy_indexing_ops_tests omarchy_reduce_ops_tests omarchy_shape_ops_tests omarchy_linalg_ops_tests omarchy_copy_offset_tests omarchy_distributed_tests omarchy_compiled_tape_tests omarchy_fft_ops_tests omarchy_fft_general_tests omarchy_eig_ops_tests omarchy_take_fill_tests omarchy_conv_tests omarchy_complex_ops_tests omarchy_select_layout_tests omarchy_fast_regression_tests omarchy_scatter_determinism_tests omarchy_eq_math_tests omarchy_fused_chain_tests omarchy_error_contract_tests omarchy_ane_bundle_tests"
for s in $SUITES; do
  bin=".work/build-accept/tests/omarchy/$s"
  if [ ! -x "$bin" ]; then echo "SUITE_MISSING_BIN=$s"; echo "SUITE_MISSING_BIN=$s" >> "$BAT/summary.txt"; continue; fi
  if timeout 1800 "$bin" > "$BAT/$s.log" 2>&1; then
    echo "SUITE_PASS=$s" >> "$BAT/summary.txt"
  else
    echo "SUITE_FAIL=$s" >> "$BAT/summary.txt"
  fi
  tail -2 "$BAT/$s.log" | head -1
done
for prof in m1-honeykrisp-fork m1-stock-no-coopmat subgroup-size-64 small-shared-memory no-cooperative-matrix; do
  if MLX_OMARCHY_ALLOW_NON_APPLE=1 timeout 1800 \
      .work/build-accept/tests/omarchy/omarchy_capability_sim_tests "$prof" \
      > "$BAT/capsim-$prof.log" 2>&1; then
    echo "SUITE_PASS=omarchy_capability_sim_tests/$prof" >> "$BAT/summary.txt"
  else
    echo "SUITE_FAIL=omarchy_capability_sim_tests/$prof" >> "$BAT/summary.txt"
  fi
done
grep -c "SUITE_PASS" "$BAT/summary.txt" || true
grep "SUITE_FAIL\|SUITE_MISSING" "$BAT/summary.txt" || echo "NO_BATTERY_FAILURES"

kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ) window complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"

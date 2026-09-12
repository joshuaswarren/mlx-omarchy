#!/usr/bin/env bash
# PrefillNativeOrder window 2: final-wheel qualification.
# Six-leg digest matrix on both drivers (discarded warmup + 3 reps each,
# alternating), paired performance against the base wheel (origin/main,
# a0775c37) on the fork driver, and the targeted battery for the touched
# routes (matmul family, capability-sim profiles, fast ops, runtime).
set -uo pipefail
ROOT="$HOME/src/mlx-omarchy-fma"
ROOT_BASE="$HOME/src/mlx-omarchy-fma-base"
R="$ROOT/receipts/2026-09-12-prefill-native-order"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
BASE_WHEEL="$(ls "$ROOT_BASE"/dist/mlx_omarchy-*.whl | head -n1)"
PY="$ROOT/.work/venv-run/bin/python"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
CAND_SHORT="$(git -C "$ROOT" rev-parse --short=7 HEAD)"
BASE_SHORT=a0775c3
mkdir -p "$R"

echo "=== window2 start $(date -u +%FT%TZ) pid $$"
exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED (window2 announced start)"
T0=$(date +%s)

{
  echo "driver-fork: $(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)"
  echo "driver-stock: $(pacman -Q mesa 2>/dev/null || echo MISSING)"
  echo "kernel: $(uname -r)"
  echo "machine: $(uname -m) $(uname -p)"
  echo "stock-icd-sha256: $(sha256sum "$STOCK_ICD" | cut -d' ' -f1)"
  echo "wheel-candidate: $(basename "$WHEEL")"
  echo "wheel-candidate-sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "wheel-base: $(basename "$BASE_WHEEL")"
  echo "wheel-base-sha256: $(sha256sum "$BASE_WHEEL" | cut -d' ' -f1)"
  echo "source-commit-candidate: $(git -C "$ROOT" rev-parse HEAD)"
  echo "source-commit-base: a0775c374a2427f3c713c2177eae9dc9e754bc6d"
} > "$R/drivers-window2.txt"
cat "$R/drivers-window2.txt"

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

run_one() {  # drv label outfile wheel
  local drv="$1" label="$2" out="$3" wheel="$4"
  local -a ICD=()
  if [ "$drv" = stock ]; then
    ICD=(env VK_DRIVER_FILES="$STOCK_ICD")
  else
    ICD=(env)
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
  local rc=$?
  echo "$out exit=$rc $(date -u +%H:%M:%S)"
  grep -cE '"status": "measured"' "$R/$out.json" 2>/dev/null || true
}

HF_HUB_OFFLINE=1 run_one fork  "jwm1 honeykrisp-fork ${CAND_SHORT} warmup-discarded" cand-fork-warmup "$WHEEL"
HF_HUB_OFFLINE=1 run_one stock "jwm1 stock-mesa ${CAND_SHORT} warmup-discarded" cand-stock-warmup "$WHEEL"
HF_HUB_OFFLINE=1 run_one fork  "jwm1 honeykrisp-fork ${BASE_SHORT} warmup-discarded" base-fork-warmup "$BASE_WHEEL"
for rep in r1 r2 r3; do
  HF_HUB_OFFLINE=1 run_one stock "jwm1 stock-mesa ${CAND_SHORT} $rep" "cand-stock-$rep" "$WHEEL"
  HF_HUB_OFFLINE=1 run_one fork  "jwm1 honeykrisp-fork ${CAND_SHORT} $rep" "cand-fork-$rep" "$WHEEL"
  HF_HUB_OFFLINE=1 run_one fork  "jwm1 honeykrisp-fork ${BASE_SHORT} $rep" "base-fork-$rep" "$BASE_WHEEL"
done

echo "=== digest + provenance assertions $(date -u +%FT%TZ)"
"$PY" - "$R" "$CAND_SHORT" <<'PYCHECK'
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
cand_short = sys.argv[2]
failures = []
report = {}
for kind, drv, rep in [
    (k, drv, rep)
    for k in ("cand", "base")
    for drv in ("fork", "stock")
    for rep in ("r1", "r2", "r3")
]:
    if kind == "base" and (drv != "fork"):
        continue
    mj = d / f"{kind}-{drv}-{rep}.json"
    if not mj.exists():
        failures.append(f"{kind}-{drv}-{rep}: MISSING")
        continue
    run = json.loads(mj.read_text())
    legs = {}
    for leg in run["legs"]:
        lid = leg["leg_id"]
        if leg["status"] != "measured":
            failures.append(f"{kind}-{drv}-{rep} {lid}: status={leg['status']}")
            continue
        got = leg["metrics"]["generated_ids_sha256_16"]
        want = Q4.get(lid) or BF16[drv][lid]
        ok = got == want
        stamp = cand_short if kind == "cand" else "a0775c3"
        stamped = stamp in leg["metrics"]["provenance_line"]
        legs[lid] = {"digest": got, "expected": want, "held": ok,
                     "provenance_stamped": stamped}
        if not ok:
            failures.append(
                f"{kind}-{drv}-{rep} {lid}: got {got} want {want} (MOVEMENT)")
        if not stamped:
            failures.append(f"{kind}-{drv}-{rep} {lid}: provenance missing {stamp}")
    report[f"{kind}-{drv}-{rep}"] = legs
(d / "digest-matrix.json").write_text(json.dumps(report, indent=1) + "\n")
for f in failures:
    print("FAIL", f)
print("ALL_DIGESTS_HELD" if not failures else "DIGEST_FAILURES")
sys.exit(1 if failures else 0)
PYCHECK
rc=$?
[ $rc -ne 0 ] && echo "DIGEST ASSERTION FAILED rc=$rc"

echo "=== performance fractions $(date -u +%FT%TZ)"
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
series = {
    ("cand", "fork"), ("cand", "stock"), ("base", "fork"),
}
for kind, drv in series:
    cells = {}
    for lid, (ndec, npre) in NATIVE.items():
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
    out[f"{kind}-{drv}"] = cells
(d / "perf-fractions.json").write_text(json.dumps(out, indent=1) + "\n")
for side, cells in out.items():
    for lid, c in cells.items():
        if "error" in c:
            print(side, lid, c["error"])
        else:
            print(f"{side} {lid}: dec {c['decode_fraction']} pre "
                  f"{c['prefill_fraction']} "
                  f"({c['prefill_tok_s_median']} tok/s)")
PYPERF

echo "=== targeted battery $(date -u +%FT%TZ)"
BAT="$R/battery"; mkdir -p "$BAT"
if [ ! -x ".work/build-accept/tests/omarchy/omarchy_matmul_family_tests" ]; then
  echo "test build missing; run: cmake -S .work/mlx -B .work/build-accept -DMLX_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build .work/build-accept -j6"
fi
for s in omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests omarchy_primitive_tests; do
  bin=".work/build-accept/tests/omarchy/$s"
  if [ ! -x "$bin" ]; then echo "SUITE_MISSING_BIN=$s" | tee -a "$BAT/summary.txt"; continue; fi
  if timeout 1800 "$bin" > "$BAT/$s.log" 2>&1; then
    echo "SUITE_PASS=$s" | tee -a "$BAT/summary.txt"
  else
    echo "SUITE_FAIL=$s" | tee -a "$BAT/summary.txt"
  fi
done
for prof in m1-honeykrisp-fork m1-stock-no-coopmat subgroup-size-64 small-shared-memory no-cooperative-matrix; do
  if MLX_OMARCHY_ALLOW_NON_APPLE=1 timeout 1800 \
      .work/build-accept/tests/omarchy/omarchy_capability_sim_tests "$prof" \
      > "$BAT/capsim-$prof.log" 2>&1; then
    echo "SUITE_PASS=omarchy_capability_sim_tests/$prof" | tee -a "$BAT/summary.txt"
  else
    echo "SUITE_FAIL=omarchy_capability_sim_tests/$prof" | tee -a "$BAT/summary.txt"
  fi
done

kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ) window2 complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"

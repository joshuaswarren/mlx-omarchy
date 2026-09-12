#!/usr/bin/env bash
# Corrected-instrument re-measurement (receipts/2026-09-12-composed-main-qualification/corrections).
# Same wheel (ab08be8b), same protocol as the qualification window, with the
# one change under test: bench_matrix now reads prefill_s from bench_decode's
# JSON result line (6-decimal span) instead of the 3-decimal printed line.
set -uo pipefail
ROOT="$HOME/src/mlx-omarchy-composed"
R="$ROOT/receipts/2026-09-12-composed-main-qualification/corrections"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
PY="$ROOT/.work/venv-run/bin/python"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
mkdir -p "$R"

echo "=== corrected window start $(date -u +%FT%TZ) pid $$"
exec 9>/tmp/m1-gpu.lock
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
T0=$(date +%s)

ok=0
for i in $(seq 1 60); do
  l=$(cut -d" " -f1 /proc/loadavg)
  if awk -v x="$l" 'BEGIN{exit !(x < 1.0)}'; then
    ok=$((ok+1)); [ "$ok" -ge 3 ] && break
  else
    ok=0
  fi
  sleep 20
done
echo "quiet gate done ok=$ok load=$(cut -d" " -f1 /proc/loadavg)"

( while :; do echo "$(date +%s) $(cut -d" " -f1-3 /proc/loadavg)"; sleep 10; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

run_one() {  # drv label outfile
  local drv="$1" label="$2" out="$3"
  local -a ICD=()
  [ "$drv" = stock ] && ICD=(env VK_DRIVER_FILES="$STOCK_ICD") || ICD=(env)
  echo "=== run $out $(date -u +%H:%M:%S) ==="
  HF_HUB_OFFLINE=1 timeout 1500 "${ICD[@]}" "$PY" scripts/bench_matrix.py --mode run \
    --python "$PY" \
    --wheel "$WHEEL" \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --host-label "$label" \
    --timeout 900 \
    --out "$R/$out.json" \
    > "$R/$out.log" 2>&1
  echo "$out exit=$? $(date -u +%H:%M:%S)"
}

run_one fork  "jwm1 honeykrisp-fork ab08be8b corrected-warmup" fork-warmup
run_one stock "jwm1 stock-mesa ab08be8b corrected-warmup" stock-warmup
for rep in r1 r2 r3; do
  run_one fork  "jwm1 honeykrisp-fork ab08be8b corrected-$rep" "fork-$rep"
  run_one stock "jwm1 stock-mesa ab08be8b corrected-$rep" "stock-$rep"
done

"$PY" - "$R" <<'PYCHECK'
import json, sys
from pathlib import Path
Q4 = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
}
BF16 = {
    "fork": {"qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
             "qwen25-0.5b-bf16:long-decode-128": "8690dc83246b39f8",
             "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5"},
    "stock": {"qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
              "qwen25-0.5b-bf16:long-decode-128": "46108ad71157cb4d",
              "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5"},
}
d = Path(sys.argv[1])
fails = []
spans = {}
for drv in ("fork", "stock"):
    for rep in ("r1", "r2", "r3"):
        run = json.loads((d / f"{drv}-{rep}.json").read_text())
        for leg in run["legs"]:
            lid = leg["leg_id"]
            if lid not in Q4 and not lid.startswith("qwen25-0.5b-bf16"):
                continue
            got = leg["metrics"]["generated_ids_sha256_16"]
            want = Q4.get(lid) or BF16[drv][lid]
            if got != want:
                fails.append(f"{drv}-{rep} {lid}: {got} != {want}")
            if leg["status"] == "measured":
                spans.setdefault(f"{drv}:{lid}", []).append(
                    leg["metrics"]["prefill_s"])
(d / "corrected-spans.json").write_text(json.dumps(spans, indent=1) + "\n")
print("ALL_DIGESTS_HELD" if not fails else "DIGEST_FAILURES")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
PYCHECK
rc=$?
[ $rc -ne 0 ] && echo "DIGEST ASSERTION FAILED rc=$rc"

kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ) corrected window complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"
exit $rc

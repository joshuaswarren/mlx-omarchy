#!/usr/bin/env bash
# Weight-memory-type window (receipts/2026-09-12-weight-memory-type).
#
# One wheel (d389c24), ONE env flag is the whole change under test:
#   MLX_OMARCHY_BIG_UNCACHED=1 routes allocations >= 1 MiB to the
#   honeykrisp memory type WITHOUT HOST_CACHED (type 1) instead of the
#   host-cached default (type 0). Both types are
#   DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT; the driver exposes no
#   non-host-visible DEVICE_LOCAL type at all.
#
# Protocol (Main's spec, 2026-09-12):
#   - digests: env ON, both drivers, 3 reps after a discarded warmup,
#     asserted against the canonical pins;
#   - perf: paired INTERLEAVED env-OFF (A) / env-ON (B) matrix runs in
#     this one window, fork driver; every run asserts all six digests.
# Run under ONE top-level flock: bash receipts/2026-09-12-weight-memory-type/run-window.sh
set -uo pipefail
ROOT="$HOME/src/mlx-omarchy-wmt"
R="$ROOT/receipts/2026-09-12-weight-memory-type"
WHEEL="$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)"
PY="$ROOT/.work/venv-run-wmt/bin/python"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
mkdir -p "$R"

echo "=== weight-memory-type window start $(date -u +%FT%TZ) pid $$"
echo "wheel $(basename "$WHEEL")"
sha256sum "$WHEEL"
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

run_one() {  # drv label outfile extra_env...
  local drv="$1" label="$2" out="$3"; shift 3
  local -a ICD=()
  [ "$drv" = stock ] && ICD=(env VK_DRIVER_FILES="$STOCK_ICD") || ICD=(env)
  echo "=== run $out $(date -u +%H:%M:%S) ==="
  HF_HUB_OFFLINE=1 timeout 1500 "${ICD[@]}" "$@" \
    "$PY" scripts/bench_matrix.py --mode run \
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

# Discarded warmups (env ON), with the per-class allocation dump on the
# fork warmup for the receipt.
run_one fork  "jwm1 honeykrisp-fork d389c24 bigunc-warmup memtypelog" \
    fork-warmup MLX_OMARCHY_BIG_UNCACHED=1 MLX_OMARCHY_LOG_MEMTYPE=1
run_one stock "jwm1 stock-mesa d389c24 bigunc-warmup" \
    stock-warmup MLX_OMARCHY_BIG_UNCACHED=1

# Digest runs: env ON, three reps per driver.
for rep in d1 d2 d3; do
  run_one fork  "jwm1 honeykrisp-fork d389c24 bigunc-$rep" \
      "fork-$rep" MLX_OMARCHY_BIG_UNCACHED=1
  run_one stock "jwm1 stock-mesa d389c24 bigunc-$rep" \
      "stock-$rep" MLX_OMARCHY_BIG_UNCACHED=1
done

# Paired interleaved perf: A = flag OFF (baseline), B = flag ON.
for rep in p1 p2 p3 p4 p5 p6; do
  run_one fork "jwm1 honeykrisp-fork d389c24 base-$rep" "base-$rep"
  run_one fork "jwm1 honeykrisp-fork d389c24 bigunc-$rep" \
      "bigunc-$rep" MLX_OMARCHY_BIG_UNCACHED=1
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
digest_runs = [f"{drv}-{rep}" for drv in ("fork", "stock")
               for rep in ("d1", "d2", "d3")]
perf_pairs = [(f"base-{p}", f"bigunc-{p}")
              for p in ("p1", "p2", "p3", "p4", "p5", "p6")]
def check_run(name, drv):
    run = json.loads((d / f"{name}.json").read_text())
    out = {}
    for leg in run["legs"]:
        lid = leg["leg_id"]
        if lid not in Q4 and not lid.startswith("qwen25-0.5b-bf16"):
            continue
        got = leg["metrics"]["generated_ids_sha256_16"]
        want = Q4.get(lid) or BF16[drv][lid]
        if got != want:
            fails.append(f"{name} {lid}: {got} != {want}")
        if leg["status"] == "measured":
            out[lid] = leg["metrics"]["decode_tok_s"]
        else:
            fails.append(f"{name} {lid}: status={leg['status']}")
    return out
digest_summary = {}
for name in digest_runs:
    digest_summary[name] = check_run(name, name.split("-")[0])
perf = {"A": [], "B": []}
for a, b in perf_pairs:
    perf["A"].append(check_run(a, "fork"))
    perf["B"].append(check_run(b, "fork"))
(d / "perf-runs.json").write_text(json.dumps(
    {"digest_runs": digest_summary, "perf_A": perf["A"], "perf_B": perf["B"]},
    indent=1) + "\n")
print("ALL_DIGESTS_HELD" if not fails else "DIGEST_FAILURES")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
PYCHECK
rc=$?
[ $rc -ne 0 ] && echo "DIGEST ASSERTION FAILED rc=$rc"

kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ) weight-memory-type window complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"
exit $rc

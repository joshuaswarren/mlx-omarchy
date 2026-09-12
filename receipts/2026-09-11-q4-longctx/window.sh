#!/usr/bin/env bash
# Q4LongCtxScaling window: decode SDPA packed-vs-scalar paired evidence.
#   1. queue on /tmp/m1-gpu.lock (single top-level flock, whole window)
#   2. quiet CPU gate (1-min loadavg < 1.0, 3 checks 20 s apart)
#   3. BIT-IDENTITY GATE: sdpa_equiv_sweep.py (17 k_len, 3 regimes)
#   4. paired decode-sdpa chain microbench at k=30/262/1053
#   5. digest-gated paired legs: 3 Q4 workloads x {packed, scalar} x 2 reps,
#      alternating arms; every leg asserts the canonical generated-id pin
# Wheel must already be built into dist/ and installed in .work/venv-run.
set -u
LOG=/tmp/q4longctx-window.log
exec >>"$LOG" 2>&1
echo "=== wrapper start $(date -u +%FT%TZ) pid $$"

timeout 3000 flock -w 2900 /tmp/m1-gpu.lock bash -s <<'INNER'
echo "=== lock held $(date -u +%FT%TZ) by $$"
cd ~/src/mlx-omarchy-q4longctx
R=receipts/2026-09-11-q4-longctx
PY=$HOME/src/mlx-omarchy-q4longctx/.work/venv-run/bin/python
PIN=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3

# quiet gate
ok=0
for i in $(seq 1 60); do
  l=$(cut -d" " -f1 /proc/loadavg)
  if awk -v x="$l" 'BEGIN{exit !(x < 1.0)}'; then
    ok=$((ok+1))
    [ "$ok" -ge 3 ] && break
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


echo "=== qmm m=262 probe $(date -u +%FT%TZ)"
MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 "$PY" "$R/qmm_m262_probe.py" \
  --out "$R/qmm_m262.json"
echo "=== equiv gate $(date -u +%FT%TZ)"
MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 "$PY" "$R/sdpa_equiv_sweep.py" \
  --out "$R/equiv.json" || { echo "EQUIV GATE FAILED"; kill $SAMPLER; exit 1; }

echo "=== sdpa chain microbench $(date -u +%FT%TZ)"
MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 "$PY" "$R/sdpa_chain_micro.py" \
  --out "$R/sdpa_chain.json" --reps 20

echo "=== paired digest-gated legs $(date -u +%FT%TZ)"
for rep in r1 r2; do
  for arm in packed scalar; do
    out="$R/leg-${arm}-${rep}.json"
    log="$R/leg-${arm}-${rep}.log"
    [ -s "$out" ] && { echo "$out cached"; continue; }
    envm=""
    [ "$arm" = scalar ] && envm="MLX_OMARCHY_SDPA_DECODE_SCALAR=1"
    env -u VK_DRIVER_FILES -u HK_PERFTEST HF_HUB_OFFLINE=1 \
      MLX_DISABLE_COMPILE=1 $envm \
      python3 scripts/bench_matrix.py --mode run \
      --manifest "$R/bench_matrix_q4only.json" \
      --python "$PY" \
      --wheel "dist/$(ls dist | grep 'cp314.*aarch64.whl' | head -1)" \
      --expect-pins "qwen25-0.5b-4bit=$PIN" \
      --host-label "jwm1-q4longctx-$arm-$rep" \
      --timeout 900 --out "$out" > "$log" 2>&1 \
      && echo "$arm $rep ok" || echo "$arm $rep FAILED"
  done
done

echo "=== digest assertions $(date -u +%FT%TZ)"
python3 - <<'PYCHECK'
import json, sys
EXPECTED = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}
ok = True
for arm in ("packed", "scalar"):
    for rep in ("r1", "r2"):
        data = json.load(open(f"receipts/2026-09-11-q4-longctx/leg-{arm}-{rep}.json"))
        for leg in data["legs"]:
            wl = leg["workload_id"]
            got = leg["metrics"]["generated_ids_sha256_16"]
            good = got == EXPECTED[wl]
            ok = ok and good
            print(f"{arm} {rep} {wl}: {got} {'OK' if good else 'MISMATCH expected ' + EXPECTED[wl]}")
sys.exit(0 if ok else 1)
PYCHECK
rc=$?
[ $rc -ne 0 ] && echo "DIGEST ASSERTION FAILED"
[ $rc -ne 0 ] && exit $rc

echo "=== window done $(date -u +%FT%TZ)"
INNER
echo "=== wrapper exit rc=$? $(date -u +%FT%TZ)"

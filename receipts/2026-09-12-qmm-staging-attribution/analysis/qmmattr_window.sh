#!/usr/bin/env bash
# Staging-attribution measurement window
# (receipts/2026-09-12-qmm-staging-attribution).
# One top-level flock on /tmp/m1-gpu.lock, quiet gate inside the lock,
# then: per-probe ISA dumps (single-cell runner), then a discarded
# warmup round plus three measured interleaved rounds of the
# kernel-isolated probe for base / hoist / nodequant.
# Print wheel + driver versions next to every number (drivers.txt).
set -euo pipefail
ROOT="$HOME/src/mlx-omarchy-qmmattr"
R="$ROOT/receipts/2026-09-12-qmm-staging-attribution"
PROBE="${PROBE:-$HOME/benchq/qmm-coop-bench/qmm_coop_bench_probe.py}"
PROBE_SHA="$(sha256sum "$PROBE" | cut -d' ' -f1)"
DUMPRUN="$ROOT/receipts/2026-09-12-qmm-staging-attribution/analysis/qmm_dump_one.py"
PY="$ROOT/.work/venv-run/bin/python"
mkdir -p "$R/logs" "$R/dumps"

echo "=== wrapper start $(date -u +%FT%TZ) pid $$"
exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"

{
  echo "driver-fork: $(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)"
  echo "kernel: $(uname -r)"
  echo "machine: $(uname -m) $(uname -p)"
  echo "wheel: $(basename "$(ls "$ROOT"/dist/mlx_omarchy-*.whl | head -n1)")"
  echo "wheel-sha256: $(sha256sum "$ROOT"/dist/mlx_omarchy-*.whl | cut -d' ' -f1)"
  echo "source-commit: $(git -C "$ROOT" rev-parse HEAD)"
  echo "probe-script-sha256: $PROBE_SHA"
  echo "venv-python: $($PY --version 2>&1)"
} > "$R/drivers.txt"
cat "$R/drivers.txt"

# quiet gate: 1-min loadavg < 1.0 on three checks 20 s apart
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
echo "quiet gate done ok=$ok load=$(cut -d" " -f1 /proc/loadavg)"

SAMPLER_PID=""
( while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 10; done ) > "$R/logs/load-sampler.log" &
SAMPLER_PID=$!
trap '[ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null' EXIT

# --- ISA dumps: one cell, one dispatch per probe, cache disabled ---
for arm in base hoist nodequant unrollbase; do
  case "$arm" in
    base)      PROBE_ENV=();;
    hoist)     PROBE_ENV=(MLX_OMARCHY_QMM_COOP_PROBE=hoist);;
    nodequant) PROBE_ENV=(MLX_OMARCHY_QMM_COOP_PROBE=nodequant);;
    unrollbase) PROBE_ENV=(MLX_OMARCHY_QMM_COOP_PROBE=unrollbase);;
  esac
  echo "$(date -u +%FT%TZ) ISA dump: $arm"
  env "${PROBE_ENV[@]}" AGX_SIMDMAT=1 AGX_MESA_DEBUG=shaders \
    MESA_SHADER_CACHE_DISABLE=true \
    "$PY" "$DUMPRUN" > "$R/logs/dump-$arm-stdout.log" 2> "$R/dumps/dump-$arm.log"
  tail -1 "$R/logs/dump-$arm-stdout.log"
done

# --- discard-warmup + 3 measured interleaved rounds ---
run_round() {  # $1 = round label
  local r="$1"
  for arm in base hoist nodequant unrollbase; do
    case "$arm" in
      base)      PROBE_ENV=();;
      hoist)     PROBE_ENV=(MLX_OMARCHY_QMM_COOP_PROBE=hoist);;
      nodequant) PROBE_ENV=(MLX_OMARCHY_QMM_COOP_PROBE=nodequant);;
    unrollbase) PROBE_ENV=(MLX_OMARCHY_QMM_COOP_PROBE=unrollbase);;
    esac
    echo "$(date -u +%FT%TZ) probe $arm round=$r"
    env "${PROBE_ENV[@]}" AGX_SIMDMAT=1 MESA_SHADER_CACHE_DISABLE=true \
      "$PY" "$PROBE" > "$R/logs/probe-$arm-$r.log" 2>&1
    grep -c tflops "$R/logs/probe-$arm-$r.log" || true
  done
}
run_round warmup
run_round r1
run_round r2
run_round r3

# --- digest gate: hoist must match base cell-for-cell, every round ---
python3 - "$R" <<'PYEOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1]) / "logs"
fail = 0
for r in ("r1", "r2", "r3"):
    base = [json.loads(l) for l in (root / f"probe-base-{r}.log").read_text().splitlines() if l.startswith("{")]
    for arm in ("hoist", "unrollbase"):
        rows = [json.loads(l) for l in (root / f"probe-{arm}-{r}.log").read_text().splitlines() if l.startswith("{")]
        assert len(base) == len(rows) == 8, (len(base), len(rows))
        for b, h in zip(base, rows):
            assert b["shape"] == h["shape"]
            if b["f16_digest"] != h["f16_digest"]:
                print(f"DIGEST MISMATCH {r} {arm} {b['shape']}: base={b['f16_digest']} probe={h['f16_digest']}")
                fail += 1
print("DIGEST-GATE:", "FAIL" if fail else "PASS (hoist and unrollbase bit-identical to base on all cells x 3 rounds)")
sys.exit(1 if fail else 0)
PYEOF

echo "$(date -u +%FT%TZ) WINDOW DONE"

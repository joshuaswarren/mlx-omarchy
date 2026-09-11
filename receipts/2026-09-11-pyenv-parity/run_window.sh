#!/usr/bin/env bash
# PythonHostParity measurement window (jwm1), v2.
# One bench_matrix call per config x rep (runs the 3 canonical Q4 workloads,
# non-explicit only), then a hostphases split leg. Digests checked per leg.
# usage: run_window.sh <window-name> <reps> cfg...  cfg=NAME=venvpy:wheel:trace(0|1)
set -u
WIN="$1"; REPS="$2"; shift 2
ROOT="$HOME/src/mlx-HostPathOverhead"
R="$ROOT/receipts/2026-09-11-pyenv-parity"
HP="$ROOT/receipts/2026-09-10-hostpath"
OUT="$R/legs/$WIN"
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
PIN="qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
declare -A CANON=( [short-decode-32]=7fd25a869ff21678 [long-decode-128]=4cc08910089477fd [longctx-1024-decode-32]=7da83f06ec9f001d )
mkdir -p "$OUT"
LOG="$OUT/window.log"
exec >>"$LOG" 2>&1
trap 'echo "[forensic] TERM at $(date -u +%FT%TZ)"; exit 143' TERM
trap 'echo "[forensic] HUP at $(date -u +%FT%TZ)"; exit 129' HUP
echo "=== window $WIN start $(date -u +%FT%TZ) pid $$ configs: $*"

loadavg() { cut -d' ' -f1 /proc/loadavg; }
quiet_gate() {
  for i in $(seq 1 60); do
    la=$(loadavg)
    if awk -v v="$la" 'BEGIN{exit !(v<1.0)}'; then
      echo "[gate] quiet at loadavg=$la $(date -u +%FT%TZ)"; return 0
    fi
    sleep 20
  done
  echo "[gate] NOT quiet after 20min, proceeding"
}

fail=0
digest_check() { # file
  "$ROOT/.work/venv-hpo/bin/python" - "$1" <<'PYDC'
import json, sys
canon = {"short-decode-32": "7fd25a869ff21678",
         "long-decode-128": "4cc08910089477fd",
         "longctx-1024-decode-32": "7da83f06ec9f001d"}
doc = json.load(open(sys.argv[1]))
bad = 0
for leg in doc.get("legs", [doc]):
    wl = leg.get("workload_id")
    m = leg.get("metrics", {})
    d = m.get("generated_ids_sha256_16")
    if wl in canon and d != canon[wl]:
        print(f"[DIGEST] FAIL {wl} got={d} want={canon[wl]}")
        bad = 1
    elif wl in canon:
        print(f"[DIGEST] ok {wl} {d}")
sys.exit(bad)
PYDC
  [ $? -ne 0 ] && fail=1
  return 0
}

run_matrix() { # name rep venvpy wheel
  local tag="$1-r$2"; local f="$OUT/$tag.json"
  if [ -f "$f" ]; then echo "[skip] $tag"; return 0; fi
  HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
    "$3" "$ROOT/scripts/bench_matrix.py" --mode run \
      --manifest "$HP/manifest-q4.json" \
      --python "$3" --wheel "$4" \
      --expect-pins "$PIN" \
      --host-label "jwm1-pyhp-$tag" --timeout 900 \
      --out "$f" > "$OUT/$tag.log" 2>&1
  echo "[matrix] $tag rc=$? $(date -u +%FT%TZ)"
  digest_check "$f"
}

split_leg() { # name rep venvpy wheel traceflag
  local tag="split-$1-r$2"; local f="$OUT/$tag.json"
  if [ -f "$f" ]; then echo "[skip] $tag"; return 0; fi
  if [ "$5" = 1 ]; then export MLX_OMARCHY_HOST_TRACE="$OUT/$tag-trace.json"; else unset MLX_OMARCHY_HOST_TRACE || true; fi
  HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
    "$3" "$HP/hostphases.py" --model "$MODEL" --prompt "Hi" --tokens 32 \
      --trace-out "$OUT/$tag-trace.json" --out "$f" > "$OUT/$tag.log" 2>&1
  echo "[split] $tag rc=$? $(date -u +%FT%TZ)"
}

declare -a CFGS=("$@")
declare -A CFG_PY CFG_WHEEL CFG_TRACE
for c in "${CFGS[@]}"; do
  name="${c%%=*}"; rest="${c#*=}"
  CFG_PY[$name]="${rest%%:*}"; rest="${rest#*:}"
  CFG_WHEEL[$name]="${rest%%:*}"
  CFG_TRACE[$name]="${rest#*:}"
done

exec 9>/tmp/m1-gpu.lock || { echo "[lock] cannot open lock file"; exit 3; }
flock -w 14400 9 || { echo "[lock] FAILED to acquire"; exit 3; }
echo "[lock] acquired $(date -u +%FT%TZ)"
( while true; do echo "$(date -u +%FT%TZ) load=$(loadavg)"; sleep 10; done ) > "$OUT/loadavg.log" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
quiet_gate

for rep in $(seq 1 "$REPS"); do
  for c in "${CFGS[@]}"; do
    name="${c%%=*}"
    run_matrix "$name" "$rep" "${CFG_PY[$name]}" "${CFG_WHEEL[$name]}"
    split_leg "$name" "$rep" "${CFG_PY[$name]}" "${CFG_WHEEL[$name]}" "${CFG_TRACE[$name]}"
  done
done

kill $SAMPLER 2>/dev/null
echo "[micro] binding microbench per config"
for c in "${CFGS[@]}"; do
  name="${c%%=*}"
  f="$OUT/micro-$name.json"
  if [ ! -f "$f" ]; then
    HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 "${CFG_PY[$name]}" "$R/microbench_binding.py" --ops 2000 --out "$f" > "$OUT/micro-$name.log" 2>&1
    echo "[micro] $name rc=$?"
  fi
done

if [ $fail -ne 0 ]; then echo "=== window $WIN END: DIGEST FAILURE $(date -u +%FT%TZ)"; exit 4; fi
echo "=== window $WIN end $(date -u +%FT%TZ) load=$(loadavg)"

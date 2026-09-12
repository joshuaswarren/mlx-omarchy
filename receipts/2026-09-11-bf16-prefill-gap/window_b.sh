#!/usr/bin/env bash
# Bf16PrefillGap verification window (lock held by caller; <=25 min).
# Phase A: provenance (wheel version + sha256 next to every number).
# Phase B: bit-identity probe base vs candidate (sdpa causal / pair
#          staging / f32 k-tail) — must be byte-identical per case.
# Phase C: paired digest+timing matrix, fork driver, base vs candidate.
set -uo pipefail
cd ~/src/mlx-Bf16PrefillGap
R=receipts-work
PYB=$R/venv-base/bin/python
PYC=$R/venv-cand/bin/python
WB=$(ls -1 $R/base-wheel.txt | head -1)
WHEELB=$(cat $R/base-wheel.txt)
WHEELC=$(cat $R/cand-wheel.txt)
mkdir -p $R/winb

for w in "$WHEELB" "$WHEELC"; do
  [[ -f $w ]] || { echo "FATAL: wheel missing $w"; exit 3; }
  echo "wheel $w"
  sha256sum "$w"
  unzip -p "$w" mlx_omarchy/version.py 2>/dev/null | grep -m1 version || true
done
driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)
echo "driver: $driver_pkg"
[[ $driver_pkg == "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1" ]] || {
  echo "FATAL: fork driver is not the pinned honeykrisp build"; exit 5; }

echo "== phase B: bit-identity probe =="
$PYB $R/probe_bitexact.py base > $R/winb/probe-base.ndjson 2>$R/winb/probe-base.log || {
  echo "FATAL: base probe crashed"; tail -5 $R/winb/probe-base.log; exit 3; }
$PYC $R/probe_bitexact.py cand > $R/winb/probe-cand.ndjson 2>$R/winb/probe-cand.log || {
  echo "FATAL: cand probe crashed"; tail -5 $R/winb/probe-cand.log; exit 3; }
python3 - <<'PYEOF'
import json, sys
def load(p):
    return {r["case"]: r["sha256"]
            for r in map(json.loads, open(p)) if r.get("case")}
b = load("receipts-work/winb/probe-base.ndjson")
c = load("receipts-work/winb/probe-cand.ndjson")
bad = 0
for k in sorted(b):
    ok = b[k] == c.get(k)
    bad += not ok
    print(("PASS" if ok else "FAIL") + f" {k} {b[k][:16]}")
print(f"BITIDENTITY {'PASS' if bad == 0 else f'FAIL({bad})'} {len(b)} cases")
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || { echo "FATAL: bit-identity failed - aborting before matrix"; exit 3; }

echo "== phase C: paired matrix (fork) =="
run_cell() {  # venv-python wheel label
  local py=$1 wheel=$2 label=$3
  local run="$R/winb/$label"
  mkdir -p "$run"
  MLX_DISABLE_COMPILE=1 timeout 1200 scripts/bench_matrix.py --mode run \
    --python "$py" --wheel "$wheel" \
    --host-label "jwm1-$label" --timeout 600 \
    --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
    --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
    --out "$run/matrix.json" > "$run.log" 2>&1
  local ok
  ok=$(grep -c 'verified=match' "$run.log" || true)
  echo "$label: verified=match x${ok} rc=$?"
}
run_cell "$PYB" "$WHEELB" base
run_cell "$PYC" "$WHEELC" cand

echo "== digest + timing extraction =="
python3 - <<'PYEOF'
import json
PINS = {
    ("qwen25-0.5b-4bit", "short-decode-32"): "7fd25a869ff21678",
    ("qwen25-0.5b-4bit", "long-decode-128"): "4cc08910089477fd",
    ("qwen25-0.5b-4bit", "longctx-1024-decode-32"): "7da83f06ec9f001d",
    ("qwen25-0.5b-bf16", "short-decode-32"): "f26175202f3dabe9",
    ("qwen25-0.5b-bf16", "long-decode-128"): "8690dc83246b39f8",
    ("qwen25-0.5b-bf16", "longctx-1024-decode-32"): "ff502900d2a179a5",
}
for label in ("base", "cand"):
    m = json.load(open(f"receipts-work/winb/{label}/matrix.json"))
    for leg in m.get("legs", []):
        met = leg.get("metrics", {})
        model = leg.get("model_id", "")
        wl = leg.get("workload_id", "")
        digest = met.get("generated_ids_sha256_16", "")
        pin = PINS.get((model, wl), "?")
        ok = "HOLD" if digest == pin else "MOVE"
        prov = met.get("provenance_line", "")
        print(f"{label:5s} {model} {wl:22s} digest={digest} pin={pin} "
              f"{ok} prefill={met.get('prefill_tok_s')} tok/s "
              f"decode={met.get('decode_tok_s')} tok/s | {prov[:80]}")
PYEOF

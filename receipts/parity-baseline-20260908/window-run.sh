#!/usr/bin/env bash
# Parity-baseline hardware window on jwm1-linux.
# ONE flock on /tmp/m1-gpu.lock for the whole process tree; the lock file
# is never created/deleted here, only locked. Prints hostname first.
set -euo pipefail
exec 9>/tmp/m1-gpu.lock
flock -w 900 9 || { echo "window: could not acquire /tmp/m1-gpu.lock" >&2; exit 9; }

echo "hostname: $(hostname)"
ROOT="$HOME/src/mlx-parity-baseline-20260908"
cd "$ROOT"
REC="receipts/parity-baseline-20260908"
WHEEL_RELEASE="$ROOT/dist-parity-release/mlx_omarchy-0.32.2.dev202609081618+c254867-cp314-cp314-linux_aarch64.whl"
WHEEL_DIAG="$ROOT/dist-parity-diag/mlx_omarchy-0.32.2.dev202609082046+diag.c254867-cp314-cp314-linux_aarch64.whl"
PYB="$ROOT/.venv-benchmark/bin/python"
PYP="$ROOT/.venv-profile/bin/python"
WHEEL_DIAG="$ROOT/dist-parity-diag/mlx_omarchy-0.32.2.dev202609081624+diag.c254867-cp314-cp314-linux_aarch64.whl"

echo "== window start: $(date -u +%FT%TZ) =="
uptime

echo "== [1/6] driver + metadata receipt =="
vulkaninfo --summary > "$REC/vulkaninfo-summary.txt" 2>&1
grep -E "deviceName|driverName|driverInfo|driverID|apiVersion" "$REC/vulkaninfo-summary.txt" | sed -n '1,6p'
HF_HUB_OFFLINE=1 "$PYB" scripts/bench_matrix.py --mode metadata \
  --python "$PYB" --host-label jwm1-linux-parity-baseline \
  --out "$REC/metadata.json" >/dev/null
python3 - "$REC/metadata.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("host:", d["host"].get("gpu"), d["host"].get("os"), d["host"].get("kernel"))
print("power:", d["power"])
print("clean:", d["clean_check"]["status"])
print("packages:", {k: v for k, v in d["packages"].items() if k.startswith("mlx") or k in ("numpy", "transformers")})
EOF

echo "== [2/6] baseline 5x3x2 =="
HF_HUB_OFFLINE=1 "$PYB" "$REC/run-baseline.py" "$PYB" "$WHEEL_RELEASE" 5

echo "== [3/6] profiling legs (diag wheel) =="
HF_HUB_OFFLINE=1 "$PYP" "$REC/run-profile.py" "$PYP" "$WHEEL_DIAG" "$REC/profile"

echo "== [4/6] CPU oracle legs =="
for leg in long short ctx1024; do
  tokens=128; [ "$leg" = short ] && tokens=32
  HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
  MLX_PAIR_IDS="$REC/oracle-cpu-$leg.ids.jsonl" \
  MLX_OMARCHY_ALLOW_NON_APPLE=0 \
  "$PYB" "$REC/cpu-oracle.py" --model "$SNAP" --prompt-id "$leg" \
    --tokens "$tokens" > "$REC/oracle-cpu-$leg.out" 2> "$REC/oracle-cpu-$leg.err" \
    || { echo "cpu-oracle $leg FAILED"; cat "$REC/oracle-cpu-$leg.err"; exit 7; }
  grep -E "generated_ids sha256|provenance" "$REC/oracle-cpu-$leg.out" | sed -n '1,2p'
done

echo "== [5/6] margin probe (GPU, long128) =="
HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
  "$PYB" "$REC/margin-probe.py" "$SNAP" --prompt-id long --tokens 128 \
  > "$REC/margins-long128.json"

echo "== [6/6] window summary =="
python3 - "$REC" <<'EOF'
import hashlib, json, sys
from pathlib import Path
rec = Path(sys.argv[1])
s = json.loads((rec / "baseline-summary.json").read_text())
print("wheel:", s["wheel_sha256"][:16])
for leg in s["legs"]:
    print(f'{leg["leg_id"]}: decode {leg["decode_tok_s_median"]} '
          f'prefill {leg["prefill_tok_s_median"]} '
          f'digest {leg["generated_ids_sha256_16"]} '
          f'stable={leg["ids_stable_across_reps"]}')
for f in sorted(rec.glob("oracle-cpu-*.out")):
    for line in f.read_text().splitlines():
        if line.startswith("generated_ids"):
            print(f.name, line)
m = json.loads((rec / "margins-long128.json").read_text())
print("margin-probe digest:", m["ids_digest_16"])
print("min margin step:", json.dumps(m["min_margin_step"]))
EOF
echo "== window end: $(date -u +%FT%TZ) =="

#!/usr/bin/env bash
# Continuation of the parity-baseline window after the SNAP regression:
# steps 4-6 only (CPU oracle legs, margin probe, summary). Baseline 5x3x2
# and the three profiling legs completed in window.log and are untouched.
set -euo pipefail
exec 9>/tmp/m1-gpu.lock
flock -w 900 9 || { echo "window2: could not acquire /tmp/m1-gpu.lock" >&2; exit 9; }

echo "hostname: $(hostname)"
ROOT="$HOME/src/mlx-parity-baseline-20260908"
cd "$ROOT"
REC="receipts/parity-baseline-20260908"
PYB="$ROOT/.venv-benchmark/bin/python"
SNAP="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"

echo "== window2 start: $(date -u +%FT%TZ) =="
uptime

echo "== [4/6] CPU oracle legs =="
for leg in long short ctx1024; do
  tokens=128; [ "$leg" = short ] && tokens=32
  HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
  MLX_PAIR_IDS="$REC/oracle-cpu-$leg.ids.jsonl" \
  "$PYB" "$REC/cpu-oracle.py" --model "$SNAP" --prompt-id "$leg" \
    --tokens "$tokens" > "$REC/oracle-cpu-$leg.out" 2> "$REC/oracle-cpu-$leg.err" \
    || { echo "cpu-oracle $leg FAILED"; cat "$REC/oracle-cpu-$leg.err"; exit 7; }
  grep -E "generated_ids sha256" "$REC/oracle-cpu-$leg.out" | sed -n '1p'
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
echo "== window2 end: $(date -u +%FT%TZ) =="

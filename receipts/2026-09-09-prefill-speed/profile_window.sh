#!/usr/bin/env bash
# Prefill-only GPU profile of one wheel on the M1 (run under the GPU lock).
# Usage: profile_window.sh <checkout> <wheel> <label>
set -euo pipefail
ROOT="$1"; WHEEL="$2"; LABEL="$3"
OUT="$ROOT/receipts/2026-09-09-prefill-speed/profiles/$LABEL"
mkdir -p "$OUT"
cd "$ROOT"
hostname; date -u +%FT%TZ
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3)
PY=.venv-accept/bin/python
$PY -m pip install -q --no-deps --force-reinstall "$WHEEL"
sha256sum "$WHEEL" | tee "$OUT/wheel.sha256"
$PY scripts/mlx_provenance.py --wheel "$WHEEL" 2>&1 | tee "$OUT/provenance.txt" || true
python3 - "$ROOT/scripts/bench_matrix.json" "$OUT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
p = d["prompts"]
open(sys.argv[2] + "/prompt-long.txt", "w").write(p["long"]["text"])
e = p["ctx1024"]
parts = [e["base"]] + [f"{e['item']} Entry {i} of {e['items']}." for i in range(1, e["items"] + 1)]
open(sys.argv[2] + "/prompt-ctx1024.txt", "w").write(" ".join(parts))
EOF
for leg in long ctx1024; do
  prompt=$(cat "$OUT/prompt-$leg.txt")
  # warm run (shader cache, model load) then the recorded run
  MLX_DISABLE_COMPILE=1 timeout 600 $PY scripts/profile_generate.py --model "$MODEL" \
      --prompt "$prompt" --max-tokens 4 --temp 0 --seed 0 --markers /tmp/warm-markers.jsonl > /dev/null 2>&1 || true
  MLX_DISABLE_COMPILE=1 MLX_OMARCHY_GPU_PROFILE="$OUT/$leg.profile.jsonl" timeout 600 $PY \
      scripts/profile_generate.py --model "$MODEL" --prompt "$prompt" --max-tokens 4 \
      --temp 0 --seed 0 --markers "$OUT/$leg.markers.jsonl" 2>&1 | tail -3 | tee "$OUT/$leg.run.log"
  python3 scripts/profile_analyze.py "$OUT/$leg.profile.jsonl" --markers "$OUT/$leg.markers.jsonl" \
      --compute-h overlay/mlx/backend/omarchy/compute.h > "$OUT/$leg.analysis.txt" 2>&1 || true
  python3 receipts/2026-09-09-prefill-speed/slot_profile.py "$OUT/$leg.profile.jsonl" \
      --markers "$OUT/$leg.markers.jsonl" --compute-h overlay/mlx/backend/omarchy/compute.h \
      --json "$OUT/$leg.slot.json" | tee "$OUT/$leg.slot.txt"
done
date -u +%FT%TZ

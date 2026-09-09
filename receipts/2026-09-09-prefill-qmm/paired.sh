#!/usr/bin/env bash
# Usage: paired.sh ROOT BASE_WHEEL CAND_WHEEL PAIRS
set -euo pipefail
ROOT="$1"; BASE_WHEEL="$2"; CAND_WHEEL="$3"; PAIRS="$4"
OUT="$ROOT/receipts/2026-09-09-prefill-qmm/paired"
BASE_PY="$HOME/src/mlx-PrefillSpeed/.venv-base/bin/python"
CAND_PY="$HOME/src/mlx-PrefillSpeed/.venv-cand/bin/python"
mkdir -p "$OUT"
unset MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH
sha256sum "$BASE_WHEEL" "$CAND_WHEEL" | tee "$OUT/wheels.sha256"
"$BASE_PY" -m pip install -q --no-deps --force-reinstall "$BASE_WHEEL"
"$CAND_PY" -m pip install -q --no-deps --force-reinstall "$CAND_WHEEL"
base_stamp="$("$BASE_PY" -c 'import importlib.metadata as m; print(m.version("mlx-omarchy"))')"
cand_stamp="$("$CAND_PY" -c 'import importlib.metadata as m; print(m.version("mlx-omarchy"))')"
echo "base $base_stamp cand $cand_stamp" | tee "$OUT/stamps.txt"
test "$base_stamp" != "$cand_stamp"
for ((i = 1; i <= PAIRS; i++)); do
  for side in base cand; do
    py="$BASE_PY"; wheel="$BASE_WHEEL"
    if [[ "$side" == cand ]]; then py="$CAND_PY"; wheel="$CAND_WHEEL"; fi
    echo "== pair $i $side $(date -u +%FT%TZ)"
    timeout 1500 python3 "$ROOT/scripts/bench_matrix.py" --mode run --python "$py" \
      --wheel "$wheel" --host-label "jwm1-PrefillQmm-paired-$side" --timeout 600 \
      --out "$OUT/pair-$i-$side.json" > "$OUT/pair-$i-$side.log" 2>&1
  done
done
python3 - "$OUT" "$PAIRS" <<'PY'
import json, statistics, sys
out, pairs = sys.argv[1], int(sys.argv[2])
legs = {}
for i in range(1, pairs + 1):
    for side in ("base", "cand"):
        data = json.load(open(f"{out}/pair-{i}-{side}.json"))
        for leg in data["legs"]:
            if not leg.get("measured"):
                continue
            metrics = leg["metrics"]
            entry = legs.setdefault(leg["leg_id"], {
                "base": [], "cand": [], "ids": {"base": set(), "cand": set()}})
            entry[side].append((metrics["prefill_tok_s"], metrics["decode_tok_s"]))
            entry["ids"][side].add(metrics["generated_ids_sha256_16"])
summary = {}
for leg, entry in legs.items():
    bp = statistics.median(x[0] for x in entry["base"])
    cp = statistics.median(x[0] for x in entry["cand"])
    bd = statistics.median(x[1] for x in entry["base"])
    cd = statistics.median(x[1] for x in entry["cand"])
    summary[leg] = {
        "baseline_prefill": bp, "candidate_prefill": cp,
        "baseline_decode": bd, "candidate_decode": cd,
        "prefill_ratio": cp / bp, "decode_ratio": cd / bd,
        "base_ids": sorted(entry["ids"]["base"]),
        "cand_ids": sorted(entry["ids"]["cand"]),
        "samples": {"base": entry["base"], "cand": entry["cand"]},
    }
    print(f"{leg:44s} prefill {bp:8.2f} -> {cp:8.2f} ({cp/bp:.3f})  "
          f"decode {bd:7.2f} -> {cd:7.2f} ({cd/bd:.3f})")
json.dump(summary, open(f"{out}/paired-summary.json", "w"), indent=1)
PY

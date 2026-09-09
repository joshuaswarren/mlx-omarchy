#!/usr/bin/env bash
# Paired alternating bench_matrix runs of a baseline and a candidate
# commit on the M1 (run under the GPU lock).
# Usage: paired.sh <base_commit> <cand_commit> <pairs>
set -euo pipefail
BASE="$1"; CAND="$2"; PAIRS="$3"
ROOT="$HOME/src/mlx-PrefillSpeed"
cd "$ROOT"
hostname; date -u +%FT%TZ
git fetch -q origin wave/PrefillSpeed
unset MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH
WHEELS="$HOME/prefillspeed-wheels"
mkdir -p "$WHEELS"
build() {
  local commit="$1"
  git checkout -q "$commit"
  local short; short="$(git rev-parse --short=7 HEAD)"
  local existing=("$WHEELS"/mlx_omarchy-*+"$short"-*.whl)
  if [[ -f "${existing[0]}" ]]; then echo "${existing[0]}"; return; fi
  rm -rf dist
  DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 timeout 3600 scripts/build-wheel.sh > "$WHEELS/build-$short.log" 2>&1
  local w=(dist/mlx_omarchy-*.whl); test "${#w[@]}" -eq 1
  cp "${w[0]}" "$WHEELS/"
  echo "$WHEELS/$(basename "${w[0]}")"
}
BASE_WHEEL="$(build "$BASE")"
CAND_WHEEL="$(build "$CAND")"
git checkout -q "$CAND"
OUT="receipts/2026-09-09-prefill-speed/paired-$(git rev-parse --short=7 "$BASE")-vs-$(git rev-parse --short=7 "$CAND")"
mkdir -p "$OUT"
sha256sum "$BASE_WHEEL" "$CAND_WHEEL" | tee "$OUT/wheels.sha256"
for side in base cand; do
  if [[ ! -x ".venv-$side/bin/python" ]]; then
    python3 -m venv ".venv-$side"
    ".venv-$side/bin/python" -m pip install -q mlx-lm==0.31.3
  fi
done
.venv-base/bin/python -m pip install -q --no-deps --force-reinstall "$BASE_WHEEL"
.venv-cand/bin/python -m pip install -q --no-deps --force-reinstall "$CAND_WHEEL"
base_stamp="$(.venv-base/bin/python -c 'import importlib.metadata as m; print(m.version("mlx-omarchy"))')"
cand_stamp="$(.venv-cand/bin/python -c 'import importlib.metadata as m; print(m.version("mlx-omarchy"))')"
echo "base $base_stamp cand $cand_stamp" | tee "$OUT/stamps.txt"
test "$base_stamp" != "$cand_stamp"
for ((i = 1; i <= PAIRS; i++)); do
  for side in base cand; do
    wheel="$BASE_WHEEL"; [[ "$side" == cand ]] && wheel="$CAND_WHEEL"
    echo "== pair $i $side $(date -u +%FT%TZ)"
    timeout 1500 python3 scripts/bench_matrix.py --mode run --python ".venv-$side/bin/python" \
      --wheel "$wheel" --host-label "jwm1-PrefillSpeed-paired-$side" --timeout 600 \
      --out "$OUT/pair-$i-$side.json" > "$OUT/pair-$i-$side.log" 2>&1 || true
  done
done
python3 - "$OUT" "$PAIRS" <<'EOF'
import json, statistics, sys
out, pairs = sys.argv[1], int(sys.argv[2])
legs = {}
for i in range(1, pairs + 1):
    for side in ("base", "cand"):
        d = json.load(open(f"{out}/pair-{i}-{side}.json"))
        for leg in d["legs"]:
            if not leg.get("measured"):
                continue
            m = leg["metrics"]
            e = legs.setdefault(leg["leg_id"], {"base": [], "cand": [], "ids": {"base": set(), "cand": set()}})
            e[side].append((m["prefill_tok_s"], m["decode_tok_s"]))
            e["ids"][side].add(m["generated_ids_sha256_16"])
summary = {}
for leg, e in legs.items():
    bp = statistics.median(x[0] for x in e["base"]); cp = statistics.median(x[0] for x in e["cand"])
    bd = statistics.median(x[1] for x in e["base"]); cd = statistics.median(x[1] for x in e["cand"])
    summary[leg] = {"baseline_prefill": bp, "candidate_prefill": cp, "baseline_decode": bd,
                    "candidate_decode": cd, "prefill_ratio": cp / bp, "decode_ratio": cd / bd,
                    "base_ids": sorted(e["ids"]["base"]), "cand_ids": sorted(e["ids"]["cand"]),
                    "samples": {"base": e["base"], "cand": e["cand"]}}
    print(f"{leg:44s} prefill {bp:8.2f} -> {cp:8.2f} ({cp/bp:.3f})  decode {bd:7.2f} -> {cd:7.2f} ({cd/bd:.3f})  ids base {sorted(e['ids']['base'])} cand {sorted(e['ids']['cand'])}")
json.dump(summary, open(f"{out}/paired-summary.json", "w"), indent=1)
EOF
date -u +%FT%TZ

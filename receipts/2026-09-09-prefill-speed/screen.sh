#!/usr/bin/env bash
# Screen one commit on the M1: diagnostics profile of both prefill legs
# plus a one-rep bench_matrix run (ids + tok/s) on the release wheel.
# Usage: screen.sh <commit> [profile|bench|both]   (run under the GPU lock)
set -euo pipefail
COMMIT="$1"; MODE="${2:-both}"
ROOT="$HOME/src/mlx-PrefillSpeed"
cd "$ROOT"
git fetch -q origin wave/PrefillSpeed
git checkout -q "$COMMIT"
test "$(git rev-parse --short=7 HEAD)" = "$(git rev-parse --short=7 "$COMMIT")"
SHORT="$(git rev-parse --short=7 HEAD)"
OUT="receipts/2026-09-09-prefill-speed/screens/$SHORT"
mkdir -p "$OUT"
hostname; date -u +%FT%TZ; echo "commit $SHORT"
unset MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH
rm -rf dist
if [[ "$MODE" == "profile" || "$MODE" == "both" ]]; then
  DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 timeout 3600 scripts/build-wheel.sh --diagnostics > "$OUT/build-diag.log" 2>&1
  diag=(dist/mlx_omarchy-*+diag.*.whl); test "${#diag[@]}" -eq 1
  bash receipts/2026-09-09-prefill-speed/profile_window.sh "$ROOT" "${diag[0]}" "$SHORT" > "$OUT/profile.log" 2>&1
  grep -A3 "== prefill" "$OUT/../../profiles/$SHORT/long.slot.txt" "$OUT/../../profiles/$SHORT/ctx1024.slot.txt"
  rm -f "${diag[0]}"
fi
if [[ "$MODE" == "bench" || "$MODE" == "both" ]]; then
  DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 timeout 3600 scripts/build-wheel.sh > "$OUT/build-release.log" 2>&1
  rel=(dist/mlx_omarchy-*.whl); test "${#rel[@]}" -eq 1
  sha256sum "${rel[0]}" | tee "$OUT/wheel.sha256"
  .venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${rel[0]}"
  timeout 1500 python3 scripts/bench_matrix.py --mode run --python .venv-accept/bin/python \
      --wheel "${rel[0]}" --host-label "jwm1-PrefillSpeed-$SHORT" --timeout 600 \
      --out "$OUT/matrix.json" > "$OUT/matrix.log" 2>&1 || true
  python3 - "$OUT/matrix.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for leg in d["legs"]:
    m = leg.get("metrics") or {}
    if leg.get("measured"):
        print(f"{leg['leg_id']:44s} prefill {m['prefill_tok_s']:8.2f} decode {m['decode_tok_s']:7.2f} ids {m['generated_ids_sha256_16']}")
    else:
        print(f"{leg['leg_id']:44s} {leg.get('status')}")
EOF
fi
date -u +%FT%TZ

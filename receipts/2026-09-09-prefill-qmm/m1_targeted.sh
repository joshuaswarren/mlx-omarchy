#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/src/mlx-PrefillQmm}"
OUT="$ROOT/receipts/2026-09-09-prefill-qmm/m1"
BIN="$ROOT/.work/build-tests/tests/omarchy/omarchy_matmul_family_tests"
mkdir -p "$OUT"
timeout 1500 "$BIN" --test-case="qmm coopmat*,qmm packed-word*,qmm tile*,qmm_vec*,register-blocked*,dense*" \
  > "$OUT/omarchy_matmul_family_tests.log" 2>&1

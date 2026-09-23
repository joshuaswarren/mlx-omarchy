#!/usr/bin/env bash
# Apply the vendored mlx-lm serve patches to a venv's mlx_lm package.
#
# GDN fast route: ON by default. Routes gated-delta updates to
# mx.fast.gated_delta_update (provided by the mlx-omarchy wheel); falls
# back to the upstream kernel when the entry point is absent.
# GDN raw route: ON by default. Decode steps (T == 1) additionally route
# to mx.fast.gated_delta_update_raw when the wheel ships it, folding the
# gate chain into the fused kernel prologue; falls back to the composed
# fast route when the entry point is absent.
# Conv-ring: OFF by default (decode-only experimental optimization);
# set MLX_OMARCHY_CONV_RING=1 to enable.
#
# Idempotent: an already-applied patch is reported and skipped.
set -euo pipefail

VENV="${1:?usage: apply-mlx-lm-patches.sh /path/to/venv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="$(dirname "$(ls -d "$VENV"/lib/python3.*/site-packages/mlx_lm 2>/dev/null | head -n 1 || true)")"
[[ -d "$SITE/mlx_lm" ]] || { echo "mlx_lm not found under $VENV" >&2; exit 3; }

apply() {
  local name="$1"
  if patch --dry-run --directory="$SITE" --strip=1 --forward --fuzz=0 \
      < "$ROOT/patches/$name" >/dev/null 2>&1; then
    patch --directory="$SITE" --strip=1 --forward --fuzz=0 \
      < "$ROOT/patches/$name"
    echo "applied: $name"
  elif patch --dry-run --directory="$SITE" --strip=1 --reverse \
      < "$ROOT/patches/$name" >/dev/null 2>&1; then
    echo "already applied: $name"
  else
    echo "patch does not apply (mlx-lm version mismatch?): $name" >&2
    return 1
  fi
}

apply mlx-lm-gated-delta-fast-route.patch
apply mlx-lm-gated-delta-raw.patch
if [[ "${MLX_OMARCHY_CONV_RING:-0}" == 1 ]]; then
  apply mlx-lm-convring.patch
else
  echo "conv-ring: OFF (set MLX_OMARCHY_CONV_RING=1 to enable)"
fi

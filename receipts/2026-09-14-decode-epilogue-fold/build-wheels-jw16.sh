#!/usr/bin/env bash
# Build the two A/B wheels on jw16 (10 cores) for installation on both
# laptops: base = main-land2 at the commit the fold branched from,
# cand = wave/DecodeEpilogueFold. Each side has its own checkout, its
# own .work (no shared cmake cache), and its own dist. The two jw16
# deviations from scripts/build-wheel.sh follow
# receipts/2026-09-14-jw16-max-gpu-attribution.md: Arch's cblas.h lives
# under /usr/include/openblas, and FetchContent is served from a local
# seed. Neither reaches overlay/mlx/backend/omarchy/.
set -euo pipefail
ROOT=/var/tmp/DecodeEpilogueFold
SEED=/var/tmp/qmm-tilem/.deps-seed
TARBALL=/var/tmp/qmm-tilem/.work/mlx-0.32.2-1f8e74e3f12f31365464a6867c6579f0e9b29d85.tar.gz
JOBS="${JOBS:-8}"

build_side() {
  local side="$1"
  local tree="$ROOT/$side"
  local work="$tree/.work"
  local dist="$ROOT/dist-$side"
  local full
  full="$(git -C "$tree" rev-parse HEAD)"
  echo "== $side identity =="
  git -C "$tree" log --oneline -1
  git -C "$tree" status --short --untracked-files=no
  if compgen -G "$dist/mlx_omarchy-*+${full:0:7}*.whl" >/dev/null; then
    echo "[receipt] $side wheel already present, skipping rebuild"
    sha256sum "$dist"/mlx_omarchy-*.whl
    return
  fi
  rm -rf "$work" "$dist"
  mkdir -p "$work" "$dist"
  cp "$TARBALL" "$work/"
  export MLX_OMARCHY_WORK_DIR="$work"
  "$tree/scripts/prepare-mlx.sh" >/dev/null
  local venv="$work/venv-build"
  python3 -m venv --system-site-packages "$venv"
  local py="$venv/bin/python"
  export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas -DFETCHCONTENT_SOURCE_DIR_JSON=$SEED/json-src -DFETCHCONTENT_SOURCE_DIR_GGUFLIB=$SEED/gguflib-src -DFETCHCONTENT_SOURCE_DIR_FMT=$SEED/fmt-src -DFETCHCONTENT_SOURCE_DIR_NANOBIND=$SEED/nanobind-src"
  export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
  export PATH="$venv/bin:$PATH"
  echo "== $side wheel =="
  "$py" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$dist" "$work/mlx" 2>&1 | tail -3
  local w
  w="$(echo "$dist"/mlx_omarchy-*.whl)"
  echo "[receipt] $side wheel: $w"
  sha256sum "$w"
  local stamp
  stamp="$(basename "$w")"
  stamp="${stamp#*+}"
  stamp="${stamp%%-*}"
  if [[ "$full" != "$stamp"* ]]; then
    echo "ERROR: $side wheel stamp +$stamp is not a prefix of $full" >&2
    exit 2
  fi
  echo "[receipt] $side stamp ok: +$stamp"
}

build_side base
build_side cand

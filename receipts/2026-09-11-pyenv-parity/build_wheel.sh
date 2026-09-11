#!/usr/bin/env bash
# PythonHostParity: build the mlx-omarchy wheel (clone tip) for one interpreter.
# usage: build_wheel.sh <python-bin> <workdir-suffix> [extra CMAKE_ARGS...]
# Work dir is version-suffixed (.work-<tag>-<majmin>) so each interpreter gets
# its own scratch tree and a stale venv can never be reused across ABIs.
set -euo pipefail
ROOT="$HOME/src/mlx-HostPathOverhead"
PYBIN="$1"; TAG="$2"; shift 2
EXTRA_CMAKE="${*:-}"

renice -n 19 -p $$ >/dev/null || true
MAJMIN="$("$PYBIN" -c "import sys; print(f'{sys.version_info[0]}{sys.version_info[1]}')")"
export MLX_OMARCHY_WORK_DIR="$ROOT/.work-$TAG-$MAJMIN"
mkdir -p "$MLX_OMARCHY_WORK_DIR"
if [ -f "$ROOT/.work/mlx-"*.tar.gz ]; then
  cp -n "$ROOT"/.work/mlx-*.tar.gz "$MLX_OMARCHY_WORK_DIR"/ 2>/dev/null || true
fi

export PATH="$(dirname "$PYBIN"):$PATH"
# Guard: python3 on PATH must be the requested interpreter, since
# build-wheel.sh creates the build venv with plain `python3`.
want="$("$PYBIN" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")"
got="$(python3 -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || echo none)"
if [ "$want" != "$got" ]; then
  echo "[build_wheel] FATAL: python3 on PATH is $got, requested $want" >&2
  exit 2
fi
# Guard: any existing build venv in the work dir must match the interpreter.
VB="$MLX_OMARCHY_WORK_DIR/venv-build"
if [ -x "$VB/bin/python" ]; then
  vver="$("$VB/bin/python" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || echo none)"
  if [ "$vver" != "$want" ]; then
    echo "[build_wheel] FATAL: stale $VB is $vver, requested $want" >&2
    exit 2
  fi
fi

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF $EXTRA_CMAKE"
unset MLX_OMARCHY_LOCAL_VERSION || true

echo "[build_wheel] tag=$TAG-$MAJMIN python=$($PYBIN -V 2>&1) extra='$EXTRA_CMAKE' start $(date -u +%FT%TZ) pid $$"
DEV_RELEASE=1 "$ROOT/scripts/build-wheel.sh"
echo "[build_wheel] tag=$TAG-$MAJMIN done $(date -u +%FT%TZ)"

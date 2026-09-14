#!/usr/bin/env bash
# Build the fat-shapes candidate wheel on this host from the pushed branch.
# Usage: build-fat-wheel.sh <src-repo-path> <root> [profiling]
# profiling=ON adds -DMLX_OMARCHY_GPU_PROFILING=ON (default OFF: clean rates).
set -euo pipefail
SRC=${1:?src repo}
ROOT=${2:?work root}
PROF=${3:-OFF}
BRANCH=agent/qmm-prefill-fat-shapes

git -C "$SRC" fetch origin "$BRANCH"
COMMIT=$(git -C "$SRC" rev-parse --verify origin/"$BRANCH")
echo "measured commit: $COMMIT"
if [[ -e "$ROOT/tree/.git" ]]; then
  git -C "$ROOT/tree" checkout --detach "$COMMIT"
else
  git -C "$SRC" worktree add --detach "$ROOT/tree" "$COMMIT"
fi

export MLX_OMARCHY_WORK_DIR="$ROOT/.work"
export MLX_OMARCHY_LOCAL_VERSION="fat.$COMMIT"
export MLX_OMARCHY_SOURCE_COMMIT="$COMMIT"
VENV="$MLX_OMARCHY_WORK_DIR/venv-build"
echo "== prepare =="
"$ROOT/tree/scripts/prepare-mlx.sh"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi
py="$VENV/bin/python"
"$py" -m pip install -q 'setuptools>=80' 2>/dev/null || true
"$py" - <<'EOF'
import setuptools
assert int(setuptools.__version__.split(".")[0]) >= 80
print("build deps ok", setuptools.__version__)
EOF

# seed third-party sources from the 2026-09-14 attribution build if present
SEED=${SEED:-/var/tmp/mlx-omarchy-prof-b41e2b74/.deps-seed}
FC=""
if [[ -d "$SEED" ]]; then
  FC="-DFETCHCONTENT_SOURCE_DIR_JSON=$SEED/json-src -DFETCHCONTENT_SOURCE_DIR_GGUFLIB=$SEED/gguflib-src -DFETCHCONTENT_SOURCE_DIR_FMT=$SEED/fmt-src -DFETCHCONTENT_SOURCE_DIR_NANOBIND=$SEED/nanobind-src"
fi

rm -rf "$ROOT/dist"; mkdir -p "$ROOT/dist"
export DEV_RELEASE=1
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DMLX_OMARCHY_GPU_PROFILING=$PROF -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas $FC"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export PATH="$VENV/bin:$PATH"
echo "== wheel =="
"$py" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$ROOT/dist" "$MLX_OMARCHY_WORK_DIR/mlx"
for w in "$ROOT"/dist/mlx_omarchy-*.whl; do
  echo "[receipt] wheel: $w"; sha256sum "$w"
done

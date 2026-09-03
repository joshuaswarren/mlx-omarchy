#!/usr/bin/env bash
# Build the mlx-omarchy wheel end to end.
# This is the U5 packaging gate: scripts/build-wheel.sh
#
# Prepares the pinned upstream tree with the Omarchy-only backend, builds the
# python bindings, and writes exactly one wheel into dist/ at the repo root.
#
# Options:
#   --diagnostics  Build the dev diagnostics wheel instead of a release
#                  wheel. Compiles in the GPU profiling harness
#                  (-DMLX_OMARCHY_GPU_PROFILING=ON), stamps the version
#                  with a "+diag" local segment, and stages the
#                  mlx-omarchy-info binary next to the wheel. The harness
#                  runs slower and this artifact is not for production.
#                  Without the flag the build is unchanged: profiling
#                  harness compiled OUT, release wheel as before.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${MLX_OMARCHY_WORK_DIR:-$ROOT/.work}"
VENV_DIR="$WORK_DIR/venv-build"
DIST_DIR="$ROOT/dist"

DIAGNOSTICS=0
for arg in "$@"; do
  case "$arg" in
    --diagnostics) DIAGNOSTICS=1 ;;
    *) echo "unknown option: $arg (supported: --diagnostics)" >&2; exit 2 ;;
  esac
done
echo "== prepare upstream tree =="
"$ROOT/scripts/prepare-mlx.sh"

echo "== build venv ($VENV_DIR) =="
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
venv_python="$VENV_DIR/bin/python"

# Offline-first: keep system packages when they satisfy the pyproject build
# requirements (setuptools>=80, typing_extensions, cmake>=3.25); pip install
# only what is missing.
if ! "$venv_python" - <<'EOF'
import setuptools, typing_extensions
assert int(setuptools.__version__.split(".")[0]) >= 80
EOF
then
  "$venv_python" -m pip install 'setuptools>=80' typing_extensions
fi

cmake_version="$(cmake --version 2>/dev/null | sed -n '1s/^cmake version //p' || true)"
if [[ -z "$cmake_version" || "$(printf '%s\n' 3.25 "$cmake_version" | sort -V | head -n1)" != "3.25" ]]; then
  "$venv_python" -m pip install 'cmake>=3.25'
fi

echo "== wheel build =="
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# setup.py appends CMAKE_ARGS to its cmake invocation; it splits the value on
# spaces, so keep each -D flag space separated.
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF"
if [[ $DIAGNOSTICS -eq 1 ]]; then
  # Diagnostics wheel: compile in the env-gated GPU profiling harness and
  # stamp the version so the artifact cannot be mistaken for a release
  # wheel. DEV_RELEASE=1 keeps setup.py from appending its own git-hash
  # local segment; patches/mlx-version-time.patch appends
  # "+$MLX_OMARCHY_LOCAL_VERSION" instead.
  export CMAKE_ARGS="$CMAKE_ARGS -DMLX_OMARCHY_GPU_PROFILING=ON"
  export DEV_RELEASE=1
  export MLX_OMARCHY_LOCAL_VERSION=diag
fi

# Honor an explicit override, else the machine's actual core count. A
# hardcoded 16 on the single-core M1 queues 96 compiler processes on one
# CPU (cmake+ninja+cc1plus fan-out) and thrashes 16 GB into the OOM zone.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
export PATH="$VENV_DIR/bin:$PATH"

"$venv_python" -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$DIST_DIR" "$WORK_DIR/mlx"

shopt -s nullglob
wheels=("$DIST_DIR"/mlx_omarchy-*.whl)
shopt -u nullglob
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "expected exactly one mlx_omarchy wheel in $DIST_DIR, found ${#wheels[@]}" >&2
  exit 1
fi
wheel="${wheels[0]}"
if [[ $DIAGNOSTICS -eq 1 ]]; then
  echo "== diagnostics staging =="
  # The wheel must carry the profiling harness (the getenv literal only
  # exists when MLX_OMARCHY_GPU_PROFILING was ON) and the info tool. The
  # tool binary is staged from the wheel itself so the asset matches the
  # artifact byte for byte.
  python3 - "$wheel" "$DIST_DIR" <<'EOF'
import sys, zipfile
wheel, dist = sys.argv[1], sys.argv[2]
has_profile = False
tool = None
with zipfile.ZipFile(wheel) as zf:
    for name in zf.namelist():
        data = zf.read(name)
        if not has_profile and b"MLX_OMARCHY_GPU_PROFILE" in data:
            has_profile = True
        if name.endswith("bin/mlx-omarchy-info"):
            tool = data
if not has_profile:
    sys.exit("diagnostics wheel was built with the profiling harness compiled OUT")
if tool is None:
    sys.exit("diagnostics wheel does not ship mlx/bin/mlx-omarchy-info")
with open(f"{dist}/mlx-omarchy-info", "wb") as fh:
    fh.write(tool)
EOF
  chmod 755 "$DIST_DIR/mlx-omarchy-info"
  echo "[receipt] profiling harness: compiled IN (MLX_OMARCHY_GPU_PROFILE literal found in wheel)"
  echo "[receipt] staged tool: $DIST_DIR/mlx-omarchy-info"
fi


echo "== receipt =="
echo "[receipt] wheel: $wheel"
echo "[receipt] size: $(stat -c '%s' "$wheel") bytes"
echo "[receipt] sha256: $(sha256sum "$wheel" | cut -d' ' -f1)"

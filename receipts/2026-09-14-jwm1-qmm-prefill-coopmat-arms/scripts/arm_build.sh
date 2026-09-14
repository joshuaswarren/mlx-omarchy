#!/usr/bin/env bash
# arm_build.sh <tag> <x_pitch|base> <w_pitch>
# Incremental candidate build in the 2026-09-13 profile-enabled b41e2b74
# tree. The tree's build directory is keyed to its own absolute path, so
# arms are built in place (no clone, no path rewriting) and each arm gets
# its own wheel dir, local version and venv. "base" restores the pristine
# shipped shader and gate constant.
set -euo pipefail
BASE=/var/tmp/mlx-omarchy-profile-enabled-b41e2b74
SRC=/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74
WORK=/var/tmp/qmm-coop-arms
TAG=${1:?tag}
XP=${2:?x pitch or "base"}
PY="$SRC/.work/venv-build/bin/python"
OMA=$BASE/mlx/mlx/backend/omarchy
mkdir -p "$WORK/dist-$TAG"

if [[ $XP == base ]]; then
  cp "$SRC/overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp" \
     "$OMA/shaders/qmm_coopmat.comp"
  cp "$SRC/overlay/mlx/backend/omarchy/primitives.cpp" "$OMA/primitives.cpp"
  echo "restored pristine shader and primitives.cpp"
else
  WP=${3:?w pitch}
  python3 - "$BASE" "$SRC" "$XP" "$WP" <<'PY'
import sys
from pathlib import Path
base, src, xp, wp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
oma = Path(base) / "mlx" / "mlx" / "backend" / "omarchy"
template = Path("/home/joshuawarren/benchq/qmmpad/qmm_coopmat.padded.comp")
s = template.read_text()
for name, value, ref, width in (("X_PITCH", xp, "STEP_K", 16),
                                ("W_PITCH", wp, "TILE_N", 32)):
    hits = [l for l in s.splitlines() if l.startswith(f"const uint {name}")]
    assert len(hits) == 1, hits
    s = s.replace(hits[0], f"const uint {name} = {ref} + {value - width}u;")
(oma / "shaders" / "qmm_coopmat.comp").write_text(s)
prim = oma / "primitives.cpp"
t = Path(src, "overlay/mlx/backend/omarchy/primitives.cpp").read_text()
old = ("constexpr uint32_t kQmmCoopmatSharedBytes =\n"
       "      (32u * 16u + 16u * 32u) * sizeof(float);")
new = ("constexpr uint32_t kQmmCoopmatSharedBytes =\n"
       f"      (32u * {xp}u + 16u * {wp}u) * sizeof(float);")
assert t.count(old) == 1, t.count(old)
prim.write_text(t.replace(old, new))
print("pitches", xp, wp, "shared bytes", (32 * xp + 16 * wp) * 4)
PY
fi

export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_OMARCHY_GPU_PROFILING=ON"
export CMAKE_BUILD_PARALLEL_LEVEL=4
export DEV_RELEASE=1
export MLX_OMARCHY_LOCAL_VERSION="diag.$TAG"
export MLX_OMARCHY_SOURCE_COMMIT=b41e2b74c330f910b24cab0e7516e306527858f0
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
echo "[$(date -Iseconds)] pip wheel $TAG"
"$PY" -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$WORK/dist-$TAG" "$BASE/mlx" > "$WORK/build-$TAG.log" 2>&1
echo "[$(date -Iseconds)] wheel done"
shopt -s nullglob
wheels=("$WORK/dist-$TAG"/mlx_omarchy-*.whl)
echo "wheel_count=${#wheels[@]}"
for w in "${wheels[@]}"; do
  echo "[receipt] wheel: $w"
  sha256sum "$w"
done

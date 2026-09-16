#!/usr/bin/env bash
# jwm1 qkv arm wheel: agent/rmsnorm-gemv-epilogue tip c3553eb2 (the
# attention-scoped RMSNorm knob commit). Same canonical jwm1 CMAKE flags
# and venv recipe as build-jwm1.sh. CPU-only build; no GPU lock required.
set -euo pipefail
ROOT=/var/tmp/SwigluEpilogue
REPO=/home/joshuawarren/src/mlx-omarchy
TARBALL=$REPO/.work/mlx-0.32.2-1f8e74e3f12f31365464a6867c6579f0e9b29d85.tar.gz
JOBS="${JOBS:-4}"
side=qkv
tree="$ROOT/$side"; work="$tree/.work"; dist="$ROOT/dist-$side"
cd "$REPO"
git worktree add --detach "$tree" c3553eb2 2>/dev/null || true
full="$(git -C "$tree" rev-parse HEAD)"
echo "== $side identity: $(git -C "$tree" log --oneline -1) =="
if [[ "$full" != c3553eb2* ]]; then echo "ERROR: tree at $full, want c3553eb2*" >&2; exit 2; fi
if compgen -G "$dist/mlx_omarchy-*+c3553eb2*.whl" >/dev/null; then
  echo "[receipt] $side wheel already present"
  sha256sum "$dist"/mlx_omarchy-*.whl
else
  if [[ -e "$work" ]]; then echo "ERROR: $work exists; move it aside first" >&2; exit 2; fi
  mkdir -p "$work" "$dist"
  cp "$TARBALL" "$work/"
  MLX_OMARCHY_WORK_DIR="$work" "$tree/scripts/prepare-mlx.sh" >/dev/null
  venv="$work/venv-build"
  python3 -m venv --system-site-packages "$venv"
  if ! "$venv/bin/python" -c "import setuptools, typing_extensions" 2>/dev/null; then
    "$venv/bin/python" -m pip install "setuptools>=80" typing_extensions
  fi
  export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTING=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF"
  export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
  export PATH="$venv/bin:$PATH"
  "$venv/bin/python" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$dist" "$work/mlx" 2>&1 | tail -3
  w="$(echo "$dist"/mlx_omarchy-*.whl)"
  echo "[receipt] $side wheel: $w"
  sha256sum "$w"
  stamp="$(basename "$w")"; stamp="${stamp#*+}"; stamp="${stamp%%-*}"
  if [[ "$full" != "$stamp"* ]]; then echo "ERROR: stamp mismatch $stamp vs $full" >&2; exit 2; fi
  echo "[receipt] $side stamp ok: +$stamp"
fi
venv="$ROOT/venv-$side"
if [[ ! -e "$venv" ]]; then
  python3 -m venv --system-site-packages "$venv"
  "$venv/bin/pip" install -q "mlx-lm==0.31.3" 2>&1 | tail -1 || true
fi
w="$(echo "$dist"/mlx_omarchy-*.whl)"
"$venv/bin/pip" install -q --force-reinstall --no-deps "$w" 2>&1 | tail -1
"$venv/bin/python" -c "import mlx.core, pathlib; print('$side wheel installed:', pathlib.Path(mlx.core.__file__).parents[1])"
echo ALL-DONE

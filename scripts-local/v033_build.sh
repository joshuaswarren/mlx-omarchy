#!/bin/sh
# v0.3.3 aarch64 release wheel build at tag b18704e (documented procedure).
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
git fetch --quiet origin main
git fetch --quiet origin tag v0.3.3 2>/dev/null || git fetch --quiet origin refs/tags/v0.3.3:refs/tags/v0.3.3
git checkout --quiet --detach v0.3.3
echo "building release wheel at $(git rev-parse --short HEAD) (tag v0.3.3)"
OUTDIR="$HOME/benchq/wheels/v033"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.whl
DEV_RELEASE=1 sh scripts/build-wheel.sh 2>&1 | grep -E "source commit|sha256:|wheel:"
cp dist/mlx_omarchy-*.whl "$OUTDIR"/
echo "artifact: $OUTDIR/$(basename "$OUTDIR"/*.whl)"
sha256sum "$OUTDIR"/*.whl

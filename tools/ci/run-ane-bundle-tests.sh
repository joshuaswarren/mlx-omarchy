#!/usr/bin/env bash
# U6 Linux bundle-validation gate: build and run the ANE bundle/manifest test
# binary. tools/ci/run-ane-bundle-tests.sh
#
# These tests run on any Linux host. They touch no device: the loader under
# test is the pre-device validation layer that must reject a malformed bundle
# before Linux maps or submits a descriptor.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${MLX_OMARCHY_BUILD_DIR:-$ROOT/.work/build}"

"$ROOT/scripts/prepare-mlx.sh" >/dev/null

cmake --build "$BUILD_DIR" --target omarchy_ane_bundle_tests -j4

echo "== omarchy_ane_bundle_tests =="
"$BUILD_DIR/tests/omarchy/omarchy_ane_bundle_tests"

echo "[receipt] omarchy_ane_bundle_tests passed on $(uname -m), linux bundle validation gate ok"

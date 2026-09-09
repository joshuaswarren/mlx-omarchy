#!/usr/bin/env bash
set -euo pipefail
root="$HOME/src/mlx-post-drain-profile-770ae465"
base="$HOME/src/mlx-rope-drain-770ae465"
test ! -e "$root"
git clone --quiet --shared "$base" "$root"
cd "$root"
test "$(git rev-parse --short=8 HEAD)" = 770ae465
out="$root/receipts/2026-09-09-post-drain-profile"
mkdir -p "$out"
unset HK_PERF MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GATED_BARRIERS
CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh --diagnostics > "$out/build.log" 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
python3.14 -m venv .venv-profile
.venv-profile/bin/python -m pip install --no-deps -r "$base/receipts/2026-09-08-dense-final/requirements-baseline.txt" "${wheels[0]}" mlx-lm==0.31.3 > "$out/install.log" 2>&1
.venv-profile/bin/python /tmp/post-drain-profile.py "$root"
printf 'POST_DRAIN_PROFILE_DONE\n'

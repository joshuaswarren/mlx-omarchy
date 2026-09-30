#!/usr/bin/env bash
# H139: base multi-row GEMV plus a next-stage weight touch (binding 25).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/shaders/qmm_vec_base.comp"
[ "$(sed -n 574p "$src")" = 'void main() {' ] || { echo "anchor 574 moved" >&2; exit 1; }
sed -e '574s|.*|layout(set = 0, binding = 25, std430) readonly buffer Touch { uint v[]; } touch;\nvoid main() {\n  { uint tw = 0u; uint tb = gl_WorkGroupID.x * 128u + gl_LocalInvocationID.x;\n    for (uint i = 0u; i < 16u; ++i) tw += touch.v[tb + i * 32768u];\n    if (tw == 0x9e3779b9u \&\& gl_WorkGroupID.x == 0u \&\& gl_LocalInvocationID.x == 0u) output0.values[0] = uint16_t(1); }|' "$src" > "$1"

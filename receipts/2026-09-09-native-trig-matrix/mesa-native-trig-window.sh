#!/usr/bin/env bash
set -euo pipefail
root="$HOME/src/mesa-pkg-20260908/src"
source="$root/mesa/src/asahi/compiler/agx_nir_lower_math.c"
restore() {
    git -C "$root/mesa" show 6f6afc896844730f6d6c47f12a91145351cb4c28:src/asahi/compiler/agx_nir_lower_math.c > "$source"
    ninja -C "$root/build" -j4 src/asahi/vulkan/libvulkan_asahi.so > /tmp/mesa-native-trig-restore.log 2>&1
}
test -z "$(git -C "$root/mesa" status --short)"
trap restore EXIT
cp /tmp/mesa-native-trig-control.c "$source"
ninja -C "$root/build" -j4 src/asahi/vulkan/libvulkan_asahi.so > /tmp/mesa-native-trig-build.log 2>&1
cp "$root/build/src/asahi/vulkan/libvulkan_asahi.so" /tmp/mesa-native-trig-libvulkan_asahi.so
VK_DRIVER_FILES=/tmp/mesa-native-trig-icd.json MESA_SHADER_CACHE_DISABLE=true timeout 60 "$HOME/src/mlx-rms-round-screen/.venv-accept/bin/python" /tmp/mlx-rope-basis.py /tmp/mlx-hardware-wrapped-rope-basis

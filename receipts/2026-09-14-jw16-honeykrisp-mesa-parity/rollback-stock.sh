#!/bin/bash
# Roll jw16 back from the Honeykrisp fork package to stock extra/ 1:26.2.2-1.
# `pacman -U` alone cannot do this: --noconfirm answers N to the
# "remove mesa-honeykrisp-omarchy?" conflict prompt, so the fork package is
# removed explicitly first (-Rdd, because everything depends on mesa).
set -u
STOCK=/home/joshuawarren/src/xbuild-drivers-jw16/stock-26.2.2

sudo pacman -Rdd --noconfirm mesa-honeykrisp-omarchy || exit 1
sudo pacman -U --noconfirm \
  "$STOCK/mesa-1:26.2.2-1-aarch64.pkg.tar.xz" \
  "$STOCK/vulkan-asahi-1:26.2.2-1-aarch64.pkg.tar.xz" \
  "$STOCK/vulkan-mesa-implicit-layers-1:26.2.2-1-aarch64.pkg.tar.xz" || exit 1

pacman -Q mesa vulkan-asahi vulkan-mesa-implicit-layers
ls /usr/share/vulkan/icd.d/
sha256sum /usr/lib/libvulkan_asahi.so
vulkaninfo --summary 2>/dev/null | grep -E "driverName|driverInfo" | head -2

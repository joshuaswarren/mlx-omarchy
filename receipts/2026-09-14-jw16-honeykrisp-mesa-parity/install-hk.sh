#!/bin/bash
# Install the Honeykrisp fork mesa package on jw16, with automatic rollback
# to the staged stock 1:26.2.2-1 packages if the install transaction fails.
set -u
PKG=/home/joshuawarren/src/mesa-pkg-jw16-20260914/mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz
STOCK=/home/joshuawarren/src/xbuild-drivers-jw16/stock-26.2.2

rollback() {
  echo "!! install failed, rolling back to stock 1:26.2.2-1"
  sudo pacman -U --noconfirm \
    "$STOCK/mesa-1:26.2.2-1-aarch64.pkg.tar.xz" \
    "$STOCK/vulkan-asahi-1:26.2.2-1-aarch64.pkg.tar.xz" \
    "$STOCK/vulkan-mesa-implicit-layers-1:26.2.2-1-aarch64.pkg.tar.xz"
  echo "rollback rc=$?"
  pacman -Q mesa vulkan-asahi vulkan-mesa-implicit-layers 2>&1
  exit 1
}

echo "== before"
pacman -Q mesa vulkan-asahi vulkan-mesa-implicit-layers mesa-honeykrisp-omarchy 2>&1
ls /usr/share/vulkan/icd.d/

echo "== removing the split stock packages the fork package replaces"
sudo pacman -Rdd --noconfirm mesa vulkan-asahi vulkan-mesa-implicit-layers || rollback

echo "== installing fork package"
sudo pacman -U --noconfirm "$PKG" || rollback

echo "== after"
pacman -Q mesa-honeykrisp-omarchy
pacman -Q mesa vulkan-asahi vulkan-mesa-implicit-layers 2>&1
ls -la /usr/share/vulkan/icd.d/
sha256sum /usr/lib/libvulkan_asahi.so

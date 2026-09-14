#!/usr/bin/env bash
# Rollback verification and state for the agx_insert_waits A/B on jwm1.
#
# No driver was installed by this work, so the rollback is a contingency that
# was staged and verified BEFORE any measurement, not one that was exercised.
# A staged rollback nobody checked is not a rollback, so this proves the
# staged package reproduces the installed driver byte for byte.
set -euo pipefail

PKG=/home/joshuawarren/src/mesa-pkg-20260908/mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz

echo "== installed driver now"
pacman -Q mesa-honeykrisp-omarchy
sha256sum /usr/lib/libvulkan_asahi.so

echo
echo "== staged rollback package"
sha256sum "$PKG"

echo
echo "== proof the package reproduces the installed driver"
T=$(mktemp -d)
tar -xJf "$PKG" -C "$T" usr/lib/libvulkan_asahi.so
sha256sum "$T/usr/lib/libvulkan_asahi.so"
if cmp -s "$T/usr/lib/libvulkan_asahi.so" /usr/lib/libvulkan_asahi.so; then
  echo "VERIFIED: byte-identical"
else
  echo "FAILED: package does not match the installed driver"
  exit 1
fi

echo
echo "== rollback command, if it were ever needed"
echo "   sudo pacman -U --noconfirm $PKG"

echo
echo "== shared ICD state (must be untouched)"
ls -la /usr/share/vulkan/icd.d/
cat /usr/share/vulkan/icd.d/asahi_icd.aarch64.json

echo
echo "== GPU lock state"
ls -lai /tmp/m1-gpu.lock
if fuser /tmp/m1-gpu.lock >/dev/null 2>&1; then
  echo "   held by another process"
else
  echo "   free"
fi

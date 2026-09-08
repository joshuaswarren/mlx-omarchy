#!/usr/bin/env bash
# Fallback for the failed cross deployment: build hk/prefill-parity
# NATIVELY on jwm1-linux and install the private ICD against it.
# CPU-only build; do NOT run while anyone holds timed GPU legs.
# After this, scripts-local/m1_window.sh phase-1 works unchanged
# (the ICD library path below matches what the window script expects).
set -euo pipefail

M1="joshuawarren@100.84.184.102"
SRC=~/src/mesa
WT=~/src/mesa-prefill-native
COMMIT=323b747d6b0

ssh "$M1" 'hostname; cd ~/src
if [ ! -d '"$WT"'/.git ]; then
  git clone -q --shared ~/src/mesa-coopmat '"$WT"' 2>/dev/null || git clone -q ~/src/mesa '"$WT"'
fi
cd '"$WT"'
git fetch -q ~/src/mesa 2>/dev/null || true
git checkout -q '"$COMMIT"' 2>/dev/null || {
  echo "commit '"$COMMIT"' not present on M1 - fetch from mesa-xbuild first:"
  echo "  ssh mesa-xbuild \"cd ~/mesa-prefill-parity && git bundle create /tmp/prefill.bundle 6f6afc89684..'"$COMMIT"'\""
  echo "  scp mesa-xbuild:/tmp/prefill.bundle ~/src/ && git -C '"$WT"' fetch ~/src/prefill.bundle"
  exit 2
}
ninja -C build 2>/dev/null >/dev/null || {
  meson setup build -Dbuildtype=debugoptimized \
    -Dprefix=$HOME/mesa-prefill-native-prefix \
    -Dvulkan-drivers=asahi -Dgallium-drivers=asahi \
    -Dvideo-codecs= -Dllvm=disabled -Ddraw-use-llvm=false \
    -Dmesa-clc=system -Dprecomp-compiler=system
  ninja -C build
}
ls -la build/src/asahi/vulkan/libvulkan_asahi.so
cat > /tmp/asahi_prefill_icd.json <<EOF
{"ICD":{"library_path":"'"$WT"'/build/src/asahi/vulkan/libvulkan_asahi.so","api_version":"1.4.359"}}
EOF
VK_ICD_FILENAMES=/tmp/asahi_prefill_icd.json AGX_SIMDMAT=1 vulkaninfo --summary 2>/dev/null | grep -E "driverName|deviceName|driverInfo" | head -4
echo NATIVE_DRIVER_READY
'

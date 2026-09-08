#!/usr/bin/env bash
# Deploy + verify the inline-fragment prefill candidate on jwm1-linux.
# Run ONLY while holding flock /tmp/m1-gpu.lock on the M1 (CPU-heavy
# build steps included). Private ICD only; the system driver is never
# touched.
set -euo pipefail

WT="$HOME/.config/superpowers/worktrees/mlx-omarchy/parity-prefill"
M1="joshuawarren@100.84.184.102"
DEST="~/src/mlx-prefill-parity-20260908"
SO_LOCAL="/tmp/libvulkan_asahi_prefill.so"

echo "== 1. fetch cross-built driver from mesa-xbuild"
scp -q mesa-xbuild:~/mesa-prefill-parity/build/src/asahi/vulkan/libvulkan_asahi.so "$SO_LOCAL"

echo "== 2. rsync worktree to M1"
rsync -a --delete \
  --exclude .work --exclude .git --exclude receipts --exclude dist \
  "$WT/" "$M1:$DEST/"

echo "== 3. install driver + private ICD"
scp -q "$SO_LOCAL" "$M1:~/src/mesa-prefill-parity-libvulkan_asahi.so"
ssh "$M1" 'hostname
mkdir -p ~/src
cat > /tmp/asahi_prefill_icd.json <<EOF
{
    "ICD": {
        "library_path": "/home/joshuawarren/src/mesa-prefill-parity-libvulkan_asahi.so",
        "api_version": "1.4.359"
    }
}
EOF
chmod +x ~/src/mesa-prefill-parity-libvulkan_asahi.so
VK_ICD_FILENAMES=/tmp/asahi_prefill_icd.json AGX_SIMDMAT=1 vulkaninfo --summary 2>/dev/null | grep -E "driverName|deviceName|driverInfo" | head -4'

echo "== 4. build wheel in the rsynced tree"
ssh "$M1" "cd $DEST && DEV_RELEASE=1 MLX_OMARCHY_SOURCE_COMMIT=\$(git rev-parse --short=8 HEAD 2>/dev/null || echo 82ececc6) scripts/build-wheel.sh 2>&1 | tail -3"

echo "== 5. venv install"
ssh "$M1" "cd $DEST && python3 -m venv --upgrade-deps ~/venv-prefill-inline 2>/dev/null || python3 -m venv ~/venv-prefill-inline
~/venv-prefill-inline/bin/pip install --quiet dist/mlx_omarchy-*.whl mlx-lm
~/venv-prefill-inline/bin/python -c 'import mlx.core as mx; print(mx.__version__)'"

echo "== 6. correctness: coopmat cases, inline hooks ON"
ssh "$M1" "cd $DEST && VK_ICD_FILENAMES=/tmp/asahi_prefill_icd.json AGX_SIMDMAT=1 AGX_QMM_INLINE_A=1 AGX_QMM_INLINE_B=1 MLX_OMARCHY_ALLOW_NON_APPLE=0 .work/build/tests/omarchy/omarchy_matmul_family_tests --tc='*qmm coopmat*' 2>&1 | tail -3"

echo "== 7. correctness: same cases, hooks OFF (staged leg)"
ssh "$M1" "cd $DEST && VK_ICD_FILENAMES=/tmp/asahi_prefill_icd.json AGX_SIMDMAT=1 MLX_OMARCHY_ALLOW_NON_APPLE=0 .work/build/tests/omarchy/omarchy_matmul_family_tests --tc='*qmm coopmat*' 2>&1 | tail -3"

echo "== 8. runtime suite, hooks ON"
ssh "$M1" "cd $DEST && VK_ICD_FILENAMES=/tmp/asahi_prefill_icd.json AGX_SIMDMAT=1 AGX_QMM_INLINE_A=1 AGX_QMM_INLINE_B=1 .work/build/tests/omarchy/omarchy_runtime_tests 2>&1 | tail -3"

echo "deploy+verify done; next: flock /tmp/m1-gpu.lock and run prefill_inline_ab.py"

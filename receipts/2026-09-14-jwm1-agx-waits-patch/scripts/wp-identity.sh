#!/usr/bin/env bash
# Static identity for the agx_insert_waits A/B. No GPU work: every fact here
# is read from files or package metadata, so it is safe to run while another
# agent holds /tmp/m1-gpu.lock.
echo "== host"
hostname; uname -a; date -Is

echo
echo "== system driver (UNCHANGED by this work - no install was performed)"
pacman -Q mesa-honeykrisp-omarchy
sha256sum /usr/lib/libvulkan_asahi.so
echo "system ICD:"; cat /usr/share/vulkan/icd.d/asahi_icd.aarch64.json

echo
echo "== staged rollback package (byte-verified against the installed driver)"
PKG=/home/joshuawarren/src/mesa-pkg-20260908/mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz
ls -la "$PKG"; sha256sum "$PKG"

echo
echo "== A/B arms (locally built, selected per-command by VK_DRIVER_FILES)"
for w in mesa-wt-waitbatch-base mesa-wt-waitbatch; do
  echo "-- ~/src/$w"
  git -C "/home/joshuawarren/src/$w" log --format="   HEAD %H%n   %s" -1
  sha256sum "/home/joshuawarren/src/$w/build/src/asahi/vulkan/libvulkan_asahi.so"
  python3 -c "
import json
o=json.load(open('/home/joshuawarren/src/$w/build/meson-info/intro-buildoptions.json'))
k={'buildtype','optimization','debug','b_ndebug'}
print('   ' + ', '.join('%s=%s'%(e['name'],e['value']) for e in o if e['name'] in k))
"
done

echo
echo "== patched branch history above the packaged base 6f6afc89684"
git -C /home/joshuawarren/src/mesa-wt-waitbatch log --format="   %h %ad %s" --date=short 6f6afc89684..HEAD

echo
echo "== mlx build (identical across both arms - the driver is the only variable)"
VENV=/var/tmp/mlx-omarchy-profile-enabled-b41e2b74/venv
"$VENV/bin/python" -c "
import importlib.metadata as m, sys
print('   python', sys.version.split()[0])
try: print('   mlx-omarchy', m.version('mlx-omarchy'))
except Exception as e: print('   mlx-omarchy version unavailable:', e)
"
find "$VENV" -name "libmlx.so" -print0 | xargs -0 -r sha256sum

echo
echo "== GPU lock"
ls -lai /tmp/m1-gpu.lock
fuser -v /tmp/m1-gpu.lock 2>&1 | head -3 || echo "   fuser: no holder"

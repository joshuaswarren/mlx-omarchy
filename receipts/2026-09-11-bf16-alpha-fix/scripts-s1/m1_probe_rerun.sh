#!/usr/bin/env bash
# Probe re-run after the S1 window: the first probe binary was invalid
# (it linked libmlx.a while the trap build had left the PRE-FIX shader
# embedded in it - rebuild-script ordering bug). This restores libmlx.a
# to the FIXED shader, rebuilds the probe under a new name, and reruns
# the f64 RNE probe under its own flock window.
set -euo pipefail
cd ~/src/mlx-omarchy-alpha-m1
R=receipts/2026-09-11-bf16-alpha-fix
RUN="taskset -c 0,1 nice -n 10"
SHADER=.work/mlx/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 7200s)"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded 7200s"; exit 4; }
echo "$(date -Is) lock acquired"
driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)
echo "driver: $driver_pkg"
[[ $driver_pkg == "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1" ]] || {
  echo "FATAL: fork driver is not the pinned honeykrisp build"; exit 5; }

echo "== rebuild libmlx.a with the fixed shader =="
cmp s1-logs/shader-fixed.comp.bak overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp \
  || { echo "FATAL: repo shader is not the fixed one"; exit 2; }
cmp s1-logs/shader-fixed.comp.bak "$SHADER" \
  || { echo "FATAL: worktree shader is not the fixed one"; exit 2; }
$RUN cmake --build .work/build-m1-alpha -j2 --target omarchy_fast_ops_tests \
  > s1-logs/build-restore-rebased.log 2>&1
tail -2 s1-logs/build-restore-rebased.log

echo "== relink probe against the fixed libmlx.a =="
cd .work/build-m1-alpha
/usr/bin/c++ -DMLX_STATIC -I"$HOME/src/mlx-omarchy-alpha-m1/.work/mlx" \
  -I"$HOME/src/mlx-omarchy-alpha-m1/.work/build-m1-alpha/_deps/doctest-src" \
  -std=gnu++20 "$HOME/src/mlx-omarchy-alpha-m1/$R/matmul_alpha_f64_probe.cpp" \
  -o /tmp/probe-alpha-fixed libmlx.a mlx/io/libgguflib.a -ldl -lpthread
cd ~/src/mlx-omarchy-alpha-m1
p_new=$(sha256sum /tmp/probe-alpha-fixed | cut -d' ' -f1)
p_old=$(sha256sum /tmp/probe-alpha | cut -d' ' -f1)
echo "probe-invalid(trap shader): $p_old"
echo "probe-fixed:                $p_new" | tee s1-logs/probe-fixed.sha256
[[ $p_new != "$p_old" ]] || { echo "FATAL: probe hash unchanged - shader not refreshed"; exit 2; }

echo "== run the f64 RNE probe (fixed kernel) =="
mv "$R/m1-logs/probe.log" "$R/m1-logs/probe-invalid-trapshader.log"
/tmp/probe-alpha-fixed 2>&1 | tee "$R/m1-logs/probe.log" | grep -E "PROBE|Status"
rc=${PIPESTATUS[0]}
echo "probe rc=$rc"
sha256sum /tmp/probe-alpha-fixed /tmp/probe-alpha >> s1-logs/suite-binaries-rebased.sha256
echo "$(date -Is) probe re-run complete rc=$rc (lock released)"
exit "$rc"

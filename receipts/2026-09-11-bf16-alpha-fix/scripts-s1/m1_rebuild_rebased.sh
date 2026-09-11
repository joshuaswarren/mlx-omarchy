#!/usr/bin/env bash
# Rebuild the staged S1 artifacts from the REBASED bf16-alpha-fix tip
# (3da2b3ee; rebase onto origin/main b4271903 changed only receipts/, but
# every artifact must carry the rebased provenance). CPU-only, OUTSIDE the
# GPU lock; cores 0-1 only, niced, per the standing arms-window courtesy.
set -euo pipefail
cd ~/src/mlx-omarchy-alpha-m1
RUN="taskset -c 0,1 nice -n 10"
R=receipts/2026-09-11-bf16-alpha-fix
mkdir -p s1-logs "$R/m1-logs"

echo "== sync tree to rebased tip =="
git fetch origin 2>&1 | tail -1
git checkout -f 3da2b3ee 2>&1 | tail -1
git rev-parse HEAD | tee s1-logs/s1-commit-rebased.txt
case $(git rev-parse HEAD) in 3da2b3ee*) ;; *) echo "FATAL: not on rebased tip"; exit 2;; esac
if git status --porcelain | grep -v '^??'; then echo "FATAL: dirty tree"; exit 2; fi

echo "== wheel (DEV_RELEASE, stamps +3da2b3e) =="
$RUN env DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=2 \
  scripts/build-wheel.sh > s1-logs/build-wheel-rebased.log 2>&1
tail -6 s1-logs/build-wheel-rebased.log
new_wheel=$(ls -t dist/mlx_omarchy-*.whl | head -1)
sha256sum "$new_wheel" | tee s1-logs/alpha-wheel.sha256
realpath "$new_wheel" | tee s1-logs/alpha-wheel-path.txt
case $new_wheel in *+3da2b3e*) ;; *) echo "FATAL: wheel not stamped with rebased tip"; exit 2;; esac

echo "== venv reinstall =="
.venv-alpha/bin/pip install -q --force-reinstall --no-deps \
  "$(cat s1-logs/alpha-wheel-path.txt)"
.venv-alpha/bin/pip list 2>/dev/null | grep -i "mlx-omarchy"

echo "== suite binaries =="
$RUN cmake --build .work/build-m1-alpha -j2 --target \
  omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests \
  > s1-logs/build-suite-rebased.log 2>&1
tail -3 s1-logs/build-suite-rebased.log
cp .work/build-m1-alpha/tests/omarchy/omarchy_matmul_family_tests /tmp/fam-alpha
cp .work/build-m1-alpha/tests/omarchy/omarchy_fast_ops_tests /tmp/fast-alpha
cp .work/build-m1-alpha/tests/omarchy/omarchy_runtime_tests /tmp/rt-alpha

echo "== trap binary: pre-fix shader (ee8d26fb) + relaxed gate =="
SHADER=.work/mlx/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp
cp "$SHADER" s1-logs/shader-fixed.comp.bak
git show ee8d26fb:overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp > "$SHADER"
$RUN cmake --build .work/build-m1-alpha -j2 --target omarchy_fast_ops_tests \
  > s1-logs/build-trap-rebased.log 2>&1
tail -3 s1-logs/build-trap-rebased.log
cp .work/build-m1-alpha/tests/omarchy/omarchy_fast_ops_tests /tmp/fast-trap
cp s1-logs/shader-fixed.comp.bak "$SHADER"
# Rebuild libmlx.a with the RESTORED fixed shader before anything else
# links against it: the trap build leaves the pre-fix shader embedded in
# libmlx.a, and a probe linked at that point would silently measure the
# trap (this exact ordering bug invalidated the first S1 probe run).
$RUN cmake --build .work/build-m1-alpha -j2 --target omarchy_fast_ops_tests \
  > s1-logs/build-restore-rebased.log 2>&1
tail -2 s1-logs/build-restore-rebased.log
cmp s1-logs/shader-fixed.comp.bak "$SHADER" && echo "shader restored OK"
cmp s1-logs/shader-fixed.comp.bak overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp \
  && echo "worktree shader == repo overlay (fixed) OK"

echo "== probe (f64 RNE) =="
cd .work/build-m1-alpha
/usr/bin/c++ -DMLX_STATIC -I"$HOME/src/mlx-omarchy-alpha-m1/.work/mlx" \
  -I"$HOME/src/mlx-omarchy-alpha-m1/.work/build-m1-alpha/_deps/doctest-src" \
  -std=gnu++20 "$HOME/src/mlx-omarchy-alpha-m1/$R/matmul_alpha_f64_probe.cpp" \
  -o /tmp/probe-alpha libmlx.a mlx/io/libgguflib.a -ldl -lpthread \
  > "$HOME/src/mlx-omarchy-alpha-m1/s1-logs/probe-build-rebased.log" 2>&1
cd ~/src/mlx-omarchy-alpha-m1

echo "== artifact receipts =="
sha256sum /tmp/fam-alpha /tmp/fast-alpha /tmp/rt-alpha /tmp/fast-trap /tmp/probe-alpha \
  | tee s1-logs/suite-binaries-rebased.sha256
a=$(sha256sum /tmp/fast-alpha | cut -d' ' -f1)
t=$(sha256sum /tmp/fast-trap | cut -d' ' -f1)
[[ $a != "$t" ]] || { echo "FATAL: trap hash equals fixed hash - no distinct shader"; exit 2; }
echo "trap hash distinct OK"
echo "-- test-case present in both binaries:"
/tmp/fast-alpha --list-test-matching="*scores scale through MatmulBF16Coopmat*" 2>&1 | tail -2
/tmp/fast-trap --list-test-matching="*scores scale through MatmulBF16Coopmat*" 2>&1 | tail -2
echo "REBASE-REBUILD-OK"

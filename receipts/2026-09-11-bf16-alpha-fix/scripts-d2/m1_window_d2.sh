#!/usr/bin/env bash
# D2 window on jwm1-linux (bf16-alpha-fix decision): differential alpha
# instrument + unchanged original-probe rerun + suites + digest gates.
#
# Pre-registered gates (declared here, BEFORE the window):
#   GATE-A (landing blocker): every IDENT digest equal between the fixed
#     and trap instrument binaries, and every dumped .bin byte-identical
#     across builds (alpha==1 fixed-vs-pre-fix bit-identity).
#   GATE-B (fix correct at alpha!=1): fixed fast route vs the
#     storage-modelling f64 oracle: mean_ulp <= 4.0 and max_ulp <= 32.
#   GATE-B2 (pre-fix kernel wrong at alpha!=1): trap fast route vs the
#     same oracle violates those bounds.
#   GATE-C (materially closer): trap fast mean_ulp >= 100 x fixed fast
#     mean_ulp against the storage oracle.
#   GATE-D: digest gates 36/36 (verify_digest_gates.py rc=0).
#   Context (not a gate): fixed fast vs the PLAIN f64 oracle is expected
#     to show the documented storage signature (max around 242) next to
#     the unchanged original probe's published 242.
# The original probe is re-run UNCHANGED (S1 binary, sha-verified) on the
# same fixed build; its bound is not touched.
set -uo pipefail
cd ~/src/mlx-omarchy-alpha-m1
R=receipts/2026-09-11-bf16-alpha-fix
PY=.venv-alpha/bin/python
WHEEL=$(cat s1-logs/alpha-wheel-path.txt)
RUN="taskset -c 0,1 nice -n 10"
SHADER=.work/mlx/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp
PROBE_SHA=30e2db7b9f5f36a212056db655571658fc14057ab21575306a95db806701dedb
FIXED_TIP=1636b378e9aa126fbfb030c4a95ee27f3b408382

link_probe() {  # out path
  cd .work/build-m1-alpha
  /usr/bin/c++ -DMLX_STATIC -I"$HOME/src/mlx-omarchy-alpha-m1/.work/mlx" \
    -I"$HOME/src/mlx-omarchy-alpha-m1/.work/build-m1-alpha/_deps/doctest-src" \
    -std=gnu++20 "$HOME/src/mlx-omarchy-alpha-m1/$R/matmul_alpha_differential_probe.cpp" \
    -o "$1" libmlx.a mlx/io/libgguflib.a -ldl -lpthread
  cd ~/src/mlx-omarchy-alpha-m1
}

echo "== phase 0: CPU build (outside the GPU lock) =="
[[ $(git rev-parse HEAD) == "$FIXED_TIP" ]] || { echo "FATAL: tree not at rebased tip"; exit 2; }
if git status --porcelain | grep -v '^??'; then echo "FATAL: dirty tree"; exit 2; fi
cmp -s s1-logs/shader-fixed.comp.bak "$SHADER" || { echo "FATAL: worktree shader not fixed"; exit 2; }
cmp -s s1-logs/shader-fixed.comp.bak overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp \
  || { echo "FATAL: overlay shader not fixed"; exit 2; }
[[ -x /tmp/probe-alpha-fixed && -x /tmp/fam-alpha && -x /tmp/fast-alpha && -x /tmp/rt-alpha ]] \
  || { echo "FATAL: S1 binaries missing"; exit 2; }
[[ -f $WHEEL ]] || { echo "FATAL: alpha wheel missing"; exit 2; }
grep -q "$PROBE_SHA" s1-logs/probe-fixed.sha256 || { echo "FATAL: probe record mismatch"; exit 2; }

# Refresh libmlx.a with the FIXED shader before anything links against it
# (S1 ordering lesson), then link the fixed instrument.
$RUN cmake --build .work/build-m1-alpha -j2 --target omarchy_fast_ops_tests \
  > s1-logs/build-d2-fixed.log 2>&1 || { echo "FATAL: fixed rebuild"; exit 2; }
link_probe /tmp/diff-alpha-fixed || { echo "FATAL: fixed instrument link"; exit 2; }

# Trap: pre-fix shader (ee8d26fb) + the fix's relaxed gate.
cp "$SHADER" s1-logs/shader-d2-fixed.comp.bak
git show ee8d26fb:overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp > "$SHADER"
$RUN cmake --build .work/build-m1-alpha -j2 --target omarchy_fast_ops_tests \
  > s1-logs/build-d2-trap.log 2>&1 || { echo "FATAL: trap rebuild"; exit 2; }
link_probe /tmp/diff-alpha-trap || { echo "FATAL: trap instrument link"; exit 2; }
cp s1-logs/shader-d2-fixed.comp.bak "$SHADER"
$RUN cmake --build .work/build-m1-alpha -j2 --target omarchy_fast_ops_tests \
  > s1-logs/build-d2-restore.log 2>&1 || { echo "FATAL: restore rebuild"; exit 2; }
cmp -s s1-logs/shader-d2-fixed.comp.bak "$SHADER" || { echo "FATAL: shader restore"; exit 2; }

a=$(sha256sum /tmp/diff-alpha-fixed | cut -d' ' -f1)
t=$(sha256sum /tmp/diff-alpha-trap | cut -d' ' -f1)
[[ $a != "$t" ]] || { echo "FATAL: instrument hashes identical"; exit 2; }
p=$(sha256sum /tmp/probe-alpha-fixed | cut -d' ' -f1)
[[ $p == "$PROBE_SHA" ]] || { echo "FATAL: original probe binary changed"; exit 2; }
echo "diff-alpha-fixed:  $a"
echo "diff-alpha-trap:   $t"
echo "probe-alpha-fixed: $p (unchanged S1 binary)"
sha256sum "$WHEEL"
echo "PHASE0-OK"

# ---------- phase 1: GPU window ----------
mkdir -p "$R/m2-logs"
# Keep the S1 working matrix aside; fresh runs repopulate matrix/ for the
# (unchanged) verifier. The S1 matrix is committed upstream on the branch.
rm -rf "$R/matrix-s1-workdir"
mv "$R/matrix" "$R/matrix-s1-workdir"

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 7200s)"
flock -w 7200 9 || { echo "FATAL: lock wait exceeded 7200s"; exit 4; }
echo "$(date -Is) lock acquired; load: $(uptime)"
driver_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo MISSING)
echo "driver: $driver_pkg"
[[ $driver_pkg == "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1" ]] || {
  echo "FATAL: fork driver is not the pinned honeykrisp build"; exit 5; }

FAILURES=0
{
  echo "== phase 1: original f64 probe, UNCHANGED binary, same fixed build =="
  /tmp/probe-alpha-fixed > "$R/m2-logs/probe-d2.log" 2>&1
  rc=$?
  echo "probe rc=$rc (max-ULP bound was pre-registered and failed in S1; rerun only)"
  grep -E "PROBE|Status" "$R/m2-logs/probe-d2.log" || true

  echo "== phase 2: differential instrument, fixed then trap =="
  mkdir -p /tmp/diffdump-fixed /tmp/diffdump-trap
  rm -f /tmp/diffdump-fixed/*.bin /tmp/diffdump-trap/*.bin
  MLX_ALPHA_DIFF_LABEL=fixed MLX_ALPHA_DIFF_DUMP=/tmp/diffdump-fixed \
    /tmp/diff-alpha-fixed 2>&1 | tee "$R/m2-logs/diff-fixed.log" | grep -E "META|IDENT|DIFF"
  MLX_ALPHA_DIFF_LABEL=trap MLX_ALPHA_DIFF_DUMP=/tmp/diffdump-trap \
    /tmp/diff-alpha-trap 2>&1 | tee "$R/m2-logs/diff-trap.log" | grep -E "META|IDENT|DIFF"

  echo "== phase 3: cross-build bit-identity (GATE-A evidence) =="
  failA=0
  for f in /tmp/diffdump-fixed/*.bin; do
    b=$(basename "$f")
    if cmp -s "$f" "/tmp/diffdump-trap/$b"; then
      echo "IDENTICAL $b"
    else
      echo "DIFFER $b"; failA=1
    fi
  done
  sha256sum /tmp/diffdump-fixed/*.bin | sed 's|/tmp/diffdump-fixed/||'
  if [[ $failA == 0 ]]; then
    echo "GATE-A raw-bytes verdict: PASS"
  else
    echo "GATE-A raw-bytes verdict: FAIL"
    FAILURES=1
  fi

  echo "== phase 4: suites (current tree binaries) =="
  timeout 1800 /tmp/fam-alpha --out="$R/m2-logs/d2-family.log" 2>&1 | tail -2
  src=$?
  echo "family rc=$src"
  [[ $src == 0 ]] || FAILURES=1
  grep -q 'row-mismatch' "$R/m2-logs/d2-family.log" && { echo "FATAL: family row mismatch"; exit 3; }
  timeout 1800 /tmp/fast-alpha --out="$R/m2-logs/d2-fastops.log" 2>&1 | tail -3
  sfs=$?
  echo "fastops rc=$sfs"
  [[ $sfs == 0 ]] || FAILURES=1
  timeout 900 /tmp/rt-alpha --out="$R/m2-logs/d2-runtime.log" 2>&1 | tail -2
  srt=$?
  echo "runtime rc=$srt"
  [[ $srt == 0 ]] || FAILURES=1
  grep -qE 'row-mismatch|FAILED' "$R/m2-logs/d2-runtime.log" && { echo "FATAL: runtime failure"; exit 3; }

  echo "== phase 5: digest gates (S1 wheel, fork + stock x warmup,r1-r3) =="
  run_matrix() {  # driver rep
    local driver=$1 rep=$2
    local run="$R/matrix/${rep}-${driver}-alpha"
    mkdir -p "$run"
    if [[ $driver == stock ]]; then
      env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
        MLX_DISABLE_COMPILE=1 timeout 1800 ~/src/mlx-bf16-prefill-attn/scripts/bench_matrix.py --mode run \
        --python "$PY" --wheel "$WHEEL" \
        --host-label "jwm1-${rep}-${driver}-alpha" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run.log" 2>&1
    else
      env MLX_DISABLE_COMPILE=1 timeout 1800 ~/src/mlx-bf16-prefill-attn/scripts/bench_matrix.py --mode run \
        --python "$PY" --wheel "$WHEEL" \
        --host-label "jwm1-${rep}-${driver}-alpha" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run.log" 2>&1
    fi
    local ok
    ok=$(grep -c 'verified=match' "$run.log" || true)
    echo "$rep-$driver-alpha: verified=match x${ok}"
  }
  run_matrix fork warmup  # discarded
  for rep in r1 r2 r3; do
    run_matrix fork "$rep"
    run_matrix stock "$rep"
  done
  $PY "$R/verify_digest_gates.py" "$R" > "$R/m2-logs/verify-d2.log" 2>&1
  vrc=$?
  echo "verifier rc=$vrc"
  [[ $vrc == 0 ]] || FAILURES=1
  tail -5 "$R/m2-logs/verify-d2.log"
  cp "$R/digest-gates.json" "$R/m2-logs/digest-gates-d2.json"
  echo "D2-WINDOW-COMPLETE FAILURES=$FAILURES"
  exit "$FAILURES"
} 2>&1 | tee "$R/m2-logs/window_d2.log"
rc=${PIPESTATUS[0]}
echo "$(date -Is) D2 window complete rc=$rc (lock released)"
exit "$rc"

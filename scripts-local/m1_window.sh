#!/usr/bin/env bash
# M1 window orchestration for the prefill-parity batch. One flock per
# process tree; the lock file is never unlinked; hostname printed in
# every remote call.
set -euo pipefail

M1="joshuawarren@100.84.184.102"
WT="$HOME/.config/superpowers/worktrees/mlx-omarchy/parity-prefill"
DEST="mlx-prefill-parity-20260908"
ICD=/tmp/asahi_prefill_icd.json

run_locked() {  # run_locked <label> <remote-script>
  echo "===== $1"
  ssh "$M1" "hostname; flock /tmp/m1-gpu.lock bash -s" <<REMOTE
$2
REMOTE
}

echo "== rsync worktree"
rsync -a --delete --exclude .work --exclude .git --exclude dist \
  "$WT/" "$M1:$DEST/"
ssh mesa-xbuild 'cat ~/mesa-prefill-parity/build/src/asahi/vulkan/libvulkan_asahi.so' \
  > /tmp/libvk_prefill.so
scp -q /tmp/libvk_prefill.so "$M1:~/src/mesa-prefill-parity-libvulkan_asahi.so"
cat > /tmp/asahi_prefill_icd_local.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "/home/joshuawarren/src/mesa-prefill-parity-libvulkan_asahi.so",
        "api_version": "1.4.359"
    }
}
EOF
scp -q /tmp/asahi_prefill_icd_local.json "$M1:$ICD"

run_locked "wheel build" '
cd ~/'"$DEST"' || exit 1
export DEV_RELEASE=1 MLX_OMARCHY_SOURCE_COMMIT=5c7876ac
scripts/build-wheel.sh > /tmp/wheel_build.log 2>&1 || { tail -30 /tmp/wheel_build.log; exit 1; }
tail -2 /tmp/wheel_build.log
rm -rf ~/venv-prefill-inline
python3 -m venv ~/venv-prefill-inline
~/venv-prefill-inline/bin/pip install --quiet dist/mlx_omarchy-*.whl mlx-lm==0.31.3
~/venv-prefill-inline/bin/python -c "import mlx.core as mx; print(mx.__version__)"
'

run_locked "correctness: 16-shape matrix, inline hooks ON" '
cd ~/'"$DEST"' || exit 1
export VK_ICD_FILENAMES='"$ICD"' AGX_SIMDMAT=1
export AGX_QMM_INLINE_A=1 AGX_QMM_INLINE_B=1
~/venv-prefill-inline/bin/python scripts-local/qmm_correctness.py | tail -3
'

run_locked "correctness: 16-shape matrix, inline hooks OFF (staged)" '
cd ~/'"$DEST"' || exit 1
export VK_ICD_FILENAMES='"$ICD"' AGX_SIMDMAT=1
~/venv-prefill-inline/bin/python scripts-local/qmm_correctness.py | tail -3
'

run_locked "correctness: bf16 matmul vs fp32 reference" '
cd ~/'"$DEST"' || exit 1
export VK_ICD_FILENAMES='"$ICD"' AGX_SIMDMAT=1
~/venv-prefill-inline/bin/python - <<"PY"
import mlx.core as mx
mx.random.seed(1)
bad = 0
for (m, k, n) in [(8, 64, 64), (32, 896, 896), (1053, 896, 4864), (262, 4864, 896), (30, 896, 128)]:
    a = mx.random.normal((m, k)).astype(mx.bfloat16)
    b = mx.random.normal((k, n)).astype(mx.bfloat16)
    out = a @ b
    mx.eval(out)
    ref = a.astype(mx.float32) @ b.astype(mx.float32)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    rel = err / max(mx.abs(ref).max().item(), 1e-6)
    ok = rel < 0.05
    bad += (not ok)
    print(f"bf16 matmul {m}x{k}x{n} rel_err={rel:.5f} ok={ok}")
assert bad == 0
print("BF16_ALL_OK")
PY
'

run_locked "divisor sweep ctx1024 prefill: 16 / 4 / 2 x 3 runs" '
cd ~/'"$DEST"' || exit 1
export VK_ICD_FILENAMES='"$ICD"' AGX_SIMDMAT=1 MLX_DISABLE_COMPILE=1
W=~/venv-prefill-inline/bin/python
WHL=$(ls -t dist/mlx_omarchy-*.whl | head -1)
PROMPT=$($W - <<"PY"
import json,sys
sys.path.insert(0,"scripts")
import bench_matrix
m=json.load(open("scripts/bench_matrix.json"))
print(bench_matrix.prompt_text(m,"ctx1024"))
PY
)
for D in 16 4 2; do
  for i in 1 2 3; do
    MLX_OMARCHY_BATCH_BYTES_DIVISOR=$D $W scripts/bench_decode.py \
      --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
      --raw-prompt --prompt "$PROMPT" --tokens 32 \
      --wheel "$WHL" 2>/dev/null \
      | grep -E "prefill" | tail -1 \
      | sed "s/^/divisor=$D run=$i /"
  done
done
'
echo "window phase 1 done (build+correctness+sweep)"

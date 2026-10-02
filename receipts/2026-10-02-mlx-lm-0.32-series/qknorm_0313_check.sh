set -euo pipefail
W=/tmp/mlxlm032/v0313
mkdir -p "$W/whl"
pip download -q --no-deps --dest "$W/whl" "mlx-lm==0.31.3"
for side in old new; do
  S="$W/$side/lib/python3.14/site-packages"
  mkdir -p "$S"
  python3 -m zipfile -e "$W"/whl/mlx_lm-0.31.3-*.whl "$S"
  for p in mlx-lm-gated-delta-fast-route mlx-lm-gated-delta-fast-route-repeat mlx-lm-gated-delta-raw mlx-lm-greedy-prune \
           mlx-lm-qwen35-qk-scaled mlx-lm-qwen35-gdn-conv mlx-lm-conv-silu mlx-lm-qwen35-gated-norm \
           mlx-lm-ttft-early-submit mlx-lm-last-logits; do
    patch -s -d "$S" -p1 --forward --fuzz=0 < "$HOME/src/omarchy-mlx-mlxlm032/patches/$p.patch"
  done
done
git -C "$HOME/src/omarchy-mlx-mlxlm032" show origin/main:scripts/patch-mlx-lm-qknorm.py > /tmp/mlxlm032/qknorm-main.py
python3 /tmp/mlxlm032/qknorm-main.py "$W/old"
python3 "$HOME/src/omarchy-mlx-mlxlm032/scripts/patch-mlx-lm-qknorm.py" "$W/new"
diff -r -x __pycache__ "$W/old/lib" "$W/new/lib" && echo "0.31.3: old and new patcher outputs identical"

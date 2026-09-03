#!/bin/sh
# CompiledFailClosed hardware verification at 7cf5e9f.
set -eu
V="$HOME/venv-bqm1-gate/bin/python"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
P="What is the capital of France? Answer in one word."
LOG="$HOME/benchq/logs/failclosed.log"

echo "== 3a. eager unaffected =="
if MLX_DISABLE_COMPILE=1 "$V" -m mlx_lm generate \
    --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
    > "$HOME/benchq/logs/failclosed-eager.log" 2>&1; then
  grep -q "Paris" "$HOME/benchq/logs/failclosed-eager.log" && echo "EAGER: clean Paris" || echo "EAGER: ran but no Paris"
else
  echo "EAGER: FAILED"; tail -5 "$HOME/benchq/logs/failclosed-eager.log"
fi

echo "== 1. default refusal, exact text =="
env -u MLX_DISABLE_COMPILE "$V" - <<'EOF' > "$HOME/benchq/logs/failclosed-refusal.log" 2>&1
import mlx.core as mx
def f(x):
    return mx.sigmoid(mx.broadcast_to(x, (2, x.shape[0])))
mx.compile(f)(mx.array([1.0, 2.0, 3.0]))
EOF
RC=$?
echo "refusal rc=$RC"
cat "$HOME/benchq/logs/failclosed-refusal.log"

echo "== 2. override permits compiled execution =="
if env -u MLX_DISABLE_COMPILE MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 "$V" - <<'EOF' > "$HOME/benchq/logs/failclosed-override.log" 2>&1
import mlx.core as mx
def f(x):
    return mx.sigmoid(mx.broadcast_to(x, (2, x.shape[0])))
print("out:", mx.compile(f)(mx.array([1.0, 2.0, 3.0])).tolist())
EOF
then
  echo "override rc=0 (compiled executed)"; cat "$HOME/benchq/logs/failclosed-override.log"
else
  echo "override rc!=0"; tail -4 "$HOME/benchq/logs/failclosed-override.log"
fi

echo "== 4. omarchy_compiled_tape_tests with override =="
cd ~/src/mlx-omarchy-bqm1-build
cmake -S .work/mlx -B .work/mlx/build -DMLX_BUILD_TESTS=ON -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > /dev/null 2>&1
cmake --build .work/mlx/build --target omarchy_compiled_tape_tests -j"$(nproc)" 2>&1 | tail -2
if MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 .work/mlx/build/tests/omarchy/omarchy_compiled_tape_tests \
    > "$HOME/benchq/logs/failclosed-tapetests.log" 2>&1; then
  echo "tape tests WITH override: rc=0"
else
  echo "tape tests WITH override: rc=$?"
fi
tail -4 "$HOME/benchq/logs/failclosed-tapetests.log"

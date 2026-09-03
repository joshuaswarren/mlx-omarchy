#!/bin/sh
# Follow-ups: (1) Bisect fragment pin on ff4b05a wheel, (2) leg-3 redo:
# compile genuinely ON at ff4b05a with CLEAN provisioning.
set -eu
V="$HOME/venv-rff4/bin/python"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
PROMPT="$HOME/benchq/prompt-france.txt"
echo "=== fragment pin (qwen2.swiglu, f16 1x7x4864) ==="
"$V" - <<'EOF'
import mlx.core as mx
from mlx_lm.utils import load
import mlx_lm.models.qwen2 as q2
model, _ = load("/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx")
g = mx.random.uniform(shape=(1, 7, 4864)).astype(mx.float16)
try:
    mx.eval(q2.swiglu(g, g))
    print("COMPILED FRAGMENT: passed (no refusal)")
except Exception as exc:
    print("COMPILED FRAGMENT: refused:", str(exc)[:200])
mx.disable_compile()
try:
    mx.eval(q2.swiglu(g, g))
    print("EAGER: passed")
except Exception as exc:
    print("EAGER: failed:", str(exc)[:200])
EOF
echo "=== leg-3 redo: compile ON at ff4b05a (clean venv) ==="
i=1
while [ "$i" -le 20 ]; do
  LOG="$HOME/benchq/logs/ABORT-CLEAN.run$i.log"
  if env -u MLX_DISABLE_COMPILE "$V" -m mlx_lm generate \
      --model "$Q4" --prompt "$(cat "$PROMPT")" \
      --max-tokens 32 --temp 0 --seed 0 > "$LOG" 2>&1 \
      && grep -q "Paris" "$LOG"; then
    PASS=${PASS:-0}; PASS=$((PASS + 1)); echo "run$i PASS"
  else
    echo "run$i FAIL — verbatim tail:"; tail -12 "$LOG"
    echo "ABORT_PROOF_CLEAN: fail at run $i"
    exit 3
  fi
  i=$((i + 1))
done
echo "ABORT_PROOF_CLEAN: 20/20 pass"

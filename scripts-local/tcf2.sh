#!/bin/sh
# TCF-2: graph/model modes + compiled-vs-eager payoff on the same wheel.
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
V="$HOME/venv-bqm1-tcf/bin/python"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
WHEEL=$(ls "$HOME/benchq/wheels/tcf"/*.whl)
MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
LIB="$HOME/venv-bqm1-tcf/lib/python3.14/site-packages/mlx/lib/libmlx.so"
LOADED=$(sha256sum "$LIB" | cut -d" " -f1)
echo "[gate] tcf member=$MEMBER loaded=$LOADED"
[ "$MEMBER" = "$LOADED" ] || { echo "FATAL: payload mismatch" >&2; exit 8; }
P="What is the capital of France? Answer in one word."

echo "== PART A: graph mode =="
env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
  MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
  "$V" scripts/differential_compile.py --mode graph \
  > "$HOME/benchq/logs/tcf2-graph.log" 2>&1
RC=$?
echo "graph rc=$RC"; tail -3 "$HOME/benchq/logs/tcf2-graph.log"
echo "== PART A: model mode (steps 1) =="
env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
  MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
  "$V" scripts/differential_compile.py --mode model --model "$Q4" \
  --prompt "What is the capital of France?" --steps 1 \
  --out-dir "$HOME/benchq/tcf2-model" \
  > "$HOME/benchq/logs/tcf2-model.log" 2>&1
RC=$?
echo "model rc=$RC"; tail -3 "$HOME/benchq/logs/tcf2-model.log"

echo "== PART B: payoff, 5 runs median per leg =="
r=1
while [ "$r" -le 5 ]; do
  if [ "$r" = 1 ]; then
    env -u MLX_DISABLE_COMPILE MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
      "$V" -m mlx_lm generate --model "$Q4" --prompt "$P" \
      --max-tokens 32 --temp 0 --seed 0 \
      > "$HOME/benchq/logs/tcf2-compiled-warm.log" 2>&1 || echo "warm rc=$?"
    echo "compiled warm-run notice count: $(grep -c 'shapeless compiled fragment' "$HOME/benchq/logs/tcf2-compiled-warm.log" || true)"
  fi
  LOGC="$HOME/benchq/logs/tcf2-compiled.run$r.log"
  MLX_DISABLE_COMPILE=1 true
  env -u MLX_DISABLE_COMPILE MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
    "$V" "$REPO/scripts/bench_decode.py" \
    --model "$Q4" --prompt "$P" --tokens 64 --wheel "$WHEEL" \
    > "$LOGC" 2>&1 || echo "compiled bench rc=$?"
  grep -q "shapeless compiled fragment" "$LOGC" && echo "compiled run$r tape-ran notice: present"
  LOGE="$HOME/benchq/logs/tcf2-eager.run$r.log"
  MLX_DISABLE_COMPILE=1 "$V" "$REPO/scripts/bench_decode.py" \
    --model "$Q4" --prompt "$P" --tokens 64 --wheel "$WHEEL" \
    > "$LOGE" 2>&1 || echo "eager bench rc=$?"
  r=$((r + 1))
done
python3 - <<'EOF'
import glob, re, statistics
for leg in ("compiled", "eager"):
    rates, prefills = [], []
    for f in sorted(glob.glob(f"/home/joshuawarren/benchq/logs/tcf2-{leg}.run*.log")):
        t = open(f).read()
        m = re.search(r"decode ([0-9.]+) tok/s over (\d+) tokens", t)
        p = re.search(r"prefill ([0-9.]+)s", t)
        if m: rates.append(float(m.group(1)))
        if p: prefills.append(float(p.group(1)))
    print(f"[{leg}] decode median={statistics.median(rates)} runs={rates}")
    if prefills:
        print(f"[{leg}] prefill median={statistics.median(prefills)}s = {36/statistics.median(prefills):.1f} tok/s")
EOF
echo "== bf16 eager context =="
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
r=1
while [ "$r" -le 5 ]; do
  LOGB="$HOME/benchq/logs/tcf2-bf16eager.run$r.log"
  MLX_DISABLE_COMPILE=1 "$V" "$REPO/scripts/bench_decode.py" \
    --model "$BF16" --prompt "$P" --tokens 64 --wheel "$WHEEL" \
    > "$LOGB" 2>&1 || echo "bf16 rc=$?"
  r=$((r + 1))
done
python3 - <<'EOF'
import glob, re, statistics
rates = []
for f in sorted(glob.glob("/home/joshuawarren/benchq/logs/tcf2-bf16eager.run*.log")):
    t = open(f).read()
    m = re.search(r"decode ([0-9.]+) tok/s over (\d+) tokens", t)
    if m: rates.append(float(m.group(1)))
print("[bf16-eager] decode median=", statistics.median(rates), "runs=", rates)
EOF

#!/bin/sh
# Shipping-state pinned-length bench: 18f59e3, 5-run median, both dtypes.
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
git fetch --quiet origin main
git checkout --quiet --detach FETCH_HEAD
echo "harness/scripts from $(git rev-parse --short HEAD); measured wheel pinned separately"
echo "testing commit $(git rev-parse --short HEAD)"
OUTDIR="$HOME/benchq/wheels/ship"
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
  sh scripts/build-wheel.sh 2>&1 | grep -E "source commit|sha256:"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
fi
VENV="$HOME/venv-bqm1-ship"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir "$OUTDIR"/*.whl
  "$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q --no-cache-dir -r "$REQS"
fi
WHEEL=$(ls "$OUTDIR"/*.whl)
MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
LOADED=$(sha256sum "$LIB" | cut -d" " -f1)
echo "[gate] wheel=$(basename "$WHEEL")"
echo "[gate] member=$MEMBER"
echo "[gate] loaded=$LOADED"
[ "$MEMBER" = "$LOADED" ] || { echo "FATAL: payload mismatch" >&2; exit 8; }
"$VENV/bin/python" -c "import mlx.core as mx; print('mx.__version__', mx.__version__)"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
PC="What is the capital of France? Answer in one word."
echo "START $(date '+%F %T') load: $(uptime)"
r=1
while [ "$r" -le 5 ]; do
  for PR in q4 bf16; do
    if [ "$PR" = q4 ]; then M="$Q4"; else M="$BF16"; fi
    MLX_DISABLE_COMPILE=1 "$VENV/bin/python" "$REPO/scripts/bench_decode.py" \
      --model "$M" --prompt "$PC" --tokens 64 --wheel "$WHEEL" \
      > "$HOME/benchq/logs/ship-$PR.run$r.log" 2>&1
    echo "ship $PR run$r: $(tail -1 "$HOME/benchq/logs/ship-$PR.run$r.log")"
  done
  r=$((r + 1))
done
echo "END $(date '+%F %T') load: $(uptime)"
python3 - <<'EOF'
import glob, re, statistics
for pr in ("q4", "bf16"):
    rates, prefills = [], []
    for f in sorted(glob.glob(f"/home/joshuawarren/benchq/logs/ship-{pr}.run*.log")):
        t = open(f).read()
        m = re.search(r"decode ([0-9.]+) tok/s over (\d+) tokens", t)
        p = re.search(r"prefill ([0-9.]+)s", t)
        if m:
            rates.append(float(m.group(1)))
        if p:
            prefills.append(float(p.group(1)))
    print(f"[{pr}] decode median={statistics.median(rates)} tok/s over 63 tokens, runs={rates}")
    if prefills:
        print(f"[{pr}] prefill median={statistics.median(prefills)}s for 36-token prompt = {36/statistics.median(prefills):.1f} tok/s")
EOF

#!/bin/sh
# TCF-2 (rebased): compiled-default vs eager on a fresh origin/main wheel.
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
git fetch --quiet origin main
git checkout --quiet --detach FETCH_HEAD
echo "testing commit $(git rev-parse --short HEAD)"
OUTDIR="$HOME/benchq/wheels/tcf2b"
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
  sh scripts/build-wheel.sh 2>&1 | grep -E "source commit|sha256:"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
fi
VENV="$HOME/venv-bqm1-tcf2b"
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
PC="What is the capital of France? Answer in one word."

echo "== B1 compiled-default: NO env vars, 5 runs =="
r=1
while [ "$r" -le 5 ]; do
  LOG="$HOME/benchq/logs/tcf2b-B1.run$r.log"
  "$VENV/bin/python" "$REPO/scripts/bench_decode.py" \
    --model "$Q4" --prompt "$P" --tokens 64 --wheel "$WHEEL" \
    > "$LOG" 2> "${LOG%.log}.err" || echo "B1 run$r rc=$?"
  WARN=$(grep -c "Compiled tapes are disabled" "${LOG%.log}.err" || true)
  NOTICE=$(grep -c "shapeless compiled fragment" "${LOG%.log}.err" || true)
  echo "B1 run$r: $(grep 'decode ' "$LOG" | head -1) | disabled-warning=$WARN (want 0) | tape-ran-notice=$NOTICE"
  r=$((r + 1))
done

echo "== B2 eager: MLX_DISABLE_COMPILE=1, 5 runs =="
r=1
while [ "$r" -le 5 ]; do
  LOG="$HOME/benchq/logs/tcf2b-B2.run$r.log"
  MLX_DISABLE_COMPILE=1 "$VENV/bin/python" "$REPO/scripts/bench_decode.py" \
    --model "$Q4" --prompt "$P" --tokens 64 --wheel "$WHEEL" \
    > "$LOG" 2> "${LOG%.log}.err" || echo "B2 run$r rc=$?"
  echo "B2 run$r: $(grep 'decode ' "$LOG" | head -1)"
  r=$((r + 1))
done

python3 - <<'EOF'
import glob, re, statistics
for leg in ("B1", "B2"):
    rates, prefills = [], []
    for f in sorted(glob.glob(f"/home/joshuawarren/benchq/logs/tcf2b-{leg}.run*.log")):
        t = open(f).read()
        m = re.search(r"decode ([0-9.]+) tok/s over (\d+) tokens", t)
        p = re.search(r"prefill ([0-9.]+)s", t)
        if m: rates.append(float(m.group(1)))
        if p: prefills.append(float(p.group(1)))
    print(f"[{leg}] decode median={statistics.median(rates)} tok/s over 63 tokens runs={rates}")
    if prefills:
        print(f"[{leg}] prefill median={statistics.median(prefills)}s = {36/statistics.median(prefills):.1f} tok/s")
EOF

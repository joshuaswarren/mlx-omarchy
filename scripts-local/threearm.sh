#!/bin/sh
# Three-arm pinned measurement: v0.3.2 asset / 7c25feb / v0.3.3 wheel.
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
git fetch --quiet origin main
git checkout --quiet --detach FETCH_HEAD
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
PC="What is the capital of France? Answer in one word."

arm() {
  TAG=$1
  WHEELDIR=$2
  VENV="$HOME/venv-bqm1-arm-$TAG"
  if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q --no-cache-dir "$WHEELDIR"/*.whl
    "$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
    "$VENV/bin/pip" install -q --no-cache-dir -r "$REQS"
  fi
  WHEEL=$(ls "$WHEELDIR"/*.whl)
  MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
  LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
  LOADED=$(sha256sum "$LIB" | cut -d" " -f1)
  echo "[gate] arm=$TAG member=$MEMBER loaded=$LOADED"
  [ "$MEMBER" = "$LOADED" ] || { echo "FATAL: arm $TAG payload mismatch" >&2; exit 8; }
  r=1
  while [ "$r" -le 5 ]; do
    for PR in q4 bf16; do
      if [ "$PR" = q4 ]; then M="$Q4"; else M="$BF16"; fi
      MLX_DISABLE_COMPILE=1 "$VENV/bin/python" "$REPO/scripts/bench_decode.py" \
        --model "$M" --prompt "$PC" --tokens 64 --wheel "$WHEEL" \
        > "$HOME/benchq/logs/arm-$TAG-$PR.run$r.log" 2>&1
      echo "arm $TAG $PR run$r: $(tail -1 "$HOME/benchq/logs/arm-$TAG-$PR.run$r.log")"
    done
    r=$((r + 1))
  done
}

echo "START $(date '+%F %T')"
arm v032 "$HOME/benchq/wheels/arm-v032"
arm c25feb "$HOME/benchq/wheels/tapeiso"
arm v033 "$HOME/benchq/wheels/v033"
echo "== MEDIANS =="
python3 - <<'EOF'
import glob, re, statistics
for tag in ("v032", "c25feb", "v033"):
    for pr in ("q4", "bf16"):
        rates, prefills = [], []
        for f in sorted(glob.glob(f"/home/joshuawarren/benchq/logs/arm-{tag}-{pr}.run*.log")):
            t = open(f).read()
            m = re.search(r"decode ([0-9.]+) tok/s over (\d+) tokens", t)
            p = re.search(r"prefill ([0-9.]+)s", t)
            if m: rates.append(float(m.group(1)))
            if p: prefills.append(float(p.group(1)))
        print(f"[{tag}-{pr}] decode median={statistics.median(rates)} tok/s over 63 tokens runs={rates}")
        if prefills:
            print(f"[{tag}-{pr}] prefill median={statistics.median(prefills)}s = {36/statistics.median(prefills):.1f} tok/s")
EOF
echo "DONE $(date '+%F %T')"

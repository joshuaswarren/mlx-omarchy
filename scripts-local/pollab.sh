#!/bin/sh
# Strict before/after decode A/B for the dispatcher poll-interval change.
# usage: pollab.sh <base-sha> <fix-sha>
# Builds both wheels first (no measurement during compiles), then runs the
# standing decode protocol on a quiet box: 5 runs, median, bf16 + 4-bit.
# Builds and venvs are skipped when their outputs already exist.
set -eu
BASE=$1
FIX=$2
cd ~/src/mlx-omarchy-bqm1-build
git fetch --quiet origin "$BASE" "$FIX" || true
BUILDLOG=~/benchq/pollab-build.log
touch "$BUILDLOG"

build_one() {
  SHA=$1
  OUTDIR=$2
  if ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
    echo "== skip build $SHA (wheel present) =="
    return 0
  fi
  echo "== build $SHA ==" | tee -a "$BUILDLOG"
  git checkout --quiet --detach "$SHA"
  sh scripts/build-wheel.sh 2>&1 | tee -a "$BUILDLOG" | grep "\[receipt\]"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
}

build_one "$BASE" "$HOME/benchq/wheels/pollab-base"
build_one "$FIX" "$HOME/benchq/wheels/pollab-fix"

echo "== venvs =="
if [ ! -x ~/venv-bqm1-pollbase/bin/python ]; then
  python3 -m venv ~/venv-bqm1-pollbase
  ~/venv-bqm1-pollbase/bin/pip install -q ~/benchq/wheels/pollab-base/*.whl
  ~/venv-bqm1-pollbase/bin/pip install -q --no-deps mlx-lm==0.31.3
  ~/venv-bqm1-pollbase/bin/pip install -q -r ~/benchq/support-reqs.txt
fi
if [ ! -x ~/venv-bqm1-pollfix/bin/python ]; then
  python3 -m venv ~/venv-bqm1-pollfix
  ~/venv-bqm1-pollfix/bin/pip install -q ~/benchq/wheels/pollab-fix/*.whl
  ~/venv-bqm1-pollfix/bin/pip install -q --no-deps mlx-lm==0.31.3
  ~/venv-bqm1-pollfix/bin/pip install -q -r ~/benchq/support-reqs.txt
fi
~/venv-bqm1-pollbase/bin/python -c "import mlx.core as mx; print('base', mx.__version__)"
~/venv-bqm1-pollfix/bin/python -c "import mlx.core as mx; print('fix', mx.__version__)"

echo "== settle before measuring =="
sleep 45
echo "START $(date '+%F %T') load: $(uptime)"
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
PROMPT=~/benchq/prompt-france.txt
sh ~/benchq/run_leg.sh ~/venv-bqm1-pollbase/bin/python "$BF16" "$PROMPT" - 5 DBASE-BF16
sh ~/benchq/run_leg.sh ~/venv-bqm1-pollbase/bin/python "$Q4" "$PROMPT" - 5 DBASE-Q4
sh ~/benchq/run_leg.sh ~/venv-bqm1-pollfix/bin/python "$BF16" "$PROMPT" - 5 DFIX-BF16
sh ~/benchq/run_leg.sh ~/venv-bqm1-pollfix/bin/python "$Q4" "$PROMPT" - 5 DFIX-Q4
echo "END $(date '+%F %T') load: $(uptime)"
python3 ~/benchq/parse_legs.py \
  'Prompt: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  'Generation: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  DBASE-BF16 DBASE-Q4 DFIX-BF16 DFIX-Q4

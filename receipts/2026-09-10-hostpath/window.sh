#!/usr/bin/env bash
# HostPathOverhead measurement window, fully queued (jwm1):
#   1. queue on /tmp/m1-gpu.lock (FIFO, single top-level flock)
#   2. wait for a quiet CPU (1-min loadavg < 1.0, 3 checks 20 s apart)
#   3. sample loadavg every 10 s while measuring
#   4. arms:
#      A baseline-parity  instrumented wheel, gates OFF (digests + tok/s)
#      C replay           MLX_OMARCHY_REPLAY=1 (digests + tok/s)
#      B trace            hostphases.py, host trace on
#      D replay+trace     hostphases.py with REPLAY=1
#      E tiny model       host floor (instrumented wheel + release wheel)
set -u
LOG=/tmp/hpo-window.log
exec >>"$LOG" 2>&1
echo "=== wrapper start $(date -u +%FT%TZ) pid $$"

R=~/src/mlx-HostPathOverhead/receipts/2026-09-10-hostpath
ROOT=~/src/mlx-HostPathOverhead
VENV=$ROOT/.work/venv-hpo/bin/python
REL_VENV=~/src/mlx-main-b6d662a8/.work/venv-run/bin/python
MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
PIN=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
WHEEL=$(ls $ROOT/dist/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl | head -1)
OUT=$ROOT/receipts/2026-09-10-hostpath/legs
mkdir -p "$OUT"

timeout 14400 flock -w 14300 /tmp/m1-gpu.lock bash -s <<INNER
echo "=== lock held \$(date -u +%FT%TZ) by \$$"
cd $ROOT

ok=0
for i in \$(seq 1 135); do
  l=\$(cut -d" " -f1 /proc/loadavg)
  if awk -v x="\$l" 'BEGIN{exit !(x < 1.0)}'; then
    ok=\$((ok+1))
    [ "\$ok" -ge 3 ] && break
  else
    ok=0
  fi
  echo "quiet-gate wait \$i load=\$l \$(date -u +%FT%TZ)"
  sleep 20
done
echo "quiet gate done ok=\$ok load=\$(cut -d" " -f1 /proc/loadavg)"

( while :; do echo "\$(date +%s) \$(cut -d" " -f1-3 /proc/loadavg)"; sleep 10; done ) &
SAMPLER=\$!
trap 'kill \$SAMPLER 2>/dev/null' EXIT

if [ ! -x "$VENV" ]; then
  python3 -m venv $ROOT/.work/venv-hpo
  $ROOT/.work/venv-hpo/bin/pip install -q "$WHEEL" mlx-lm
fi

run_leg () {  # tag workload; bench_matrix --select for one leg
  tag=\$1; wl=\$2
  if [ -f "$OUT/\$tag.json" ]; then echo "\$tag cached"; return; fi
  env -u MLX_OMARCHY_REPLAY -u MLX_OMARCHY_HOST_TRACE HF_HUB_OFFLINE=1 \
    MLX_DISABLE_COMPILE=1 \
    $VENV scripts/bench_matrix.py --mode run \
      --manifest $R/manifest-q4.json \
      --python $VENV --wheel "$WHEEL" \
      --select \$wl \
      --expect-pins qwen25-0.5b-4bit=\$PIN \
      --host-label jwm1-hostpath-\$tag --timeout 600 \
      --out "$OUT/\$tag.json" > "$OUT/\$tag.log" 2>&1
  echo "\$tag rc=\$?"
}

# A: baseline parity of the instrumented wheel, gates OFF (3 reps)
for rep in 1 2 3; do
  for wl in short-decode-32 long-decode-128 longctx-1024-decode-32; do
    run_leg a\$rep-baseline-\$wl \$wl
  done
done

# C: replay prototype, digests must stay canonical (3 reps)
for rep in 1 2 3; do
  for wl in short-decode-32 long-decode-128 longctx-1024-decode-32; do
    if [ -f "$OUT/c\$rep-replay-\$wl.json" ]; then echo cached; else
      MLX_OMARCHY_REPLAY=1 run_leg c\$rep-replay-\$wl \$wl
    fi
  done
done

# B: host-trace legs (trace on), short workload
for rep in 1 2 3; do
  if [ ! -f "$OUT/b\$rep-trace.json" ]; then
    env -u MLX_OMARCHY_REPLAY HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
      MLX_OMARCHY_HOST_TRACE=$OUT/b\$rep-trace-trace.json \
      $VENV $R/hostphases.py --model $MODEL --prompt "Hi" --tokens 32 \
      --trace-out $OUT/b\$rep-trace-trace.json \
      --out $OUT/b\$rep-trace.json > $OUT/b\$rep-trace.log 2>&1
    echo "b\$rep-trace rc=\$?"
  fi
done

# D: replay + trace
for rep in 1 2; do
  if [ ! -f "$OUT/d\$rep-replaytrace.json" ]; then
    HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 MLX_OMARCHY_REPLAY=1 \
      MLX_OMARCHY_HOST_TRACE=$OUT/d\$rep-replaytrace-trace.json \
      $VENV $R/hostphases.py --model $MODEL --prompt "Hi" --tokens 32 \
      --trace-out $OUT/d\$rep-replaytrace-trace.json \
      --out $OUT/d\$rep-replaytrace.json > $OUT/d\$rep-replaytrace.log 2>&1
    echo "d\$rep rc=\$?"
  fi
done

# E: tiny model host floor
if [ ! -f /tmp/tiny-qwen/model.safetensors ]; then
  $VENV $R/tiny_model.py --real $MODEL --out /tmp/tiny-qwen \
    > $OUT/tiny-build.log 2>&1 || echo "tiny build FAILED rc=\$?"
fi
for rep in 1 2 3; do
  if [ ! -f "$OUT/e\$rep-tiny.json" ]; then
    env -u MLX_OMARCHY_REPLAY HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
      MLX_OMARCHY_HOST_TRACE=$OUT/e\$rep-tiny-trace.json \
      $VENV $R/hostphases.py --model /tmp/tiny-qwen --prompt "Hi" \
      --tokens 32 --trace-out $OUT/e\$rep-tiny-trace.json \
      --out $OUT/e\$rep-tiny.json > $OUT/e\$rep-tiny.log 2>&1
    echo "e\$rep rc=\$?"
  fi
done
# E2: tiny on the release wheel venv (cross-wheel check)
if [ ! -f "$OUT/e1rel-tiny.json" ]; then
  env -u MLX_OMARCHY_REPLAY -u MLX_OMARCHY_HOST_TRACE HF_HUB_OFFLINE=1 \
    MLX_DISABLE_COMPILE=1 \
    $REL_VENV $R/hostphases.py --model /tmp/tiny-qwen --prompt "Hi" \
    --tokens 32 --trace-out /dev/null \
    --out $OUT/e1rel-tiny.json > $OUT/e1rel-tiny.log 2>&1
  echo "e1rel rc=\$?"
fi

echo "=== window done \$(date -u +%FT%TZ)"
INNER
echo "=== wrapper exit rc=$? $(date -u +%FT%TZ)"

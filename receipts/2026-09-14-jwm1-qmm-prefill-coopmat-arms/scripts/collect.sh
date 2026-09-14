#!/usr/bin/env bash
# collect.sh <stage-dir>
# Stage every artifact and identity fact for the receipt.
set -euo pipefail
STAGE=${1:?stage dir}
W=/var/tmp/qmm-coop-arms
BASE=/var/tmp/mlx-omarchy-profile-enabled-b41e2b74
SRC=/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74
Q=/home/joshuawarren/benchq/qmmpad
mkdir -p "$STAGE/arms" "$STAGE/dumps" "$STAGE/profiles" "$STAGE/scripts"

cp "$W/ab-all.jsonl" "$W/ab-summary.md" "$STAGE/"
for a in base clamp uvec4; do
  gzip -c "$W/dump-$a.log" > "$STAGE/dumps/dump-$a.log.gz"
done
for p in base clamp; do
  mkdir -p "$STAGE/profiles/$p"
  cp "$W/prof-$p/analysis.txt" "$W/prof-$p/markers.jsonl" \
     "$W/prof-$p/lock.json" "$W/prof-$p/run.log" "$STAGE/profiles/$p/"
  gzip -c "$W/prof-$p/profile.jsonl" > "$STAGE/profiles/$p/profile.jsonl.gz"
done
cp "$Q"/qmm_coopmat.*.comp "$STAGE/arms/"
cp "$Q/primitives.uvec4.cpp" "$STAGE/arms/"
cp "$SRC/overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp" \
   "$STAGE/arms/qmm_coopmat.base.comp"
cp "$Q"/*.sh "$Q"/*.py "$STAGE/scripts/"
cp "$W"/../qmm-coop-arms/ab-all.jsonl "$STAGE/" 2>/dev/null || true

{
  echo "== host"
  hostname; uname -srm; nproc
  echo "== kernel cmdline"
  cat /proc/cmdline
  echo "== mesa package"
  pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || pacman -Q mesa
  echo "== vulkan device"
  vulkaninfo --summary 2>/dev/null | sed -n '1,40p'
  echo "== agx / ane modules"
  lsmod | grep -E '^(ane|asahi)' || true
  echo "== power"
  for f in /sys/class/power_supply/*/online /sys/class/power_supply/macsmc-battery/status; do
    [ -e "$f" ] && echo "$f=$(cat "$f")"
  done
  echo "== source commit"
  git -C "$SRC" rev-parse HEAD
  git -C "$SRC" status --short | head -5
  echo "== model"
  sha256sum /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx/config.json
  sha256sum /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx/model.safetensors.index.json 2>/dev/null || true
  echo "== prompt"
  sha256sum /tmp/jwm1-phase-profile-20260913/prompt.txt
  wc -c /tmp/jwm1-phase-profile-20260913/prompt.txt
  echo "== probe"
  sha256sum /home/joshuawarren/benchq/qmm-coop-bench/qmm_coop_bench_probe.py
  sha256sum "$Q/probe.py"
  sha256sum /home/joshuawarren/benchq/qmm-coop-bench/qmm_dump_probe.py
  echo "== wheels"
  sha256sum "$BASE/dist"/*.whl
  sha256sum "$W"/dist-*/*.whl
  echo "== installed libmlx per arm"
  for v in "$BASE/venv" "$W"/venv-*; do
    printf '%s ' "$v"
    sha256sum "$v"/lib/python3.14/site-packages/mlx/lib/libmlx.so | awk '{print $1}'
  done
  echo "== arm shader sha256"
  sha256sum "$STAGE/arms"/*.comp "$STAGE/arms"/*.cpp
  echo "== base tree restored?"
  sha256sum "$BASE/mlx/mlx/backend/omarchy/shaders/qmm_coopmat.comp" \
            "$SRC/overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp"
  sha256sum "$BASE/mlx/mlx/backend/omarchy/primitives.cpp" \
            "$SRC/overlay/mlx/backend/omarchy/primitives.cpp"
  echo "== lock state"
  flock -n /tmp/m1-gpu.lock -c true && echo "flock -n: free" || echo "flock -n: held"
  fuser -v /tmp/m1-gpu.lock 2>&1 || echo "fuser: empty"
  stat -c 'inode=%i path=%n' /tmp/m1-gpu.lock
} > "$STAGE/identity.txt" 2>&1
echo "staged $(find "$STAGE" -type f | wc -l) files in $STAGE"
du -sh "$STAGE"

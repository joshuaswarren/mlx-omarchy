#!/usr/bin/env bash
# Deferred-post-barrier A/B window (wave/Bf16EncoderBoundary 490f2f52 vs
# main a12eafd9). ONE top-level flock on /tmp/m1-gpu.lock covering the
# source builds AND every GPU run. Inside the window, in order:
#   1. driver + host identity capture
#   2. wheel builds for both trees (pristine base, candidate commit)
#   3. venvs: copies of the parity venv-run, each force-reinstalled with
#      its arm's wheel (--no-deps --no-index)
#   4. quiet gate (1-min loadavg < 1.0 on three checks 20 s apart)
#   5. engagement probe on both wheels (trace counters)
#   6. canonical bench_matrix runs: discarded warmup per wheel, two A/A
#      baseline runs, interleaved base/candidate 3x3 (counterbalanced),
#      one stock-driver run per wheel
#   7. digest assertions against the committed pins on every measured leg
#   8. paired medians summary
set -uo pipefail
BASE_COMMIT=a12eafd994af881dbc2b39c0b9807523f27b171a
CAND_COMMIT=490f2f52bc220658466be13246d6846ba1fd33fd
B="$HOME/src/mlx-enc-base"
C="$HOME/src/mlx-enc-cand"
R="$HOME/enc-ab-window"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
PARITY_WORK="$HOME/src/mlx-omarchy-parity/.work"

exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock pid $$"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
T0=$(date +%s)
mkdir -p "$R"

{
  echo "host=$(hostname) kernel=$(uname -r) nproc=$(nproc)"
  pacman -Q mesa-honeykrisp-omarchy 2>/dev/null
  echo "base_commit=$BASE_COMMIT"
  echo "cand_commit=$CAND_COMMIT"
} > "$R/drivers.txt"
cat "$R/drivers.txt"

build_wheel() {  # tree commit label
  local tree="$1" commit="$2" label="$3"
  echo "=== build $label $(date -u +%H:%M:%S) ==="
  mkdir -p "$tree/.work"
  cp "$PARITY_WORK"/mlx-*.tar.gz "$tree/.work/" 2>/dev/null
  (cd "$tree" \
    && MLX_OMARCHY_SOURCE_COMMIT="$commit" DEV_RELEASE=1 \
       CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
       ./scripts/build-wheel.sh) > "$R/build-$label.log" 2>&1
  local rc=$?
  echo "build $label rc=$rc $(date -u +%H:%M:%S)"
  [ $rc -ne 0 ] && { echo "FATAL: build $label failed"; exit 5; }
  sha256sum "$tree"/dist/mlx_omarchy-*.whl >> "$R/drivers.txt"
}

prepare_env() {  # tree label wheelvenv
  local tree="$1" label="$2"
  rm -rf "$tree/.work/venv-$label"
  cp -a "$PARITY_WORK/venv-run" "$tree/.work/venv-$label"
  "$tree/.work/venv-$label/bin/python" -m pip install --quiet \
    --force-reinstall --no-deps --no-index \
    "$tree"/dist/mlx_omarchy-*.whl >> "$R/venv-$label.log" 2>&1
  local rc=$?
  echo "venv $label rc=$rc"
  [ $rc -ne 0 ] && { echo "FATAL: venv $label install failed"; exit 6; }
}

build_wheel "$B" "$BASE_COMMIT" base
build_wheel "$C" "$CAND_COMMIT" cand
prepare_env "$B" base
prepare_env "$C" cand

BASE_WHEEL=$(ls "$B"/dist/mlx_omarchy-*.whl)
CAND_WHEEL=$(ls "$C"/dist/mlx_omarchy-*.whl)
BASE_PY="$B/.work/venv-base/bin/python"
CAND_PY="$C/.work/venv-cand/bin/python"

# quiet gate before any measurement
ok=0
for i in $(seq 1 270); do
  load=$(cut -d" " -f1 /proc/loadavg)
  pass=$(awk -v l="$load" 'BEGIN{print (l<1.0)?1:0}')
  if [ "$pass" = 1 ]; then
    ok=$((ok+1))
    [ $ok -ge 3 ] && break
    sleep 20
  else
    ok=0
    sleep 20
  fi
done
echo "quiet gate done ok=$ok load=$(cut -d' ' -f1 /proc/loadavg)"
[ $ok -ge 3 ] || { echo "FATAL: quiet gate never passed"; exit 7; }

( while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 10; done ) > "$R/loadavg.txt" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

matrix_run() {  # tree py wheel label out icd
  local tree="$1" py="$2" wheel="$3" label="$4" out="$5" icd="$6"
  local -a ICD=()
  [ "$icd" = stock ] && ICD=(env VK_DRIVER_FILES="$STOCK_ICD")
  echo "=== matrix $out $(date -u +%H:%M:%S) ==="
  (
    cd "$tree" || exit 8
    timeout 1500 "${ICD[@]}" HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
      "$py" scripts/bench_matrix.py --mode run \
      --python "$py" \
      --wheel "$wheel" \
      --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
      --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
      --host-label "jwm1 $label" \
      --timeout 900 \
      --out "$R/$out.json" > "$R/$out.log" 2>&1
    echo "$out exit=$? $(date -u +%H:%M:%S)"
  )
}

probe_run() {  # py wheel label
  local py="$1" wheel="$2" label="$3"
  timeout 300 "$py" "$R/probe.py" > "$R/probe-$label.txt" 2>&1
  echo "probe $label exit=$?"
  grep '^PROBE' "$R/probe-$label.txt" || tail -3 "$R/probe-$label.txt"
}

probe_run "$BASE_PY" "$BASE_WHEEL" base
probe_run "$CAND_PY" "$CAND_WHEEL" cand

# discarded warmups, one full matrix per wheel
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd warmup-discarded" base-warmup fork
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand 490f2f5 warmup-discarded" cand-warmup fork
# A/A noise pair on the baseline
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd aa1" base-aa1 fork
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd aa2" base-aa2 fork
# interleaved, counterbalanced: b c c b b c
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r1" base-r1 fork
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand 490f2f5 r1" cand-r1 fork
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand 490f2f5 r2" cand-r2 fork
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r2" base-r2 fork
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r3" base-r3 fork
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand 490f2f5 r3" cand-r3 fork
# stock-driver smoke, one run per wheel
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd stock" base-stock stock
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand 490f2f5 stock" cand-stock stock

kill $SAMPLER 2>/dev/null

echo "=== digest + summary $(date -u +%FT%TZ)"
python3 - "$R" <<'PYCHECK'
import glob
import json
import statistics
import sys

r = sys.argv[1]
FORK = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
    "qwen25-0.5b-bf16:long-decode-128": "8690dc83246b39f8",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
STOCK = dict(FORK)
STOCK.update({
    "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
    "qwen25-0.5b-bf16:long-decode-128": "46108ad71157cb4d",
})
failures = []
summary = {}
for path in sorted(glob.glob(r + "/*.json")):
    if path.endswith(("summary.json", "probe.json")):
        continue
    run = path.rsplit("/", 1)[-1][:-5]
    try:
        d = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        failures.append(f"{run}: unparsable ({e})")
        continue
    pins = STOCK if run.endswith("-stock") else FORK
    for leg in d["legs"]:
        if not leg.get("measured"):
            continue
        lid = leg["leg_id"]
        want = pins.get(lid)
        got = leg["metrics"].get("generated_ids_sha256_16")
        if want is None:
            failures.append(f"{run}:{lid}: no pin for measured leg")
        elif got != want:
            failures.append(f"{run}:{lid}: digest {got} != pin {want}")
        if leg["metrics"].get("provenance_line", "").find("verified=match") < 0:
            failures.append(f"{run}:{lid}: provenance not verified=match")
        summary.setdefault(run, {})[lid] = {
            "decode_tok_s": leg["metrics"]["decode_tok_s"],
            "prefill_tok_s": leg["metrics"]["prefill_tok_s"],
        }

paired = {}
for arm in ("base", "cand"):
    for rep in ("r1", "r2", "r3"):
        for lid, m in summary.get(f"{arm}-{rep}", {}).items():
            paired.setdefault(lid, {}).setdefault(arm, []).append(
                m["decode_tok_s"])
print("== paired decode tok/s (medians of 3, higher is better)")
for lid, arms in sorted(paired.items()):
    b = statistics.median(arms.get("base", [float('nan')]))
    c = statistics.median(arms.get("cand", [float('nan')]))
    ratio = (c / b) if b else float("nan")
    print(f"{lid}: base={b:.2f} cand={c:.2f} cand/base={ratio:.4f}")
aa = {}
for lid, m in summary.get("base-aa1", {}).items():
    aa[lid] = [m["decode_tok_s"]]
for lid, m in summary.get("base-aa2", {}).items():
    aa.setdefault(lid, []).append(m["decode_tok_s"])
print("== A/A baseline pair (in-window noise)")
for lid, vals in sorted(aa.items()):
    if len(vals) == 2:
        print(f"{lid}: {vals[0]:.2f} vs {vals[1]:.2f} "
              f"({abs(vals[0]-vals[1])/min(vals)*100:.2f}%)")
json.dump({"summary": summary, "paired": paired},
          open(r + "/summary.json", "w"), indent=1)
if failures:
    print("FAILURES:")
    for f in failures:
        print(" " + f)
    sys.exit(1)
print("ALL_DIGEST_ASSERTIONS_PASSED")
PYCHECK
rc=$?
echo "checker rc=$rc"
echo "$(date -u +%FT%TZ) window complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"
exit $rc

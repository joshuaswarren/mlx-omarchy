#!/bin/bash
# TDT joint pair-load A/B window (jw16; after EncoderCompilerCoverage handoff).
# ONE wheel built from origin/main (corrected libmlx bytes, 925cfa64 lineage);
# TWO pkg trees (origin/main = baseline, agent/tdt-joint-pair-load = candidate)
# so the arms differ by exactly the pair-load commit. Interleaved, >=6 measured
# runs per arm, full transcript/hidden/mel/control gates + per-run decoder
# traces. Caller stops llm-inference.service; this script holds one flock for
# the whole battery and restarts nothing.
#
# Usage: run on jw16 as
#   TDT_AB_LIBMLX_SHA=<libmlx16 sha of the built wheel> ./tdt-pairload-ab-window.sh
set -uo pipefail
W=${TDT_AB_OUT:-/var/tmp/tdt-pairload-ab}
BASE_COMMIT=${TDT_AB_BASE:-origin/main}
CAND_COMMIT=${TDT_AB_CAND:-agent/tdt-joint-pair-load}
FORK=${TDT_AB_FORK:-$HOME/src/mlx-omarchy}
RUN=/var/tmp/ParakeetE2EJw16
RUNNER=/var/tmp/encwall-v071/base/vulkan_encoder.py
MODEL=$HOME/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
MEASURED=${TDT_AB_MEASURED:-6}
mkdir -p "$W"
unset PYTHONPATH
export MLX_OMARCHY_SPIRV_CACHE=/var/tmp/MelFrontendPerf/spirv-ab.3KDGHZ
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE ANE_OP_WALL || true

exec 9>/tmp/m1-gpu.lock
flock -x -w 3600 9 || { echo "LOCK-FAIL"; exit 4; }
echo "lock held $(date -Iseconds) inode $(stat -c %i /tmp/m1-gpu.lock)"

# ---- prep: venv from the corrected-libmlx wheel + two pkg trees
cd "$FORK"
git fetch origin --prune || exit 4
git rev-parse --verify "$BASE_COMMIT^{commit}" >/dev/null || exit 4
git rev-parse --verify "$CAND_COMMIT^{commit}" >/dev/null || exit 4
[ "$(git merge-base "$BASE_COMMIT" "$CAND_COMMIT")" = "$(git rev-parse "$BASE_COMMIT^{commit}")" ] \
  || { echo "CANDIDATE-NOT-BASED-ON-BASE"; exit 4; }

if [ -n "${TDT_AB_WHEEL:-}" ]; then
  WHL=$TDT_AB_WHEEL
else
  (cd "$FORK" && scripts/build-wheel.sh) || exit 4
  WHL=$(ls -t "$FORK"/dist/mlx_omarchy-*aarch64.whl | head -1)
fi
if [ ! -x "$W/venv/bin/python" ]; then
  "${TDT_AB_PYTHON:-python3.14}" -m venv "$W/venv"
  "$W/venv/bin/pip" install --quiet "$WHL" || exit 4
fi
PY=$W/venv/bin/python
[ -n "${TDT_AB_LIBMLX_SHA:-}" ] && {
  "$PY" /var/tmp/v063-jw16/scripts/venv-identity-guard.py --expect "$TDT_AB_LIBMLX_SHA" "$W/venv" || exit 4
}
"$PY" -c 'import importlib.metadata as M; print("venv dist:", M.version("mlx-omarchy"))'

for arm in base cand; do
  ref=$([ "$arm" = base ] && echo "$BASE_COMMIT" || echo "$CAND_COMMIT")
  rm -rf "$W/pkg-$arm"; mkdir -p "$W/pkg-$arm"
  git archive "$ref" overlay/tools/coreml | tar -x -C "$W/pkg-$arm"
  mv "$W/pkg-$arm/overlay/tools/coreml" "$W/pkg-$arm/coreml"
done
diff -rq "$W/pkg-base/coreml" "$W/pkg-cand/coreml" | sed 's/^/pkg-delta: /'

run_one () { # arm idx pkg
  local name=$1 pkg=$2
  local out=$W/out-$name scratch=$W/scratch-$name
  rm -rf "$out" "$scratch"; mkdir -p "$out" "$scratch"
  MLX_OMARCHY_PLACED=AC ANE_ISLAND_MODE=resident-batch \
  "$PY" "$RUN/fused_e2e.py" \
    --audio /var/tmp/ParakeetE2E/audio/fixture.flac \
    --golden /var/tmp/EncoderParityAne/capture \
    --model "$MODEL" \
    --pkg "$pkg" \
    --encoder-runner "$RUNNER" \
    --source /var/tmp/EncoderParityAne/encoder-source \
    --ane-reference /var/tmp/EncoderParityAne/capture/encoder_hidden.npy \
    --bundles /var/tmp/jw16-conv-place/bundles-conv \
    --worker /var/tmp/jw16-oproj-place/mlx-omarchy-ane-worker \
    --libane /var/tmp/jw16-oproj-place/libane-strict-fill.so \
    --scratch "$scratch" --out "$out" \
    --deadline-ms 20000 > "$W/log-$name.txt" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "RUN-FAILED $name rc=$rc"; tail -8 "$W/log-$name.txt"; exit 3; fi
  cp "$out/token_ids.json" "$W/decoder-trace-$name.json" 2>/dev/null
  echo "run $name done $(date -Iseconds)"
}

# warm-up each arm (excluded), then INTERLEAVED measured runs
run_one base-warm "$W/pkg-base/coreml"
run_one cand-warm "$W/pkg-cand/coreml"
for i in $(seq 1 "$MEASURED"); do
  run_one "base-$i" "$W/pkg-base/coreml"
  run_one "cand-$i" "$W/pkg-cand/coreml"
done

python3 - "$W" "$MEASURED" <<'PYEOF'
import hashlib, json, statistics, sys
W, N = sys.argv[1], int(sys.argv[2])
TRANSCRIPT = "db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790"
HIDDEN = "38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7"
MEL = "5b54f4a9a2ba3434cd69b6e48e6780d3bcb6c635d9ce85cda3d85c60f2455bde"
STAGES = ["audio_load", "mel_frontend", "encoder_ane", "decoder_load", "tdt_decode", "detokenize"]

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def collect(arm):
    rows = []
    for i in range(1, N + 1):
        r = json.load(open(f"{W}/out-{arm}-{i}/e2e-report.json"))
        st = {s["stage"]: s.get("wall_ms") for s in r["stages"] if isinstance(s, dict) and "stage" in s}
        ane = r.get("ane") or {}
        ex = r.get("execution") or {}
        row = {
            "run": i, "status": r["status"],
            "control": ex.get("control"), "fallback": ex.get("tdt_fallback_reason"),
            "prefix": r["layers"]["layer_6_decoder_sequence"].get("matching_prefix_length"),
            "bounds": r["layers"]["layer_5_encoder"]["all_bounds_pass"],
            "cpu_ev": r["execution"]["cpu_tensor_events"], "timeouts": ane.get("timeouts"),
            "transcript_sha": sha(f"{W}/out-{arm}-{i}/transcript.txt"),
            "hidden_sha": sha(f"{W}/out-{arm}-{i}/encoder_hidden.npy"),
            "mel_sha": sha(f"{W}/out-{arm}-{i}/mel.npy"),
            "stages": {k: st.get(k) for k in STAGES},
        }
        rows.append(row)
    return rows

def gate(rows, arm):
    for row in rows:
        bad = []
        if row["status"] != "match": bad.append(f"status={row['status']}")
        if row["prefix"] != 104: bad.append(f"prefix={row['prefix']}")
        if row["control"] != "gpu-loop": bad.append(f"control={row['control']}")
        if row["fallback"] is not None: bad.append(f"fallback={row['fallback']}")
        if row["transcript_sha"] != TRANSCRIPT: bad.append("transcript")
        if row["hidden_sha"] != HIDDEN: bad.append("hidden")
        if row["mel_sha"] != MEL: bad.append("mel")
        if not row["bounds"]: bad.append("bounds")
        if row["cpu_ev"] != 0: bad.append(f"cpu_ev={row['cpu_ev']}")
        if row["timeouts"] != 0: bad.append(f"timeouts={row['timeouts']}")
        if bad:
            print(f"GATE-FAIL {arm}-{row['run']}: {', '.join(bad)}")
            sys.exit(2)
    print(f"{arm} gate: {N}/{N} pins-EXACT")

summary = {}
for arm in ("base", "cand"):
    rows = collect(arm)
    gate(rows, arm)
    tdt = sorted(r["stages"]["tdt_decode"] for r in rows)
    summary[arm] = {
        "rows": rows,
        "tdt_decode_median_ms": round(statistics.median(tdt), 1),
        "tdt_decode_per_run_ms": [round(r["stages"]["tdt_decode"], 1) for r in rows],
        "stage_medians_ms": {k: round(statistics.median([r["stages"][k] for r in rows]), 1) for k in STAGES},
    }
delta = summary["cand"]["tdt_decode_median_ms"] - summary["base"]["tdt_decode_median_ms"]
summary["tdt_decode_delta_ms"] = round(delta, 1)
json.dump(summary, open(f"{W}/summary.json", "w"), indent=2)
print(json.dumps({a: {"tdt_decode_median_ms": summary[a]["tdt_decode_median_ms"],
                      "per_run": summary[a]["tdt_decode_per_run_ms"]} for a in ("base", "cand")}, indent=2))
print(f"TDT-AB-DONE delta={summary['tdt_decode_delta_ms']}ms")
PYEOF
echo "lock released $(date -Iseconds)"

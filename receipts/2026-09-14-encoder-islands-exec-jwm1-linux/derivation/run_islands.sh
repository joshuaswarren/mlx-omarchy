#!/bin/bash
# One guarded submit per encoder island on jwm1-linux. No retry, no loop, no
# reboot, no module unload, no GPU lock, no 1x896. If an island fails, the next
# island still gets its single submit; a failure is never re-run.
set -u

BASE=/var/tmp/jwm1-encoder-islands
WORKER=/var/tmp/jwm1-select-island-exec2/mlx-omarchy-ane-worker
LIBANE=/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so
OUT="$BASE/out"
mkdir -p "$OUT"

snapshot() {
  local tag=$1
  {
    echo "tag=$tag"
    echo "at=$(date -Is)"
    echo "device=$(stat -c '%F %a' /dev/accel/accel0 2>&1)"
    echo "module=$(lsmod | awk '$1=="ane"{print $1" refs="$3}')"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime_start=$(uptime -s)"
    echo "quarantine_bytes=$(stat -c %s /run/lock/mlx-omarchy-ane/quarantine 2>/dev/null)"
    echo "workers=$(pgrep -c -x mlx-omarchy-ane-worker || echo 0)"
    echo "dmesg_tm_failed=$(dmesg 2>/dev/null | grep -c 'tm execution failed')"
    echo "dmesg_errno110=$(dmesg 2>/dev/null | grep -cE 'errno[ =-]*110')"
  } > "$OUT/state-$tag.txt"
  cat "$OUT/state-$tag.txt"
}

submit() {
  local island=$1; shift
  echo "=== island $island"
  date -Is > "$OUT/$island.started_at"
  # shellcheck disable=SC2068
  timeout --signal=KILL 120 "$WORKER" "$@" \
    > "$OUT/$island.stdout" 2> "$OUT/$island.stderr"
  local rc=$?
  date -Is > "$OUT/$island.finished_at"
  echo "$rc" > "$OUT/$island.exit"
  echo "exit=$rc"
  cat "$OUT/$island.stdout"
  cat "$OUT/$island.stderr" >&2
}

snapshot pre

# Order is B (5 TDs), C (208), A (416): if the largest program wedges the TM,
# the two cheaper islands are already banked. Nothing is re-run either way.
submit B \
  --bundle "$BASE/bundles/island-select-8head" --libane "$LIBANE" \
  --deadline-ms 10000 --iterations 1 \
  --input ninf_rt="$BASE/stage/B_ninf_rt.bin" \
  --input matrix_bd_5="$BASE/stage/B_matrix_bd_5.bin" \
  --input cond="$BASE/stage/B_cond.bin" \
  --save attention_mask_9="$OUT/B_attention_mask_9.out.bin" \
  --expect attention_mask_9="$BASE/stage/B_ref_attention_mask_9.bin"

snapshot post-B

submit C \
  --bundle "$BASE/bundles/island-pv" --libane "$LIBANE" \
  --deadline-ms 10000 --iterations 1 \
  --input probs="$BASE/stage/C_probs.bin" \
  --input v_heads="$BASE/stage/C_v_heads.bin" \
  --save attn_output_1="$OUT/C_attn_output_1.out.bin"

snapshot post-C

submit A \
  --bundle "$BASE/bundles/island-attn-a-kt" --libane "$LIBANE" \
  --deadline-ms 10000 --iterations 1 \
  --input q_v="$BASE/stage/A_q_v.bin" \
  --input pos_kT="$BASE/stage/A_pos_kT.bin" \
  --input q_scaled="$BASE/stage/A_q_scaled.bin" \
  --input k_headsT="$BASE/stage/A_k_headsT.bin" \
  --save attention_scores_1="$OUT/A_attention_scores_1.out.bin" \
  --save matmul_0="$OUT/A_matmul_0.out.bin"

snapshot post

dmesg 2>/dev/null | tail -60 > "$OUT/dmesg-tail.txt"
echo "--- dmesg tail"
tail -25 "$OUT/dmesg-tail.txt"
echo "--- outputs"
ls -l "$OUT"

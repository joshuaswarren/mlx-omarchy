#!/usr/bin/env bash
# Six one-shot H13 LUT submits on jwm1: 3 sigmoid, 3 tanh, CHW {512,1,1}.
# One submit per program instance, deadline 5 s, no retry on any failure.
set -u
D=/var/tmp/jwm1-decoder-activations
WORKER=/var/tmp/AneWorkerValidation-c05ba1df/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
LIBANE=/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id) refcnt_before=$(cat /sys/module/ane/refcnt)"
for OP in sigmoid tanh; do
  for P in 0 1 2; do
    echo "=== $OP part $P $(date -Iseconds)"
    sha256sum "$WORKER" "$LIBANE" | cut -c1-16
    "$WORKER" --bundle "$D/bundle-$OP" --libane "$LIBANE" --deadline-ms 5000 --iterations 1 \
      --input "x=$D/x_${OP}_$P.bin" --expect "y=$D/y_expect_${OP}_$P.bin" --save "y=$D/y_out_${OP}_$P.bin"
    RC=$?
    echo "WORKER_EXIT=$RC $(date -Iseconds) refcnt=$(cat /sys/module/ane/refcnt)"
    if [ "$RC" -ne 0 ] && [ "$RC" -ne 1 ]; then
      echo "STOP: worker exit $RC is not the expect-mismatch exit; no retry"
      break 2
    fi
    if [ ! -s "$D/y_out_${OP}_$P.bin" ]; then
      echo "STOP: no output saved; no retry"
      break 2
    fi
  done
done
echo "refcnt_after=$(cat /sys/module/ane/refcnt)"
pgrep -af mlx-omarchy-ane-worker | grep -v run_remote || echo workers=none
fuser -v /dev/accel/accel0 2>&1 || echo accel0_holders=none
sha256sum "$D"/y_out_*.bin

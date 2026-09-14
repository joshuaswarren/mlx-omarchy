#!/bin/bash
# Device state snapshot for the three-island encoder parity run.
#
# Every line is measured from a readable interface on this host. Worker
# liveness comes from the canonical helper (tools/ane_worker_liveness.py),
# not from pgrep: `pgrep -x` cannot match a 22-character name against a
# 15-byte comm and `pgrep -f` matches the calling shell.
#
# quarantine_bytes is deliberately absent. The 2026-09-14 two-island receipts
# carry quarantine_bytes=0, but no readable interface on this host exposes it:
# there is no /sys/kernel/debug/ane, and /sys/kernel/debug/accel/26bc04000.ane
# holds only `name`. Rather than copy a zero this run did not measure, the
# field is omitted and named in the receipt.
set -u
ROOT=/var/tmp/EncoderParity3Islands
TAG="$1"
{
  echo "tag=$TAG"
  echo "at=$(date -Is)"
  echo "device=$(stat -c '%F %a %u:%g' /dev/accel/accel0)"
  echo "module_refcnt=$(cat /sys/module/ane/refcnt)"
  echo "module_loaded=$(lsmod | awk '$1=="ane"{print "yes"}')"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "uptime_start=$(uptime -s)"
  echo "ane_sys_runtime_status=$(cat /sys/devices/genpd_provider/ane_sys/power/runtime_status 2>/dev/null)"
  echo "dmesg_tm_failed=$(dmesg | grep -ci 'tm execution failed')"
  echo "dmesg_errno110=$(dmesg | grep -c 'errno 110')"
  echo "worker_liveness=$("$HOME"/venv-agxgen/bin/python "$ROOT/tools/ane_worker_liveness.py")"
} > "$ROOT/state-$TAG.txt"
cat "$ROOT/state-$TAG.txt"

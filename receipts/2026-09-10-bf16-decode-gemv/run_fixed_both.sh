#!/usr/bin/env bash
set -euo pipefail

rm -rf receipt-bf16/fixed-after-fork receipt-bf16/fixed-after-stock
receipt-bf16/linux_capture.sh \
  .venv-candidate/bin/python receipt-bf16/fixed-after-fork
VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
  receipt-bf16/linux_capture.sh \
  .venv-candidate/bin/python receipt-bf16/fixed-after-stock

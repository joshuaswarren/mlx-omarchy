#!/bin/sh
set -eu
exec flock -w 2400 /tmp/m1-gpu.lock timeout 2400 env \
  VK_ICD_FILENAMES=/tmp/asahi_coopmat_icd.json \
  AGX_SIMDMAT=1 \
  /bin/sh -c '
    /tmp/prefill_profile.py \
      /home/joshuawarren/src/mlx-GemvNativeForm/.venv-accept/bin/python \
      /tmp/prefill-isa-before-6c0f8ca \
      /home/joshuawarren/src/mlx-GemvNativeForm 8 16 \
      > /tmp/prefill-isa-before-6c0f8ca.out
    /tmp/prefill_profile.py \
      /home/joshuawarren/src/mlx-PrefillQmmIsa/.venv-accept/bin/python \
      /tmp/prefill-isa-after-6c0f8ca \
      /home/joshuawarren/src/mlx-PrefillQmmIsa 8 16 \
      > /tmp/prefill-isa-after-6c0f8ca.out
  '

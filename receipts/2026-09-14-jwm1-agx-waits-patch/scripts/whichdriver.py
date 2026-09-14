#!/usr/bin/env python3
"""Prove which libvulkan_asahi.so an arm actually loaded.

Runs one tiny GPU op so the Vulkan loader really binds the ICD, then reads
/proc/self/maps for the mapped driver. Transport success (the env var being
set) is not evidence; the mapped path is.
"""
import hashlib
import json
import os
import sys

import mlx.core as mx

x = mx.ones((8, 8), dtype=mx.float16)
mx.eval(x @ x)

mapped = set()
with open("/proc/self/maps") as fh:
    for line in fh:
        path = line.rstrip("\n").split(" ", 5)[-1].strip()
        if path.endswith("libvulkan_asahi.so"):
            mapped.add(path)

out = {
    "env_vk_driver_files": os.environ.get("VK_DRIVER_FILES"),
    "mapped_driver": sorted(mapped),
}
for p in sorted(mapped):
    try:
        with open(p, "rb") as fh:
            out["sha256_" + p] = hashlib.sha256(fh.read()).hexdigest()
    except OSError as exc:
        out["sha256_" + p] = "unreadable: %s" % exc

print(json.dumps(out, indent=2))
sys.exit(0 if len(mapped) == 1 else 1)

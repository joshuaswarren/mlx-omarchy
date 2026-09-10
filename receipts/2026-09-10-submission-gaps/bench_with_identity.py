#!/usr/bin/env python3
import importlib.metadata
import json
import os
import runpy
import sys

import mlx.core as mx

info = mx.device_info()
print(json.dumps({
    "mlx_omarchy": importlib.metadata.version("mlx-omarchy"),
    "cooperative_matrix_f32_8": info.get("cooperative_matrix_f32_8"),
    "device": info.get("device_name"),
    "VK_DRIVER_FILES": os.environ.get("VK_DRIVER_FILES"),
}), flush=True)
sys.argv = ["bench_matrix.py", *sys.argv[1:]]
runpy.run_path("scripts/bench_matrix.py", run_name="__main__")

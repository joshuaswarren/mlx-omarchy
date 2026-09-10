#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

root, python, output = map(Path, sys.argv[1:4])
output.mkdir(exist_ok=True)
model = subprocess.check_output(
    [
        str(python),
        "-c",
        "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen2.5-0.5B-Instruct-4bit', revision='a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3', local_files_only=True))",
    ],
    text=True,
).strip()
for label, gate in (("before", "0"), ("after", "1")):
    trace = output / f"{label}.jsonl"
    markers = output / f"{label}-markers.jsonl"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MLX_")}
    env.update(
        HF_HUB_OFFLINE="1",
        MLX_DISABLE_COMPILE="1",
        MLX_OMARCHY_SDPA_DECODE_NATIVE=gate,
        MLX_OMARCHY_GPU_PROFILE=str(trace),
    )
    with (output / f"{label}.log").open("w") as log:
        subprocess.run(
            [
                str(python),
                "scripts/profile_generate.py",
                "--model", model,
                "--prompt", "Explain how a computer executes a program, step by step, in detail.",
                "--max-tokens", "32",
                "--markers", str(markers),
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )
    with (output / f"{label}-analysis.txt").open("w") as log:
        subprocess.run(
            [
                str(python),
                "scripts/profile_analyze.py", str(trace),
                "--markers", str(markers),
                "--compute-h", "overlay/mlx/backend/omarchy/compute.h",
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    print("PROFILE_PASS", label, len(markers.read_text().splitlines()), trace.stat().st_size, flush=True)

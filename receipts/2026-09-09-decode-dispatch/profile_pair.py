#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

root, python, output = map(Path, sys.argv[1:4])
label, fusion = sys.argv[4:6]
output.mkdir(parents=True, exist_ok=True)
models = [
    ("4bit", "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"),
    ("bf16", "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"),
]
for name, revision in models:
    model = snapshot_download(
        "mlx-community/Qwen2.5-0.5B-Instruct-" + name,
        revision=revision,
        local_files_only=True,
    )
    trace = output / f"{label}-{name}.jsonl"
    markers = output / f"{label}-{name}-markers.jsonl"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MLX_")}
    env.update(
        HF_HUB_OFFLINE="1",
        MLX_DISABLE_COMPILE="1",
        MLX_OMARCHY_FUSED_CHAIN=fusion,
        MLX_OMARCHY_GPU_PROFILE=str(trace),
    )
    with (output / f"{label}-{name}.log").open("w") as log:
        subprocess.run(
            [
                str(python),
                "scripts/profile_generate.py",
                "--model", str(model),
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
    events = [json.loads(line) for line in markers.read_text().splitlines()]
    assert trace.stat().st_size > 0
    with (output / f"{label}-{name}-analysis.txt").open("w") as log:
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
    print("PROFILE_CAPTURED", label, name, len(events), trace.stat().st_size, flush=True)

#!/usr/bin/env python3
"""Profiling legs for the parity baseline: wall/CPU/GPU/kernel breakdown
for the three gate workloads on the -diag wheel (built with
-DMLX_OMARCHY_GPU_PROFILING=ON). Same pinned model, chat template, seed,
eager policy as run-baseline.py so kernel shares attribute to the same
computation.

Usage (cwd = repo root, under the GPU lock):
  python3 run-profile.py DIAG_PYTHON DIAG_WHEEL [OUTDIR]

Writes <OUTDIR>/<workload>.profile.jsonl, .markers.jsonl,
.analysis.txt and profile-summary.json.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAP = ("~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-"
        "Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")
WORKLOADS = [("short-32", "short", 32), ("long-128", "long", 128),
             ("ctx1024-32", "ctx1024", 32)]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: run-profile.py DIAG_PYTHON DIAG_WHEEL [OUTDIR]")
    diag_py = Path(sys.argv[1]).resolve()
    diag_wheel = Path(sys.argv[2]).resolve()
    if not diag_wheel.is_file():
        sys.exit(f"diag wheel missing: {diag_wheel}")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "profile"
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / "scripts" / "bench_matrix.json").read_text())
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_matrix

    env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
    env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1")
    model = os.path.expanduser(SNAP)
    if not Path(model).is_dir():
        sys.exit(f"model snapshot missing: {SNAP}")

    summary = {"schema": "parity-baseline-profile/1", "workloads": []}
    for name, prompt_id, tokens in WORKLOADS:
        prompt = bench_matrix.prompt_text(manifest, prompt_id)
        prof = out / f"{name}.profile.jsonl"
        markers = out / f"{name}.markers.jsonl"
        label = out / f"{name}.stdout.txt"
        lenv = dict(env)
        lenv["MLX_OMARCHY_GPU_PROFILE"] = str(prof)
        lenv["MLX_OMARCHY_GPU_PROFILE_LABEL"] = name
        cmd = [str(diag_py), str(ROOT / "scripts" / "profile_generate.py"),
               "--model", model, "--prompt", prompt,
               "--max-tokens", str(tokens), "--temp", "0", "--seed", "0",
               "--markers", str(markers)]
        with open(label, "w") as log:
            subprocess.run(cmd, cwd=ROOT, env=lenv, stdout=log,
                           stderr=subprocess.STDOUT, check=True,
                           timeout=1800)
        analysis = subprocess.run(
            [str(diag_py), str(ROOT / "scripts" / "profile_analyze.py"),
             str(prof), "--markers", str(markers), "--compute-h",
             str(ROOT / "overlay" / "mlx" / "backend" / "omarchy" /
                 "compute.h")],
            cwd=ROOT, env=env, capture_output=True, text=True,
            check=True, timeout=600)
        (out / f"{name}.analysis.txt").write_text(analysis.stdout)
        entry = {"workload": name, "prompt_id": prompt_id,
                 "max_tokens": tokens,
                 "profile_lines": sum(1 for _ in open(prof)),
                 "stdout": label.read_text().strip()[-400:]}
        summary["workloads"].append(entry)
        print(f"profile {name}: {entry['profile_lines']} events", flush=True)

    summary["diag_wheel"] = str(diag_wheel)
    summary["diag_wheel_sha256"] = hashlib.sha256(
        diag_wheel.read_bytes()).hexdigest()
    (out / "profile-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("profile: done")


if __name__ == "__main__":
    main()

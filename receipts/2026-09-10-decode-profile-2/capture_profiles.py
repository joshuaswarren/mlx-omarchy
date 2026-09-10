#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "scripts"))
    from bench_matrix import prompt_text

    manifest = json.loads((root / "scripts/bench_matrix.json").read_text())
    model = (Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")
    if not model.is_dir():
        raise SystemExit(f"missing pinned model: {model}")

    provenance = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "wheel": args.wheel.name,
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "model_revision": model.name,
        "python": str(args.python),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for case, prompt_id in (("short", "short"), ("ctx1024", "ctx1024")):
        profile = args.out / f"{case}.profile.jsonl"
        markers = args.out / f"{case}.markers.jsonl"
        log = args.out / f"{case}.generate.log"
        env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
        env.update(
            HF_HUB_OFFLINE="1",
            MLX_DISABLE_COMPILE="1",
            MLX_OMARCHY_GPU_PROFILE=str(profile),
            MLX_OMARCHY_GPU_PROFILE_LABEL=f"{args.label}-{case}",
        )
        cmd = [
            str(args.python), str(root / "scripts/profile_generate.py"),
            "--model", str(model), "--prompt", prompt_text(manifest, prompt_id),
            "--max-tokens", "16", "--temp", "0", "--seed", "0",
            "--markers", str(markers),
        ]
        with log.open("w") as fh:
            subprocess.run(cmd, cwd=root, env=env, stdout=fh,
                           stderr=subprocess.STDOUT, check=True, timeout=600)
        if not profile.stat().st_size or not markers.stat().st_size:
            raise SystemExit(f"empty profile output for {case}")
        print(f"PROFILE_CAPTURED {case} {profile.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()

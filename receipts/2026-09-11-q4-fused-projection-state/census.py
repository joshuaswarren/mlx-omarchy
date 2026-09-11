#!/usr/bin/env python3
"""Authoritative per-token dispatch census for one Q4 decode leg.

Runs scripts/profile_generate.py under MLX_OMARCHY_GPU_PROFILE, then
receipts/2026-09-09-decode-fusion/dispatch-table.py on a late
inter-token interval (requested 10, clamped to what the run produced -
EOS can end generation early), and writes a compact census JSON plus a
one-line stdout summary. usage:

  census.py OUT_DIR LEG PROMPT_ID [--fused-gemv V] [--max-tokens N]

PROMPT_ID resolves through manifest-q4fp.json ("text" entries verbatim,
"numbered" templates expanded exactly like scripts/bench_matrix.py).
Run at the repo root (cwd = checkout) with the diagnostics wheel
installed in .venv-accept, under the GPU lock.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
DISPATCH_TABLE = HERE.parent / "2026-09-09-decode-fusion" / "dispatch-table.py"
MODEL_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MODEL_REV = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("leg")
ap.add_argument("prompt_id")
ap.add_argument("--fused-gemv", default=None)
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--token", type=int, default=10)
args = ap.parse_args()

out = Path(args.out_dir)
out.mkdir(exist_ok=True)
py = ROOT / ".venv-accept/bin/python"
assert py.is_file(), py

entry = json.loads((HERE / "manifest-q4fp.json").read_text())[
    "prompts"][args.prompt_id]
if "text" in entry:
    prompt = entry["text"]
elif entry.get("template") == "numbered":
    prompt = " ".join([entry["base"]] + [
        f"{entry['item']} Entry {i} of {entry['items']}."
        for i in range(1, entry["items"] + 1)])
else:
    raise KeyError(f"prompt {args.prompt_id}: unsupported prompt entry")

env = {k: v for k, v in os.environ.items()
       if not k.startswith("MLX_") and k != "VK_DRIVER_FILES"}
env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1",
           MESA_SHADER_CACHE_DISABLE="true",
           MLX_OMARCHY_GPU_PROFILE=str(out / f"{args.leg}.jsonl"))
if args.fused_gemv is not None:
    env["MLX_OMARCHY_FUSED_GEMV"] = args.fused_gemv

model = subprocess.run(
    [str(py), "-c",
     "from huggingface_hub import snapshot_download; "
     f"print(snapshot_download('{MODEL_REPO}', revision='{MODEL_REV}', "
     "local_files_only=True))"],
    capture_output=True, text=True, check=True, env=env).stdout.strip()

with open(out / f"{args.leg}.log", "w") as log:
    subprocess.run(
        [str(py), "scripts/profile_generate.py", "--model", model,
         "--prompt", prompt, "--max-tokens", str(args.max_tokens),
         "--temp", "0", "--seed", "0",
         "--markers", str(out / f"{args.leg}-markers.jsonl")],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        check=True, timeout=300)

markers = [json.loads(line)
           for line in open(out / f"{args.leg}-markers.jsonl")]
intervals = len([m for m in markers if m["p"] == "tok"]) - 1
token = max(1, min(args.token, intervals))

table = subprocess.run(
    [str(py), str(DISPATCH_TABLE),
     str(out / f"{args.leg}.jsonl"),
     str(out / f"{args.leg}-markers.jsonl"),
     "overlay/mlx/backend/omarchy/compute.h",
     "--token", str(token),
     "--json", str(out / f"{args.leg}-token{token}.json")],
    cwd=ROOT, capture_output=True, text=True, check=True)
(out / f"{args.leg}-table.txt").write_text(table.stdout)

data = json.loads((out / f"{args.leg}-token{token}.json").read_text())
census = {
    "leg": args.leg,
    "fused_gemv": args.fused_gemv,
    "prompt_id": args.prompt_id,
    "dispatches_per_token": data["dispatches"],
    "submissions_per_token": len(data["submissions"]),
    "gpu_busy_us": data["gpu_busy_us"],
    "gap_us": data["gap_us"],
    "span_us": data["span_us"],
    "by_kernel": data["by_kernel"],
    "token_interval": token,
    "model": f"{MODEL_REPO}@{MODEL_REV}",
}
(out / f"{args.leg}-census.json").write_text(json.dumps(census, indent=1) + "\n")
print(f"CENSUS {args.leg} fused_gemv={args.fused_gemv}: "
      f"{census['dispatches_per_token']} dispatches/token, "
      f"{census['submissions_per_token']} submissions, "
      f"GPU busy {census['gpu_busy_us']} us, gaps {census['gap_us']} us, "
      f"span {census['span_us']} us", flush=True)

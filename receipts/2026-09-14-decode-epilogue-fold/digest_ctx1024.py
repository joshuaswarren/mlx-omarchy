#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path("/var/tmp/DecodeEpilogueFold")
MODEL = ("/home/joshuawarren/.cache/huggingface/hub/"
         "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/"
         "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")
BENCH = ROOT / "cand/scripts/bench_decode.py"
MANIFEST = json.loads((ROOT / "cand/scripts/bench_matrix.json").read_text())
entry = MANIFEST["prompts"]["ctx1024"]
prompt = " ".join([entry["base"]] + [
    f"{entry['item']} Entry {i} of {entry['items']}."
    for i in range(1, entry["items"] + 1)])
WBASE = str(next((ROOT / "dist-base").glob("mlx_omarchy-*.whl")))
WCAND = str(next((ROOT / "dist-cand").glob("mlx_omarchy-*.whl")))

def run(name, py, wheel, extra_env=None):
    env = dict(os.environ)
    env.update({"MLX_DISABLE_COMPILE": "1", "HF_HUB_OFFLINE": "1"})
    if extra_env:
        env.update(extra_env)
    cmd = [py, str(BENCH), "--model", MODEL, "--prompt", prompt,
           "--tokens", "32", "--temp", "0.0", "--seed", "0",
           "--warmup-tokens", "4", "--wheel", wheel]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=180)
    if proc.returncode != 0:
        sys.exit(f"{name} rc={proc.returncode}\n{proc.stderr[-1500:]}")
    ids = prov = decode = None
    for line in proc.stdout.splitlines():
        if line.startswith("generated_ids"):
            ids = line
        if line.startswith("provenance:"):
            prov = line
        if line.startswith("decode "):
            decode = line
    print(f"== {name} ==")
    print(prov)
    print(decode)
    print(ids)

run("base", str(ROOT / "venv-base/bin/python"), WBASE)
run("cand", str(ROOT / "venv-cand/bin/python"), WCAND)
run("cand-off", str(ROOT / "venv-cand/bin/python"), WCAND,
    {"MLX_OMARCHY_FOLD_EPILOGUE": "0"})

#!/usr/bin/env python3
"""Prefill A/B for the inline-fragment QMM coopmat kernel on jwm1-linux.

Five alternating process pairs per workload on the private prefill-parity
ICD: leg OFF is MLX_OMARCHY_NO_QMM_INLINE=1 (staged 2 KiB kernel,
5935a011 behaviour), leg ON sets the mesa inline hooks
AGX_QMM_INLINE_A=1 AGX_QMM_INLINE_B=1 and the MLX gate picks
QmmInlineCoopmatF16. Same wheel both legs; the driver is constant.

Every run is gated on the 1-minute load average below 1.5 with no model
or test process running, matching the 2026-09-08-qmm-prefill-coopmat
receipt protocol. Results print as one JSON line per run.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "scripts" / "bench_decode.py"
MATRIX = ROOT / "scripts" / "bench_matrix.json"

sys.path.insert(0, str(ROOT / "scripts"))
import bench_matrix  # noqa: E402

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
WORKLOADS = [
    ("short_30_32", "short", 32),
    ("long_262_128", "long", 128),
    ("ctx1024_1053_32", "ctx1024", 32),
]
PAIRS = 5


def load_ok():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0]) < 1.5


def busy_procs():
    out = subprocess.run(
        ["pgrep", "-af", "python|omarchy_|bench_decode"],
        capture_output=True, text=True).stdout
    bad = [l for l in out.splitlines()
           if "prefill_inline_ab" not in l and l.strip()]
    return bad


def wait_quiet():
    for _ in range(600):
        if load_ok() and not busy_procs():
            return
        time.sleep(10)
    raise SystemExit("load never dropped below 1.5")


def run_leg(prompt, tokens, env_extra):
    env = dict(os.environ)
    env["MLX_DISABLE_COMPILE"] = "1"
    env.pop("VK_ICD_FILENAMES", None)
    env.pop("AGX_SIMDMAT", None)
    env.pop("AGX_QMM_INLINE_A", None)
    env.pop("AGX_QMM_INLINE_B", None)
    env.pop("MLX_OMARCHY_NO_QMM_INLINE", None)
    env.update(env_extra)
    cmd = [sys.executable, str(BENCH),
           "--model", MODEL,
           "--raw-prompt", "--prompt", prompt,
           "--tokens", str(tokens),
           "--wheel", env["MLX_BENCH_WHEEL"]]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"bench failed rc={proc.returncode}\n"
                         f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    line = [l for l in proc.stdout.splitlines() if l.startswith("{")][-1]
    result = json.loads(line)
    result["leg_env"] = env_extra
    result["wall_s"] = round(time.time() - started, 2)
    print(json.dumps(result), flush=True)


def main():
    wheel = os.environ.get("MLX_BENCH_WHEEL")
    if not wheel:
        raise SystemExit("set MLX_BENCH_WHEEL to the venv's mlx wheel path")
    manifest = json.loads(MATRIX.read_text())
    for name, prompt_id, tokens in WORKLOADS:
        prompt = bench_matrix.prompt_text(manifest, prompt_id)
        for pair in range(PAIRS):
            for leg, env_extra in [
                ("off", {"MLX_OMARCHY_NO_QMM_INLINE": "1"}),
                ("on", {"AGX_QMM_INLINE_A": "1",
                        "AGX_QMM_INLINE_B": "1"}),
            ]:
                wait_quiet()
                result = run_leg(prompt, tokens, env_extra)
                result["pair"] = pair
                result["leg"] = leg
                result["workload"] = name
                print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

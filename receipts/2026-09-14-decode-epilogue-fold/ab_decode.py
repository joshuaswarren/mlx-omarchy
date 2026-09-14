#!/usr/bin/env python3
"""Interleaved decode A/B of two installed wheels on one laptop.

Runs under ONE flock hold on /tmp/m1-gpu.lock (the caller takes it).
Each round runs every arm on both legs (short 30/32, ctx1024 1053/32),
alternating the arm order round by round, every leg a fresh
bench_decode.py process with the receipt protocol (MLX_DISABLE_COMPILE=1,
HF_HUB_OFFLINE=1, --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4,
--wheel <this arm's wheel>). The generated-ID digest of every run is
asserted against the pinned pair; a mismatch stops the run. The medians
over the rounds are the reported numbers.

Arms are "name=python=wheel" triples. bench_decode.py's provenance
refusal (exit 3) proves each arm's interpreter really loaded that
wheel, and the wheel stamps are asserted distinct up front.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PINNED = {"short": "7fd25a869ff21678", "ctx1024": "7da83f06ec9f001d"}
TOKENS = 32


def prompt_text(manifest, prompt_id):
    entry = manifest["prompts"][prompt_id]
    if "text" in entry:
        return entry["text"]
    parts = [entry["base"]] + [
        f"{entry['item']} Entry {i} of {entry['items']}."
        for i in range(1, entry["items"] + 1)]
    return " ".join(parts)


def run_leg(arm, bench, model, prompt, extra_env):
    cmd = [arm["python"], str(bench), "--model", model, "--prompt", prompt,
           "--tokens", str(TOKENS), "--temp", "0.0", "--seed", "0",
           "--warmup-tokens", "4", "--wheel", arm["wheel"]]
    env = dict(os.environ)
    env.update({"MLX_DISABLE_COMPILE": "1", "HF_HUB_OFFLINE": "1"})
    env.update(extra_env)
    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          env=env)
    wall = time.monotonic() - start
    if proc.returncode != 0:
        sys.exit(f"{arm['name']} failed rc={proc.returncode}:\n"
                 f"{proc.stderr[-2000:]}")
    result = None
    provenance = None
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            result = json.loads(line)
        if line.startswith("provenance:"):
            provenance = line
    if result is None or provenance is None:
        sys.exit(f"{arm['name']}: no result/provenance line:\n{proc.stdout}")
    result["provenance"] = provenance
    result["wall_s"] = round(wall, 3)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=python=wheel (repeatable)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--env", action="append", default=[],
                    help="NAME=VALUE applied to every arm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        name, python, wheel = spec.split("=", 2)
        if not Path(wheel).is_file() or not Path(python).is_file():
            sys.exit(f"arm {name}: missing python or wheel")
        arms.append({"name": name, "python": python, "wheel": wheel,
                     "stamp": Path(wheel).name.split("+")[1].split("-")[0]})
    stamps = [a["stamp"] for a in arms]
    if len(set(stamps)) != len(stamps):
        sys.exit(f"arms do not differ: {stamps}")
    extra_env = dict(e.split("=", 1) for e in args.env)
    manifest = json.load(open(args.manifest))
    prompts = {leg: prompt_text(manifest, leg) for leg in PINNED}

    runs = []
    for rnd in range(args.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))
        for leg in PINNED:
            for arm in order:
                r = run_leg(arm, args.bench, args.model, prompts[leg],
                            extra_env)
                if r["ids_sha256_16"] != PINNED[leg]:
                    sys.exit(f"DIGEST MISMATCH {arm['name']} {leg}: "
                             f"{r['ids_sha256_16']} != {PINNED[leg]}")
                want = {"short": 30, "ctx1024": 1053}[leg]
                if r["prompt_tokens"] != want:
                    sys.exit(f"{arm['name']} {leg}: prompt_tokens "
                             f"{r['prompt_tokens']} != {want}")
                r.update(round=rnd, arm=arm["name"], leg=leg)
                runs.append(r)
                print(f"round {rnd} {leg:8s} {arm['name']:6s} "
                      f"decode {r['decode_tps']:9.4f} tok/s  prefill "
                      f"{r['prefill_tps']:10.4f} tok/s  ids "
                      f"{r['ids_sha256_16']}", flush=True)

    summary = {}
    for leg in PINNED:
        for arm in arms:
            rows = [r for r in runs if r["leg"] == leg
                    and r["arm"] == arm["name"]]
            summary[f"{leg}/{arm['name']}"] = {
                "decode_tps_median": statistics.median(
                    r["decode_tps"] for r in rows),
                "decode_tps_all": [r["decode_tps"] for r in rows],
                "prefill_tps_median": statistics.median(
                    r["prefill_tps"] for r in rows),
                "ids_sha256_16": sorted({r["ids_sha256_16"] for r in rows}),
                "provenance": sorted({r["provenance"] for r in rows}),
            }
    out = {"arms": arms, "rounds": args.rounds, "env": extra_env,
           "pinned": PINNED, "runs": runs, "summary": summary}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    for key, row in summary.items():
        print(f"{key:16s} decode median {row['decode_tps_median']:9.4f} "
              f"{row['decode_tps_all']}")


if __name__ == "__main__":
    main()

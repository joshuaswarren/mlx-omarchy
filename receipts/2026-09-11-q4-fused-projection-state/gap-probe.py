#!/usr/bin/env python3
"""Release-side GPU-gap probe: same binary, three instrument modes.

For each mode runs scripts/profile_generate.py on one leg and derives
per-token numbers from the SAME run (host wall always from the
script's own CLOCK_MONOTONIC tok markers):

  off    profiling env absent  -> the release execution path of this
                                  binary; markers give the true wall.
  noiso  MLX_OMARCHY_PROFILE_NOISOBAR=1 with GPU_PROFILE -> per-dispatch
         device timestamps WITHOUT the profiler's isolation barrier;
         busy = sum(t1-t0), span = last t1 - first t0, gap = span-busy
         on the native stream.
  iso    GPU_PROFILE default    -> the shipped diag instrument; the
         difference to noiso is the instrument's own cost.

usage: gap-probe.py OUT_DIR [LEG]
Run at the repo root with the probe diag wheel in .venv-accept, under
the GPU lock.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
MODEL_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MODEL_REV = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
LEG = sys.argv[2] if len(sys.argv) > 2 else "short"
OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
py = ROOT / ".venv-accept/bin/python"
assert py.is_file(), py

entry = json.loads((HERE / "manifest-q4fp.json").read_text())["prompts"][LEG]
prompt = entry.get("text") or " ".join(
    [entry["base"]]
    + [f"{entry['item']} Entry {i} of {entry['items']}."
       for i in range(1, entry["items"] + 1)])

env0 = {k: v for k, v in os.environ.items()
        if not k.startswith("MLX_") and k != "VK_DRIVER_FILES"}
env0.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1",
            MESA_SHADER_CACHE_DISABLE="true")
model = subprocess.run(
    [str(py), "-c",
     "from huggingface_hub import snapshot_download; "
     f"print(snapshot_download('{MODEL_REPO}', revision='{MODEL_REV}', "
     "local_files_only=True))"],
    capture_output=True, text=True, check=True, env=env0).stdout.strip()

MODES = [
    ("off", False, {}),
    ("noiso", True, {"MLX_OMARCHY_PROFILE_NOISOBAR": "1"}),
    ("iso", True, {}),
]


def analyze(disp, subs, toks, period_ns):
    """Per-token busy/gap/span between consecutive tok markers."""
    out = []
    for a, b in zip(toks, toks[1:]):
        chosen = [e for e in disp if a <= subs.get(e["s"], -1) < b]
        if not chosen:
            continue
        busy = sum(e["t1"] - e["t0"] for e in chosen)
        span = chosen[-1]["t1"] - chosen[0]["t0"]
        gaps = []
        nbound = 0.0
        prev = None
        for e in chosen:
            if prev is not None:
                g = e["t0"] - prev["t1"]
                gaps.append(g)
                if e["s"] != prev["s"]:
                    nbound += g
            prev = e
        out.append({
            "dispatches": len(chosen),
            "wall_ms": (b - a) / 1e6,
            "busy_ms": busy * period_ns / 1e6,
            "span_ms": span * period_ns / 1e6,
            "gap_ms": (span - busy) * period_ns / 1e6,
            "gap_at_boundary_ms": nbound * period_ns / 1e6,
            "gap_mean_us": (sum(gaps) / len(gaps)) * period_ns / 1e3
            if gaps else 0.0,
            "submissions": len({e["s"] for e in chosen}),
        })
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


summary = {"leg": LEG, "modes": {}}
for mode, prof, extra in MODES:
    env = dict(env0)
    if prof:
        env["MLX_OMARCHY_GPU_PROFILE"] = str(OUT / f"{LEG}-{mode}.jsonl")
        env.update(extra)
    with open(OUT / f"{LEG}-{mode}.log", "w") as log:
        subprocess.run(
            [str(py), "scripts/profile_generate.py", "--model", model,
             "--prompt", prompt, "--max-tokens", "32", "--temp", "0",
             "--seed", "0", "--markers", str(OUT / f"{LEG}-{mode}-m.jsonl")],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=300)
    marks = [json.loads(line)
             for line in open(OUT / f"{LEG}-{mode}-m.jsonl")]
    toks = [e["t"] for e in marks if e.get("p") == "tok"]
    walls = [(b - a) / 1e6 for a, b in zip(toks, toks[1:])]
    per = []
    if prof:
        period = 1.0
        disp, subs = [], {}
        for line in open(OUT / f"{LEG}-{mode}.jsonl"):
            e = json.loads(line)
            if e["k"] == "meta":
                period = e.get("period_ns", 1.0)
            elif e["k"] == "d":
                disp.append(e)
            elif e["k"] == "s":
                subs[e["s"]] = e["t"]
        per = analyze(disp, subs, toks, period)
    k = max(1, len(walls) // 2)
    steady_walls = walls[k:]
    steady_per = per[k:] if per else []
    entry_out = {
        "median_wall_ms": median(steady_walls),
        "intervals": len(walls),
        "per_token": steady_per,
    }
    summary["modes"][mode] = entry_out
    if prof:
        bg = median([t["busy_ms"] for t in steady_per])
        gg = median([t["gap_ms"] for t in steady_per])
        gb = median([t["gap_at_boundary_ms"] for t in steady_per])
        gm = median([t["gap_mean_us"] for t in steady_per])
        sp = median([t["span_ms"] for t in steady_per])
        nd = median([t["dispatches"] for t in steady_per])
        ns = median([t["submissions"] for t in steady_per])
        print(f"PROBE {mode}: wall {entry_out['median_wall_ms']:.2f} "
              f"ms/tok, busy {bg:.2f}, span {sp:.2f}, gap {gg:.2f} ms "
              f"(boundaries {gb:.2f}, mean {gm:.1f} us) "
              f"over {nd:.0f} dispatches, {ns:.0f} subs", flush=True)
    else:
        print(f"PROBE {mode}: wall {entry_out['median_wall_ms']:.2f} "
              f"ms/tok (profiling off = release path)", flush=True)

(OUT / f"{LEG}-gap-probe.json").write_text(
    json.dumps(summary, indent=1) + "\n")
print("GAP_PROBE_DONE", flush=True)

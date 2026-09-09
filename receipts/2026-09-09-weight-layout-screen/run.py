#!/usr/bin/env python3
"""Parity-baseline 5x3x2: five isolated repetitions of the three gate
workloads for BOTH pinned models (Qwen2.5-0.5B-Instruct-4bit and -bf16),
with exact generated-ID capture, provenance and contention asserts.

Usage:
  python3 run-baseline.py VENV_PYTHON WHEEL [REPS]

Writes per-rep rep<N>.json / rep<N>.log / rep<N>.ids.jsonl and
baseline-summary.json next to this script. Any assert aborts nonzero;
a partial run is never reported as complete. Run under the GPU lock
with cwd = repo root.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/joshuawarren/src/mlx-rope-drain-770ae465')
PIN_4BIT = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
PIN_BF16 = "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: run-baseline.py VENV_PYTHON WHEEL [REPS]")
    py = Path(sys.argv[1]).absolute()
    wheel = Path(sys.argv[2]).absolute()
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    if not py.is_file() or not wheel.is_file():
        sys.exit(f"missing python {py} or wheel {wheel}")

    manifest = json.loads((ROOT / "scripts" / "bench_matrix.json").read_text())
    manifest["generation"]["engine_script"] = "capture-ids-parity.py"
    hook = ROOT / "scripts" / "capture-ids-parity.py"
    hook.write_text((HERE / "capture-ids.py").read_text())
    mpath = HERE / "manifest-baseline.json"
    mpath.write_text(json.dumps(manifest, indent=2))

    summary = {
        "schema": "parity-baseline/1",
        "reps": reps,
        "assignment": "first hardware window baseline at c2548675",
        "wheel": None,
        "pins": {"qwen25-0.5b-4bit": PIN_4BIT, "qwen25-0.5b-bf16": PIN_BF16},
        "legs": [],
    }
    summary["wheel"] = str(wheel)
    summary["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    per_workload = {}
    for rep in range(1, reps + 1):
        prefix = HERE / f"rep{rep}"
        ids_path = Path(str(prefix) + ".ids.jsonl")
        ids_path.write_text("")
        env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
        env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1",
                   MLX_PAIR_IDS=str(ids_path))
        cmd = [str(py), "scripts/bench_matrix.py", "--mode", "run",
               "--manifest", str(mpath), "--python", str(py),
               "--wheel", str(wheel),
               "--expect-pins", f"qwen25-0.5b-4bit={PIN_4BIT}",
               "--expect-pins", f"qwen25-0.5b-bf16={PIN_BF16}",
               "--host-label", "jwm1-linux-parity-baseline",
               "--timeout", "600", "--out", str(prefix) + ".json"]
        with open(str(prefix) + ".log", "w") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True,
                           timeout=3600)
        data = json.loads(Path(str(prefix) + ".json").read_text())
        assert data["clean_check"]["status"] == "clean", (rep, "contention")
        if data.get("power"):
            assert data["power"]["source"] == "AC", (rep, data["power"])
        prov = data["binary_provenance"]["omarchy"]
        assert prov["verified"] == "match", (rep, prov.get("mismatch"))
        measured = [l for l in data["legs"] if l["status"] == "measured"]
        assert len(measured) == 6, (rep, [l["status"] for l in data["legs"]])
        records = [json.loads(x) for x in ids_path.read_text().splitlines()]
        assert len(records) == 6, (rep, len(records))
        for leg, record in zip(measured, records):
            assert len(record["ids"]) == leg["tokens"] == record["requested"], \
                (rep, leg["workload_id"], "token count")
            assert record["prompt_tokens"] == leg["metrics"]["prompt_tokens"], \
                (rep, leg["workload_id"], "prompt tokens")
            per = per_workload.setdefault(leg["leg_id"], [])
            per.append({
                "rep": rep,
                "decode_tok_s": leg["metrics"]["decode_tok_s"],
                "prefill_tok_s": leg["metrics"]["prefill_tok_s"],
                "prompt_tokens": leg["metrics"]["prompt_tokens"],
                "generated_ids_sha256_16":
                    leg["metrics"]["generated_ids_sha256_16"],
                "ids": record["ids"],
            })
        print(f"rep {rep}: ok", flush=True)

    for leg_id, runs in sorted(per_workload.items()):
        digests = {r["generated_ids_sha256_16"] for r in runs}
        summary["legs"].append({
            "leg_id": leg_id,
            "reps": len(runs),
            "decode_tok_s_median": statistics.median(
                r["decode_tok_s"] for r in runs),
            "decode_tok_s_min": min(r["decode_tok_s"] for r in runs),
            "decode_tok_s_max": max(r["decode_tok_s"] for r in runs),
            "prefill_tok_s_median": statistics.median(
                r["prefill_tok_s"] for r in runs),
            "prefill_tok_s_min": min(r["prefill_tok_s"] for r in runs),
            "prefill_tok_s_max": max(r["prefill_tok_s"] for r in runs),
            "prompt_tokens": runs[0]["prompt_tokens"],
            "generated_ids_sha256_16": sorted(digests),
            "ids_stable_across_reps": len(digests) == 1,
        })
    (HERE / "baseline-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["legs"], indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Uninstrumented Q4 decode token attribution: ablation arms.

Runs the canonical q4 bench_matrix legs on the release wheel (baseline,
digest-gated) and on the measurement-ablation wheel with
MLX_OMARCHY_ABLATE selecting one kernel class at a time (plus "all").
Ablation keeps dispatch shape, grid, loop walks, and output stores but
removes payload loads and math, so the wall-clock delta against the
baseline wheel is that class's marginal cost at identical submission
structure. Rounds interleave arms so machine drift spreads across arms.

Run from the worktree root under the GPU lock on a quiet machine:
  timeout 1800 flock -w 1700 python3 \
    receipts/2026-09-10-decode-attribution/run_arms.py \
    --python .work/venv-run/bin/python \
    --ablate-python .work/venv-ablate/bin/python \
    --wheel ~/src/mlx-main-b6d662a8/dist/<release>.whl \
    --ablate-wheel dist/<ablate>.whl --out receipts/2026-09-10-decode-attribution/arms
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import sys

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
EXPECTED = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}
ALL_LEGS = ["short-decode-32", "long-decode-128", "longctx-1024-decode-32"]
# arm -> reps per leg
ARMS = {
    "baseline": {"short-decode-32": 12, "long-decode-128": 12,
                 "longctx-1024-decode-32": 12},
    "gemv": {"short-decode-32": 12, "longctx-1024-decode-32": 8},
    "rope": {"short-decode-32": 12, "longctx-1024-decode-32": 8},
    "rms": {"short-decode-32": 12, "longctx-1024-decode-32": 8},
    "kvwrite": {"short-decode-32": 12, "long-decode-128": 8,
                "longctx-1024-decode-32": 8},
    "attn": {"short-decode-32": 12, "long-decode-128": 8,
             "longctx-1024-decode-32": 8},
    "swiglu": {"short-decode-32": 12, "longctx-1024-decode-32": 8},
    "sampler": {"short-decode-32": 12, "longctx-1024-decode-32": 8},
    "all": {"short-decode-32": 12, "long-decode-128": 8,
            "longctx-1024-decode-32": 8},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--python", type=Path, required=True,
                    help="venv python holding the release wheel")
    ap.add_argument("--ablate-python", type=Path, required=True,
                    help="venv python holding the ablation wheel")
    ap.add_argument("--wheel", type=Path, required=True,
                    help="unmodified release wheel (baseline arm)")
    ap.add_argument("--ablate-wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of arms to run (default all)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = {k: v for k, v in ARMS.items()
            if args.arms is None or k in args.arms}

    manifest = json.loads((root / "scripts/bench_matrix.json").read_text())
    manifest["models"] = [m for m in manifest["models"]
                          if m["id"] == "qwen25-0.5b-4bit"]
    manifest["generation"]["engine_script"] = "bench_decode_identity.py"
    manifest_path = out / "manifest-q4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    ndjson = (out / "legs.ndjson").open("a")
    max_rep = max(max(reps.values()) for reps in arms.values())
    for rep in range(1, max_rep + 1):
        for arm, reps in arms.items():
            for workload, nreps in reps.items():
                if rep > nreps:
                    continue
                t_leg = time.time()
                stem = f"r{rep}-{arm}-{workload}"
                out_json = out / f"{stem}.json"
                log_path = out / f"{stem}.log"
                if out_json.exists():
                    print(f"{stem} cached", flush=True)
                else:
                    env = {k: v for k, v in os.environ.items()
                           if not k.startswith("MLX_")}
                    env.pop("VK_DRIVER_FILES", None)
                    env.pop("HK_PERFTEST", None)
                    env["HF_HUB_OFFLINE"] = "1"
                    env["MLX_DISABLE_COMPILE"] = "1"
                    python = Path(os.path.abspath(
                        str(args.python if arm == "baseline"
                            else args.ablate_python)))
                    wheel = Path(os.path.abspath(
                        str(args.wheel if arm == "baseline"
                            else args.ablate_wheel)))
                    cmd = [
                        sys.executable,  # runner process: stdlib only
                        "scripts/bench_matrix.py", "--mode", "run",
                        "--manifest", str(manifest_path),
                        "--python", str(python),
                        "--wheel", str(wheel),
                        "--expect-pins", f"qwen25-0.5b-4bit={PIN}",
                        "--host-label", f"jwm1-decode-attrib-{arm}-r{rep}",
                        "--timeout", str(args.timeout),
                        "--out", str(out_json),
                    ]
                    with log_path.open("w") as log:
                        subprocess.run(cmd, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=True,
                                       timeout=args.timeout + 120)
                data = json.loads(out_json.read_text())
                assert data["clean_check"]["status"] == "clean", \
                    data["clean_check"]
                assert data["binary_provenance"]["omarchy"]["verified"] == \
                    "match", data["binary_provenance"]
                legs = [l for l in data["legs"] if l["workload_id"] == workload]
                assert len(legs) == 1 and legs[0]["status"] == "measured", \
                    [(l["leg_id"], l["status"]) for l in data["legs"]]
                leg = legs[0]
                m = leg["metrics"]
                digest = m["generated_ids_sha256_16"]
                if arm == "baseline":
                    assert digest == EXPECTED[workload], (workload, digest)
                record = {
                    "arm": arm, "rep": rep, "workload": workload,
                    "decode_tok_s": m["decode_tok_s"],
                    "prefill_s": m.get("prefill_s"),
                    "prompt_tokens": m.get("prompt_tokens"),
                    "digest": digest,
                    "duration_s": leg.get("duration_s"),
                    "host_label": leg.get("host_label"),
                    "wall_s": t_leg,
                }
                ndjson.write(json.dumps(record, sort_keys=True) + "\n")
                print(f"{stem} ok {m['decode_tok_s']} tok/s {digest}",
                      flush=True)
    ndjson.close()
    print("ALL_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()

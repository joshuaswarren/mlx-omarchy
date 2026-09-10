#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
EXPECTED = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--driver", choices=("fork", "stock"), required=True)
    args = ap.parse_args()

    root = args.root.resolve()
    python = args.python if args.python.is_absolute() else root / args.python
    wheel = args.wheel if args.wheel.is_absolute() else root / args.wheel
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "scripts/bench_matrix.json").read_text())
    manifest["models"] = [m for m in manifest["models"] if m["id"] == "qwen25-0.5b-4bit"]
    manifest["generation"]["engine_script"] = "bench_decode_identity.py"
    manifest_path = args.out / "manifest-q4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    summary = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "driver": args.driver,
        "VK_DRIVER_FILES": os.environ.get("VK_DRIVER_FILES"),
        "order": [],
        "runs": [],
    }
    grouped = defaultdict(list)
    for pair in range(1, args.pairs + 1):
        order = ("default", "gated") if pair % 2 else ("gated", "default")
        summary["order"].append({"pair": pair, "arms": list(order)})
        for arm in order:
            stem = f"pair{pair}-{arm}"
            out_json = args.out / f"{stem}.json"
            log_path = args.out / f"{stem}.log"
            env = {key: value for key, value in os.environ.items() if not key.startswith("MLX_")}
            env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1")
            if arm == "gated":
                env["MLX_OMARCHY_GATED_BARRIERS"] = "1"
            cmd = [
                str(python), "scripts/bench_matrix.py", "--mode", "run",
                "--manifest", str(manifest_path), "--python", str(python),
                "--wheel", str(wheel), "--expect-pins",
                f"qwen25-0.5b-4bit={PIN}", "--host-label",
                f"jwm1-decode-bound-3-{args.driver}-{stem}", "--timeout", "600",
                "--out", str(out_json),
            ]
            with log_path.open("w") as log:
                subprocess.run(cmd, cwd=root, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1200)
            data = json.loads(out_json.read_text())
            assert data["clean_check"]["status"] == "clean", data["clean_check"]
            assert data["binary_provenance"]["omarchy"]["verified"] == "match"
            legs = [leg for leg in data["legs"] if leg["status"] == "measured"]
            assert len(legs) == 3, [(leg["leg_id"], leg["status"]) for leg in data["legs"]]
            run = {"pair": pair, "arm": arm, "legs": []}
            for leg in legs:
                workload = leg["workload_id"]
                metrics = leg["metrics"]
                digest = metrics["generated_ids_sha256_16"]
                assert digest == EXPECTED[workload], (workload, digest)
                row = {
                    "workload": workload,
                    "prompt_tokens": metrics["prompt_tokens"],
                    "decode_tok_s": metrics["decode_tok_s"],
                    "digest": digest,
                }
                run["legs"].append(row)
                grouped[(arm, workload)].append(metrics["decode_tok_s"])
            summary["runs"].append(run)
            print(f"{stem} ok", flush=True)

    summary["medians"] = {
        arm: {
            workload: statistics.median(grouped[(arm, workload)])
            for workload in EXPECTED
        }
        for arm in ("default", "gated")
    }
    summary["gated_change_pct"] = {
        workload: round(
            (summary["medians"]["gated"][workload] /
             summary["medians"]["default"][workload] - 1) * 100, 4)
        for workload in EXPECTED
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["medians"], sort_keys=True), flush=True)
    print(json.dumps(summary["gated_change_pct"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

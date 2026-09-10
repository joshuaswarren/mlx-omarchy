#!/usr/bin/env python3
"""Driver A/B decode legs with digest gate for the dispatch-floor receipt.

Runs the canonical q4 bench_matrix legs (short/long/longctx decode) under each
driver arm and asserts the six canonical digests. Arms are selected purely via
environment (VK_DRIVER_FILES / HK_PERFTEST); the mlx wheel is identical for all
arms, so any digest flip is a driver-side arithmetic change.

Run from the checkout root under the GPU lock on a quiet machine:
  python3 receipts/2026-09-10-dispatch-floor/run_driver_legs.py
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
EXPECTED = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}
WT_ICD = "/home/joshuawarren/src/mesa-wt-dispatchfloor/dispatchfloor-icd.json"
STOCK_ICD = "/home/joshuawarren/stock-mesa/stock-icd.json"
ARMS = {
    "wt-default": {"VK_DRIVER_FILES": WT_ICD},
    "wt-nocdmbarrier": {"VK_DRIVER_FILES": WT_ICD, "HK_PERFTEST": "nocdmbarrier"},
    "wt-usccdmbarrier": {"VK_DRIVER_FILES": WT_ICD, "HK_PERFTEST": "usccdmbarrier"},
    "stock": {"VK_DRIVER_FILES": STOCK_ICD},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((root / "scripts/bench_matrix.json").read_text())
    manifest["models"] = [m for m in manifest["models"] if m["id"] == "qwen25-0.5b-4bit"]
    manifest["generation"]["engine_script"] = "bench_decode_identity.py"
    manifest_path = args.out / "manifest-q4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    summary = {"arms": {}, "runs": []}
    for rep in range(1, args.reps + 1):
        for arm, extra in ARMS.items():
            stem = f"r{rep}-{arm}"
            out_json = args.out / f"{stem}.json"
            log_path = args.out / f"{stem}.log"
            env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
            env.pop("VK_DRIVER_FILES", None)
            env.pop("HK_PERFTEST", None)
            env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1")
            env.update({k: v for k, v in extra.items() if v is not None})
            cmd = [
                str(args.python), "scripts/bench_matrix.py", "--mode", "run",
                "--manifest", str(manifest_path), "--python", str(args.python),
                "--wheel", str(args.wheel), "--expect-pins",
                f"qwen25-0.5b-4bit={PIN}", "--host-label",
                f"jwm1-dispatch-floor-{arm}-r{rep}", "--timeout", "600",
                "--out", str(out_json),
            ]
            with log_path.open("w") as log:
                subprocess.run(cmd, cwd=root, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1200)
            data = json.loads(out_json.read_text())
            assert data["clean_check"]["status"] == "clean", data["clean_check"]
            assert data["binary_provenance"]["omarchy"]["verified"] == "match"
            legs = [l for l in data["legs"] if l["status"] == "measured"]
            assert len(legs) == 3, [(l["leg_id"], l["status"]) for l in data["legs"]]
            run = {"rep": rep, "arm": arm, "legs": []}
            for leg in legs:
                metrics = leg["metrics"]
                digest = metrics["generated_ids_sha256_16"]
                assert digest == EXPECTED[leg["workload_id"]], (
                    leg["workload_id"], digest)
                run["legs"].append({
                    "workload": leg["workload_id"],
                    "prompt_tokens": metrics["prompt_tokens"],
                    "decode_tok_s": metrics["decode_tok_s"],
                    "digest": digest,
                })
            summary["runs"].append(run)
            summary["arms"].setdefault(arm, []).append(
                {l["workload"]: l["decode_tok_s"] for l in run["legs"]})
            print(f"{stem} ok", flush=True)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ALL_DIGEST_LEGS_OK", flush=True)


if __name__ == "__main__":
    main()

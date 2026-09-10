#!/usr/bin/env python3
"""Paired alternating qmm scheduling screen (default / SMALLN_TILE /
SPLITK=auto) on one exact wheel, per driver. Mirrors the decode-bound-3
paired protocol: rotations balance arm order, every leg asserts the
pinned Q4 generated-id digest, and both prefill and decode throughput
are recorded so the 1K prefill question gets a paired answer."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
# Canonical plain-engine Q4 pins (bench_decode.py). These were reproduced
# by the default arm of this screen on wheel 1323dc8 and by the canonical
# b6d662a8 matrix. Do NOT substitute the bench_decode_identity wrapper:
# its device_info() before model load perturbs buffer offsets and can
# flip the long-decode-128 digest on any build (observed 2026-09-10).
EXPECTED = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}
ARM_ENV = {
    "default": {},
    "tile128": {"MLX_OMARCHY_QMM_SMALLN_TILE": "128"},
    "splitauto": {"MLX_OMARCHY_QMM_SPLITK": "auto"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rotations", type=int, default=3)
    ap.add_argument("--driver", choices=("fork", "stock"), required=True)
    args = ap.parse_args()

    root = args.root.resolve()
    python = args.python if args.python.is_absolute() else root / args.python
    wheel = args.wheel if args.wheel.is_absolute() else root / args.wheel
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "scripts/bench_matrix.json").read_text())
    manifest["models"] = [m for m in manifest["models"] if m["id"] == "qwen25-0.5b-4bit"]
    manifest_path = args.out / "manifest-q4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    summary = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "driver": args.driver,
        "VK_DRIVER_FILES": os.environ.get("VK_DRIVER_FILES"),
        "arm_env": ARM_ENV,
        "order": [],
        "runs": [],
    }
    grouped = defaultdict(list)
    arms = ["default", "tile128", "splitauto"]
    for rotation in range(1, args.rotations + 1):
        order = arms[(rotation - 1) % len(arms):] + arms[: (rotation - 1) % len(arms)]
        summary["order"].append({"rotation": rotation, "arms": list(order)})
        for arm in order:
            stem = f"rot{rotation}-{arm}"
            out_json = args.out / f"{stem}.json"
            log_path = args.out / f"{stem}.log"
            env = {key: value for key, value in os.environ.items() if not key.startswith("MLX_")}
            env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1")
            env.update(ARM_ENV[arm])
            cmd = [
                str(python), "scripts/bench_matrix.py", "--mode", "run",
                "--manifest", str(manifest_path), "--python", str(python),
                "--wheel", str(wheel), "--expect-pins",
                f"qwen25-0.5b-4bit={PIN}", "--host-label",
                f"jwm1-qmm-splitk-parity-{args.driver}-{stem}", "--timeout", "600",
                "--out", str(out_json),
            ]
            # The default arm retries: a digest mismatch there means a
            # fragmented-heap window flipped dispatches to the tile
            # fallback (environmental, not deterministic). A knob arm
            # gets one attempt: a deviation is the arm's own routing
            # changing generated ids, which is the finding itself.
            attempts = []
            max_attempts = 3 if arm == "default" else 1
            digests = {}
            for attempt in range(1, max_attempts + 1):
                with log_path.open("w") as log:
                    subprocess.run(cmd, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True,
                                   timeout=1200)
                data = json.loads(out_json.read_text())
                assert data["clean_check"]["status"] == "clean", data["clean_check"]
                assert data["binary_provenance"]["omarchy"]["verified"] == "match"
                legs = [leg for leg in data["legs"] if leg["status"] == "measured"]
                assert len(legs) == 3, [(leg["leg_id"], leg["status"]) for leg in data["legs"]]
                digests = {leg["workload_id"]: leg["metrics"]["generated_ids_sha256_16"]
                           for leg in legs}
                if all(digests[w] == EXPECTED[w] for w in EXPECTED):
                    break
                if arm != "default":
                    break
                attempts.append({"attempt": attempt, "digests": digests})
                print(f"{stem} digest retry {attempt}: {digests}", flush=True)
            digest_ok = all(digests[w] == EXPECTED[w] for w in EXPECTED)
            assert digest_ok or arm != "default", (stem, digests)
            run = {"rotation": rotation, "arm": arm, "legs": [],
                   "digest_ok": digest_ok,
                   "digests": digests,
                   "digest_retries": len(attempts)}
            for leg in legs:
                workload = leg["workload_id"]
                metrics = leg["metrics"]
                digest = metrics["generated_ids_sha256_16"]
                run["legs"].append({
                    "workload": workload,
                    "prompt_tokens": metrics["prompt_tokens"],
                    "prefill_tok_s": metrics["prefill_tok_s"],
                    "decode_tok_s": metrics["decode_tok_s"],
                    "digest": digest,
                })
                grouped[(arm, workload)].append(metrics["prefill_tok_s"])
            summary["runs"].append(run)
            print(f"{stem} ok digest_ok={digest_ok}", flush=True)

    workloads = list(EXPECTED)
    summary["prefill_medians"] = {
        arm: {workload: statistics.median(grouped[(arm, workload)])
              for workload in workloads}
        for arm in arms
    }
    # Order-aware paired change: each rotation has exactly one default leg;
    # pair each non-default arm leg against that rotation's default leg.
    default_by_rotation = {
        run["rotation"]: {leg["workload"]: leg["prefill_tok_s"]
                          for leg in run["legs"]}
        for run in summary["runs"] if run["arm"] == "default"
    }
    paired = defaultdict(list)
    for run in summary["runs"]:
        if run["arm"] == "default":
            continue
        base = default_by_rotation[run["rotation"]]
        for leg in run["legs"]:
            paired[(run["arm"], leg["workload"])].append(
                (leg["prefill_tok_s"] / base[leg["workload"]] - 1) * 100)
    summary["paired_change_pct"] = {
        f"{arm}/{workload}": [round(v, 4) for v in values]
        for (arm, workload), values in sorted(paired.items())
    }
    summary["median_paired_change_pct"] = {
        f"{arm}/{workload}": round(statistics.median(values), 4)
        for (arm, workload), values in sorted(paired.items())
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["prefill_medians"], sort_keys=True), flush=True)
    print(json.dumps(summary["median_paired_change_pct"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

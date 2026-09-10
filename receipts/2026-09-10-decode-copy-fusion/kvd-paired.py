#!/usr/bin/env python3
"""Paired fork/stock KV-write decode A/B at the 30/262/1053 prompt legs,
following the paired_barriers protocol (decode-bound-3): alternating arm
order per pair, per-leg canonical-digest hard gate, medians + change pct.

Arms on ONE release wheel of wave/DecodeCopyFusion:
  fork   default env            -> producer-direct KV writes (KV_DIRECT=1)
  stock  MLX_OMARCHY_KV_DIRECT=0 -> merged SliceUpdatePair dispatch path

The six canonical Q4 digests (3 legs x 2 arms) are hard gates. A digest
mismatch retries the leg once; a persisting mismatch aborts the window.
Every measured leg must also report a clean (uncontended) machine.

usage: kvd-paired.py ROOT PYTHON WHEEL OUT_DIR [PAIRS]
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
EXPECTED = {
    "short-decode-32": ("7fd25a869ff21678", 30),
    "long-decode-128": ("4cc08910089477fd", 262),
    "longctx-1024-decode-32": ("7da83f06ec9f001d", 1053),
}
ARMS = {
    "fork": {},
    "stock": {"MLX_OMARCHY_KV_DIRECT": "0"},
}


def base_env(extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MLX_", "HK_"))}
    env.pop("VK_DRIVER_FILES", None)
    env.pop("PYTHONPATH", None)
    env["MESA_SHADER_CACHE_DISABLE"] = "true"
    env["HF_HUB_OFFLINE"] = "1"
    env["MLX_DISABLE_COMPILE"] = "1"
    env.update(extra)
    return env


def run_leg(args, stem, arm, workload, retry_of):
    """One bench_matrix leg; returns metrics dict or None on digest
    mismatch after the built-in single retry."""
    out_json = args.out / f"{stem}.json"
    log_path = args.out / f"{stem}.log"
    manifest = json.loads((args.root / "scripts/bench_matrix.json").read_text())
    manifest["models"] = [m for m in manifest["models"]
                          if m["id"] == "qwen25-0.5b-4bit"]
    manifest["generation"]["engine_script"] = "bench_decode_identity.py"
    manifest["workloads"] = [w for w in manifest["workloads"]
                             if w["id"] == workload]
    manifest_path = args.out / "manifest-q4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cmd = [
        str(args.python), "scripts/bench_matrix.py", "--mode", "run",
        "--manifest", str(manifest_path), "--python", str(args.python),
        "--wheel", str(args.wheel), "--expect-pins",
        f"qwen25-0.5b-4bit={PIN}", "--host-label",
        f"jwm1-kv-direct-{stem}", "--timeout", "600",
        "--out", str(out_json),
    ]
    with log_path.open("w") as log:
        subprocess.run(cmd, cwd=args.root, env=base_env(ARMS[arm]),
                       stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=1200)
    data = json.loads(out_json.read_text())
    assert data["clean_check"]["status"] == "clean", data["clean_check"]
    assert data["binary_provenance"]["omarchy"]["verified"] == "match", \
        data["binary_provenance"]
    legs = [leg for leg in data["legs"] if leg["status"] == "measured"]
    if len(legs) != 1:
        print(f"{stem}: no measured leg, status "
              f"{[ (l['leg_id'], l['status']) for l in data['legs'] ]}",
              flush=True)
        return None
    leg = legs[0]
    metrics = leg["metrics"]
    digest = metrics["generated_ids_sha256_16"]
    expected_digest, expected_prompt = EXPECTED[workload]
    if metrics["prompt_tokens"] != expected_prompt:
        print(f"{stem}: prompt tokens {metrics['prompt_tokens']} != "
              f"{expected_prompt}", flush=True)
        return None
    if leg.get("contended"):
        print(f"{stem}: contended machine", flush=True)
        return None
    if digest != expected_digest:
        print(f"{stem}: DIGEST MISMATCH {digest} != {expected_digest}",
              flush=True)
        return None
    metrics["digest"] = digest
    return metrics


def run_leg_with_retry(args, stem, arm, workload):
    metrics = run_leg(args, stem, arm, workload, retry_of=None)
    retries = 0
    while metrics is None and retries < 1:
        retries += 1
        print(f"{stem}: retry {retries} after mismatch/contension",
              flush=True)
        time.sleep(5)
        metrics = run_leg(args, f"{stem}.retry{retries}", arm, workload,
                          retry_of=retries)
    assert metrics is not None, f"{stem}: digest gate failed after retry"
    metrics["retries"] = retries
    return metrics


def main():
    root = Path(sys.argv[1]).resolve()
    python = Path(sys.argv[2])
    wheel = Path(sys.argv[3])
    class A:
        pass
    args = A()
    args.root = root
    args.python = python if python.is_absolute() else root / python
    args.wheel = wheel if wheel.is_absolute() else root / wheel
    args.out = Path(sys.argv[4])
    pairs = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "wheel": args.wheel.name,
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "expected_digests": {k: v[0] for k, v in EXPECTED.items()},
        "arm_env": ARMS,
        "order": [],
        "runs": [],
    }
    grouped = {arm: {w: [] for w in EXPECTED} for arm in ARMS}
    for pair in range(1, pairs + 1):
        order = ("fork", "stock") if pair % 2 else ("stock", "fork")
        summary["order"].append({"pair": pair, "arms": list(order)})
        for arm in order:
            stem = f"pair{pair}-{arm}"
            run = {"pair": pair, "arm": arm, "legs": []}
            for workload in EXPECTED:
                metrics = run_leg_with_retry(args, f"{stem}-{workload}",
                                             arm, workload)
                run["legs"].append({
                    "workload": workload,
                    "prompt_tokens": metrics["prompt_tokens"],
                    "decode_tok_s": metrics["decode_tok_s"],
                    "prefill_tok_s": metrics.get("prefill_tok_s"),
                    "digest": metrics["digest"],
                    "retries": metrics["retries"],
                })
                grouped[arm][workload].append(metrics["decode_tok_s"])
                print(f"{stem}-{workload} ok "
                      f"{metrics['decode_tok_s']} tok/s "
                      f"retries={metrics['retries']}", flush=True)
            summary["runs"].append(run)
    summary["medians"] = {
        arm: {w: statistics.median(v) for w, v in grouped[arm].items()}
        for arm in ARMS
    }
    summary["fork_change_pct"] = {
        w: round((summary["medians"]["fork"][w] /
                  summary["medians"]["stock"][w] - 1) * 100, 4)
        for w in EXPECTED
    }
    summary["all_digests_canonical"] = all(
        leg["digest"] == EXPECTED[leg["workload"]][0]
        for run in summary["runs"] for leg in run["legs"])
    (args.out / "paired-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["medians"], sort_keys=True), flush=True)
    print(json.dumps(summary["fork_change_pct"], sort_keys=True), flush=True)
    print("PAIRED_DONE", flush=True)


if __name__ == "__main__":
    main()

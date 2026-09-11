#!/usr/bin/env python3
"""Paired fork/stock decode legs for the Q4 split-K receipt (2026-09-11).

Four arms: {fork, stock} x {base, splitk}. Base = default binaries; splitk = MLX_OMARCHY_QMM_VEC_Q4_SPLITK=<S (--splits)>. Canonical digests are asserted on every
base leg and RECORDED (never asserted) on splitk legs: the splitk digests are
the datum this receipt measures. Run under the GPU lock on a quiet M1:
  python3 receipts-workdata/run_splitk_legs.py --wheel <wheel> --reps 3
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
EXPECTED_BASE = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}
NATIVE = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "254d73fd93164b98",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}
STOCK_ICD = "/home/joshuawarren/stock-mesa/stock-icd.json"


def arms_for(splits):
    env = {"MLX_OMARCHY_QMM_VEC_Q4_SPLITK": str(splits)}
    return [
        ("fork", "base", None),
        ("fork", "splitk", dict(env)),
        ("stock", "base", {"VK_DRIVER_FILES": STOCK_ICD}),
        ("stock", "splitk", {
            "VK_DRIVER_FILES": STOCK_ICD,
            **env}),
    ]


def loadavg():
    return round(os.getloadavg()[0], 2)


def quiet_gate():
    samples = []
    for _ in range(3):
        la = loadavg()
        samples.append(la)
        if la >= 1.0:
            return False, samples
        time.sleep(2)
    return True, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[1])
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "legs")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--splits", type=int, default=4, choices=[2, 4, 8])
    args = ap.parse_args()
    ARMS = arms_for(args.splits)

    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((root / "scripts/bench_matrix.json").read_text())
    manifest["models"] = [m for m in manifest["models"]
                          if m["id"] == "qwen25-0.5b-4bit"]
    manifest["generation"]["engine_script"] = "bench_decode_identity.py"
    manifest_path = args.out / "manifest-q4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    ok, samples = quiet_gate()
    if not ok:
        raise SystemExit(f"quiet gate failed, loadavg samples {samples}")

    summary = {"native_baseline_tok_s": {
                   "short-decode-32": 150.57,
                   "long-decode-128": 146.77,
                   "longctx-1024-decode-32": 140.38},
               "native_digests": NATIVE, "expected_base": EXPECTED_BASE,
               "runs": []}
    for rep in range(1, args.reps + 1):
        for driver, variant, extra in ARMS:
            stem = f"r{rep}-{driver}-{variant}"
            out_json = args.out / f"{stem}.json"
            log_path = args.out / f"{stem}.log"
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("MLX_")}
            env.pop("VK_DRIVER_FILES", None)
            env.pop("HK_PERFTEST", None)
            env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1")
            env.update({k: v for k, v in (extra or {}).items()})
            cmd = [
                str(args.python), "scripts/bench_matrix.py", "--mode",
                "run", "--manifest", str(manifest_path), "--python",
                str(args.python), "--wheel", str(args.wheel),
                "--expect-pins", f"qwen25-0.5b-4bit={PIN}",
                "--host-label", f"jwm1-splitk-{driver}-{variant}",
                "--timeout", "600", "--out", str(out_json),
            ]
            la = loadavg()
            with log_path.open("w") as log:
                log.write(f"loadavg_1m_before={la}\n")
                log.flush()
                subprocess.run(cmd, cwd=root, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True,
                               timeout=1800)
            data = json.loads(out_json.read_text())
            assert data["clean_check"]["status"] == "clean", \
                data["clean_check"]
            assert data["binary_provenance"]["omarchy"]["verified"] == \
                "match", data["binary_provenance"]
            legs = [l for l in data["legs"] if l["status"] == "measured"]
            assert len(legs) == 3, [(l["leg_id"], l["status"])
                                    for l in data["legs"]]
            for leg in legs:
                metrics = leg["metrics"]
                digest = metrics["generated_ids_sha256_16"]
                workload = leg["workload_id"]
                if variant == "base":
                    assert digest == EXPECTED_BASE[workload], \
                        (workload, digest, "base digest moved")
                run = {
                    "rep": rep, "driver": driver, "variant": variant,
                    "workload": workload,
                    "prompt_tokens": metrics["prompt_tokens"],
                    "decode_tok_s": metrics["decode_tok_s"],
                    "prefill_tok_s": metrics.get("prefill_tok_s"),
                    "prefill_s": metrics.get("prefill_s"),
                    "digest": digest,
                    "loadavg_1m_before": la,
                }
                summary["runs"].append(run)
                print(json.dumps(run), flush=True)

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("SPLITK_LEGS_DONE", flush=True)


if __name__ == "__main__":
    main()

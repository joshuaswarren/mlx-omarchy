#!/usr/bin/env python3
"""Fork/stock six-leg digest run at 30/262/1053 prompt tokens, plus an
unpacked-GEMV-group screen, for the Q4 row-packing negative receipt.

Arms:
  fork      default driver (VK_DRIVER_FILES unset), three q4 workloads
  stock     stock Mesa ICD via VK_DRIVER_FILES, same three workloads
  unpacked  fork driver with MLX_OMARCHY_FUSED_GEMV=0, short workload
            only: same-x decode GEMV groups disabled, so the q/k/v and
            gate/up rows go back to one dispatch each. Positive control:
            digests must stay canonical while decode tok/s drops.

Every measured leg asserts the exact canonical generated-ids digest.
Adapted from receipts/2026-09-10-decode-bound-3/paired_barriers.py.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
EXPECTED = {
    "short-decode-32": ("7fd25a869ff21678", 30),
    "long-decode-128": ("4cc08910089477fd", 262),
    "longctx-1024-decode-32": ("7da83f06ec9f001d", 1053),
}
STOCK_ICD = "/home/joshuawarren/stock-mesa/stock-icd.json"
IDENTITY = (
    "import json,os; import mlx.core as mx; i=mx.device_info(); "
    "print(json.dumps({'cooperative_matrix_f32_8': i.get("
    "'cooperative_matrix_f32_8'), 'device_name': i.get('device_name'), "
    "'mlx_version': mx.__version__, 'VK_DRIVER_FILES': os.environ.get("
    "'VK_DRIVER_FILES')}, sort_keys=True))"
)


def base_env(driver):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MLX_", "HK_"))}
    env.pop("VK_DRIVER_FILES", None)
    env.pop("PYTHONPATH", None)
    env["MESA_SHADER_CACHE_DISABLE"] = "true"
    env["HF_HUB_OFFLINE"] = "1"
    env["MLX_DISABLE_COMPILE"] = "1"
    if driver == "stock":
        env["VK_DRIVER_FILES"] = STOCK_ICD
    return env


def identity(python, driver):
    out = subprocess.run([str(python), "-c", IDENTITY], env=base_env(driver),
                         capture_output=True, text=True, check=True,
                         timeout=120).stdout.strip().splitlines()[-1]
    info = json.loads(out)
    expected_coop = 0 if driver == "stock" else 1
    assert info["cooperative_matrix_f32_8"] == expected_coop, info
    return info


def manifest_for(root, out_dir, workloads):
    manifest = json.loads(
        (root / "scripts" / "bench_matrix.json").read_text())
    manifest["models"] = [m for m in manifest["models"]
                          if m["id"] == "qwen25-0.5b-4bit"]
    manifest["generation"]["engine_script"] = "bench_decode_identity.py"
    if workloads is not None:
        manifest["workloads"] = [w for w in manifest["workloads"]
                                 if w["id"] in workloads]
    path = out_dir / "manifest-q4.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def run_arm(args, name, driver, workloads, extra_env):
    python = args.python
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    info = identity(python, driver)
    env = base_env(driver)
    env.update(extra_env)
    manifest_path = manifest_for(args.root, out_dir, workloads)
    cmd = [
        str(python), "scripts/bench_matrix.py", "--mode", "run",
        "--manifest", str(manifest_path), "--python", str(python),
        "--wheel", str(args.wheel), "--expect-pins",
        f"qwen25-0.5b-4bit={PIN}", "--host-label",
        f"jwm1-q4-row-packing-{name}", "--timeout", "600",
        "--out", str(out_dir / "run.json"),
    ]
    with (out_dir / "run.log").open("w") as log:
        subprocess.run(cmd, cwd=args.root, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=1500)
    data = json.loads((out_dir / "run.json").read_text())
    assert data["clean_check"]["status"] == "clean", data["clean_check"]
    assert data["binary_provenance"]["omarchy"]["verified"] == "match", \
        data["binary_provenance"]
    legs = [leg for leg in data["legs"] if leg["status"] == "measured"]
    assert len(legs) == len(workloads), [
        (leg["leg_id"], leg["status"]) for leg in data["legs"]]
    rows = []
    for leg in legs:
        metrics = leg["metrics"]
        digest = metrics["generated_ids_sha256_16"]
        expected_digest, expected_prompt = EXPECTED[leg["workload_id"]]
        assert digest == expected_digest, (leg["workload_id"], digest)
        assert metrics["prompt_tokens"] == expected_prompt, (
            leg["workload_id"], metrics["prompt_tokens"])
        rows.append({
            "workload": leg["workload_id"],
            "prompt_tokens": metrics["prompt_tokens"],
            "decode_tokens": metrics["decode_tokens"],
            "decode_tok_s": metrics["decode_tok_s"],
            "prefill_tok_s": metrics.get("prefill_tok_s"),
            "digest": digest,
            "contended": leg.get("contended"),
        })
    print(f"{name} ok: " + json.dumps(
        {r["workload"]: r["decode_tok_s"] for r in rows}, sort_keys=True),
        flush=True)
    return {
        "arm": name,
        "driver": driver,
        "env_fingerprint": {
            key: env[key] for key in sorted(env)
            if key in ("VK_DRIVER_FILES", "MLX_DISABLE_COMPILE",
                       "MLX_OMARCHY_FUSED_GEMV", "HF_HUB_OFFLINE",
                       "MESA_SHADER_CACHE_DISABLE")
        },
        "device_identity": info,
        "clean_check": data["clean_check"],
        "provenance": data["binary_provenance"]["omarchy"]["verified"],
        "legs": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip(),
        "wheel": args.wheel.name,
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "expected_digests": {k: v[0] for k, v in EXPECTED.items()},
        "arms": [],
    }
    summary["arms"].append(run_arm(args, "fork", "fork", set(EXPECTED), {}))
    summary["arms"].append(
        run_arm(args, "stock", "stock", set(EXPECTED), {}))
    summary["arms"].append(
        run_arm(args, "unpacked", "fork", {"short-decode-32"},
                {"MLX_OMARCHY_FUSED_GEMV": "0"}))
    by_name = {arm["arm"]: arm for arm in summary["arms"]}
    summary["digests"] = {
        arm: {row["workload"]: row["digest"]
              for row in by_name[arm]["legs"]}
        for arm in ("fork", "stock", "unpacked")
    }
    summary["decode_tok_s"] = {
        arm: {row["workload"]: row["decode_tok_s"]
              for row in by_name[arm]["legs"]}
        for arm in ("fork", "stock", "unpacked")
    }
    short_fork = by_name["fork"]["legs"][0]["decode_tok_s"]
    summary["unpacked_screen"] = {
        "fork_short_decode_tok_s": short_fork,
        "unpacked_short_decode_tok_s":
            by_name["unpacked"]["legs"][0]["decode_tok_s"],
        "change_pct": round(
            (by_name["unpacked"]["legs"][0]["decode_tok_s"] / short_fork
             - 1) * 100, 4),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["digests"], sort_keys=True), flush=True)
    print(json.dumps(summary["decode_tok_s"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

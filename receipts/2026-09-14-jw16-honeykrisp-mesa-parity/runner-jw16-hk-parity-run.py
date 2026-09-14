#!/usr/bin/env python3
"""jw16 Q4 decode/prefill + fp16 matmul capture under one held GPU lock.

Same protocol as receipts/2026-09-14-jw16-gpu-parity-rerun.md: fresh
subprocess per leg, MLX_DISABLE_COMPILE=1, greedy temp 0, seed 0, EOS
suppressed, 32 pinned tokens, 4 warmup tokens, bench_matrix prompts.
Repeated REPS times per leg, alternating legs, so the receipt can quote a
median instead of one warm run.

Usage: python3 this.py <phase-label> [reps]
Emits one JSON document on stdout.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HOME = Path("/home/joshuawarren")
REPO = HOME / "src/mlx-omarchy"
SCRIPTS = REPO / "scripts"
PY = HOME / ".local/share/mlx-omarchy-test-venv/bin/python"
WHEEL = REPO / "dist/mlx_omarchy-0.32.2.dev202609122106+b41e2b74-cp314-cp314-linux_aarch64.whl"
MODEL = (HOME / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit"
         / "snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")
LOCK = "/tmp/m1-gpu.lock"

sys.path.insert(0, str(SCRIPTS))
from bench_matrix import prompt_text  # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {"cmd": cmd, "rc": p.returncode,
            "out": p.stdout.strip(), "err": p.stderr.strip()[:400]}


def power():
    vals = {}
    for f in sorted(Path("/sys/class/power_supply").glob("*/online")):
        vals[str(f)] = f.read_text().strip()
    for f in sorted(Path("/sys/class/power_supply").glob("*/status")):
        vals[str(f)] = f.read_text().strip()
    return vals


def leg(prompt_id, manifest):
    env = dict(os.environ)
    env["MLX_DISABLE_COMPILE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [str(PY), str(SCRIPTS / "bench_decode.py"),
           "--model", str(MODEL),
           "--prompt", prompt_text(manifest, prompt_id),
           "--tokens", "32", "--temp", "0.0", "--seed", "0",
           "--warmup-tokens", "4", "--wheel", str(WHEEL)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    rec = {"prompt_id": prompt_id, "rc": p.returncode,
           "wall_s": round(time.time() - t0, 3),
           "stdout": p.stdout, "stderr": p.stderr[-800:]}
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            rec["json"] = json.loads(line)
    return rec


def matmul():
    """2048^2 fp16 matmul(a,b)+bias: 3 warmups, 20 measured medians."""
    code = r'''
import json, statistics, time
import mlx.core as mx
mx.set_default_device(mx.gpu)
N = 2048
mx.random.seed(20260913)
a = mx.random.normal((N, N)).astype(mx.float16)
b = mx.random.normal((N, N)).astype(mx.float16)
bias = mx.zeros((N,), dtype=mx.float16)
ea = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float16)
eb = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float16)
ebias = mx.array([1.0, 1.0], dtype=mx.float16)
exact = (mx.matmul(ea, eb) + ebias)
mx.eval(exact)
warm, meas = [], []
for i in range(23):
    t0 = time.perf_counter_ns()
    c = mx.matmul(a, b) + bias
    mx.eval(c)
    dt = (time.perf_counter_ns() - t0) / 1e6
    (warm if i < 3 else meas).append(round(dt, 6))
med = statistics.median(meas)
print(json.dumps({"warmups_ms": warm, "measured_ms": meas,
                  "median_ms": med,
                  "tflops": (2 * N ** 3 + N ** 2) / (med / 1e3) / 1e12,
                  "exact_2x2": exact.tolist(),
                  "device": str(mx.default_device())}))
'''
    env = dict(os.environ)
    env["MLX_DISABLE_COMPILE"] = "1"
    p = subprocess.run([str(PY), "-c", code], capture_output=True, text=True,
                       env=env)
    rec = {"rc": p.returncode, "stderr": p.stderr[-600:]}
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            rec.update(json.loads(line))
    return rec


def main():
    manifest = json.loads((SCRIPTS / "bench_matrix.json").read_text())
    out = {
        "phase": PHASE,
        "reps": REPS,
        "host": platform.node(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "started_unix": int(time.time()),
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lock_path": LOCK,
        "lock_inode": os.stat(LOCK).st_ino,
        "nested_flock_rc": subprocess.run(
            ["flock", "-n", LOCK, "-c", "true"]).returncode,
        "power_before": power(),
        "driver": sh("vulkaninfo --summary 2>/dev/null | "
                     "grep -E 'driverName|driverInfo|deviceName|apiVersion'"),
        "mesa_pkg": sh("pacman -Q mesa mesa-honeykrisp-omarchy 2>&1"),
        "icd_dir": sh("ls -la /usr/share/vulkan/icd.d/"),
        "vk_env": {k: v for k, v in os.environ.items()
                   if k.startswith(("VK_", "AGX_", "MESA_"))},
        "wheel_sha256": sh(f"sha256sum {WHEEL}")["out"],
        "harness_sha256": sh(f"sha256sum {SCRIPTS}/bench_decode.py "
                             f"{SCRIPTS}/bench_matrix.json")["out"],
        "llm_service": sh("systemctl is-active llm-inference.service")["out"],
        "legs": [],
    }
    for rep in range(1, REPS + 1):
        for pid in ("short", "ctx1024"):
            rec = leg(pid, manifest)
            rec["rep"] = rep
            out["legs"].append(rec)
            out[f"power_after_{pid}_{rep}"] = power()
    out["matmul"] = matmul()
    out["power_after"] = power()
    out["finished_unix"] = int(time.time())

    summary = {}
    for pid in ("short", "ctx1024"):
        js = [l["json"] for l in out["legs"]
              if l["prompt_id"] == pid and l.get("json")]
        if not js:
            continue
        summary[pid] = {
            "n": len(js),
            "decode_tps": [j["decode_tps"] for j in js],
            "prefill_tps": [j["prefill_tps"] for j in js],
            "decode_tps_median": statistics.median(j["decode_tps"] for j in js),
            "prefill_tps_median": statistics.median(j["prefill_tps"] for j in js),
            "prompt_tokens": sorted({j["prompt_tokens"] for j in js}),
            "ids_sha256_16": sorted({j["ids_sha256_16"] for j in js}),
            "device": sorted({j["device"] for j in js}),
        }
    out["summary"] = summary
    print(json.dumps(out, indent=1, sort_keys=True))


main()

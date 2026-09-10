#!/usr/bin/env python3
"""Profile QmmPrefillCoopmatF16 at the four Qwen projection shapes."""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

SHAPES = [
    (262, 896, 896),
    (262, 896, 4864),
    (262, 4864, 896),
    (262, 896, 128),
    (1053, 896, 896),
    (1053, 896, 4864),
    (1053, 4864, 896),
    (1053, 896, 128),
]

WORKER = r'''
import json, sys, time
import mlx.core as mx
import numpy as np
out, warmups, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(17)
saved = {}
for m, k, n in json.loads(sys.argv[4]):
    x = mx.array(rng.standard_normal((m, k)).astype(np.float16))
    w = mx.array(rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint32))
    s = mx.array(rng.uniform(-1, 1, size=(n, k // 64)).astype(np.float16))
    b = mx.array(rng.uniform(-1, 1, size=(n, k // 64)).astype(np.float16))
    mx.eval(x, w, s, b)
    def qmm():
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(*[qmm() for _ in range(warmups)])
    t0 = time.perf_counter_ns()
    ys = [qmm() for _ in range(reps)]
    mx.eval(*ys)
    wall_ms = (time.perf_counter_ns() - t0) / 1e6
    saved[f"m{m}_k{k}_n{n}"] = np.asarray(ys[-1]).view(np.uint16)
    print(json.dumps({"m": m, "k": k, "n": n, "wall_ms_per_eval": wall_ms / reps}), flush=True)
np.savez(out, **saved)
'''


def main():
    python, out_dir, repo = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    warmups = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = out_dir / "profile.jsonl"
    npz = out_dir / "outputs.npz"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
    env.update({
        "MLX_OMARCHY_GPU_PROFILE": str(profile),
        "MESA_SHADER_CACHE_DISABLE": "true",
    })
    run = subprocess.run(
        [str(python), "-c", WORKER, str(npz), str(warmups), str(reps), json.dumps(SHAPES)],
        env=env, check=True, capture_output=True, text=True, timeout=1800,
    )
    (out_dir / "run.log").write_text(run.stdout + run.stderr)
    sys.path.insert(0, str(repo / "scripts"))
    from profile_analyze import parse_kernel_names
    names = parse_kernel_names(repo / "overlay/mlx/backend/omarchy/compute.h")
    rows = [json.loads(line) for line in profile.read_text().splitlines()]
    period_ns = next(row["period_ns"] for row in rows if row.get("k") == "meta")
    dispatches = [row for row in rows if row.get("k") == "d" and names[row["e"]] == "QmmPrefillCoopmatF16"]
    results = []
    per_shape = warmups + reps
    if len(dispatches) != len(SHAPES) * per_shape:
        raise RuntimeError(("dispatch count", len(dispatches)))
    for i, (m, k, n) in enumerate(SHAPES):
        hits = dispatches[i * per_shape + warmups:(i + 1) * per_shape]
        if any(row["n"] != m * n for row in hits):
            raise RuntimeError((m, k, n, [row["n"] for row in hits]))
        samples = [(row["t1"] - row["t0"]) * period_ns / 1e6 for row in hits]
        if len(samples) != reps:
            raise RuntimeError((m, k, n, len(samples)))
        results.append({
            "m": m, "k": k, "n": n, "kernel": "QmmPrefillCoopmatF16",
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
        })
    report = {"warmups": warmups, "reps": reps, "results": results}
    (out_dir / "kernel-times.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

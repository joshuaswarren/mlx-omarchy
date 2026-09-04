#!/usr/bin/env python3
"""Single-primitive shape probes for the wedge bisection.

This isolates each primitive that scales with sequence length, recording
whether the single mx.eval at given (q_len, ...) wedges. The wedge
candidate list is:

  - SDPA scores buffer (one big f32 tensor, scales as q_len*k_len)
  - The QK^T matmul dispatch (workgroup count scales with q_len*k_len)
  - The softmax reduction (axis scales with k_len)
  - The PV matmul (sizes scale with q_len, v_len)
  - Layernorm / RoPE buffers

Each probe is a fresh subprocess because the wedge poisons state.

Usage on Honeykrisp:
  wedge_primitive_probe.py single --python ... --output-dir ... --n 2048
  wedge_primitive_probe.py sweep --python ... --output-dir ...
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap

CHILD_SDPA = textwrap.dedent('''
    import json, os, sys, time
    import numpy as np
    import mlx.core as mx

    n = int(sys.argv[1])
    out_path = sys.argv[2]

    # Qwen 0.5B-Instruct shape: 14 query heads, 2 kv heads, head_dim 64.
    # Single-batch SDPA forward: B=1, H=14, KVH=2, q_len=k_len=n, D=64.
    # Repeats = H/KVH = 7.
    H, KVH, D = 14, 2, 64
    mx.random.seed(0)
    q = mx.random.normal((1, H, n, D))
    k = mx.random.normal((1, KVH, n, D))
    v = mx.random.normal((1, KVH, n, D))

    if int(os.environ.get("MLX_DISABLE_COMPILE", "1")):
        mx.disable_compile()
    else:
        mx.enable_compile()

    t0 = time.monotonic()
    try:
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / (D ** 0.5), mask=None)
        mx.eval(out)
        dt = time.monotonic() - t0
        np.savez(out_path, out=np.asarray(out.astype(mx.float32)))
        json.dump({
            "status": "ok", "wall_s": dt, "n": n,
            "shape": [1, H, n, D],
            "scores_bytes": 1 * KVH * (H // KVH) * n * n * 4,
        }, open(out_path + ".json", "w"))
    except Exception as ex:
        dt = time.monotonic() - t0
        msg = str(ex)
        is_wedge = "failed to advance" in msg or "did not complete" in msg
        json.dump({
            "status": "wedge" if is_wedge else "error",
            "wall_s": dt, "n": n, "error": msg[:2000],
        }, open(out_path + ".json", "w"))
        sys.exit(124 if is_wedge else 1)
''')


CHILD_QK_MATMUL = textwrap.dedent('''
    import json, os, sys, time
    import numpy as np
    import mlx.core as mx

    n = int(sys.argv[1])
    out_path = sys.argv[2]

    H, KVH, D = 14, 2, 64
    mx.random.seed(0)
    # QK^T scores shape [1, KVH, H/KVH, n, n] in f32.
    # Use a fused-equivalent matmul on a flat [B*K, n, D] @ [B*K, D, n].
    qs = mx.random.normal((KVH * (H // KVH), n, D))
    kt = mx.random.normal((KVH * (H // KVH), D, n))
    scores = mx.matmul(qs, kt)
    if int(os.environ.get("MLX_DISABLE_COMPILE", "1")):
        mx.disable_compile()
    else:
        mx.enable_compile()
    t0 = time.monotonic()
    try:
        mx.eval(scores)
        dt = time.monotonic() - t0
        np.savez(out_path, scores=np.asarray(scores.astype(mx.float32)))
        json.dump({
            "status": "ok", "wall_s": dt, "n": n,
            "scores_bytes": int(scores.nbytes),
        }, open(out_path + ".json", "w"))
    except Exception as ex:
        dt = time.monotonic() - t0
        msg = str(ex)
        is_wedge = "failed to advance" in msg or "did not complete" in msg
        json.dump({
            "status": "wedge" if is_wedge else "error",
            "wall_s": dt, "n": n, "error": msg[:2000],
        }, open(out_path + ".json", "w"))
        sys.exit(124 if is_wedge else 1)
''')


CHILD_SOFTMAX = textwrap.dedent('''
    import json, os, sys, time
    import numpy as np
    import mlx.core as mx

    n = int(sys.argv[1])
    out_path = sys.argv[2]

    H, KVH, D = 14, 2, 64
    mx.random.seed(0)
    # The QK^T post-softmax input shape [1, KVH, H/KVH, n, n] f32.
    x = mx.random.normal((1, KVH, H // KVH, n, n))
    if int(os.environ.get("MLX_DISABLE_COMPILE", "1")):
        mx.disable_compile()
    else:
        mx.enable_compile()
    t0 = time.monotonic()
    try:
        out = mx.softmax(x, axis=-1)
        mx.eval(out)
        dt = time.monotonic() - t0
        np.savez(out_path, out=np.asarray(out.astype(mx.float32)))
        json.dump({
            "status": "ok", "wall_s": dt, "n": n,
            "scores_bytes": int(x.nbytes),
        }, open(out_path + ".json", "w"))
    except Exception as ex:
        dt = time.monotonic() - t0
        msg = str(ex)
        is_wedge = "failed to advance" in msg or "did not complete" in msg
        json.dump({
            "status": "wedge" if is_wedge else "error",
            "wall_s": dt, "n": n, "error": msg[:2000],
        }, open(out_path + ".json", "w"))
        sys.exit(124 if is_wedge else 1)
''')


def _run(python, body, args_for_child, out_path, *,
         watchdog_ns=10_000_000_000, wall_timeout_s=60, log_path=None):
    env = dict(os.environ)
    env["MLX_OMARCHY_HANG_NO_PROGRESS_NS"] = str(int(watchdog_ns))
    env["MLX_OMARCHY_MAX_WALL_NS"] = str(int(watchdog_ns))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    child = out_path + ".py"
    with open(child, "w") as f:
        f.write(body)
    cmd = [python, child] + list(args_for_child) + [out_path]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=wall_timeout_s)
        if log_path:
            with open(log_path, "w") as f:
                f.write(r.stdout)
                f.write("\n--- stderr ---\n")
                f.write(r.stderr)
    except subprocess.TimeoutExpired:
        return {"status": "killed", "wall_s": wall_timeout_s}, 124
    try:
        meta = json.load(open(out_path + ".json"))
    except Exception:
        meta = {"status": "no_meta",
                "stdout_tail": r.stdout[-400:],
                "stderr_tail": r.stderr[-400:]}
    return meta, r.returncode


def cmd_single(args):
    body = {"sdpa": CHILD_SDPA, "qk_matmul": CHILD_QK_MATMUL,
            "softmax": CHILD_SOFTMAX}[args.primitive]
    out = os.path.join(args.output_dir, f"{args.primitive}_n{args.n:05d}.npz")
    log = os.path.join(args.output_dir, f"{args.primitive}_n{args.n:05d}.log")
    meta, code = _run(args.python, body, [args.n], out,
                      watchdog_ns=args.watchdog_ns,
                      wall_timeout_s=args.wall_timeout_s, log_path=log)
    meta["exit"] = code
    print(json.dumps(meta, indent=2))


def cmd_sweep(args):
    body = {"sdpa": CHILD_SDPA, "qk_matmul": CHILD_QK_MATMUL,
            "softmax": CHILD_SOFTMAX}[args.primitive]
    points = sorted({int(x) for x in args.points.split(",") if x.strip()})
    results = []
    for n in points:
        out = os.path.join(args.output_dir, f"{args.primitive}_n{n:05d}.npz")
        log = os.path.join(args.output_dir, f"{args.primitive}_n{n:05d}.log")
        meta, code = _run(args.python, body, [n], out,
                          watchdog_ns=args.watchdog_ns,
                          wall_timeout_s=args.wall_timeout_s, log_path=log)
        meta["exit"] = code
        results.append(meta)
        kind = meta.get("status", "?")
        wall = meta.get("wall_s", -1)
        print(f"  n={n:>5} status={kind:<6} wall={wall:6.2f}s exit={code}",
              flush=True)
    summary = os.path.join(args.output_dir,
                           f"{args.primitive}_summary.json")
    json.dump(results, open(summary, "w"), indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--watchdog-ns", type=int, default=10_000_000_000)
    p.add_argument("--wall-timeout-s", type=int, default=120)
    p.add_argument("--primitive", required=True,
                   choices=["sdpa", "qk_matmul", "softmax"])
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("single")
    ps.add_argument("--n", type=int, required=True)
    ps.set_defaults(func=cmd_single)

    pw = sub.add_parser("sweep")
    pw.add_argument(
        "--points",
        default="128,256,512,768,1024,1280,1536,1800,2000,2048,2200,"
                "2400,3000,4096",
    )
    pw.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Wedge bisection harness for the single-full-sequence-eval hang.

Discriminator (receipts/2026-09-04-hang-watchdog-hardware.md):
  differential_compile --mode realpath --steps 1 (one mx.eval over the full
  2,048-token eager forward) wedges Honeykrisp at the 10 s watchdog with
  the counter frozen at 0. mlx_lm.generate at 6,009 tokens completes in
  183 s. Trigger is ONE submission's shape, not total work.

Two-phase bisect, every probe a fresh subprocess because the wedge
poisons the process.

Phase 1 (threshold): sweep prompt length. The smallest N that wedges
gives the threshold.

Phase 2 (single-factor): at the threshold, vary exactly one of {compile
on/off, watchdog interval, n_tokens} to confirm a single-factor reading.

Usage on Honeykrisp:
  wedge_bisect.py phase1 --python /path/to/python --model /path/to/qwen \
      --output-dir ~/benchq/wedge
  wedge_bisect.py phase2 --python /path/to/python --model /path/to/qwen \
      --threshold 2048 --output-dir ~/benchq/wedge
  wedge_bisect.py single --python /path/to/python --model /path/to/qwen \
      --tokens 2048 --output-dir ~/benchq/wedge

Exit code per probe: 0 = OK, 124 = wedge (watchdog or wall), 1 = other
error. Process stdout/stderr are captured to <probe>.log.
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap

CHILD_TEMPLATE = textwrap.dedent('''
    import json, os, sys, time
    import numpy as np

    import mlx.core as mx

    model_path = sys.argv[1]
    n_tokens = int(sys.argv[2])
    out_path = sys.argv[3]
    disable_compile = bool(int(sys.argv[4]))

    mx.random.seed(0)

    from mlx_lm.utils import load
    model, tok = load(model_path)

    # Build a deterministic prompt of exactly n_tokens. Reuse the same
    # in-distribution prefix until we hit the target length. The wedge
    # shape is a full-sequence forward; an out-of-distribution padding
    # token would still exercise every primitive, but real ids rule out
    # any "first out-of-vocab token triggers it" theory.
    prompt = "What is the capital of France?"
    ids = tok.encode(prompt)
    while 0 < len(ids) < n_tokens:
        ids = ids + ids[: n_tokens - len(ids)]
    ids = ids[:n_tokens]

    if disable_compile:
        mx.disable_compile()
    else:
        mx.enable_compile()

    t0 = time.monotonic()
    try:
        logits = model(mx.array([ids]), cache=None)
        mx.eval(logits)
        dt = time.monotonic() - t0
        out = np.asarray(logits[:, -1, :].astype(mx.float32))
        np.savez(out_path, logits=out)
        json.dump({
            "status": "ok",
            "wall_s": dt,
            "n_tokens": n_tokens,
            "disable_compile": disable_compile,
        }, open(out_path + ".json", "w"))
    except Exception as ex:
        dt = time.monotonic() - t0
        msg = str(ex)
        is_wedge = (
            "failed to advance" in msg
            or "did not complete" in msg
        )
        json.dump({
            "status": "wedge" if is_wedge else "error",
            "wall_s": dt,
            "n_tokens": n_tokens,
            "disable_compile": disable_compile,
            "error": msg[:2000],
        }, open(out_path + ".json", "w"))
        sys.exit(124 if is_wedge else 1)
''')


def run_one(python, model, n_tokens, out_path, *, disable_compile,
            watchdog_ns=10_000_000_000, wall_timeout_s=60, log_path=None):
    """Run one probe in a fresh subprocess. Returns (meta, exit_code)."""
    env = dict(os.environ)
    env["MLX_OMARCHY_HANG_NO_PROGRESS_NS"] = str(int(watchdog_ns))
    env["MLX_OMARCHY_MAX_WALL_NS"] = str(int(watchdog_ns))
    if disable_compile:
        env["MLX_DISABLE_COMPILE"] = "1"
    else:
        env.pop("MLX_DISABLE_COMPILE", None)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    child = out_path + ".py"
    with open(child, "w") as f:
        f.write(CHILD_TEMPLATE)
    cmd = [python, child, model, str(n_tokens), out_path,
           "1" if disable_compile else "0"]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=wall_timeout_s)
        if log_path:
            with open(log_path, "w") as f:
                f.write(r.stdout)
                f.write("\n--- stderr ---\n")
                f.write(r.stderr)
    except subprocess.TimeoutExpired:
        return {"status": "killed", "wall_s": wall_timeout_s,
                "n_tokens": n_tokens, "disable_compile": disable_compile}, 124
    try:
        meta = json.load(open(out_path + ".json"))
    except Exception:
        meta = {"status": "no_meta", "stdout_tail": r.stdout[-400:],
                "stderr_tail": r.stderr[-400:]}
    return meta, r.returncode


def _print_result(name, meta, code):
    kind = meta.get("status", "?")
    wall = meta.get("wall_s", -1)
    err = meta.get("error", "")
    print(f"  {name:<28} status={kind:<6} wall={wall:6.2f}s exit={code}"
          f"{'  err=' + err[:80] if err else ''}", flush=True)


def phase1(args):
    points = sorted({int(x) for x in args.points.split(",") if x.strip()})
    print(f"phase1: sweeping {len(points)} token lengths", flush=True)
    results = []
    for n in points:
        out = os.path.join(args.output_dir, f"phase1_n{n:05d}.npz")
        log = os.path.join(args.output_dir, f"phase1_n{n:05d}.log")
        meta, code = run_one(
            args.python, args.model, n, out,
            disable_compile=True,  # eager by default - matches receipt
            watchdog_ns=args.watchdog_ns,
            wall_timeout_s=args.wall_timeout_s,
            log_path=log,
        )
        meta["exit"] = code
        results.append(meta)
        _print_result(f"n={n}", meta, code)
    summary = os.path.join(args.output_dir, "phase1_summary.json")
    json.dump(results, open(summary, "w"), indent=2)
    threshold = next(
        (r["n_tokens"] for r in results if r.get("status") == "wedge"),
        None,
    )
    print(f"phase1: first wedge at n_tokens={threshold}")
    return threshold


def phase2(args):
    print(f"phase2: single-factor probes at threshold={args.threshold}",
          flush=True)
    factors = [
        ("baseline_eager", dict(disable_compile=True,
                                n_tokens=args.threshold)),
        ("baseline_compiled", dict(disable_compile=False,
                                   n_tokens=args.threshold)),
        ("threshold_quarter", dict(disable_compile=True,
                                   n_tokens=max(1, args.threshold // 4))),
        ("threshold_double", dict(disable_compile=True,
                                  n_tokens=args.threshold * 2)),
        ("short_watchdog", dict(disable_compile=True,
                                n_tokens=args.threshold,
                                watchdog_ns=2_000_000_000)),
    ]
    results = []
    for name, kw in factors:
        out = os.path.join(args.output_dir, "phase2", f"{name}.npz")
        log = os.path.join(args.output_dir, "phase2", f"{name}.log")
        n_tokens = kw.pop("n_tokens")
        meta, code = run_one(
            args.python, args.model, n_tokens, out,
            watchdog_ns=kw.pop("watchdog_ns", args.watchdog_ns),
            disable_compile=kw.pop("disable_compile"),
            wall_timeout_s=args.wall_timeout_s,
            log_path=log,
        )
        meta["factor"] = name
        meta["exit"] = code
        results.append(meta)
        _print_result(name, meta, code)
    summary = os.path.join(args.output_dir, "phase2", "phase2_summary.json")
    json.dump(results, open(summary, "w"), indent=2)


def single(args):
    out = os.path.join(args.output_dir, "single.npz")
    log = os.path.join(args.output_dir, "single.log")
    meta, code = run_one(
        args.python, args.model, args.tokens, out,
        disable_compile=args.disable_compile,
        watchdog_ns=args.watchdog_ns,
        wall_timeout_s=args.wall_timeout_s,
        log_path=log,
    )
    print(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--watchdog-ns", type=int, default=10_000_000_000)
    p.add_argument("--wall-timeout-s", type=int, default=120)
    p.add_argument("--disable-compile", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("phase1")
    p1.add_argument(
        "--points",
        default="1,16,32,64,128,256,512,768,1024,1280,1536,1800,1900,"
                "2000,2048,2100,2200,2400,3000,4096",
    )
    p1.set_defaults(func=phase1)

    p2 = sub.add_parser("phase2")
    p2.add_argument("--threshold", type=int, required=True)
    p2.set_defaults(func=phase2)

    ps = sub.add_parser("single")
    ps.add_argument("--tokens", type=int, required=True)
    ps.set_defaults(func=single)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

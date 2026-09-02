#!/usr/bin/env python3
# bf16 compiled-tape REUSE probe — isolates the "cached tape invoked
# repeatedly with varying inputs and interleaved eager work" variable.
# M1Bf16Tape, 2026-09-02. No commits.
#
# Variants, all with ONE compiled function created once:
#   A: same arrays every call (narrow probe's original shape)
#   B: fresh random arrays of the same shape every call
#   C: fresh arrays + an eager matmul+add chain interleaved between calls
#   D: two different compiled functions interleaved
# Each call is compared bit-for-bit against the eager result for those
# exact inputs. Reports the first invocation where divergence appears.
import functools
import json
import os
import sys

import mlx.core as mx
import numpy as np

DIM = 896


def banner(msg):
    sys.stderr.write(f"[bf16tape_reuse] {msg}\n")
    sys.stderr.flush()


def silu_mul_eager(x, y):
    neg_x = mx.negative(x)
    e = mx.exp(neg_x)
    one_plus = mx.add(e, 1.0)
    sig = mx.divide(1.0, one_plus)
    return mx.multiply(mx.multiply(x, sig), y)


@functools.partial(mx.compile, shapeless=True)
def swiglu_a(gate, x):
    return silu_mul_eager(gate, x)


@functools.partial(mx.compile, shapeless=True)
def swiglu_b(gate, x):
    return silu_mul_eager(gate, x)


def fresh(seed):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.normal(size=(1, 4, DIM)).astype(np.float32)).astype(mx.bfloat16)
    y = mx.array(rng.normal(size=(1, 4, DIM)).astype(np.float32)).astype(mx.bfloat16)
    return x, y


def check(out, x, y):
    ref = silu_mul_eager(x, y)
    mx.eval(ref)
    o = np.asarray(out.astype(mx.float32))
    r = np.asarray(ref.astype(mx.float32))
    return int(np.sum(o != r))


def main():
    info = mx.device_info()
    banner(f"device={info.get('device_name')!r} v={mx.__version__} nproc_env={os.environ.get('MLX_DISABLE_COMPILE')}")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    results = {}

    # Variant A: same arrays every call.
    x, y = fresh(0)
    mx.eval(x); mx.eval(y)
    bad_a = None
    for i in range(n):
        d = check(swiglu_a(x, y), x, y)
        if d > 0 and bad_a is None:
            bad_a = i
            banner(f"A: FIRST divergence at invocation {i} diff={d}")
    results["A_same_arrays"] = bad_a
    banner(f"A same-arrays: {'CLEAN' if bad_a is None else f'diverged at {bad_a}'}")

    # Variant B: fresh arrays every call.
    bad_b = None
    for i in range(n):
        x, y = fresh(100 + i)
        d = check(swiglu_a(x, y), x, y)
        if d > 0 and bad_b is None:
            bad_b = i
            banner(f"B: FIRST divergence at invocation {i} diff={d}")
    results["B_fresh_arrays"] = bad_b
    banner(f"B fresh-arrays: {'CLEAN' if bad_b is None else f'diverged at {bad_b}'}")

    # Variant C: fresh arrays + interleaved eager matmul chain.
    w = mx.array(np.random.default_rng(7).normal(size=(DIM, DIM)).astype(np.float32)).astype(mx.bfloat16)
    bad_c = None
    for i in range(n):
        x, y = fresh(200 + i)
        inter = mx.add(mx.matmul(x, w), 1.0)  # eager interleaved work
        mx.eval(inter)
        d = check(swiglu_a(x, y), x, y)
        if d > 0 and bad_c is None:
            bad_c = i
            banner(f"C: FIRST divergence at invocation {i} diff={d}")
    results["C_interleaved_eager"] = bad_c
    banner(f"C interleaved: {'CLEAN' if bad_c is None else f'diverged at {bad_c}'}")

    # Variant D: two compiled functions interleaved, fresh arrays.
    bad_d = None
    for i in range(n):
        x, y = fresh(300 + i)
        d1 = check(swiglu_a(x, y), x, y)
        x2, y2 = fresh(400 + i)
        d2 = check(swiglu_b(x2, y2), x2, y2)
        if (d1 > 0 or d2 > 0) and bad_d is None:
            bad_d = i
            banner(f"D: FIRST divergence at invocation {i} d1={d1} d2={d2}")
    results["D_two_fns"] = bad_d
    banner(f"D two-fns: {'CLEAN' if bad_d is None else f'diverged at {bad_d}'}")

    summary = {
        "kind": "reuse",
        "n_per_variant": n,
        "first_divergence": results,
        "reproduced": any(v is not None for v in results.values()),
    }
    sys.stdout.write(json.dumps(summary) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# bf16 compiled-tape defect bisect — MLP-replicating probe.
# M1Bf16Tape, 2026-09-02. No commits.
#
# Replicates the mlx-lm Qwen2 MLP path: bf16 Linear -> compiled swiglu ->
# bf16 Linear, matching hidden/intermediate sizes from the model, with the
# Linear weights pinned to the same shape/seed the real model uses. Compares
# the compiled chain output against a fully-eager baseline.
#
# Usage: python bf16tape_probe_mlp.py [N_REPS]
import functools
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def banner(msg):
    sys.stderr.write(f"[bf16tape_mlp] {msg}\n")
    sys.stderr.flush()


@functools.partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    neg_x = mx.negative(gate)
    e = mx.exp(neg_x)
    one_plus = mx.add(e, 1.0)
    sig = mx.divide(1.0, one_plus)
    silu_gate = mx.multiply(gate, sig)
    return mx.multiply(silu_gate, x)


def mlp_eager(gate_proj, up_proj, down_proj, x):
    g = gate_proj(x)
    u = up_proj(x)
    s = swiglu_eager(g, u)
    return down_proj(s)


def swiglu_eager(gate, x):
    neg_x = mx.negative(gate)
    e = mx.exp(neg_x)
    one_plus = mx.add(e, 1.0)
    sig = mx.divide(1.0, one_plus)
    silu_gate = mx.multiply(gate, sig)
    return mx.multiply(silu_gate, x)


def mlp_compiled(gate_proj, up_proj, down_proj, x):
    """Mimics what mlx-lm's MLP.__call__ actually does: compiled swiglu,
    eager matmuls."""
    g = gate_proj(x)
    u = up_proj(x)
    s = swiglu(g, u)  # this is the compiled fragment
    return down_proj(s)


def main():
    info = mx.device_info()
    sys.stderr.write(f"[bf16tape_mlp] device_name={info.get('device_name')!r} arch={info.get('architecture')!r}\n")
    sys.stderr.write(f"[bf16tape_mlp] v={mx.__version__} default={mx.default_device()}\n")

    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    banner(f"reps={n_reps}")

    dim = 896
    hidden_dim = 4864

    # Build bf16 Linear modules with deterministic seed.
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    gate_proj.weight = gate_proj.weight.astype(mx.bfloat16)
    up_proj.weight = up_proj.weight.astype(mx.bfloat16)
    down_proj.weight = down_proj.weight.astype(mx.bfloat16)

    # Use a deterministic bf16 input.
    np.random.seed(0)
    x_np = np.random.normal(size=(1, 4, dim)).astype(np.float32)
    x = mx.array(x_np).astype(mx.bfloat16)
    mx.eval(x)

    # Eager reference.
    y_eager = mlp_eager(gate_proj, up_proj, down_proj, x)
    mx.eval(y_eager)
    eager_np = np.asarray(y_eager.astype(mx.float32))
    banner(f"eager shape={list(y_eager.shape)} sample[0:3]={eager_np.flatten()[:3].tolist()}")

    mismatches = 0
    nondet_runs = 0
    last_np = None
    first_sample = None

    for i in range(n_reps):
        y = mlp_compiled(gate_proj, up_proj, down_proj, x)
        mx.eval(y)
        y_np = np.asarray(y.astype(mx.float32))
        diff = int(np.sum(y_np != eager_np))
        nan_mask = np.isnan(y_np) & ~np.isnan(eager_np)
        diff -= int(np.sum(nan_mask))
        nans = int(np.sum(nan_mask))
        if diff > 0:
            mismatches += 1
            banner(f"iter {i} diff_count={diff} nans={nans}")
        if first_sample is None:
            first_sample = y_np.flatten()[:5].tolist()
        if last_np is not None:
            if not np.array_equal(y_np, last_np, equal_nan=False):
                nondet_runs += 1
                banner(f"iter {i} nondet vs iter {i-1}")
        last_np = y_np

    summary = {
        "kind": "mlp_chain",
        "dtype": "bfloat16",
        "n_reps": n_reps,
        "mismatches": mismatches,
        "nondet_runs": nondet_runs,
        "eager_sample": eager_np.flatten()[:5].tolist(),
        "first_iter_sample": first_sample,
        "compiled_env_disable": os.environ.get("MLX_DISABLE_COMPILE"),
    }
    sys.stdout.write(json.dumps(summary) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# bf16 compiled-tape defect bisect — on-device runner for jwm1.
# M1Bf16Tape, 2026-09-02. No commits.
#
# What this does
# --------------
# 1. End-to-end smoke: 8-tok greedy bf16 mlx-lm generation with MLX_DISABLE_COMPILE=1
#    (must match the fox-jumps baseline from the receipts).
# 2. End-to-end repro: 32-tok bf16 mlx-lm generation, MLX_DISABLE_COMPILE UNSET
#    and the compiled-tape gate lifted on the wheel. Capture full output.
# 3. Narrow-from-mlx-lim: replicate the swiglu fragment in pure Python and run
#    it many times via mx.compile, against eager. Count mismatches and check
#    for nondeterminism (different garbage per run with identical inputs).
# 4. Narrow further: each fusable op (Negative, Exp, Add, Divide, Multiply,
#    Sigmoid primitive) wrapped in @partial(mx.compile, shapeless=True), run
#    N times, check vs eager.
# 5. f16 control: identical chain in fp16 — if f16 stays clean, the defect is
#    bf16-specific and lives in the bf16 kernel variant, NOT in the tape
#    machinery.
#
# Usage
# -----
#   MLX_DISABLE_COMPILE=1 .work/venv-fp16/bin/python receipts/bf16tape_probe.py smoke
#   .work/venv-fp16/bin/python receipts/bf16tape_probe.py mlxlm_repro NREPS
#   .work/venv-fp16/bin/python receipts/bf16tape_probe.py narrow DTYPE NREPS
#   .work/venv-fp16/bin/python receipts/bf16tape_probe.py ops DTYPE NREPS
#
# Each invocation appends a single line of JSON-ish output for the parent
# aggregator to consume. No fancy framework — kernel-evaluate per repetition
# is the only thing that matters.
import functools
import json
import os
import sys
import time

import numpy as np

import mlx.core as mx


def banner(msg):
    sys.stderr.write(f"[bf16tape_probe] {msg}\n")
    sys.stderr.flush()


def device_check():
    info = mx.device_info()
    sys.stderr.write(f"[bf16tape_probe] device_name={info.get('device_name')!r} arch={info.get('architecture')!r}\n")
    sys.stderr.write(f"[bf16tape_probe] v={mx.__version__} default={mx.default_device()}\n")
    sys.stderr.flush()


def fixed_input_bf16(seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(1, 4, 896)).astype(np.float32)
    b = rng.normal(size=(1, 4, 896)).astype(np.float32)
    return mx.array(a).astype(mx.bfloat16), mx.array(b).astype(mx.bfloat16)


def fixed_input_f16(seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(1, 4, 896)).astype(np.float16)
    b = rng.normal(size=(1, 4, 896)).astype(np.float16)
    return mx.array(a), mx.array(b)


def silu_then_mul(x, y):
    # nn.silu(x) * y  with the upstream decomposition
    neg_x = mx.negative(x)
    e = mx.exp(neg_x)
    one_plus = mx.add(e, 1.0)
    sig = mx.divide(1.0, one_plus)
    silu_x = mx.multiply(x, sig)
    return mx.multiply(silu_x, y)


def silu_then_mul_eager(x, y):
    neg_x = mx.negative(x)
    e = mx.exp(neg_x)
    one_plus = mx.add(e, 1.0)
    sig = mx.divide(1.0, one_plus)
    silu_x = mx.multiply(x, sig)
    return mx.multiply(silu_x, y)


@functools.lru_cache(maxsize=None)
def _compiled_silu_mul():
    @functools.partial(mx.compile, shapeless=True)
    def f(x, y):
        return silu_then_mul(x, y)
    return f


def run_narrow(dtype: str, n_reps: int):
    """Compiled chain vs eager, repeated n_reps times, exact bit comparison."""
    if dtype == "bfloat16":
        x, y = fixed_input_bf16(0)
    elif dtype == "float16":
        x, y = fixed_input_f16(0)
    else:
        raise ValueError(dtype)

    # Eager reference (computed once).
    eager_out = silu_then_mul_eager(x, y)
    mx.eval(eager_out)
    eager_first = eager_out

    f = _compiled_silu_mul()

    mismatches = 0
    nans_in_mismatch = 0
    distinct_compiled_outputs = 0
    last_compiled = None
    first_compiled = None

    sys.stderr.write(f"[bf16tape_probe] run_narrow start dtype={dtype} n_reps={n_reps}\n"); sys.stderr.flush()
    for i in range(n_reps):
        sys.stderr.write(f"[bf16tape_probe] iter {i} call\n"); sys.stderr.flush()
        out = f(x, y)
        mx.eval(out)
        sys.stderr.write(f"[bf16tape_probe] iter {i} eval ok\n"); sys.stderr.flush()
        # Compare element-wise via numpy (device-side bool Sum/Copy not
        # on the supported kernel list yet). Read out eagerly. bf16
        # cannot go through np.asarray directly; cast to float32 first.
        out_np = np.asarray(out.astype(mx.float32)) if out.dtype == mx.bfloat16 else np.asarray(out)
        eager_np = np.asarray(eager_first.astype(mx.float32)) if eager_first.dtype == mx.bfloat16 else np.asarray(eager_first)
        nan_mask = np.isnan(out_np) & ~np.isnan(eager_np)
        # NaN != x is True under numpy; subtract the NaN-mismatch
        # entries so diff_count reports only "garbage" elements.
        diff_count = int(np.sum(out_np != eager_np)) - int(np.sum(nan_mask))
        nan_in_mismatch = int(np.sum(nan_mask))

        if diff_count > 0:
            mismatches += 1
            nans_in_mismatch += nan_in_mismatch

        if last_compiled is not None:
            last_np = np.asarray(last_compiled.astype(mx.float32)) if last_compiled.dtype == mx.bfloat16 else np.asarray(last_compiled)
            if not np.array_equal(out_np, last_np, equal_nan=False):
                distinct_compiled_outputs += 1
        last_compiled = out

    sys.stderr.write(f"[bf16tape_probe] run_narrow done mismatches={mismatches}\n"); sys.stderr.flush()
    summary = {
        "kind": "narrow",
        "dtype": dtype,
        "n_reps": n_reps,
        "mismatches": mismatches,
        "nans_in_mismatch": nans_in_mismatch,
        "distinct_compiled_outputs": distinct_compiled_outputs,
        "compiled_env_disable": os.environ.get("MLX_DISABLE_COMPILE"),
    }
    sys.stdout.flush()


def run_single_op(dtype: str, n_reps: int):
    """Each fusable op alone, compiled, repeated."""
    if dtype == "bfloat16":
        x, y = fixed_input_bf16(0)
    elif dtype == "float16":
        x, y = fixed_input_f16(0)
    else:
        raise ValueError(dtype)

    cases = {
        "negative": lambda: mx.negative(x),
        "exp":      lambda: mx.exp(x),
        "add_1":    lambda: mx.add(x, 1.0),
        "divide_1": lambda: mx.divide(1.0, mx.add(mx.exp(mx.negative(x)), 1.0)),
        "multiply": lambda: mx.multiply(x, y),
        "sigmoid":  lambda: mx.divide(1.0, mx.add(mx.exp(mx.negative(x)), 1.0)),
    }

    out_lines = []
    for name, fn in cases.items():
        sys.stderr.write(f"[bf16tape_probe] op={name} building eager\n"); sys.stderr.flush()
        # eager ref
        eager = fn()
        mx.eval(eager)
        sys.stderr.write(f"[bf16tape_probe] op={name} eager ok shape={list(eager.shape)}\n"); sys.stderr.flush()

        # compile
        compiled_fn = mx.compile(fn, shapeless=True)
        sys.stderr.write(f"[bf16tape_probe] op={name} compiled fn built\n"); sys.stderr.flush()
        mismatches = 0
        nondet = 0
        last_out = None
        for _ in range(n_reps):
            c = compiled_fn()
            sys.stderr.write(f"[bf16tape_probe] op={name} iter got c dtype={c.dtype}\n"); sys.stderr.flush()
            mx.eval(c)
            sys.stderr.write(f"[bf16tape_probe] op={name} iter eval ok\n"); sys.stderr.flush()
            c_np = np.asarray(c.astype(mx.float32)) if c.dtype == mx.bfloat16 else np.asarray(c)
            e_np = np.asarray(eager.astype(mx.float32)) if eager.dtype == mx.bfloat16 else np.asarray(eager)
            nan_mask = np.isnan(c_np) & ~np.isnan(e_np)
            diff = int(np.sum(c_np != e_np)) - int(np.sum(nan_mask))
            sys.stderr.write(f"[bf16tape_probe] op={name} iter diff={diff}\n"); sys.stderr.flush()
            if diff > 0:
                mismatches += 1
            if last_out is not None:
                last_np = np.asarray(last_out.astype(mx.float32)) if last_out.dtype == mx.bfloat16 else np.asarray(last_out)
                if not np.array_equal(c_np, last_np, equal_nan=False):
                    nondet += 1
            last_out = c
        out_lines.append(json.dumps({
            "kind": "single_op",
            "dtype": dtype,
            "op": name,
            "n_reps": n_reps,
            "mismatches": mismatches,
            "nondet_runs": nondet,
            "compiled_env_disable": os.environ.get("MLX_DISABLE_COMPILE"),
        }))
    sys.stdout.write("\n".join(out_lines) + "\n")
    sys.stdout.flush()

def run_smoke():
    """MLX_DISABLE_COMPILE=1 greedy smoke for the fox-jumps baseline."""
    import subprocess
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", "/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx",
        "--prompt", "Hi",
        "--max-tokens", "8",
        "--temp", "0",
    ]
    env = dict(os.environ)
    env["MLX_DISABLE_COMPILE"] = "1"
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    return r.returncode


def run_mlxlm_repro(n_reps: int):
    """mlx_lm bf16 with compile ON, repeated n_reps times. Capture outputs."""
    import subprocess
    outputs = []
    for i in range(n_reps):
        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", "/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx",
            "--prompt", "Hi",
            "--max-tokens", "32",
            "--temp", "0",
            "--seed", "0",
        ]
        env = dict(os.environ)
        env.pop("MLX_DISABLE_COMPILE", None)
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        dt = time.time() - t0
        outputs.append({
            "iter": i,
            "returncode": r.returncode,
            "wall_time_s": round(dt, 2),
            "stdout": r.stdout,
            "stderr_tail": r.stderr[-2000:] if r.stderr else "",
        })
    sys.stdout.write(json.dumps({"kind": "mlxlm_repro", "n_reps": n_reps, "outputs": outputs}) + "\n")
    sys.stdout.flush()


def main():
    device_check()
    if len(sys.argv) < 2:
        sys.stderr.write("usage: bf16tape_probe.py {smoke|mlxlm_repro N|narrow DTYPE N|ops DTYPE N}\n")
        sys.exit(2)
    mode = sys.argv[1]
    if mode == "smoke":
        rc = run_smoke()
        sys.exit(rc)
    if mode == "mlxlm_repro":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        run_mlxlm_repro(n)
        return
    if mode == "narrow":
        dtype = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        run_narrow(dtype, n)
        return
    if mode == "ops":
        dtype = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        run_single_op(dtype, n)
        return
    sys.stderr.write(f"unknown mode {mode!r}\n")
    sys.exit(2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Ordering probe: which path does a compiled call take, and when.

Companion to docs/differential-harness.md and the auto-eager hook in
overlay/mlx/backend/omarchy/device.cpp. The runtime disables compilation
at device discovery on real Apple GPUs and keeps the tape-interpreter
refusal as a backstop. This probe measures which path actually fires for
each ordering of mx.compile against the first device touch, so the
behaviour is observed rather than reasoned about.

The discriminator: a bf16 tape is refused with the named
`[omarchy] Compiled tape bfloat16` error, while eager bf16 add is
implemented and correct. So "tape error" means the compiled tape path
ran, and "output" means the call ran eager.

Scenarios, each in a fresh subprocess because the compile switch is
process-global:

  compile-first      mx.compile before any array or device touch; this is
                     the ordering the auto-eager hook must protect.
  array-first        one array created before mx.compile; the device is
                     discovered and the hook fires before compiling.
  forced-eager       mx.disable_compile() before mx.compile(); isolates
                     the upstream flag semantics the hook relies on.
  pre-armed-then-    mx.compile() first, then mx.disable_compile(); the
  disable            deterministic form of a hook that fires after a
                     function was already armed.

Output: one line per scenario naming the path that fired. Exit 0 when
the probe ran to completion - finding the compiled path is a finding,
not a failure. Exit 2 when mlx is missing, 3 on a self-test failure.

Self-test (no mlx): python3 probe_compile_ordering.py --self-test
"""

import argparse
import subprocess
import sys

TAPE_BF16 = "[omarchy] Compiled tape bfloat16"
TAPE_REFUSED = "[omarchy] Compiled tapes are refused"
LABEL_BF16 = "fused bf16 tape built"
LABEL_REFUSED = "fail-closed backstop fired"

CHILD_TEMPLATE = """
import json
import sys

import mlx.core as mx

scenario = sys.argv[1]


def build():
    # Two fusable nodes: the tracer only builds a real tape (and a
    # Compiled primitive) for multi-node graphs, so a single add here
    # would run eager and prove nothing.
    return mx.compile(lambda a: mx.abs(mx.round(mx.multiply(a, a))))
def run_and_print(fn, x):
    out = fn(x)
    mx.eval(out)
    print("RESULT: " + json.dumps([float(v) for v in out]))


x = mx.array([0.5], dtype=mx.bfloat16)
if scenario == "compile-first":
    fn = build()
    run_and_print(fn, x)
elif scenario == "array-first":
    fn = build()
    run_and_print(fn, x)
elif scenario == "forced-eager":
    mx.disable_compile()
    run_and_print(build(), x)
elif scenario == "pre-armed-then-disable":
    fn = build()
    mx.disable_compile()
    run_and_print(fn, x)
else:
    raise SystemExit(f"unknown scenario {scenario}")
"""


def classify(proc):
    """One-line verdict for a child run."""
    if proc.returncode == 0:
        return f"eager path (output {proc.stdout.strip()})"
    if TAPE_REFUSED in proc.stderr:
        return f"COMPILED PATH: {LABEL_REFUSED}"
    if TAPE_BF16 in proc.stderr:
        return f"COMPILED PATH: {LABEL_BF16}"
    tail = (proc.stderr or "").strip().splitlines()
    return "unexpected: " + (tail[-1] if tail else f"exit {proc.returncode}")


def self_test():
    ok = classify(
        subprocess.CompletedProcess([], 0, stdout="RESULT: [1.0]", stderr="")
    ).startswith("eager")
    ok = ok and LABEL_BF16 in classify(
        subprocess.CompletedProcess([], 1, stdout="", stderr=f"boot\n{TAPE_BF16}\n")
    )
    ok = ok and LABEL_REFUSED in classify(
        subprocess.CompletedProcess([], 1, stdout="", stderr=f"boot\n{TAPE_REFUSED}\n")
    )
    print("self-test: classifier checks pass" if ok else "self-test: FAILED")
    return 0 if ok else 3


def main():
    p = argparse.ArgumentParser(description="compile ordering probe")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print("mlx is not importable; run inside the wheel venv", file=sys.stderr)
        return 2

    print(f"mlx.core version: {__import__('mlx').core.__version__}")
    scenarios = [
        "compile-first",
        "array-first",
        "forced-eager",
        "pre-armed-then-disable",
    ]
    for scenario in scenarios:
        proc = subprocess.run(
            [sys.executable, "-c", CHILD_TEMPLATE, scenario],
            capture_output=True,
            text=True,
        )
        print(f"{scenario:24s} -> {classify(proc)}")

    print(
        "note: a fused tape after a disable means the function was armed\n"
        "before the disable and upstream still fuses at trace time; see\n"
        "docs/known-defects.md for the consequence on real Apple GPUs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

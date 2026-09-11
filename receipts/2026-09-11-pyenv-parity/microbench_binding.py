#!/usr/bin/env python3
"""Binding-cost decomposition microbench (host-paced decode support).

Three eager numbers, same interpreter and wheel as the decode legs:
  ctor_ns/op   - time to BUILD a lazy op graph node (x = x + 1), no eval
  eval_ns/op   - amortized mx.eval cost of the built graph (1 node)
  attr_ns/op   - pure-python attribute/dispatch loop (no C++ binding call)
No model load; backend init happens on import (no GPU work until eval).
"""
import argparse, time
import mlx.core as mx

def timed(fn, n):
    t0 = time.perf_counter_ns()
    fn(n)
    return (time.perf_counter_ns() - t0) / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = a.ops
    x = mx.ones((1, 128), mx.float32)

    def ctor_loop(k):
        y = x
        for _ in range(k):
            y = y + 1.0
        return y
    y = ctor_loop(4); mx.eval(y)          # warm
    ctor = timed(lambda k: mx.eval(ctor_loop(k)), n)

    def eval_loop(k):
        z = x + 1.0
        for _ in range(k):
            mx.eval(z)
    eval_loop(4)
    ev = timed(eval_loop, n)

    class Box: __slots__ = ("v",)
    b = Box(); b.v = 0
    def py_loop(k):
        acc = 0
        for i in range(k):
            acc += b.v
        return acc
    py_loop(4)
    attr = timed(py_loop, n)

    import sys, json
    res = {"python": sys.version.split()[0], "ops": n,
           "ctor_ns_per_op": round(ctor, 1),
           "eval_ns_per_op": round(ev, 1),
           "pyloop_ns_per_op": round(attr, 1)}
    print(json.dumps(res))
    if a.out:
        open(a.out, "w").write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()

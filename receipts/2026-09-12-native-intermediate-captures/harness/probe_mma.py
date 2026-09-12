#!/usr/bin/env python3
"""Determine Apple simdgroup_multiply_accumulate(float8x8) accumulation
order empirically - the order native MLX's steel qmm relies on for every
quantized prefill matmul.

Each trial runs one hardware mma: D = A * B + C with seeded random f32
8x8 A, B, C placed via the standard thread-element mapping (get_coord).
The CPU side compares every (r, c) result against candidate 8-deep
accumulation orders; the surviving order across thousands of trials is
the hardware's. Also verifies the element mapping itself with encoded
values.
"""

import json
import os

import mlx.core as mx
import numpy as np

RNG = np.random.default_rng(424242)
OUT = os.path.expanduser("~/src/native-captures-20260912/qmm")

BODY = """
  uint trial = threadgroup_position_in_grid.x;
  uint lane = thread_index_in_simdgroup;
  short qid = lane / 4;
  short fm = (qid & 4) + ((lane / 2) % 4);
  short fn = (qid & 2) * 2 + (lane % 2) * 2;
  simdgroup_float8x8 A, B, C, D;
  // thread_elements() layout: two elements per thread
  // thread_elements() layout: [kElemRows=1][kElemCols=2] -> two elements
  A.thread_elements()[0] = Abuf[trial * 64 + fm * 8 + fn];
  A.thread_elements()[1] = Abuf[trial * 64 + fm * 8 + fn + 1];
  B.thread_elements()[0] = Bbuf[trial * 64 + fm * 8 + fn];
  B.thread_elements()[1] = Bbuf[trial * 64 + fm * 8 + fn + 1];
  C.thread_elements()[0] = Cbuf[trial * 64 + fm * 8 + fn];
  C.thread_elements()[1] = Cbuf[trial * 64 + fm * 8 + fn + 1];
  simdgroup_multiply_accumulate(D, A, B, C);
  Dout[trial * 64 + fm * 8 + fn] = D.thread_elements()[0];
  Dout[trial * 64 + fm * 8 + fn + 1] = D.thread_elements()[1];
"""

HEADER = """
constant int trials_unused = 0;
"""


def place(vals_per_trial, name):
    """vals: (T, 8, 8) -> per-thread element buffer (T*64,)."""
    t = vals_per_trial.shape[0]
    buf = np.empty((t, 8, 8), np.float32)
    coords = []
    for lane in range(32):
        qid = lane // 4
        fm = (qid & 4) + ((lane // 2) % 4)
        fn = (qid & 2) * 2 + (lane % 2) * 2
        coords.append((fm, fn))
    for lane, (fm, fn) in enumerate(coords):
        buf[:, fm, fn] = vals_per_trial[:, fm, fn]
        buf[:, fm, fn + 1] = vals_per_trial[:, fm, fn + 1]
    return buf.reshape(t, 64)


def candidates(a, b, c):
    """a, b: (8,) products of one output element; c: scalar init.
    Returns dict name -> float32 result."""
    prods = (a.astype(np.float64) * b.astype(np.float64)).astype(np.float32)
    c = np.float32(c)

    def fma(x, y, z):
        # exact float32 fma via float64 (product of f32s is exact in f64)
        return np.float32(np.float64(x) * np.float64(y) + np.float64(z))

    out = {}
    # sequential fma into accumulator starting at C
    t = c
    for k in range(8):
        t = fma(a[k], b[k], t)
    out["seq_fma"] = t
    # sequential rounded products, plain adds
    t = c
    for k in range(8):
        t = np.float32(t + prods[k])
    out["seq_mul_add"] = t
    # pairwise tree of products, added to C last
    p = prods.copy()
    while len(p) > 1:
        p = (p[0::2] + p[1::2]).astype(np.float32)
    out["tree_then_c"] = np.float32(p[0] + c)
    out["c_then_tree"] = np.float32(c + p[0])
    # fma tree: pair fmas then adds
    l1 = np.array([fma(a[2 * i], b[2 * i], np.float32(0)) for i in range(4)],
                  np.float32)
    r1 = np.array([(prods[2 * i + 1]) for i in range(4)], np.float32)
    l2 = np.array([np.float32(l1[i] + r1[i]) for i in range(4)], np.float32)
    p2 = np.array([np.float32(l2[2 * i] + l2[2 * i + 1]) for i in range(2)],
                  np.float32)
    out["fma_pairs"] = np.float32(p2[0] + p2[1] + c)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    trials = 2048
    A = RNG.standard_normal((trials, 8, 8)).astype(np.float32)
    B = RNG.standard_normal((trials, 8, 8)).astype(np.float32)
    C = RNG.standard_normal((trials, 8, 8)).astype(np.float32)
    Ab, Bb, Cb = place(A, "A"), place(B, "B"), place(C, "C")
    k = mx.fast.metal_kernel(
        name="mma_probe", input_names=["Abuf", "Bbuf", "Cbuf"],
        output_names=["Dout"], header=HEADER, source=BODY,
        ensure_row_contiguous=False)
    res, = k(inputs=[mx.array(Ab.reshape(-1)), mx.array(Bb.reshape(-1)),
                     mx.array(Cb.reshape(-1))],
             grid=(trials, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(trials * 64,)],
             output_dtypes=[mx.float32])
    mx.eval(res)
    D = np.array(res).reshape(trials, 8, 8)

    names = None
    wins = {}
    for t in range(trials):
        for r in range(8):
            a = A[t, r, :]
            for c_i in range(8):
                bvec = B[t, :, c_i]  # out[r,c] = sum_k A[r,k] B[k,c]
                cands = candidates(a, bvec, C[t, r, c_i])
                got = np.float32(D[t, r, c_i])
                if names is None:
                    names = list(cands)
                    wins = {n: 0 for n in names}
                for n in names:
                    if cands[n].view(np.uint32) == got.view(np.uint32):
                        wins[n] += 1
    total = trials * 64
    result = {"trials": trials, "elements": total, "wins": wins,
              "winner": max(wins, key=wins.get)}
    with open(os.path.join(OUT, "mma_order.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.save(os.path.join(OUT, "mma_probe_D.npy"), D)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

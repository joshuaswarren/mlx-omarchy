#!/usr/bin/env python3
"""Determine Metal simd_sum's binary combine tree empirically.

Method: per threadgroup, place non-zero values at exactly three lanes
(i, j, k); all other lanes contribute +0.0f, which cannot change any
partial sum. The result equals (a+b)+c, (a+c)+b, or (b+c)+a in float32 -
which one reveals which pair combined at the deeper level of the
reduction tree. Every triple (i,j,k) of the 32 lanes is probed in one
launch; the tree is reconstructed offline from the first-pair relation.

Also banks a 32x32 pairwise matrix under the same trick (2 non-zeros ->
commutative, control only) and quad-level structure checks.
"""

import json
import os
import sys

import mlx.core as mx
import numpy as np

OUT = os.path.expanduser("~/src/native-captures-20260912/attention")
RNG = np.random.default_rng(777)


def run_groups(vals):
    """vals: (G, 32) float32 -> simd_sum per group."""
    g = vals.shape[0]
    x = mx.array(vals.reshape(-1))
    src = (
        "uint lid = thread_position_in_threadgroup.x;"
        " uint grp = threadgroup_position_in_grid.x;"
        " float v = x[grp*32 + lid];"
        " y[grp*32 + lid] = simd_sum(v);")
    kernel = mx.fast.metal_kernel(name="probe_sum_order",
                                  input_names=["x"], output_names=["y"],
                                  source=src)
    y, = kernel(inputs=[x], grid=(g, 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[x.shape], output_dtypes=[mx.float32])
    mx.eval(y)
    sums = np.array(y.astype(mx.float32)).reshape(g, 32)
    assert (sums == sums[:, :1]).all()
    return sums[:, 0]


def first_pair(a, b, c):
    """Which order was applied: returns label of the reduction applied."""
    ab = np.float32(np.float32(a) + np.float32(b))
    ac = np.float32(np.float32(a) + np.float32(c))
    bc = np.float32(np.float32(b) + np.float32(c))
    return {
        "(ij)k": np.float32(ab + np.float32(c)),
        "(ik)j": np.float32(ac + np.float32(b)),
        "(jk)i": np.float32(bc + np.float32(a)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    lanes = list(range(32))
    triples = [(i, j, k) for i in lanes for j in lanes if i < j
               for k in lanes if k > j]
    g = len(triples)
    print(f"probing {g} triples", flush=True)
    vals = np.zeros((g, 32), np.float32)
    noise = RNG.standard_normal((g, 3)).astype(np.float32)
    for t, (i, j, k) in enumerate(triples):
        vals[t, i] = noise[t, 0]
        vals[t, j] = noise[t, 1]
        vals[t, k] = noise[t, 2]
    got = run_groups(vals)

    # per triple, which candidate orders match (values chosen so all three
    # orders differ with overwhelming probability)
    votes = np.zeros((32, 32, 32), np.int8)  # pairwise-depth relation
    labels = []
    ambiguous = 0
    for t, (i, j, k) in enumerate(triples):
        cands = first_pair(*vals[t, [i, j, k]])
        r = np.float32(got[t])
        hits = [name for name, v in cands.items()
                if v.view(np.uint32) == r.view(np.uint32)]
        labels.append(hits[0] if len(hits) == 1 else "|".join(hits))
        if len(hits) != 1:
            ambiguous += 1
    result = {
        "triples": g,
        "ambiguous": ambiguous,
        "labels_sample": labels[:20],
    }
    np.save(os.path.join(OUT, "simd_order_triple_inputs.npy"), vals)
    np.save(os.path.join(OUT, "simd_order_triple_outputs.npy"), got)
    with open(os.path.join(OUT, "simd_order_triples.json"), "w") as f:
        json.dump({"triples": [list(t) for t in triples],
                   "labels": labels, **result}, f)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cross-check microbench: per-class decode chains on the release wheel.

Builds eager op chains at the exact per-token decode shapes of
Qwen2.5-0.5B-Instruct-4bit and times mx.eval of a freshly built graph
each repetition (mlx memoizes an already-evaluated graph, so the graph
is rebuilt per rep with a perturbed input to force real GPU work). Two
windows are recorded per rep: the python graph build and the eval sync
(the GPU chain). The eval window is the GPU estimate; the build window
quantifies the host-side graph-construction cost that the same rebuild
pays once per token in real decode. A same-length tiny-add chain gives
the harness floor; the dispatch-floor receipt's 4.5 us pipelined
cadence is reported alongside. Independent instrument against the
ablation arms (run_arms.py); where they disagree, say so.

Run under the GPU lock, MLX_DISABLE_COMPILE=1, offline:
  MLX_DISABLE_COMPILE=1 python3 chain_microbench.py --out microbench.json
"""
import argparse
import json
import statistics
import time

import mlx.core as mx

FLOOR_US = 4.5  # dispatch-floor receipt: pipelined CDM launch cadence


def q4w(n, k):
    return (mx.random.randint(0, 2**31, (n, k // 8)).astype(mx.uint32),
            mx.random.normal((n, k // 64)).astype(mx.float16),
            mx.random.normal((n, k // 64)).astype(mx.float16))


def qmm(x, wsb):
    w, s, b = wsb
    return mx.quantized_matmul(x, w, s, b, transpose=True,
                               group_size=64, bits=4)


def class_builders(seq):
    """name -> (build(delta) -> outputs, dispatches per eval).

    Chains are serial where the op preserves shape (dependent dispatches,
    matching the production submission pattern); lm_head and sampler
    cannot refeed (reduction/shape change) and stay independent roots.
    """
    H, KVH, HD, L = 14, 2, 64, 24
    FF, VOCAB = 4864, 151936
    x0 = mx.random.normal((1, H * HD)).astype(mx.float16)
    q0 = mx.random.normal((1, H, 1, HD)).astype(mx.float16)
    k0 = mx.random.normal((1, KVH, 1, HD)).astype(mx.float16)
    n0 = mx.random.normal((1, 1, H * HD)).astype(mx.float16)
    g0 = mx.random.normal((1, 1, FF)).astype(mx.float16)
    u0 = mx.random.normal((1, 1, FF)).astype(mx.float16)
    lg0 = mx.random.normal((1, VOCAB)).astype(mx.float16)
    wl = q4w(VOCAB, H * HD)
    wg = q4w(FF, H * HD)
    wd = q4w(H * HD, FF)
    w = mx.random.normal((H * HD,)).astype(mx.float16)
    kt = mx.random.normal((1, KVH, seq, HD)).astype(mx.float16)
    cache = mx.zeros((1, KVH, seq + 1, HD), mx.float16)
    kn0 = mx.random.normal((1, KVH, 1, HD)).astype(mx.float16)
    tiny0 = mx.random.normal((1, H * HD)).astype(mx.float16)

    def gemv_layer(d):
        # serial gate_up -> down pairs (896 -> 4864 -> 896), dependent
        x = x0 + d
        outs = []
        for _ in range(L):
            x = qmm(qmm(x, wg), wd)
            outs.append(x)
        return outs

    def lm_head(d):
        x = x0 + d
        return [qmm(x, wl) for _ in range(L)]

    def rope(d):
        q = q0 + d
        k = k0 + d
        outs = []
        for _ in range(L):
            q = mx.fast.rope(q, HD, traditional=False, base=1000000.0,
                             scale=1.0, offset=7)
            k = mx.fast.rope(k, HD, traditional=False, base=1000000.0,
                             scale=1.0, offset=7)
            outs.append(q)
            outs.append(k)
        return outs

    def rms(d):
        nrm = n0 + d
        outs = []
        for _ in range(2 * L + 1):
            nrm = mx.fast.rms_norm(nrm, w, 1e-6)
            outs.append(nrm)
        return outs

    def sdpa(d):
        q = q0 + d
        outs = []
        for _ in range(L):
            q = mx.fast.scaled_dot_product_attention(
                q, kt, kt, scale=1.0 / HD)
            outs.append(q)
        return outs

    def kvwrite(d):
        kn = kn0 + d
        cur = cache
        outs = []
        for _ in range(2 * L):
            cur = mx.slice_update(cur, kn, mx.array([seq]), [2])
            outs.append(cur)
        return outs

    def swiglu(d):
        g = g0 + d
        outs = []
        for _ in range(L):
            g = (mx.sigmoid(g) * g) * u0
            outs.append(g)
        return outs

    def sampler(d):
        logits = lg0 + d
        return ([mx.logsumexp(logits) for _ in range(L)]
                + [mx.argmax(logits, axis=-1) for _ in range(L)])

    def harness(d):
        tiny = tiny0 + d
        outs = []
        for _ in range(L):
            tiny = tiny + 0
            outs.append(tiny)
        return outs

    counts = {"gemv_layer": 2 * L, "lm_head": L, "rope": 2 * L,
              "rms": 2 * L + 1, "sdpa": L, "kvwrite": 2 * L, "swiglu": L,
              "sampler": 2 * L, "harness": L}
    builders = {"gemv_layer": gemv_layer, "lm_head": lm_head, "rope": rope,
                "rms": rms, "sdpa": sdpa, "kvwrite": kvwrite,
                "swiglu": swiglu, "sampler": sampler, "harness": harness}
    return {n: (b, counts[n]) for n, b in builders.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="microbench.json")
    ap.add_argument("--seqs", type=int, nargs="*", default=[48, 16, 1024])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    results = {"schema": "decode-attribution-chain-microbench/1",
               "reps": args.reps, "floor_us": FLOOR_US,
               "note": "graphs rebuilt per rep with perturbed inputs "
                       "(mlx memoizes evaluated graphs); eval window is "
                       "the GPU chain estimate, build window is the host "
                       "graph-construction cost",
               "chains": {}}
    for seq in args.seqs:
        for name, (build, ndisp) in class_builders(seq).items():
            builds, evals = [], []
            for rep in range(args.warmup + args.reps):
                d = mx.array(float(rep % 17) * 1e-3, mx.float16)
                t0 = time.perf_counter()
                outs = build(d)
                t1 = time.perf_counter()
                mx.eval(outs)
                t2 = time.perf_counter()
                if rep >= args.warmup:
                    builds.append((t1 - t0) * 1e3)
                    evals.append((t2 - t1) * 1e3)
            med_eval = statistics.median(evals)
            per_us = med_eval * 1e3 / ndisp
            results["chains"][f"{name}@seq{seq}" if name == "sdpa"
                              else name] = {
                "dispatches": ndisp,
                "eval_ms_median": round(med_eval, 4),
                "eval_ms_min": round(min(evals), 4),
                "build_ms_median": round(statistics.median(builds), 4),
                "per_dispatch_us_median": round(per_us, 3),
                "per_dispatch_us_minus_receipt_floor": round(
                    per_us - FLOOR_US, 3),
                "gpu_chain_ms_est_minus_receipt_floor": round(
                    med_eval - ndisp * FLOOR_US / 1e3, 4),
            }
            print(seq, name, results["chains"][
                f"{name}@seq{seq}" if name == "sdpa" else name], flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    print("MICROBENCH_DONE", flush=True)


if __name__ == "__main__":
    main()

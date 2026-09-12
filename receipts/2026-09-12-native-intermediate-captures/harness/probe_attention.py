#!/usr/bin/env python3
"""Native MLX attention probes - macOS only.

Banks what docs/parity-id-policy.md rule 1 needs for the decode wall:
  1. fast::exp vs fast::exp2(x*log2e) I/O tables on seeded inputs.
  2. simd_sum reduction order, decided empirically against candidate orders.
  3. A verbatim-arithmetic dump copy of sdpa_vector (D=64, no mask/sinks)
     that writes per-key scores, per-key running max/sum, exp values,
     per-lane output partials, and the cross-simdgroup combine - run at the
     canonical decode shapes including KV>=1024, where M1 Max dispatch
     would pick the 2pass path.
  4. Fixed-input sdpa dispatch captures (q/k/v/out) at boundary KV lengths
     for f16/bf16/f32, whatever kernel the dispatch selects.

The dump kernel is validated against the real dispatch output on the same
fixed inputs (dispatch_out_differences must be 0 for single-pass shapes).
"""

import argparse
import hashlib
import json
import os

import mlx.core as mx
import numpy as np

RNG = np.random.default_rng(20260912)


def sha16(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def f32(a):
    """Exact widening save helper (f16/bf16 -> f32 is lossless)."""
    return np.array(a.astype(mx.float32))


def save(out, rel, arr):
    path = os.path.join(out, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(arr))


def probe_exp(out, manifest):
    n = 1 << 21
    xs = RNG.uniform(-40.0, 1.0, n).astype(np.float32)
    xs[:16] = np.array([0, -1, -2, -4, -8, -16, -32, -38.5, 1, 0.5, -1e-6,
                        -1e-3, -20, -30, -35, -39.9], np.float32)
    x = mx.array(xs)
    y, = mx.fast.metal_kernel(
        name="probe_fast_exp", input_names=["x"], output_names=["y"],
        source="y[thread_position_in_grid.x] = metal::fast::exp(x["
               "thread_position_in_grid.x]);")(
        inputs=[x], grid=(n, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32])
    mx.eval(y)
    log2e = np.float32(np.float32(1.4426950408889634))
    xs2 = mx.array(xs * log2e)
    y2, = mx.fast.metal_kernel(
        name="probe_fast_exp2", input_names=["x"], output_names=["y"],
        source="y[thread_position_in_grid.x] = metal::fast::exp2(x["
               "thread_position_in_grid.x]);")(
        inputs=[xs2], grid=(n, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32])
    mx.eval(y2)
    a, b = f32(y), f32(y2)
    save(out, "attention/exp_inputs_f32.npy", xs)
    save(out, "attention/fast_exp_outputs.npy", a)
    save(out, "attention/fast_exp2_log2e_outputs.npy", b)
    manifest["fast_exp_vs_exp2_log2e_differences"] = int(
        np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    manifest["fast_exp_sha16"] = sha16(a)


def probe_simd_sum(out, manifest):
    g = 4096
    xs = RNG.standard_normal(g * 32).astype(np.float32) * RNG.choice(
        [1.0, 1e-3, 1e3], g * 32).astype(np.float32)
    x = mx.array(xs)
    src = (
        "uint lid = thread_index_in_simdgroup;"
        " uint grp = threadgroup_position_in_grid.x;"
        " float v = x[grp*32 + lid];"
        " y[grp*32 + lid] = simd_sum(v);"
        " m[grp*32 + lid] = simd_max(v);")
    kernel = mx.fast.metal_kernel(name="probe_simd_sum",
                                  input_names=["x"], output_names=["y", "m"],
                                  source=src)
    y, m = kernel(inputs=[x], grid=(g, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[x.shape, x.shape],
                  output_dtypes=[mx.float32, mx.float32])
    mx.eval(y)
    mx.eval(m)
    sums = f32(y).reshape(g, 32)
    assert (sums == sums[:, :1]).all(), "simd_sum not uniform across lanes"
    got = sums[:, 0]

    def cand(fn):
        res = np.empty(g, np.float32)
        xr = xs.reshape(g, 32)
        for i in range(g):
            res[i] = fn(xr[i])
        return res

    def left(v):
        a = np.float32(0)
        for e in v:
            a = np.float32(a + e)
        return a

    def tree(v):
        a = v.copy()
        while len(a) > 1:
            a = (a[0::2] + a[1::2]).astype(np.float32)
        return np.float32(a[0])

    def butterfly(v, lo2hi):
        a = v.astype(np.float32)
        offs = [1, 2, 4, 8, 16] if lo2hi else [16, 8, 4, 2, 1]
        for off in offs:
            b = a.copy()
            for l in range(32):
                b[l] = np.float32(a[l] + a[l ^ off])
            a = b
        return np.float32(a[0])

    cands = {
        "left_sequential": cand(left),
        "pairwise_tree_lowbits": cand(tree),
        "butterfly_offset_lo2hi": cand(lambda v: butterfly(v, True)),
        "butterfly_offset_hi2lo": cand(lambda v: butterfly(v, False)),
    }
    match = {k: int(np.count_nonzero(v.view(np.uint32)
                                     == got.view(np.uint32)))
             for k, v in cands.items()}
    manifest["simd_sum"] = {"groups": g, "matches_of": g, **match}
    save(out, "attention/simd_sum_probe_inputs.npy", xs)
    save(out, "attention/simd_sum_probe_outputs.npy", got)


SDPA_HEADER = """
constant int N = {N};
constant int seq_stride = 64;
constant int k_head_stride = {kh};
constant int v_head_stride = {vh};
constant int gqa_factor = {gqa};
"""

# Kernel S: one threadgroup of 32 threads per (head, key-stream). Thread
# d holds dims (2d, 2d+1), exactly upstream's per-thread dim split. The
# key index comes in as a 0-d int input; launched once per key so the
# running state is the native online-softmax chain, verbatim.
STEP_BODY = """
  typedef float U;
  int head = threadgroup_position_in_grid.x / 32;
  int stream = threadgroup_position_in_grid.x % 32;
  int lane = thread_index_in_simdgroup;
  int slot = head * 32 + stream;
  int i = key_ivals[threadgroup_position_in_grid.x];

  U max_score = max_in[slot];
  U sum_exp_score = sum_in[slot];
  U o0 = o_in[slot * 64 + lane * 2];
  U o1 = o_in[slot * 64 + lane * 2 + 1];

  if (i % 32 == stream) {
    const device T_* qp = queries + head * 64 + lane * 2;
    const device T_* kp = keys + (head / gqa_factor) * k_head_stride
        + i * 64 + lane * 2;
    const device T_* vp = values + (head / gqa_factor) * v_head_stride
        + i * 64 + lane * 2;

    U q0 = static_cast<U>(0.125f) * qp[0];
    U q1 = static_cast<U>(0.125f) * qp[1];

    U k0 = kp[0], k1 = kp[1];
    U score = 0;
    score += q0 * k0;
    score += q1 * k1;
    score = simd_sum(score);
    d_scores[slot] = score;

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    d_maxrun[slot] = max_score;
    d_sumrun[slot] = sum_exp_score;
    d_exps[slot] = exp_score;
    d_factors[slot] = factor;

    o0 = o0 * factor + exp_score * vp[0];
    o1 = o1 * factor + exp_score * vp[1];
  }

  max_out[slot] = max_score;
  sum_out[slot] = sum_exp_score;
  o_out[slot * 64 + lane * 2] = o0;
  o_out[slot * 64 + lane * 2 + 1] = o1;
"""

# Kernel C: one threadgroup of 32 threads per (head, dim-block G). Lane
# L = key-stream L; reads that stream's partial for dims (2G, 2G+1),
# scales by its factor, simd_sum over streams, divides by the combined
# denominator - upstream's transpose-combine order, verbatim.
COMBINE_BODY = """
  typedef float U;
  int head = threadgroup_position_in_grid.x / 32;
  int G = threadgroup_position_in_grid.x % 32;
  int lane = thread_index_in_simdgroup;

  U new_max = simd_max(max_in[head * 32 + lane]);
  U factor = fast::exp(max_in[head * 32 + lane] - new_max);
  U sum_exp_score = simd_sum(sum_in[head * 32 + lane] * factor);

  U o0 = o_in[(head * 32 + lane) * 64 + G * 2] * factor;
  U o1 = o_in[(head * 32 + lane) * 64 + G * 2 + 1] * factor;
  U c0 = simd_sum(o0);
  U c1 = simd_sum(o1);
  c0 = sum_exp_score == 0 ? c0 : (c0 / sum_exp_score);
  c1 = sum_exp_score == 0 ? c1 : (c1 / sum_exp_score);

  if (lane == 0) {
    d_newmax[head] = new_max;
    d_final_sum[head] = sum_exp_score;
    d_final_o[head * 64 + G * 2] = c0;
    d_final_o[head * 64 + G * 2 + 1] = c1;
    out[head * 64 + G * 2] = static_cast<T_>(c0);
    out[head * 64 + G * 2 + 1] = static_cast<T_>(c1);
  }
  d_lane_fac[head * 32 + lane] = factor;
"""


def sdpa_dump_kernel(q, k, v, out_dir, rel, manifest):
    H, N = q.shape[1], k.shape[2]
    kvh = k.shape[1]
    msl = {mx.float16: "float16_t", mx.bfloat16: "bfloat16_t",
           mx.float32: "float"}[q.dtype]
    header = SDPA_HEADER.format(N=N, kh=N * 64, vh=N * 64, gqa=H // kvh)
    step = mx.fast.metal_kernel(
        name="sdpa_step", input_names=["queries", "keys", "values",
                                       "key_ivals", "max_in", "sum_in",
                                       "o_in"],
        output_names=["max_out", "sum_out", "o_out", "d_scores", "d_maxrun",
                      "d_sumrun", "d_exps", "d_factors"],
        header=header, source=STEP_BODY.replace("T_", msl),
        ensure_row_contiguous=False)
    combine = mx.fast.metal_kernel(
        name="sdpa_combine", input_names=["queries", "keys", "values",
                                          "o_in", "max_in", "sum_in"],
        output_names=["out", "d_newmax", "d_final_sum", "d_lane_fac",
                      "d_final_o"],
        header=header, source=COMBINE_BODY.replace("T_", msl),
        ensure_row_contiguous=False)
    hs = H * 32
    tiny = np.finfo(np.float32).tiny  # Limits<float>::finite_min
    max_state = mx.array(np.full((hs,), tiny, np.float32))
    sum_state = mx.array(np.zeros((hs,), np.float32))
    o_state = mx.array(np.zeros((hs * 64,), np.float32))
    mx.eval(max_state); mx.eval(sum_state); mx.eval(o_state)
    trails = {n: np.full((N, hs), np.nan, np.float32)
              for n in ("scores", "maxrun", "sumrun", "exps", "factors")}
    for i in range(N):
        res = step(inputs=[q, k, v,
                           mx.array(np.full(hs, i, np.int32)),
                           max_state, sum_state, o_state],
                   grid=(hs, 1, 1), threadgroup=(32, 1, 1),
                   output_shapes=[(hs,), (hs,), (hs * 64,)] + [(hs,)] * 5,
                   output_dtypes=[mx.float32] * 8)
        max_state, sum_state, o_state = res[0], res[1], res[2]
        for idx, n in enumerate(("maxrun", "sumrun", "exps", "factors"),
                                start=3):
            trails[n][i] = f32(res[idx])
        trails["scores"][i] = f32(res[6])
    mx.eval(max_state); mx.eval(sum_state); mx.eval(o_state)
    cres = combine(inputs=[q, k, v, o_state, max_state, sum_state],
                   grid=(hs, 1, 1), threadgroup=(32, 1, 1),
                   output_shapes=[(H, 1, 64), (H,), (H,), (hs,), (H * 64,)],
                   output_dtypes=[q.dtype] + [mx.float32] * 4)
    mx.eval(cres[0])
    saved = {"out": f32(cres[0]), "newmax": f32(cres[1]),
             "final_sum": f32(cres[2]), "lane_fac": f32(cres[3]),
             "final_o": f32(cres[4])}
    for n in ("scores", "maxrun", "sumrun", "exps", "factors"):
        saved[n] = trails[n]
    save(out_dir, f"{rel}_o_accum.npy", f32(o_state).reshape(H, 32, 64))
    for n, a in saved.items():
        save(out_dir, f"{rel}_{n}.npy", a)
    return saved


def probe_sdpa_dump(out, manifest):
    for dt, tag in [(mx.float16, "f16"), (mx.bfloat16, "bf16")]:
        npdt = np.float16 if dt == mx.float16 else np.float32
        for kv in (8, 30, 31, 32, 64, 262, 1023, 1024, 1053, 1054):
            q = mx.array(RNG.standard_normal((1, 14, 1, 64)).astype(npdt)
                         ).astype(dt)
            k = mx.array(RNG.standard_normal((1, 2, kv, 64)).astype(npdt)
                         ).astype(dt)
            v = mx.array(RNG.standard_normal((1, 2, kv, 64)).astype(npdt)
                         ).astype(dt)
            mx.eval(q); mx.eval(k); mx.eval(v)
            saved = sdpa_dump_kernel(q, k, v, out,
                                     f"attention/dump/{tag}_kv{kv}", manifest)
            ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
            mx.eval(ref)
            rd = f32(ref)
            save(out, f"attention/dump/{tag}_kv{kv}_dispatch_out.npy", rd)
            same = int(np.count_nonzero(
                saved["out"].view(np.uint32) != rd.view(np.uint32)))
            manifest.setdefault("sdpa_dump_runs", []).append(
                {"tag": f"{tag}_kv{kv}", "kv": kv, "heads": 14,
                 "gqa": 7, "dtype": str(dt), "out_sha16": sha16(saved["out"]),
                 "dispatch_out_differences": same,
                 "dispatch_out_sha16": sha16(rd)})
            mpath = os.path.join(out, "attention", "manifest_probe.json")
            with open(mpath, "w") as f:
                json.dump(manifest, f, indent=1)
            print(f"  dump {tag} kv{kv}: dispatch diffs {same}", flush=True)


def probe_sdpa_dispatch(out, manifest):
    for dt, tag in [(mx.float16, "f16"), (mx.bfloat16, "bf16"),
                    (mx.float32, "f32")]:
        npdt = {mx.float16: np.float16, mx.bfloat16: np.float32,
                mx.float32: np.float32}[dt]
        for kv in (8, 16, 30, 31, 32, 33, 64, 262, 263, 1023, 1024, 1053,
                   1054):
            q = mx.array(RNG.standard_normal((1, 14, 1, 64)).astype(npdt)
                         ).astype(dt)
            k = mx.array(RNG.standard_normal((1, 2, kv, 64)).astype(npdt)
                         ).astype(dt)
            v = mx.array(RNG.standard_normal((1, 2, kv, 64)).astype(npdt)
                         ).astype(dt)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
            mx.eval(o)
            rel = f"attention/dispatch/{tag}_kv{kv}"
            save(out, f"{rel}_q.npy", f32(q))
            save(out, f"{rel}_k.npy", f32(k))
            save(out, f"{rel}_v.npy", f32(v))
            save(out, f"{rel}_out.npy", f32(o))
            manifest.setdefault("sdpa_dispatch_captures", []).append(
                {"tag": f"{tag}_kv{kv}", "kv": kv, "dtype": str(dt),
                 "out_sha16": sha16(f32(o))})
    mpath = os.path.join(out, "attention", "manifest_probe.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/src/native-captures-20260912")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    manifest = {"mlx_version": mx.__version__,
                "device": str(mx.metal.device_info())}
    only = args.only.split(",") if args.only else None
    steps = [("exp", probe_exp), ("sum", probe_simd_sum),
             ("dump", probe_sdpa_dump), ("dispatch", probe_sdpa_dispatch)]
    for name, fn in steps:
        if only and name not in only:
            continue
        print(f"running {name}...", flush=True)
        fn(out, manifest)
        print(f"  {name} done", flush=True)
    print(json.dumps({k: manifest.get(k) for k in
                      ("fast_exp_vs_exp2_log2e_differences", "simd_sum")},
                     indent=1))


if __name__ == "__main__":
    main()

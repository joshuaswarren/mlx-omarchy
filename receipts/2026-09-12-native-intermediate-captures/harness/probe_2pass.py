#!/usr/bin/env python3
"""Verbatim-arithmetic dump copies of sdpa_vector_2pass_1 and _2pass_2
(MLX 1f8e74e3, M1 Max config: devc 's', blocks=128 for 1024<N<=8192,
grid (kv_heads, batch, blocks), threadgroup (32, gqa, q_seq) = (32,7,1);
pass 2 grid (B*H, q_seq, 1), threadgroup 1024).

Validated against the real dispatch output at the same inputs - must be
bit-exact. Banks per-key scores/running max/sum per (head, block), the
partials buffer bits, per-block sums/maxs, pass-2 factors and the final
combine - the rule-1 intermediates for the N>=1024 decode path.
"""

import argparse
import hashlib
import json
import os

import mlx.core as mx
import numpy as np

RNG = np.random.default_rng(314159265)


def sha16(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def f32(a):
    return np.array(a.astype(mx.float32))


def save(out, rel, arr):
    path = os.path.join(out, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(arr))


PASS1_HEADER = """
constant int N = {N};
constant int blocks = {blocks};
constant float scale_ = 0.125f;
constant int k_head_stride = {kh};
constant int v_head_stride = {vh};
constant int seq_stride = 64;
"""

PASS1_BODY = """
  constexpr int BD = 32;
  constexpr int D = 64;
  constexpr int V = 64;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;
  const int kv_head_idx = threadgroup_position_in_grid.x;
  const int batch_idx = threadgroup_position_in_grid.y;
  const int block_idx = threadgroup_position_in_grid.z;
  const int gqa_factor = 7;
  const int q_seq_len = 1;
  const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
  const int q_batch_head_idx = batch_idx * 14 + q_head_idx;
  const int o_offset = q_batch_head_idx * q_seq_len;

  const device T_* qp = queries + q_batch_head_idx * D
      + thread_index_in_simdgroup * qk_per_thread;
  const device T_* kp = keys + (batch_idx * 2 + kv_head_idx) * k_head_stride
      + block_idx * seq_stride + thread_index_in_simdgroup * qk_per_thread;
  const device T_* vp = values + (batch_idx * 2 + kv_head_idx) * v_head_stride
      + block_idx * seq_stride + thread_index_in_simdgroup * v_per_thread;

  thread U q[qk_per_thread];
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale_) * qp[i];
  }
  thread U o[v_per_thread];
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  int step = 0;
  for (int i = block_idx; i < N; i += blocks, ++step) {
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * kp[j];
    }
    score = simd_sum(score);
    d_scores[(q_batch_head_idx * N + i)] = score;

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    d_maxrun[q_batch_head_idx * N + i] = max_score;
    d_sumrun[q_batch_head_idx * N + i] = sum_exp_score;
    d_exps[q_batch_head_idx * N + i] = exp_score;
    d_factors[q_batch_head_idx * N + i] = factor;

    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + vp[j] * exp_score;
    }
    kp += blocks * seq_stride;
    vp += blocks * seq_stride;
  }

  device T_* pp = partials + o_offset * blocks * V + block_idx * V
      + thread_index_in_simdgroup * v_per_thread;
  for (int i = 0; i < v_per_thread; i++) {
    pp[i] = static_cast<T_>(o[i]);
    d_lane_o[(q_batch_head_idx * blocks + block_idx) * 32 * V
             + thread_index_in_simdgroup * V + i] = o[i];
  }
  if (thread_index_in_simdgroup == 0) {
    sums_f[o_offset * blocks + block_idx] = sum_exp_score;
    maxs_f[o_offset * blocks + block_idx] = max_score;
  }
"""

PASS2A_HEADER = """
constant int nblocks = {blocks};
"""

# Pass-2 phase A: one threadgroup of 32 per head. Lane l reduces the
# block maxima at indices l, l+32, l+64, l+96 (upstream's per-lane b
# loop), then simd_max; the sum phase re-derives each block's factor and
# accumulates per lane, then simd_sum - verbatim upstream order.
P2A_BODY = """
  typedef float U;
  int head = threadgroup_position_in_grid.x;
  int lane = thread_index_in_simdgroup;
  const device float* mp = maxs_f + head * nblocks;
  const device float* sp = sums_f + head * nblocks;

  U max_score = Limits<U>::finite_min;
  for (int b = 0; b < nblocks / 32; ++b) {
    max_score = max(max_score, mp[lane + 32 * b]);
  }
  max_score = simd_max(max_score);
  gmax_out[head] = max_score;

  U sum_exp_score = 0.0;
  for (int b = 0; b < nblocks / 32; ++b) {
    U factor = fast::exp(mp[lane + 32 * b] - max_score);
    d_fac_sum[head * nblocks + lane + 32 * b] = factor;
    sum_exp_score += factor * sp[lane + 32 * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  sum_out[head] = sum_exp_score;
"""

# Pass-2 phase C: one threadgroup of 32 per (head, dim-block G); lane =
# block stream (gid in upstream). Each lane accumulates its four blocks
# sequentially, then the transpose-combine sums the 32 streams, divides,
# and casts - verbatim upstream order.
P2C_BODY = """
  typedef float U;
  int head = threadgroup_position_in_grid.x / 32;
  int G = threadgroup_position_in_grid.x % 32;
  int lane = thread_index_in_simdgroup;
  U gmax = gmax_in[head];
  U sum_exp_score = sum_in[head];

  U o0 = 0;
  U o1 = 0;
  for (int b = 0; b < nblocks / 32; ++b) {
    int blk = lane + 32 * b;
    U factor = fast::exp(maxs_f[head * nblocks + blk] - gmax);
    d_fac_o[head * nblocks + blk] = factor;
    o0 += factor * static_cast<U>(partials[(head * nblocks + blk) * 64 + G * 2]);
    o1 += factor * static_cast<U>(partials[(head * nblocks + blk) * 64 + G * 2 + 1]);
  }
  d_lane_o2[(head * 32 + G) * 32 + lane] = o0;
  d_lane_o2[(head * 32 + G) * 32 + lane + 32] = o1;

  U c0 = simd_sum(o0);
  U c1 = simd_sum(o1);
  c0 = sum_exp_score == 0 ? c0 : (c0 / sum_exp_score);
  c1 = sum_exp_score == 0 ? c1 : (c1 / sum_exp_score);
  if (lane == 0) {
    d_final_o[head * 64 + G * 2] = c0;
    d_final_o[head * 64 + G * 2 + 1] = c1;
    out[head * 64 + G * 2] = static_cast<T_>(c0);
    out[head * 64 + G * 2 + 1] = static_cast<T_>(c1);
  }
"""


def run(kv, dtype, out, manifest):
    tagdt = {mx.float16: "f16", mx.bfloat16: "bf16"}[dtype]
    npdt = np.float16 if dtype == mx.float16 else np.float32
    H, KVH, G = 14, 2, 7
    blocks = 128
    q = mx.array(RNG.standard_normal((1, H, 1, 64)).astype(npdt)).astype(dtype)
    k = mx.array(RNG.standard_normal((1, KVH, kv, 64)).astype(npdt)).astype(
        dtype)
    v = mx.array(RNG.standard_normal((1, KVH, kv, 64)).astype(npdt)).astype(
        dtype)
    mx.eval(q); mx.eval(k); mx.eval(v)
    nf = H * kv
    rel = f"attention/twopass/{tagdt}_kv{kv}"

    hdr1 = PASS1_HEADER.format(N=kv, blocks=blocks, kh=kv * 64, vh=kv * 64)
    b1 = PASS1_BODY.replace("T_", {mx.float16: "float16_t",
                                   mx.bfloat16: "bfloat16_t"}[dtype])
    shapes1 = {"scores": (nf,), "maxrun": (nf,), "sumrun": (nf,),
               "exps": (nf,), "factors": (nf,),
               "lane_o": (H * blocks * 32 * 64,)}
    outs1 = ["partials", "sums_f", "maxs_f"] + ["d_" + n for n in
                                                ("scores", "maxrun", "sumrun",
                                                 "exps", "factors", "lane_o")]
    k1 = mx.fast.metal_kernel(
        name="sdpa2p1", input_names=["queries", "keys", "values"],
        output_names=outs1, header=hdr1, source=b1,
        ensure_row_contiguous=False)
    res1 = k1(inputs=[q, k, v], grid=(KVH, 1, blocks),
              threadgroup=(32, G, 1),
              output_shapes=[(H, blocks, 64), (H * blocks,), (H * blocks,)]
              + [shapes1[n] for n in ("scores", "maxrun", "sumrun", "exps",
                                      "factors", "lane_o")],
              output_dtypes=[dtype, mx.float32, mx.float32] + [mx.float32] * 6)
    mx.eval(res1[0])

    hdrA = PASS2A_HEADER.format(blocks=blocks)
    k2a = mx.fast.metal_kernel(
        name="sdpa2p2a", input_names=["sums_f", "maxs_f"],
        output_names=["gmax_out", "sum_out", "d_fac_sum"],
        header=hdrA, source=P2A_BODY, ensure_row_contiguous=False)
    res2a = k2a(inputs=[res1[1], res1[2]], grid=(H, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(H,), (H,), (H * blocks,)],
                output_dtypes=[mx.float32] * 3)
    mx.eval(res2a)

    k2c = mx.fast.metal_kernel(
        name="sdpa2p2c", input_names=["partials", "sums_f", "maxs_f",
                                      "gmax_in", "sum_in"],
        output_names=["out", "d_fac_o", "d_lane_o2", "d_final_o"],
        header=hdrA, source=P2C_BODY.replace("T_", {mx.float16: "float16_t",
                                                    mx.bfloat16: "bfloat16_t"}[dtype]),
        ensure_row_contiguous=False)
    res2c = k2c(inputs=[res1[0], res1[1], res1[2], res2a[0], res2a[1]],
                grid=(H * 32, 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[(H, 1, 64), (H * blocks,),
                               (H * 32 * 64,), (H * 64,)],
                output_dtypes=[dtype, mx.float32, mx.float32, mx.float32])
    mx.eval(res2c[0])

    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    mx.eval(ref)
    got, refd = f32(res2c[0]), f32(ref)
    diffs = int(np.count_nonzero(got.view(np.uint32)
                                 != refd.view(np.uint32)))
    save(out, f"{rel}_q.npy", f32(q))
    save(out, f"{rel}_k.npy", f32(k))
    save(out, f"{rel}_v.npy", f32(v))
    save(out, f"{rel}_out_dump.npy", got)
    save(out, f"{rel}_out_dispatch.npy", refd)
    for arr, n in zip(res1, outs1):
        save(out, f"{rel}_p1_{n}.npy", bits_of(arr))
    for arr, n in zip(res2a, ["gmax", "final_sum", "fac_sum"]):
        save(out, f"{rel}_p2_{n}.npy", f32(arr))
    for arr, n in zip(res2c[1:], ["fac_o", "lane_o2", "final_o"]):
        save(out, f"{rel}_p2_{n}.npy", bits_of(arr))
    manifest.setdefault("twopass_runs", []).append(
        {"tag": f"{tagdt}_kv{kv}", "kv": kv, "blocks": blocks,
         "dispatch_out_differences": diffs,
         "out_sha16": sha16(got), "dispatch_sha16": sha16(refd)})
    print(f"2pass {tagdt} kv{kv}: dispatch diffs {diffs}", flush=True)


def bits_of(a):
    if a.dtype in (mx.float16, mx.bfloat16):
        return np.array(a.view(mx.uint16))
    return f32(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/src/native-captures-20260912")
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    manifest = {}
    for dtype in (mx.float16, mx.bfloat16):
        for kv in (1053, 1054, 2048):
            run(kv, dtype, out, manifest)
    with open(os.path.join(out, "attention", "manifest_2pass.json"),
              "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()

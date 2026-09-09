#!/usr/bin/env python3
"""GPU arithmetic probe for the Q4 word GEMV path: runs mx.quantized_matmul
(M = 1, transpose, group 64, bits 4, float16) on random inputs and compares
the device output against the CPU model of native qmv under several
32-lane reduction orders, stage by stage:

  reduce   scale 1, bias 0, one unit nibble per word -> pure lane reduction
  onenibble same with random x (equivalent to reduce; regression control)
  dotonly  scale 1, bias 0, nibbles random          -> dot accumulation + reduce
  scaleonly scale random, bias 0, one unit nibble    -> fma(scale, x0, 0) + reduce
  inputsum scale 0, bias random               -> half quad-sum chain + reduce
  dot      bias 0, scale random, nibbles random -> dot accumulation + reduce
  full     everything random

usage: probe-stages.py OUT_JSON [K] [TRIALS] [float16|float32] [DUMP_DIR] [STAGES]

float32 runs the f32 kernel variant (no half rounding of the quad sums, f32
store) with f16-representable x, so every product stays exact and the
comparison is bit-exact at f32 without the f16 store hiding differences.
"""
import ctypes
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

import mlx.core as mx

f32 = np.float32
SIMD = 32
BLOCK = 256


def h(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


_SRC = """
#include <math.h>
#include <stddef.h>
void fmaf_arr(const float*a,const float*b,const float*c,float*o,size_t n){
  for(size_t i=0;i<n;i++) o[i]=fmaf(a[i],b[i],c[i]);
}
"""
_so = pathlib.Path(tempfile.gettempdir()) / 'pgp_fmaf_arr.so'
if not _so.exists():
    _c = _so.with_suffix('.c')
    _c.write_text(_SRC)
    subprocess.check_call(['cc', '-O2', '-shared', '-fPIC', '-o', str(_so), str(_c), '-lm'])
_lib = ctypes.CDLL(str(_so))


def fma(a, b, c):
    """Exact float32 fma through libm fmaf (numpy has none)."""
    a, b, c = (np.ascontiguousarray(v) for v in np.broadcast_arrays(np.asarray(a, np.float32), np.asarray(b, np.float32), np.asarray(c, np.float32)))
    o = np.empty_like(a)
    _lib.fmaf_arr(a.ctypes.data_as(ctypes.c_void_p), b.ctypes.data_as(ctypes.c_void_p), c.ctypes.data_as(ctypes.c_void_p), o.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.size))
    return o


def lane_partials(x, w, scales, biases, round_quads=True, combine='fma_scale'):
    K = x.shape[0]
    N = w.shape[0]
    nblocks = (K + BLOCK - 1) // BLOCK
    kk = (np.arange(nblocks)[:, None, None] * BLOCK + np.arange(SIMD)[None, :, None] * 8 + np.arange(8)[None, None, :])
    valid = kk < K
    xs = np.where(valid, x[np.minimum(kk, K - 1)], f32(0))
    q0, q1 = xs[..., 0:4], xs[..., 4:8]
    r = h if round_quads else (lambda v: v)
    hs0 = r(r(r(q0[..., 0] + q0[..., 1]) + q0[..., 2]) + q0[..., 3])
    hs1 = r(r(r(q1[..., 0] + q1[..., 1]) + q1[..., 2]) + q1[..., 3])
    sum_ = (f32(0) + hs0) + hs1
    xt = xs / np.array([1, 16, 256, 4096] * 2, np.float32)
    widx = np.arange(nblocks)[:, None] * SIMD + np.arange(SIMD)[None, :]
    words = np.where((widx < K // 8)[None], w[:, np.minimum(widx, K // 8 - 1)], np.uint32(0))
    ws = np.stack([words & 0xFFFF, words >> 16], -1)
    m = (ws[..., None] & np.array([0x000F, 0x00F0, 0x0F00, 0xF000], np.uint32)).astype(np.float32)
    p = xt.reshape(nblocks, SIMD, 2, 4)[None] * m
    inner = ((p[..., 0] + p[..., 1]) + p[..., 2]) + p[..., 3]
    accum = (f32(0) + inner[..., 0]) + inner[..., 1]
    gidx = np.arange(nblocks)[:, None] * 4 + np.arange(SIMD)[None, :] // 8
    gvalid = gidx < K // 64
    sc = np.where(gvalid[None], scales[:, np.minimum(gidx, K // 64 - 1)], f32(0)).astype(np.float32)
    bi = np.where(gvalid[None], biases[:, np.minimum(gidx, K // 64 - 1)], f32(0)).astype(np.float32)
    if combine == 'fma_scale':
        qd = fma(sc, accum, sum_[None] * bi)
    elif combine == 'fma_bias':
        qd = fma(sum_[None] * np.ones_like(sc), bi, sc * accum)
    else:
        qd = sc * accum + sum_[None] * bi
    result = np.zeros((N, SIMD), np.float32)
    for b in range(nblocks):
        result = result + qd[:, b, :]
    return result


def reduce_lanes(part, order):
    v = part.copy()
    if order == 'sequential':
        t = v[:, 0].copy()
        for l in range(1, SIMD):
            t = t + v[:, l]
        return t
    offs = [1, 2, 4, 8, 16] if order == 'xor_up' else [16, 8, 4, 2, 1]
    for o in offs:
        v = v + v[:, np.arange(SIMD) ^ o]
    return v[:, 0]


ORDERS = ['xor_up', 'xor_down', 'sequential']
DUMP = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else None
COMBINES = ['fma_scale', 'fma_bias', 'nofma']


def run_stage(stage, rng, K, N, dtype):
    words = rng.integers(0, 2**32, size=(N, K // 8), dtype=np.uint64).astype(np.uint32)
    scales = rng.uniform(-1.0, 1.0, size=(N, K // 64)).astype(np.float16).astype(np.float32)
    biases = rng.uniform(-1.0, 1.0, size=(N, K // 64)).astype(np.float16).astype(np.float32)
    # wide dynamic range so the half quad chain and the float adds both round
    x = (rng.standard_normal(K) * np.exp2(rng.uniform(-6, 4, size=K))).astype(np.float16).astype(np.float32)
    if stage == 'reduce':
        words[:] = 0x1
        scales[:] = 1.0
        biases[:] = 0.0
    elif stage == 'inputsum':
        scales[:] = 0.0
    elif stage == 'dot':
        biases[:] = 0.0
    elif stage == 'dotonly':
        biases[:] = 0.0
        scales[:] = 1.0
    elif stage == 'scaleonly':
        biases[:] = 0.0
        words[:] = 0x1
    elif stage == 'onenibble':
        biases[:] = 0.0
        scales[:] = 1.0
        words[:] = 0x1
    mdt = mx.float16 if dtype == 'float16' else mx.float32
    out = mx.quantized_matmul(mx.array(x.reshape(1, K)).astype(mdt), mx.array(words), mx.array(scales).astype(mdt),
                              mx.array(biases).astype(mdt), transpose=True, group_size=64, bits=4)
    mx.eval(out)
    device = np.array(out.astype(mx.float32)).reshape(-1)
    if DUMP is not None:
        np.savez(DUMP / f'{stage}-{K}-{dtype}-{len(list(DUMP.glob(stage + "-*")))}.npz', x=x, words=words, scales=scales, biases=biases, device=device)
    store = h if dtype == 'float16' else (lambda v: v)
    report = {}
    for combine in COMBINES:
        part = lane_partials(x, words, scales, biases, round_quads=dtype == 'float16', combine=combine)
        for order in ORDERS:
            tot = reduce_lanes(part, order)
            report[f'{combine}:{order}'] = int(np.count_nonzero(store(tot) != device))
    return report, device


def main():
    out_path = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 896
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    dtype = sys.argv[4] if len(sys.argv) > 4 else 'float16'
    N = 64
    rng = np.random.default_rng(7)
    summary = {'k': K, 'n': N, 'trials': trials, 'dtype': dtype, 'device': str(mx.default_device()), 'stages': {}}
    stages = sys.argv[6].split(',') if len(sys.argv) > 6 else ['reduce', 'inputsum', 'dot', 'full']
    for stage in stages:
        totals = None
        for _ in range(trials):
            report, _ = run_stage(stage, rng, K, N, dtype)
            totals = report if totals is None else {k: totals[k] + report[k] for k in report}
        summary['stages'][stage] = {'outputs': trials * N, 'mismatches': totals}
        print(json.dumps({'stage': stage, **summary['stages'][stage]}), flush=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()

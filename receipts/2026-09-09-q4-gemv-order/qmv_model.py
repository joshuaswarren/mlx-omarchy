#!/usr/bin/env python3
"""CPU model of native Metal qmv_impl (affine 4-bit, group 64, T=half, U=float).

Reproduces, per output row and per simd lane, the exact float32 arithmetic of
load_vector + qdot + block accumulation, then applies candidate 32-lane
reduction orders and the final half rounding. Reports mismatch counts against
captured native outputs for each (fma placement, reduction order) combination.
"""
import ctypes
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/pgp/data')
SIMD = 32
VALUES_PER_THREAD = 8
BLOCK = SIMD * VALUES_PER_THREAD  # 256

# --- exact fmaf through libm (numpy has no fma) -------------------------------
_src = r'''
#include <math.h>
#include <stddef.h>
void fmaf_arr(const float*a,const float*b,const float*c,float*o,size_t n){
  for(size_t i=0;i<n;i++) o[i]=fmaf(a[i],b[i],c[i]);
}
'''
_so = Path('/tmp/pgp/fmaf_arr.so')
if not _so.exists():
    _c = _so.with_suffix('.c')
    _c.write_text(_src)
    subprocess.check_call(['gcc', '-O2', '-shared', '-fPIC', '-o', str(_so), str(_c), '-lm'])
_lib = ctypes.CDLL(str(_so))


def fma(a, b, c):
    a, b, c = np.broadcast_arrays(np.asarray(a, np.float32), np.asarray(b, np.float32), np.asarray(c, np.float32))
    a = np.ascontiguousarray(a); b = np.ascontiguousarray(b); c = np.ascontiguousarray(c)
    o = np.empty_like(a)
    _lib.fmaf_arr(a.ctypes.data_as(ctypes.c_void_p), b.ctypes.data_as(ctypes.c_void_p),
                  c.ctypes.data_as(ctypes.c_void_p), o.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.size))
    return o


f32 = np.float32
f16 = np.float16


def h(x):
    """Round float32 to half and back (RNE), as a half-precision add result."""
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


# --- per-lane partial sums ------------------------------------------------------
def lane_partials(x, w, scales, biases, qdot_mode, combine_mode):
    """x: (K,) float16 values as float32. w: (N, K/8) uint32. Returns (N, 32) float32
    per-lane `result` after all blocks, before simd_sum."""
    K = x.shape[0]
    N = w.shape[0]
    nblocks = (K + BLOCK - 1) // BLOCK
    assert K % 8 == 0
    # k index for block b, lane l, value i: b*256 + l*8 + i
    kk = (np.arange(nblocks)[:, None, None] * BLOCK + np.arange(SIMD)[None, :, None] * 8 + np.arange(8)[None, None, :])
    valid = kk < K
    xs = np.where(valid, x[np.minimum(kk, K - 1)], f32(0))  # (B, 32, 8)
    # load_vector: sum in half chain per quad, then float accumulate
    q0 = xs[..., 0:4]
    q1 = xs[..., 4:8]
    hs0 = h(h(h(q0[..., 0] + q0[..., 1]) + q0[..., 2]) + q0[..., 3])
    hs1 = h(h(h(q1[..., 0] + q1[..., 1]) + q1[..., 2]) + q1[..., 3])
    sum_ = (f32(0) + hs0) + hs1  # float adds
    shifts = np.array([1, 16, 256, 4096, 1, 16, 256, 4096], np.float32)
    xt = xs / shifts  # exact power-of-two divides
    # words: block b, lane l -> word index (b*256 + l*8)/8 = b*32 + l
    widx = np.arange(nblocks)[:, None] * SIMD + np.arange(SIMD)[None, :]  # (B, 32)
    wvalid = widx < K // 8
    words = w[:, np.minimum(widx, K // 8 - 1)]  # (N, B, 32)
    words = np.where(wvalid[None], words, np.uint32(0))
    lo = (words & 0xFFFF).astype(np.float32)
    hi = (words >> 16).astype(np.float32)
    masks = np.array([0x000F, 0x00F0, 0x0F00, 0xF000], np.uint32)
    ws = np.stack([words & 0xFFFF, words >> 16], -1)  # (N,B,32,2) uint32
    m = (ws[..., None] & masks).astype(np.float32)  # (N,B,32,2,4)
    xq = xt.reshape(nblocks, SIMD, 2, 4)[None]  # (1,B,32,2,4)
    # qdot inner: accum += (p0 + p1 + p2 + p3), left to right
    p = xq * m  # (N,B,32,2,4) products (exact rounding of each product)
    if qdot_mode == 'nofma':
        inner = ((p[..., 0] + p[..., 1]) + p[..., 2]) + p[..., 3]
    elif qdot_mode == 'fma_chain':  # fma(x3,m3, fma(x2,m2, fma(x1,m1, x0*m0)))
        inner = fma(xq[..., 3], m[..., 3], fma(xq[..., 2], m[..., 2], fma(xq[..., 1], m[..., 1], p[..., 0])))
    elif qdot_mode == 'fma_chain_first':  # fma(x0,m0, x1*m1) then fma(x2..), fma(x3..)
        inner = fma(xq[..., 3], m[..., 3], fma(xq[..., 2], m[..., 2], fma(xq[..., 0], m[..., 0], p[..., 1])))
    else:
        raise ValueError(qdot_mode)
    accum = (f32(0) + inner[..., 0])
    if qdot_mode == 'nofma':
        accum = accum + inner[..., 1]
    else:
        accum = accum + inner[..., 1]
    # per-block scale/bias: group index = (b*256 + l*8)//64 = b*4 + l//8
    gidx = np.arange(nblocks)[:, None] * 4 + np.arange(SIMD)[None, :] // 8
    gvalid = gidx < K // 64
    sc = scales[:, np.minimum(gidx, K // 64 - 1)]
    bi = biases[:, np.minimum(gidx, K // 64 - 1)]
    sc = np.where(gvalid[None], sc, f32(0)).astype(np.float32)
    bi = np.where(gvalid[None], bi, f32(0)).astype(np.float32)
    s = sum_[None]  # (1,B,32)
    if combine_mode == 'nofma':
        qd = sc * accum + s * bi
    elif combine_mode == 'fma_scale':
        qd = fma(sc, accum, s * bi)
    elif combine_mode == 'fma_bias':
        qd = fma(s, bi, sc * accum)
    else:
        raise ValueError(combine_mode)
    # result += qd sequentially over blocks
    result = np.zeros((N, SIMD), np.float32)
    for b in range(nblocks):
        result = result + qd[:, b, :]
    return result


# --- 32-lane reduction orders ---------------------------------------------------
def reduce_lanes(part, order):
    """part: (N, 32). Returns lane-0 total per row for the named order."""
    v = part.copy()
    if order == 'sequential':
        t = v[:, 0].copy()
        for l in range(1, SIMD):
            t = t + v[:, l]
        return t
    if order == 'sequential_rev':
        t = v[:, SIMD - 1].copy()
        for l in range(SIMD - 2, -1, -1):
            t = t + v[:, l]
        return t
    if order in ('xor_up', 'xor_down', 'shdown_up', 'shdown_down', 'shup_up', 'shup_down'):
        offs = [1, 2, 4, 8, 16] if order.endswith('_up') else [16, 8, 4, 2, 1]
        for o in offs:
            idx = np.arange(SIMD)
            if order.startswith('xor'):
                src = idx ^ o
            elif order.startswith('shdown'):
                src = np.minimum(idx + o, SIMD - 1)  # out-of-range lanes read self (value unused for lane 0)
                src = np.where(idx + o < SIMD, idx + o, idx)
            else:  # shuffle up: lane i reads i - o; lane 31 holds the total
                src = np.where(idx - o >= 0, idx - o, idx)
            v = v + v[:, src]
        return v[:, 0] if not order.startswith('shup') else v[:, SIMD - 1]
    if order == 'quad_then_xor':  # sum within quads (4 lanes) sequentially, then xor over 8 quads
        q = v.reshape(-1, 8, 4)
        t = ((q[..., 0] + q[..., 1]) + q[..., 2]) + q[..., 3]
        for o in [1, 2, 4]:
            t = t + t[:, np.arange(8) ^ o]
        return t[:, 0]
    if order == 'quad_tree_then_xor':  # xor 1,2 within quad, then xor 4,8,16
        for o in [1, 2, 4, 8, 16]:
            v = v + v[:, np.arange(SIMD) ^ o]
        return v[:, 0]
    if order == 'pairs_rev':  # xor_up but summing other operand first (v[src] + v)
        for o in [1, 2, 4, 8, 16]:
            v = v[:, np.arange(SIMD) ^ o] + v
        return v[:, 0]
    raise ValueError(order)


ORDERS = ['sequential', 'sequential_rev', 'xor_up', 'xor_down', 'shdown_up', 'shdown_down', 'shup_up', 'shup_down', 'quad_then_xor']


def main():
    tensors = load_file(str(DATA / 'model.safetensors'))
    projs = {
        'q_proj': ('self_attn', 'attn'), 'k_proj': ('self_attn', 'attn'), 'v_proj': ('self_attn', 'attn'),
        'o_proj': ('self_attn', 'attn'), 'gate_proj': ('mlp', 'mlp'), 'up_proj': ('mlp', 'mlp'), 'down_proj': ('mlp', 'mlp'),
    }
    call = int(os.environ.get('CALL', '35'))
    report = {}
    combos = list(itertools.product(['nofma', 'fma_chain', 'fma_chain_first'], ['nofma', 'fma_scale', 'fma_bias']))
    only = os.environ.get('COMBOS')
    if only:
        combos = [tuple(c.split(':')) for c in only.split(',')]
    for name, (mod, label) in projs.items():
        pre = f'model.layers.0.{mod}.{name}'
        w = tensors[pre + '.weight']
        sc = tensors[pre + '.scales'].astype(np.float32)
        bi = tensors[pre + '.biases'].astype(np.float32)
        lb = tensors.get(pre + '.bias')
        x = np.load(DATA / f'call{call}-{label}-{name}-input.npy').reshape(-1)
        assert x.size == w.shape[1] * 8, (x.shape, w.shape)
        expected = np.load(Path(os.environ.get("EXPECTED", str(DATA))) / f"call{call}-{label}-{name}-output.npy").reshape(-1)
        report[name] = {}
        for qm, cm in combos:
            part = lane_partials(x, w, sc, bi, qm, cm)
            for order in ORDERS:
                tot = reduce_lanes(part, order)
                y = h(tot)
                if lb is not None:
                    y = (y.astype(np.float64) + lb.astype(np.float64)).astype(np.float16).astype(np.float32)
                diff = int(np.count_nonzero(y != expected))
                report[name][f'{qm}:{cm}:{order}'] = diff
        best = sorted(report[name].items(), key=lambda kv: kv[1])[:6]
        print(name, 'K', x.size, 'N', w.shape[0], 'best:', best, flush=True)
    # combos that are exact for every projection
    keys = set.intersection(*(set(r) for r in report.values()))
    exact = sorted(k for k in keys if all(report[n][k] == 0 for n in report))
    print('EXACT_FOR_ALL:', exact)
    out = Path(os.environ.get('REPORT', f'/tmp/pgp/qmv-model-call{call}.json'))
    out.write_text(json.dumps({'call': call, 'mismatches': report, 'exact_for_all': exact}, indent=1))


if __name__ == '__main__':
    main()

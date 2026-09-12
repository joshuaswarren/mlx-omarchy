"""Bit-exact software model of shaders/matmul_vec.comp (USE_BF16).

Because inputs are bf16 (8-bit mantissa), a product of two inputs is exact in
f64; adding the f32 accumulator in f64 is exact (|exp| far from overflow), so
round_f32(f64 product + f64 acc) is the exact single-rounding f32 fma. This
makes the model hardware-deterministic by construction, the same standard the
chunked-softmax receipt used.

Models:
  shipped      : lane L owns k chunks {4L + 128j}, sequential fma per lane,
                 shuffle-down tree (16,8,4,2,1), bf16 RNE store.
  uvec4_widen  : candidate "load instruction count" mechanism: lane L owns
                 k chunks {8L + 256j} (one 16B load per row per iteration).
  span2        : 2 rows per workgroup, same per-column lane mapping.
"""
import numpy as np

F32 = np.float32
F64 = np.float64


def rne_f32(x):
    # round f64 to nearest f32 (ties-to-even), no overflow in our ranges
    return F32(x)


def bf16_pack(f):
    """RNE bf16 pack of a f32 (matches bf16_store incl. NaN quieting)."""
    arr = np.atleast_1d(np.asarray(f, dtype=np.float32))
    bits = arr.view(np.uint32)
    isnan = (bits & 0x7FFFFFFF) > 0x7F800000
    packed = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    packed[isnan] = ((bits[isnan] >> 16) | 0x40).astype(np.uint16)
    out = packed if np.ndim(f) else packed[0]
    return out


def bf16_unpack(u16):
    arr = np.atleast_1d(u16).astype(np.uint32) << 16
    out = arr.view(np.float32)
    return out[0] if np.ndim(u16) == 0 else out

def fma_bf16(a_bf16, b_bf16, acc_f32):
    """exact f32 fma of two bf16 values into an f32 accumulator"""
    prod = F64(bf16_unpack(a_bf16)) * F64(bf16_unpack(b_bf16))
    return rne_f32(F64(acc_f32) + prod)


def kernel_column_sum(x_u16, w_u16, k_per_lane, lane_stride, K):
    """one output column: tree over 32 lanes of the given chunk mapping.

    x_u16, w_u16: bf16 bit patterns (uint16 arrays, length K for this row).
    k_per_lane: consecutive k elements per lane per chunk (4 shipped / 8 cand).
    lane_stride: k advance per iteration (128 shipped / 256 candidate).
    """
    lanes = np.zeros(32, dtype=np.float32)
    for L in range(32):
        acc = F32(0.0)
        k = L * k_per_lane
        while k < K:
            for j in range(k_per_lane):
                if k + j < K:
                    acc = fma_bf16(w_u16[k + j], x_u16[k + j], acc)
            k += lane_stride
        lanes[L] = acc
    # shuffle-down tree (16,8,4,2,1); only lane 0's value is stored
    a = lanes.copy()
    for off in (16, 8, 4, 2, 1):
        for L in range(32 - off):
            a[L] = F32(a[L] + a[L + off])
    return a[0]


def run(K=896, n_cols=64, seed=0):
    rng = np.random.default_rng(seed)
    x = bf16_pack(rng.standard_normal(K).astype(F32))
    cols_ship, cols_cand = [], []
    diffs = 0
    for c in range(n_cols):
        w = bf16_pack(rng.standard_normal(K).astype(F32))
        s = kernel_column_sum(x, w, 4, 128, K)
        d = kernel_column_sum(x, w, 8, 256, K)
        cols_ship.append(bf16_pack(s))
        cols_cand.append(bf16_pack(d))
        if bf16_pack(s) != bf16_pack(d):
            diffs += 1
    return n_cols, diffs


def ulp_diff(a_u16, b_u16):
    ia = a_u16.astype(np.int16)
    ib = b_u16.astype(np.int16)
    return np.abs(ia.astype(np.int32) - ib.astype(np.int32))


if __name__ == "__main__":
    import sys
    # 1. benign random data: association-stable (0 diffs expected)
    for K in (896, 4864):
        n, diffs = run(K=K, n_cols=48, seed=K)
        print(f"K={K}: {n} random-bf16 columns, uvec4-widen differs on {diffs}")
    # 2. tie-rich adversarial grid: counterexample exists
    rng = np.random.default_rng(23)
    V = [1.0, 1+2**-8, 1+2**-7, 1.5, -1.0, -(1+2**-8), -1.5, 0.5, -0.5,
         2**-8, -2**-8, 2**-9, 0.0, 1+2**-15, -2**-15, 2.0**24, -(2.0**24),
         1+2**-23, 2**-16, 3.0, -3.0]
    Vb = bf16_pack(np.array(V, F32))
    K = 16
    hit = None
    for trial in range(20000):
        x = Vb[rng.integers(0, len(Vb), K)]
        w = Vb[rng.integers(0, len(Vb), K)]
        s = bf16_pack(kernel_column_sum(x, w, 4, 128, K))
        d = bf16_pack(kernel_column_sum(x, w, 8, 256, K))
        if s != d:
            hit = (x, w, s, d)
            break
    if hit is None:
        print("unexpected: no counterexample in tie-rich grid")
        sys.exit(1)
    x, w, s, d = hit
    ulp = abs(int(np.int16(np.uint16(s))) - int(np.int16(np.uint16(d))))
    print(f"K=16 tie-rich grid trial {trial}: shipped {hex(s)} vs "
          f"uvec4-widen {hex(d)} (ULP {ulp}) -> NOT bit-preserving")
    np.savez(sys.argv[1] if len(sys.argv) > 1 else "counterexample_K16.npz",
             x=x, w=w, shipped_bits=np.uint16(s), candidate_bits=np.uint16(d),
             trial=trial)
    print("span2 (same lane mapping): bit-identical by construction")

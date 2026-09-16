# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# Probe: which kernel reduction reproduces mx.matmul fp16 [1,K]@[K,N] on the
# omarchy backend bit-exactly? Candidates: fp32 ascending accumulation and
# BNNS-style exact_fma16 128-block chains.
import sys
import numpy as np
import mlx.core as mx

HEADER = """
float16_t exact_fma16(float16_t acc, float16_t x, float16_t y) {
    precise float af = float(acc);
    precise float p = float(x) * float(y);
    precise float s = af + p;
    precise float bv = s - af;
    precise float l = (af - (s - bv)) + (p - bv);
    float16_t r = float16_t(s);
    if (l == 0.0f) return r;
    uint u = floatBitsToUint(s);
    uint biased = (u >> 23) & 0xFFu;
    if (biased == 0xFFu || biased < 114u) return r;
    uint mant = u & 0x7FFFFFu;
    if (((mant >> 12) & 1u) == 0u) return r;
    if ((mant & 0xFFFu) != 0u) return r;
    precise float half16 = uintBitsToFloat((biased - 11u) << 23);
    return float16_t(s + (l > 0.0f ? half16 : -half16));
}
"""


def gemv_kernel(name, n_out):
    return mx.fast.metal_kernel(
        name=name,
        input_names=["h", "w", "bias"],
        output_names=["oa", "ob"],
        header=HEADER,
        source=f"""
            uint lane = thread_position_in_grid.x;
            uint n = lane;
            if (n >= {n_out}u) {{ return; }}
            precise float acc = 0.0f;
            float16_t bacc = float16_t(0.0f);
            float16_t bc = float16_t(0.0f);
            for (uint k = 0u; k < 640u; ++k) {{
                precise float prod = float(h[k]) * float(w[k * {n_out}u + n]);
                acc = acc + prod;
                bc = exact_fma16(bc, h[k], w[k * {n_out}u + n]);
                if (((k + 1u) & 127u) == 0u) {{
                    bacc = ((k == 127u) ? bc : float16_t(bacc + bc));
                    bc = float16_t(0.0f);
                }}
            }}
            oa[n] = float16_t(acc) + bias[n];
            ob[n] = bacc + bias[n];
        """,
        compile_options={"math_mode": "safe"},
    )


def run(count=8, seed=7, n_out=640):
    rng = np.random.default_rng(seed)
    # candidate parity on random data first
    for trial in range(count):
        h = (rng.standard_normal(640) * 0.7).astype(np.float16)
        w = (rng.standard_normal((640, n_out)) * 0.05).astype(np.float16)
        bias = (rng.standard_normal(n_out) * 0.3).astype(np.float16)
        with mx.stream(mx.gpu):
            ref = mx.matmul(mx.array(h).reshape(1, 640), mx.array(w).T) + mx.array(bias)
        mx.eval(ref)
        want = np.asarray(ref).astype(np.float16).reshape(-1)
        k = gemv_kernel(f"gemv_probe_{n_out}", n_out)
        out = k(
            inputs=[mx.array(h), mx.array(np.ascontiguousarray(w.T)), mx.array(bias)],
            output_shapes=[(n_out,), (n_out,)],
            output_dtypes=[mx.float16, mx.float16],
            grid=(n_out, 1, 1),
            threadgroup=(256, 1, 1),
            stream=mx.gpu,
        )
        mx.eval(*out)
        got_a = np.asarray(out[0])
        got_b = np.asarray(out[1])
        ba = int((got_a != want).sum())
        bb = int((got_b != want).sum())
        print(f"trial {trial}: fp32-acc {ba}/{n_out} mismatches; bnns-chain {bb}/{n_out}")
        if ba == 0 and bb == 0:
            print("both candidates match on this trial; stopping early")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(run())

# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# Probe: fp16 semantics of the omarchy Vulkan backend against the BNNS
# contract needs. Checks fp16 add/mul, bit-indexed float16_t LUT reads, and
# the exact_fma16 helper (TwoSum + explicit fp16 sticky rounding) that
# replaces the double-rounded hardware fma() path.
import sys
import numpy as np
import mlx.core as mx

_EXACT_FMA16_HEADER = """
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


def golden_fma(a, b, c):
    """rnd16(c + a*b) with an exact fp64 product (fp16 products are exact)."""
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    c64 = c.astype(np.float64)
    with np.errstate(over="ignore"):
        total = c64 + a64 * b64
    return total.astype(np.float16)


def build_kernel():
    return mx.fast.metal_kernel(
        name="fma16_probe",
        input_names=["a", "b", "c", "lut"],
        output_names=["y_fma", "y_add", "y_mul", "y_lut", "y_ex"],
        header=_EXACT_FMA16_HEADER,
        source="""
            uint i = thread_position_in_grid.x;
            float16_t av = a[i];
            float16_t bv = b[i];
            float16_t cv = c[i];
            y_fma[i] = fma(av, bv, cv);
            y_add[i] = av + cv;
            y_mul[i] = av * bv;
            uint bits = packHalf2x16(vec2(float(av), 0.0f)) & 0xFFFFu;
            y_lut[i] = lut[bits];
            y_ex[i] = exact_fma16(cv, av, bv);
        """,
        compile_options={"math_mode": "safe"},
    )


def bits16(values):
    return np.ascontiguousarray(values, dtype=np.float16).view(np.uint16)


def from_bits(bits):
    return np.array([bits], dtype=np.uint16).view(np.float16)[0]


def main() -> int:
    rng = np.random.default_rng(1234)
    count = 1 << 18
    # Dense fp16 sweep, exponent spread bounded so the fp64 golden is exact.
    a = (rng.standard_normal(count) * 2.0).astype(np.float16)
    b = (rng.standard_normal(count) * 2.0).astype(np.float16)
    c = (rng.standard_normal(count) * 8.0).astype(np.float16)
    # Extreme-spread spot values: c tiny vs product, and vice versa.
    a[:256] = np.float16(1.0)
    b[:256] = from_bits(0x3800 + np.arange(256, dtype=np.uint16))
    c[:256] = np.float16(0.0009765625)
    a[256:512] = np.float16(0.000030517578125)
    b[256:512] = np.float16(1.0)
    c[256:512] = np.float16(2048.0)
    # Adversarial ties for the sticky logic: force the fp32-visible tie case
    # (round bit set, visible sticky zero) with a nonzero far tail.
    for slot in range(512, 1024, 2):
        acc_bits = 0x3C00 + np.uint16(rng.integers(0, 0x40))  # ~[1, 2)
        a[slot] = from_bits(acc_bits)
        b[slot] = from_bits(np.uint16(acc_bits - 0x1000))  # ~[0.0625, ...)
        tail = np.uint16(rng.integers(1, 16))
        c[slot] = from_bits(np.uint16(0x0000 + tail))  # fp16 subnormal tail
        a[slot + 1] = a[slot]
        b[slot + 1] = b[slot]
        c[slot + 1] = np.float16(0.0)  # exact-tie sibling: l == 0
    a[1024] = np.float16(0.25)
    b[1024] = np.float16(-0.0078125)
    c[1024] = from_bits(0x4570)  # exact fp32 tie at 1391.5 ulp; l == 0
    lut = np.arange(65536, dtype=np.uint16).view(np.float16)
    lut_mx = mx.array(np.ascontiguousarray(lut))

    kernel = build_kernel()
    out = kernel(
        inputs=[mx.array(a), mx.array(b), mx.array(c), lut_mx],
        output_shapes=[(count,)] * 5,
        output_dtypes=[mx.float16] * 5,
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )
    mx.eval(*out)
    got_fma = np.asarray(out[0])
    got_add = np.asarray(out[1])
    got_mul = np.asarray(out[2])
    got_lut = np.asarray(out[3])
    got_ex = np.asarray(out[4])

    want_fma = golden_fma(a, b, c)
    want_add = (c.astype(np.float64) + a.astype(np.float64)).astype(np.float16)
    want_mul = (a.astype(np.float64) * b.astype(np.float64)).astype(np.float16)

    ex_bad = np.flatnonzero(got_ex != want_fma)
    add_bad = np.flatnonzero(got_add != want_add)
    mul_bad = np.flatnonzero(got_mul != want_mul)
    lut_bad = np.flatnonzero(got_lut.astype(np.float16) != lut[bits16(a)])
    fma_bad = np.flatnonzero(got_fma != want_fma)
    print(
        f"exact_fma16 {len(ex_bad)}/{count} mismatches; add {len(add_bad)}; "
        f"mul {len(mul_bad)}; lut {len(lut_bad)}; "
        f"(hw fma reference: {len(fma_bad)} mismatched)"
    )
    for name, bad, got, want in (
        ("exact", ex_bad, got_ex, want_fma),
        ("add", add_bad, got_add, want_add),
        ("mul", mul_bad, got_mul, want_mul),
    ):
        for i in bad[:3]:
            print(
                f"  {name}[{i}]: got {bits16(got[i:i+1])[0, None][0][()]:#06x} "
                f"want {bits16(want[i:i+1])[0, None][0][()]:#06x}"
            )
    verdict = not (len(ex_bad) or len(add_bad) or len(mul_bad) or len(lut_bad))
    print("PROBE", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

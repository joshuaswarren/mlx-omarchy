# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Stride-aware ANEC output decoding + matched-layout comparison helpers.

The ANEC output surface is allocated with 0x4000-aligned row strides; the
logical tensor may be denser. Decoding with the logical shape alone
misaligns every row after the first. These helpers decode with the
manifest stride and produce the comparison record (bit equality, ULP,
inf/nan masks) used by the F placement differential.
"""

import numpy as np

ALIGN = 0x4000


def stride_for(shape, dtype_bytes: int) -> int:
    """Row stride for a 4-D NCHW fp16-class tensor, 0x4000-aligned."""
    if len(shape) < 2:
        return 0
    row_bytes = shape[-2] * shape[-1] * dtype_bytes
    return (row_bytes + ALIGN - 1) // ALIGN * ALIGN


def decode_strided(raw: bytes, shape, dtype, stride: int) -> np.ndarray:
    """Decode a strided surface into the logical dense array."""
    dt = np.dtype(dtype)
    rows = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    row_elems = shape[-2] * shape[-1] if len(shape) > 2 else shape[-1]
    row_bytes = row_elems * dt.itemsize
    if stride < row_bytes:
        raise ValueError(f"stride {stride} smaller than row bytes {row_bytes}")
    out = np.empty(rows * row_elems, dtype=dt)
    for r in range(rows):
        chunk = np.frombuffer(raw, dtype=dt, count=row_elems,
                              offset=r * stride)
        out[r * row_elems:(r + 1) * row_elems] = chunk
    return out.reshape(shape)


def compare_record(dev: np.ndarray, ref: np.ndarray) -> dict:
    """Bit-level comparison record: exact/ULP + inf/nan mask agreement."""
    d16 = np.ascontiguousarray(dev).astype(np.float16)
    r16 = np.ascontiguousarray(ref).astype(np.float16)
    d32 = d16.astype(np.float32)
    r32 = r16.astype(np.float32)
    diff = np.abs(d32 - r32)
    finite = np.isfinite(diff)
    d16v, r16v = d16.view(np.uint16), r16.view(np.uint16)
    return {
        "bit_equal": bool(np.array_equal(d16v, r16v)),
        "mismatch_count": int((d16v != r16v).sum()),
        "total": int(d16v.size),
        "max_abs_finite": float(diff[finite].max()) if finite.any() else None,
        "dev_inf": int(np.isinf(d32).sum()),
        "ref_inf": int(np.isinf(r32).sum()),
        "dev_neginf": int(np.isneginf(d32).sum()),
        "ref_neginf": int(np.isneginf(r32).sum()),
        "dev_posinf": int(np.isposinf(d32).sum()),
        "ref_posinf": int(np.isposinf(r32).sum()),
        "dev_nan": int(np.isnan(d32).sum()),
        "ref_nan": int(np.isnan(r32).sum()),
    }


def test_stride_for_pads_to_alignment():
    s = stride_for((1, 1, 375, 375), 2)
    assert s % ALIGN == 0 and s >= 375 * 375 * 2


def test_decode_strided_roundtrip():
    shape = (1, 1, 4, 4)
    dt = np.float16
    dense = np.arange(16, dtype=dt).reshape(shape)
    row_bytes = 4 * 2
    stride = 0x4000  # heavily padded rows
    raw = dense.tobytes() + b"\x00" * (stride - row_bytes)
    out = decode_strided(bytes(raw), shape, dt, stride)
    assert np.array_equal(out.reshape(-1), dense.reshape(-1))


def test_decode_strided_rejects_bad_stride():
    import pytest
    with pytest.raises(ValueError, match="stride"):
        decode_strided(b"\0" * 4, (1, 1, 2, 2), np.float16, 2)


def test_compare_record_counts_and_masks():
    dev = np.array([[1.0, float("-inf")], [float("nan"), 2.0]],
                   dtype=np.float16)
    ref = np.array([[1.0, float("-inf")], [float("nan"), 3.0]],
                   dtype=np.float16)
    rec = compare_record(dev, ref)
    assert rec["bit_equal"] is False
    assert rec["mismatch_count"] == 1
    assert rec["dev_neginf"] == 1 and rec["ref_neginf"] == 1
    assert rec["dev_nan"] == 1 and rec["ref_nan"] == 1
    same = compare_record(ref, ref)
    assert same["bit_equal"] is True and same["mismatch_count"] == 0

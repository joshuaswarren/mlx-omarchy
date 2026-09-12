# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Exact parity tests for the portable vDSP_dotpr reduction."""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

from dotpr import mel_projection, portable_dotpr
from generate_dotpr_probes import probe_cases


MACOS_RESULTS = {
    "fma-two-term": 0xBDE9787B,
    "vector-four": 0x423CABE1,
    "association-eight": 0x450C31FC,
    "vector-tail-seventeen": 0x44252851,
    "production-length-257": 0xC7D65DF8,
    "signed-zero": 0x00000000,
}


def floats(bits: list[int]) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


@pytest.mark.parametrize("name,lhs_bits,rhs_bits", probe_cases())
def test_portable_dotpr_matches_macos_vdsp_bits(name, lhs_bits, rhs_bits):
    actual = portable_dotpr(floats(lhs_bits), floats(rhs_bits))
    assert actual.view(np.uint32).item() == MACOS_RESULTS[name]


def test_mel_projection_uses_the_same_reduction_for_every_pair():
    _, lhs_bits, rhs_bits = probe_cases()[0]
    lhs = np.stack([floats(lhs_bits), floats(rhs_bits)])
    rhs = np.stack([floats(rhs_bits), floats(lhs_bits)])
    actual = mel_projection(lhs, rhs)
    expected = np.asarray(
        [[portable_dotpr(row, vector) for row in lhs] for vector in rhs],
        dtype=np.float32,
    )
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize(
    "lhs,rhs,message",
    [
        (np.ones(2, dtype=np.float64), np.ones(2, dtype=np.float32), "float32"),
        (np.ones((1, 2), dtype=np.float32), np.ones(2, dtype=np.float32), "one-dimensional"),
        (np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32), "same length"),
        (np.ones(0, dtype=np.float32), np.ones(0, dtype=np.float32), "non-empty"),
    ],
)
def test_portable_dotpr_rejects_invalid_vectors(lhs, rhs, message):
    with pytest.raises(ValueError, match=message):
        portable_dotpr(lhs, rhs)

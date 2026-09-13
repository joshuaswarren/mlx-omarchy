# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Exact pinned-audio frontend checks against the macOS Parakeet fixture."""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

from mel_reference import compare_capture, hann_window
from reference import default_cache_root

CAPTURE = (
    default_cache_root()
    / "captures"
    / "b650695c-75aec2a"
    / "20260912T154759Z-librispeech"
    / "ane"
)


def test_pinned_audio_to_mel_is_bit_exact():
    if not CAPTURE.is_dir():
        pytest.skip(f"pinned macOS capture is not installed at {CAPTURE}")

    report = compare_capture(CAPTURE)

    assert report.exact
    assert report.bit_exact_count == report.total == 3001 * 128
    assert all(report.structural.values())


def test_hann_window_rejects_unqualified_lengths():
    with pytest.raises(ValueError, match="pinned 400-sample Hann window"):
        hann_window(512)

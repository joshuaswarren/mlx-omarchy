# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""The numpy MIL evaluator reproduces the authenticated golden encoder output
inside the reference lock's frozen contract."""

import hashlib
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))

from coreml.mil_adapter import emit_mlpackage
from coreml.mil_numpy import validate
from coreml.reference import ReferenceLock, default_cache_root

LOCK_PATH = TOOLS / "coreml" / "parakeet-reference.lock"
LOCK = ReferenceLock.load(LOCK_PATH)
PACKAGE = default_cache_root() / LOCK.model_repo / LOCK.model_revision / "encoder.mlpackage"
CAPTURE = (
    default_cache_root()
    / "captures"
    / "b650695c-75aec2a"
    / "20260912T154759Z-librispeech"
    / "ane"
)
GOLDEN = ("encoder_input_features.npy", "encoder_input_mask.npy", "encoder_hidden.npy")


def test_encoder_matches_golden_capture(tmp_path):
    if not PACKAGE.is_dir():
        pytest.skip("pinned encoder.mlpackage is not installed")
    if not all((CAPTURE / name).is_file() for name in GOLDEN):
        pytest.skip("authenticated LibriSpeech ANE capture is not installed")
    for name in GOLDEN:
        digest = hashlib.sha256((CAPTURE / name).read_bytes()).hexdigest()
        assert digest == LOCK.macos_reference_paths[name], name

    source = emit_mlpackage(PACKAGE, tmp_path / "source", LOCK_PATH)
    report = validate(Path(source.mil_path).parent, CAPTURE, LOCK.to_dict()["numerical_contract"])

    assert report["ops_executed"] == 3351
    assert report["encoder_hidden_shape"] == [1, 375, 640]
    assert report["encoder_mask_equal"]
    assert report["within_frozen_contract"], report["measured"]

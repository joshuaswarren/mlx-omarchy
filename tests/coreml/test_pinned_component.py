# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Contract checks for pinned decoder and joint component loading."""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))

from coreml.pinned_component import load_pinned_component
from coreml.reference import ReferenceLock, default_cache_root

LOCK = ReferenceLock.load(TOOLS / "coreml" / "parakeet-reference.lock")
PACKAGE = (
    default_cache_root()
    / LOCK.model_repo
    / LOCK.model_revision
    / "joint.mlpackage"
)


def test_pinned_joint_constants_are_named_and_exact():
    if not PACKAGE.is_dir():
        pytest.skip("pinned joint.mlpackage is not installed")

    component = load_pinned_component(PACKAGE, "joint")

    assert component.package_path == PACKAGE.resolve()
    assert len(component.block.operations) == 21
    assert component.constant("head_weight_to_fp16").shape == (8198, 640)
    assert component.constant("head_weight_to_fp16").dtype == np.float16
    assert component.constant("head_bias_to_fp16").shape == (8198,)
    np.testing.assert_array_equal(component.constant("var_14_begin_0"), [0, 0])
    np.testing.assert_array_equal(component.constant("var_14_end_0"), [1, 8193])
    np.testing.assert_array_equal(component.constant("var_14_end_mask_0"), [True, False])
    np.testing.assert_array_equal(component.constant("var_19_begin_0"), [0, 8193])
    np.testing.assert_array_equal(component.constant("var_19_end_0"), [1, 8198])
    np.testing.assert_array_equal(component.constant("var_19_end_mask_0"), [True, True])


def test_loader_rejects_component_name_that_does_not_match_package():
    if not PACKAGE.is_dir():
        pytest.skip("pinned joint.mlpackage is not installed")

    with pytest.raises(ValueError, match="decoder.mlpackage"):
        load_pinned_component(PACKAGE, "decoder")

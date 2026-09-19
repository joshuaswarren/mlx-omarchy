# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Bit-exactness of the uint32 pair-packed joint buffer.

The loop/step joint kernels read the packed joint buffer as uint32 fp16
pairs in output-lane-major order (``joint[j * 320 + k]`` unpacked with
``unpackHalf2x16``) instead of the former fp16 k-major scalars
(``joint[k * 8198 + j]``).  ``unpackHalf2x16`` widens both fp16 lanes of a
word exactly, and the reduction keeps the pinned fp32 ascending-k chain
with one fp16 rounding plus the fp16 bias add, so only addressing differs.
These tests pin that claim with a numpy model of both addressings, first
on synthetic sentinel weights, then on the real pinned joint component
against the captured joint traces.
"""

import unittest
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.pinned_component import load_pinned_component

_JOINT_OUT = 8198
_VOCAB = 8193
_DIM = 640
_PAIRS = _DIM // 2


def _pair_pack(weights, bias):
    """The pack_step_weights joint layout: fp16 pairs as uint32 words."""
    return np.concatenate(
        [
            np.ascontiguousarray(weights, dtype=np.float16)
            .view(np.uint32)
            .ravel(),
            np.ascontiguousarray(bias, dtype=np.float16)
            .view(np.uint32)
            .ravel(),
        ]
    ).astype(np.uint32)


def _unpack_pair(word):
    """GLSL unpackHalf2x16: low half first, high half second."""
    low = (np.uint32(word) & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (np.uint32(word) >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return low, high


def _relu(encoder_frame, decoder_state):
    """fp16 relu(encoder_frame + decoder_state), the joint hidden vector."""
    added = np.float16(np.float16(encoder_frame) + np.float16(decoder_state))
    return np.where(
        added > np.float16(0.0), added, np.float16(0.0)
    ).astype(np.float16)


def _logits_k_major(joint_fp16, relu):
    """Former kernel addressing: fp16 scalars, joint[k * 8198 + j]."""
    acc = np.zeros(_JOINT_OUT, np.float32)
    for k in range(_DIM):
        acc = (
            acc
            + np.float32(relu[k])
            * joint_fp16[k * _JOINT_OUT : (k + 1) * _JOINT_OUT].astype(np.float32)
        ).astype(np.float32)
    bias = joint_fp16[_DIM * _JOINT_OUT :]
    return np.float32(np.float16(acc) + bias)


def _logits_pair_major(joint_u32, relu):
    """New kernel addressing: uint32 pairs, joint[j * 320 + k]."""
    acc = np.zeros(_JOINT_OUT, np.float32)
    words = joint_u32[: _JOINT_OUT * _PAIRS].reshape(_JOINT_OUT, _PAIRS)
    for k in range(_PAIRS):
        low, high = _unpack_pair(words[:, k])
        acc = (acc + np.float32(relu[2 * k]) * low.astype(np.float32)).astype(
            np.float32
        )
        acc = (
            acc + np.float32(relu[2 * k + 1]) * high.astype(np.float32)
        ).astype(np.float32)
    bias = np.empty(_JOINT_OUT, np.float16)
    bias[0::2], bias[1::2] = _unpack_pair(joint_u32[_JOINT_OUT * _PAIRS :])
    return np.float32(np.float16(acc) + bias)


def _reference_joint_buffers(weights, bias):
    """Old (fp16 k-major) and new (uint32 pair) buffers over the same W/b."""
    old = np.concatenate(
        [
            np.ascontiguousarray(weights.T, dtype=np.float16).ravel(),
            np.asarray(bias, dtype=np.float16).ravel(),
        ]
    )
    new = _pair_pack(weights, bias)
    assert old.nbytes == new.nbytes
    return old, new


class JointPairLayoutTest(unittest.TestCase):
    def test_sentinel_addressing_roundtrip(self):
        """Every weight/bias value the new addressing yields is the pinned value."""
        rng = np.random.default_rng(20260919)
        weights = rng.standard_normal((_JOINT_OUT, _DIM)).astype(np.float16)
        bias = rng.standard_normal(_JOINT_OUT).astype(np.float16)
        packed = _pair_pack(weights, bias)

        low, high = _unpack_pair(packed[: _JOINT_OUT * _PAIRS].reshape(_JOINT_OUT, _PAIRS))
        rows = np.empty((_JOINT_OUT, _DIM), np.float16)
        rows[:, 0::2], rows[:, 1::2] = low, high
        np.testing.assert_array_equal(rows.view(np.uint16), weights.view(np.uint16))

        bias_out = np.empty(_JOINT_OUT, np.float16)
        bias_out[0::2], bias_out[1::2] = _unpack_pair(packed[_JOINT_OUT * _PAIRS :])
        np.testing.assert_array_equal(bias_out.view(np.uint16), bias.view(np.uint16))

    def test_logits_bit_identical_synthetic(self):
        """Old and new addressings produce bit-identical logits and argmaxes."""
        rng = np.random.default_rng(104)
        weights = rng.standard_normal((_JOINT_OUT, _DIM)).astype(np.float16)
        bias = rng.standard_normal(_JOINT_OUT).astype(np.float16)
        old_pack, new_pack = _reference_joint_buffers(weights, bias)
        encoder_frame = rng.standard_normal(_DIM).astype(np.float32)
        decoder_state = rng.standard_normal(_DIM).astype(np.float32)
        relu = _relu(encoder_frame, decoder_state)

        old_logits = _logits_k_major(old_pack, relu)
        new_logits = _logits_pair_major(new_pack, relu)
        np.testing.assert_array_equal(
            old_logits.view(np.uint32), new_logits.view(np.uint32)
        )
        self.assertEqual(int(np.argmax(old_logits[:_VOCAB])), int(np.argmax(new_logits[:_VOCAB])))
        self.assertEqual(
            int(np.argmax(old_logits[_VOCAB:])), int(np.argmax(new_logits[_VOCAB:]))
        )

    def test_real_pinned_joint_bit_exact(self):
        """Bit-identical logits on the pinned component and captured traces."""
        model_root = (
            Path.home()
            / ".cache/mlx-omarchy/parakeet-reference/mweinbach1"
            / "parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018"
        )
        capture = (
            Path.home()
            / ".cache/mlx-omarchy/parakeet-reference/captures"
            / "b650695c-75aec2a/20260913T105550Z-librispeech-tdt-tensors/ane"
        )
        if not model_root.exists() or not capture.exists():
            self.skipTest("pinned parakeet cache not present on this host")
        component = load_pinned_component(model_root / "joint.mlpackage", "joint")
        weights = np.asarray(
            component.constant("head_weight_to_fp16"), dtype=np.float16
        )
        bias = np.asarray(
            component.constant("head_bias_to_fp16"), dtype=np.float16
        ).ravel()
        self.assertEqual(weights.shape, (_JOINT_OUT, _DIM))
        old_pack, new_pack = _reference_joint_buffers(weights, bias)

        traces = sorted(capture.glob("tdt_trace_*_joint_encoder_frame.npy"))
        self.assertGreater(len(traces), 0)
        for frame_file in traces:
            index = frame_file.name.split("_")[2]
            encoder_frame = np.load(
                capture / f"tdt_trace_{index}_joint_encoder_frame.npy"
            ).ravel()
            decoder_state = np.load(
                capture / f"tdt_trace_{index}_joint_decoder_state.npy"
            ).ravel()
            relu = _relu(encoder_frame, decoder_state)
            old_logits = _logits_k_major(old_pack, relu)
            new_logits = _logits_pair_major(new_pack, relu)
            np.testing.assert_array_equal(
                old_logits.view(np.uint32), new_logits.view(np.uint32)
            )
            self.assertEqual(
                int(np.argmax(old_logits[:_VOCAB])),
                int(np.argmax(new_logits[:_VOCAB])),
            )
            self.assertEqual(
                int(np.argmax(old_logits[_VOCAB:])),
                int(np.argmax(new_logits[_VOCAB:])),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

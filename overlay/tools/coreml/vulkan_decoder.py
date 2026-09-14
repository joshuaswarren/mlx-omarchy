# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Parakeet-specific MLX GPU execution for the pinned decoder package.

The two recurrent layers follow Apple's MIL ``lstm`` definition: input,
forget, output, cell (IFOC) gate packing; row-major ``[4H, I]`` and
``[4H, H]`` weights; correctly-rounded fp16 sigmoid recurrent gates; and
correctly-rounded fp16 tanh cell/output activations (macOS Core ML CPU).
CoreML8 inherits those equations from the iOS 15 operation and adds fp16
support in the iOS 17 definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .pinned_component import (
    PinnedComponent,
    PinnedComponentError,
    load_pinned_component,
)
from .reference import ReferenceLock

_HIDDEN_SIZE = 640
_VOCAB_SIZE = 8193
_INPUT_IDS_SHAPE = (1, 1)
_STATE_SHAPE = (2, 1, _HIDDEN_SIZE)

_CONSTANT_SPECS = {
    "embedding_weight_to_fp16": ((_VOCAB_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_1_to_fp16": ((4 * _HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_2_to_fp16": ((4 * _HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_0_to_fp16": ((4 * _HIDDEN_SIZE,), np.dtype("<f2")),
    "concat_4_to_fp16": ((4 * _HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_5_to_fp16": ((4 * _HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_3_to_fp16": ((4 * _HIDDEN_SIZE,), np.dtype("<f2")),
    "projector_weight_to_fp16": ((_HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "projector_bias_to_fp16": ((_HIDDEN_SIZE,), np.dtype("<f2")),
}

_EXPECTED_HISTOGRAM = {
    "add": 1,
    "cast": 8,
    "const": 44,
    "gather": 1,
    "greater_equal": 1,
    "linear": 1,
    "lstm": 2,
    "select": 1,
    "split": 2,
    "squeeze": 4,
    "stack": 2,
    "transpose": 2,
}

_F16 = np.dtype("<f2")


def fp16_sigmoid(x):
    """Correctly-rounded binary16 sigmoid: round ``1/(1+exp(-x))`` once."""
    x64 = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        y = 1.0 / (1.0 + np.exp(-x64))
    return np.asarray(y, dtype=_F16)


def fp16_tanh(x):
    """Correctly-rounded binary16 tanh: round ``tanh(x)`` once."""
    return np.asarray(np.tanh(np.asarray(x, dtype=np.float64)), dtype=_F16)


def _validate_graph(component: PinnedComponent) -> None:
    histogram = dict(sorted(Counter(op.type for op in component.block.operations).items()))
    if histogram != _EXPECTED_HISTOGRAM:
        raise PinnedComponentError(
            f"pinned decoder operation inventory differs: {histogram}"
        )
    if tuple(component.block.outputs) != (
        "decoder_hidden",
        "next_hidden",
        "next_cell",
    ):
        raise PinnedComponentError("pinned decoder output bindings differ")


class DecoderResult(NamedTuple):
    decoder_hidden: object
    next_hidden: object
    next_cell: object


class _Weights(NamedTuple):
    embedding: object
    layer_0_ih: object
    layer_0_hh: object
    layer_0_bias: object
    layer_1_ih: object
    layer_1_hh: object
    layer_1_bias: object
    projector: object
    projector_bias: object


@cache
def _mlx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("the pinned Parakeet decoder requires mlx-omarchy") from exc
    return mx


def _constant(component: PinnedComponent, name: str) -> np.ndarray:
    value = component.constant(name)
    shape, dtype = _CONSTANT_SPECS[name]
    if value.shape != shape or value.dtype != dtype:
        raise PinnedComponentError(
            f"pinned decoder constant {name!r} must be {dtype.name}{shape}, "
            f"got {value.dtype.name}{value.shape}"
        )
    return value


def _load_weights(component: PinnedComponent, mx) -> _Weights:
    values = [_constant(component, name) for name in _CONSTANT_SPECS]
    with mx.stream(mx.gpu):
        arrays = [mx.array(value) for value in values]
    mx.eval(*arrays)
    return _Weights(*arrays)


def _validate_array(name, value, shape, dtype, mx) -> None:
    if not isinstance(value, mx.array):
        raise TypeError(f"{name} must be an mlx.core.array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have {dtype} dtype, got {value.dtype}")


def _cr_fp16(fn, value, mx):
    return mx.array(fn(np.asarray(value)), dtype=mx.float16)


def _lstm(sequence, hidden, cell, weight_ih, weight_hh, bias, mx):
    gates = sequence[0] @ weight_ih.T + hidden @ weight_hh.T + bias
    input_gate, forget_gate, output_gate, cell_gate = mx.split(gates, 4, axis=-1)
    next_cell = (
        _cr_fp16(fp16_sigmoid, forget_gate, mx) * cell
        + _cr_fp16(fp16_sigmoid, input_gate, mx) * _cr_fp16(fp16_tanh, cell_gate, mx)
    )
    next_hidden = _cr_fp16(fp16_sigmoid, output_gate, mx) * _cr_fp16(
        fp16_tanh, next_cell, mx
    )
    return mx.expand_dims(next_hidden, axis=0), next_hidden, next_cell


class VulkanDecoder:
    """Pinned decoder: GPU matmul, correctly-rounded fp16 activations."""

    __slots__ = ("_weights",)

    def __init__(self, component: PinnedComponent):
        _validate_graph(component)
        self._weights = _load_weights(component, _mlx())

    def __call__(self, input_ids, hidden, cell) -> DecoderResult:
        mx = _mlx()
        _validate_array("input_ids", input_ids, _INPUT_IDS_SHAPE, mx.int32, mx)
        _validate_array("hidden", hidden, _STATE_SHAPE, mx.float32, mx)
        _validate_array("cell", cell, _STATE_SHAPE, mx.float32, mx)
        weights = self._weights

        with mx.stream(mx.gpu):
            indices = input_ids.astype(mx.int16).astype(mx.int32)
            indices = mx.where(indices >= 0, indices, indices + _VOCAB_SIZE)
            indices = indices.astype(mx.int16)
            embedded = mx.take(weights.embedding, indices, axis=0)
            sequence = mx.transpose(embedded, (1, 0, 2))
            hidden_fp16 = hidden.astype(mx.float16)
            cell_fp16 = cell.astype(mx.float16)
            sequence, hidden_0, cell_0 = _lstm(
                sequence,
                hidden_fp16[0],
                cell_fp16[0],
                weights.layer_0_ih,
                weights.layer_0_hh,
                weights.layer_0_bias,
                mx,
            )
            sequence, hidden_1, cell_1 = _lstm(
                sequence,
                hidden_fp16[1],
                cell_fp16[1],
                weights.layer_1_ih,
                weights.layer_1_hh,
                weights.layer_1_bias,
                mx,
            )
            decoder_hidden = (
                mx.transpose(sequence, (1, 0, 2)) @ weights.projector.T
                + weights.projector_bias
            ).astype(mx.float32)
            next_hidden = mx.stack((hidden_0, hidden_1), axis=0).astype(mx.float32)
            next_cell = mx.stack((cell_0, cell_1), axis=0).astype(mx.float32)
        return DecoderResult(decoder_hidden, next_hidden, next_cell)


def load_decoder(package_path: Path) -> VulkanDecoder:
    """Hash-validate and load the exact pinned ``decoder.mlpackage``."""
    return VulkanDecoder(load_pinned_component(package_path, "decoder"))


def _tensor_summary(value) -> dict:
    materialized = np.asarray(value)
    return {
        "shape": list(materialized.shape),
        "dtype": str(materialized.dtype),
        "finite": bool(np.isfinite(materialized).all()),
        "sha256": hashlib.sha256(materialized.tobytes()).hexdigest(),
    }


def smoke(package_path: Path) -> dict:
    """Run two dependent decoder transitions and return a machine-readable receipt."""
    mx = _mlx()
    decoder = load_decoder(package_path)
    with mx.stream(mx.gpu):
        hidden = mx.zeros(_STATE_SHAPE, dtype=mx.float32)
        cell = mx.zeros(_STATE_SHAPE, dtype=mx.float32)
        first = decoder(mx.array([[0]], dtype=mx.int32), hidden, cell)
        second = decoder(
            mx.array([[8192]], dtype=mx.int32),
            first.next_hidden,
            first.next_cell,
        )
    mx.eval(*first, *second)
    summaries = {
        "first": {name: _tensor_summary(value) for name, value in first._asdict().items()},
        "second": {name: _tensor_summary(value) for name, value in second._asdict().items()},
    }
    if not all(item["finite"] for step in summaries.values() for item in step.values()):
        raise RuntimeError("pinned decoder smoke produced a non-finite output")
    lock = ReferenceLock.load()
    return {
        "schema": "mlx-omarchy.parakeet-vulkan-decoder-smoke/1",
        "component": "decoder.mlpackage",
        "model_revision": lock.model_revision,
        "device": str(mx.default_device()),
        "tokens": [0, 8192],
        "state_reused_on_gpu": True,
        "outputs": summaries,
        "native_coreml_parity": False,
        "native_capture_required": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(smoke(args.package), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

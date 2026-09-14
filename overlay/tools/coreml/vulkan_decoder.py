# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Parakeet-specific MLX GPU execution for the pinned decoder package.

The two recurrent layers follow Apple's MIL ``lstm`` definition: input,
forget, output, cell (IFOC) gate packing; row-major ``[4H, I]`` and
``[4H, H]`` weights; sigmoid recurrent gates; and tanh cell/output
activations. CoreML8 inherits those equations from the iOS 15 operation and
adds fp16 support in the iOS 17 definition.

The trailing projector is not an ``mx`` matmul. Its native arithmetic is
established bit-exactly against the authenticated capture as a serial fp16
accumulator over unrounded products, so it runs as the custom Vulkan kernel
below.
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
    projector_codes: object
    projector_bias_codes: object


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
    constants = {name: _constant(component, name) for name in _CONSTANT_SPECS}
    # The projector kernel reduces one output column per thread and walks the
    # reduction index serially, so the weight travels transposed: element
    # (k, column) of the transpose is what consecutive threads read at step k,
    # which is the coalesced read the layout wants. Both projector operands
    # travel as fp16 bit patterns, decoded in the shader, so no part of the
    # reduction depends on the device's fp16 arithmetic, rounding mode or
    # denormal mode.
    with mx.stream(mx.gpu):
        arrays = [
            mx.array(constants["embedding_weight_to_fp16"]),
            mx.array(constants["concat_1_to_fp16"]),
            mx.array(constants["concat_2_to_fp16"]),
            mx.array(constants["concat_0_to_fp16"]),
            mx.array(constants["concat_4_to_fp16"]),
            mx.array(constants["concat_5_to_fp16"]),
            mx.array(constants["concat_3_to_fp16"]),
            mx.array(
                np.ascontiguousarray(
                    constants["projector_weight_to_fp16"].T
                ).view(np.uint16)
            ),
            mx.array(constants["projector_bias_to_fp16"].view(np.uint16)),
        ]
    mx.eval(*arrays)
    return _Weights(*arrays)


def _validate_array(name, value, shape, dtype, mx) -> None:
    if not isinstance(value, mx.array):
        raise TypeError(f"{name} must be an mlx.core.array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have {dtype} dtype, got {value.dtype}")


def _lstm(sequence, hidden, cell, weight_ih, weight_hh, bias, mx):
    gates = sequence[0] @ weight_ih.T + hidden @ weight_hh.T + bias
    input_gate, forget_gate, output_gate, cell_gate = mx.split(gates, 4, axis=-1)
    next_cell = (
        mx.sigmoid(forget_gate) * cell
        + mx.sigmoid(input_gate) * mx.tanh(cell_gate)
    )
    next_hidden = mx.sigmoid(output_gate) * mx.tanh(next_cell)
    return mx.expand_dims(next_hidden, axis=0), next_hidden, next_cell


# Native Core ML evaluates the projector as an fp16 accumulator over unrounded
# products, reduced in strictly ascending index order, with the fp16 bias added
# after the reduction. That is bit-exact on every captured projector lane and it
# is unique: rounding each product instead, reversing the reduction order, or
# seeding the accumulator with the bias all break it, and no fp32 reduction
# reaches it. The contract is serial in the reduction index, so it is not an
# `mx` matmul; this kernel is the contract.
#
# Every step needs the exact sum of an fp16 accumulator and an exact fp16
# product rounded once to fp16. Accumulating in fp32 and narrowing each step
# rounds twice, and two roundings are not one: on the captured transitions that
# loses 94 of 9600 lanes. So each step takes a Knuth two-sum to recover the
# exact residual of the fp32 add, forces the fp32 sum odd whenever that residual
# is non-zero, and narrows once. A float16 midpoint has thirteen trailing zero
# bits in float32 and so is never odd, which is what makes the single narrowing
# correctly rounded.
#
# Both the fp16 decode and the fp16 narrowing are integer arithmetic, and the
# operands arrive as fp16 bit patterns, so the kernel needs only IEEE float32
# multiply and add from the device. It does not depend on device fp16
# arithmetic, on the fp16 rounding mode, or on fp16 denormal handling - and the
# pinned projector weight does carry 437 fp16 denormals, which a driver is
# allowed to flush.
#
# Three identifiers are unavailable inside the shader. `discard` is a GLSL
# keyword, `step` would shadow a GLSL built-in, and `half` is a Metal type name
# that the MSL-to-GLSL translation rewrites to `float16_t` wherever it appears
# as a word.
_PROJECTOR_HEADER = """
float decode_fp16(uint code) {
    uint sign = (code & 0x8000u) << 16u;
    uint exponent = (code >> 10u) & 0x1fu;
    uint mantissa = code & 0x3ffu;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            return uintBitsToFloat(sign);
        }
        uint leading = uint(findMSB(mantissa));
        return uintBitsToFloat(
            sign | ((103u + leading) << 23u)
            | ((mantissa << (23u - leading)) & 0x7fffffu));
    }
    if (exponent == 31u) {
        return uintBitsToFloat(sign | 0x7f800000u | (mantissa << 13u));
    }
    return uintBitsToFloat(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
}

float narrow_fp16(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = bits & 0x80000000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        return value;
    }
    if (magnitude < 0x33800000u) {
        return uintBitsToFloat(
            magnitude > 0x33000000u ? (sign | 0x33800000u) : sign);
    }
    int exponent = int(magnitude >> 23u) - 127;
    uint dropped = 13u + uint(max(0, -14 - exponent));
    uint spacing = 1u << dropped;
    uint truncated = magnitude & ~(spacing - 1u);
    uint remainder = magnitude - truncated;
    uint midpoint = spacing >> 1u;
    bool up = remainder > midpoint
        || (remainder == midpoint && (truncated & spacing) != 0u);
    uint rounded = up ? truncated + spacing : truncated;
    if (rounded > 0x477fe000u) {
        return uintBitsToFloat(sign | 0x7f800000u);
    }
    return uintBitsToFloat(sign | rounded);
}

float force_odd(float total, float residual) {
    uint bits = floatBitsToUint(total);
    if (residual == 0.0f || total == 0.0f || (bits & 1u) != 0u) {
        return total;
    }
    if (isinf(total) || isnan(total)) {
        return total;
    }
    bool away = (residual > 0.0f) == ((bits & 0x80000000u) == 0u);
    return uintBitsToFloat(away ? bits + 1u : bits - 1u);
}

float accumulate_fp16(float accumulator, float addend) {
    precise float total = accumulator + addend;
    precise float upper = total - addend;
    precise float lower = total - upper;
    precise float residual = (accumulator - upper) + (addend - lower);
    return narrow_fp16(force_odd(total, residual));
}
"""

_PROJECTOR_SOURCE = f"""
    uint column = thread_position_in_grid.x;
    float accumulator = 0.0f;
    for (uint k = 0u; k < {_HIDDEN_SIZE}u; ++k) {{
        precise float product = decode_fp16(uint(hidden[k]))
            * decode_fp16(uint(weight[k * {_HIDDEN_SIZE}u + column]));
        accumulator = accumulate_fp16(accumulator, product);
    }}
    projected[column] = accumulate_fp16(
        accumulator, decode_fp16(uint(bias[column])));
"""

# One output column per thread. The reduction cannot be split, so columns are
# the only parallelism and the group size is the only launch knob. Measured on
# the M1: 32, 64 and 128 are within noise of each other, 256 costs 6 percent
# and a single group of 640, which is a single core, costs 39 percent.
_PROJECTOR_GROUP = 64


@cache
def _projector_kernel(mx=None):
    mx = mx or _mlx()
    return mx.fast.metal_kernel(
        name="parakeet_decoder_projector_fp16_serial",
        input_names=["hidden", "weight", "bias"],
        output_names=["projected"],
        header=_PROJECTOR_HEADER,
        source=_PROJECTOR_SOURCE,
        compile_options={"math_mode": "safe"},
    )


def _project(hidden, weights, mx):
    """The native projector contract: ``decoder_hidden`` for one decoder step."""
    return _projector_kernel(mx)(
        inputs=[
            mx.view(hidden, mx.uint16),
            weights.projector_codes,
            weights.projector_bias_codes,
        ],
        output_shapes=[(1, 1, _HIDDEN_SIZE)],
        output_dtypes=[mx.float32],
        grid=(_HIDDEN_SIZE, 1, 1),
        threadgroup=(_PROJECTOR_GROUP, 1, 1),
        stream=mx.gpu,
    )[0]


class VulkanDecoder:
    """Callable pinned decoder whose tensor operations run on ``mx.gpu``."""

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
            _, hidden_1, cell_1 = _lstm(
                sequence,
                hidden_fp16[1],
                cell_fp16[1],
                weights.layer_1_ih,
                weights.layer_1_hh,
                weights.layer_1_bias,
                mx,
            )
            decoder_hidden = _project(hidden_1, weights, mx)
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

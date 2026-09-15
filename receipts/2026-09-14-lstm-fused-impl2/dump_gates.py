# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Dump the Vulkan-decoder GPU gate preacts for every ran_decoder capture trace.

Replicates overlay/tools/coreml/vulkan_decoder.py::__call__ up to the gate
split (before any unary), chaining layer 1 two ways:

  g0          layer-0 gates from the native fp16 input states
  g1_native   layer-1 gates with the native layer-0 next_hidden injected
  g1_cr       layer-1 gates chained through the correctly-rounded layer-0
              next_hidden (what origin/main ships today)

All arrays are the exact fp16 GPU values the shipped decoder feeds its
unaries. The whole dump is computed twice and compared bit-for-bit.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

from coreml.pinned_component import load_pinned_component

CAPTURE = Path("/home/joshuawarren/tmp/coreml-tdt-0cf2d148/capture/ane")
MODEL = Path("/home/joshuawarren/tmp/coreml-tdt-0cf2d148/model")
OUT = Path("/tmp/vulkan-tdt-142/gate_dump.npz")

_SPECS = {
    "embedding_weight_to_fp16": ((8193, 640), np.dtype("<f2")),
    "concat_1_to_fp16": ((2560, 640), np.dtype("<f2")),
    "concat_2_to_fp16": ((2560, 640), np.dtype("<f2")),
    "concat_0_to_fp16": ((2560,), np.dtype("<f2")),
    "concat_4_to_fp16": ((2560, 640), np.dtype("<f2")),
    "concat_5_to_fp16": ((2560, 640), np.dtype("<f2")),
    "concat_3_to_fp16": ((2560,), np.dtype("<f2")),
}


def sigmoid_tanh(gates_o, gates_g):
    with mx.stream(mx.gpu):
        h0 = (mx.sigmoid(gates_o) * mx.tanh(gates_g)).astype(mx.float16)
    mx.eval(h0)
    return h0


def layer_gates(sequence_row, hidden_row, wih, whh, bias):
    with mx.stream(mx.gpu):
        gates = sequence_row @ wih.T + hidden_row @ whh.T + bias
        i, f, o, g = mx.split(gates, 4, axis=-1)
    mx.eval(i, f, o, g)
    return i, f, o, g


def run_once(traces):
    component = load_pinned_component(MODEL / "decoder.mlpackage", "decoder")
    names = list(_SPECS)
    with mx.stream(mx.gpu):
        arrays = [mx.array(component.constant(n)) for n in names]
    mx.eval(*arrays)
    embedding, wih0, whh0, b0, wih1, whh1, b1 = arrays

    dump = {}
    for trace in traces:
        index = trace["index"]
        paths = trace["tensor_paths"]
        ids = np.load(CAPTURE / paths["decoder_input_ids"])
        hidden = np.load(CAPTURE / paths["decoder_hidden"])
        cell = np.load(CAPTURE / paths["decoder_cell"])
        native_next_hidden = np.load(CAPTURE / paths["decoder_next_hidden"])
        with mx.stream(mx.gpu):
            input_ids = mx.array(ids.astype(np.int32))
            hidden_mx = mx.array(hidden)
            cell_mx = mx.array(cell)
            indices = input_ids.astype(mx.int16).astype(mx.int32)
            indices = mx.where(indices >= 0, indices, indices + 8193)
            indices = indices.astype(mx.int16)
            embedded = mx.take(embedding, indices, axis=0)
            sequence = mx.transpose(embedded, (1, 0, 2))
            hidden_fp16 = hidden_mx.astype(mx.float16)
            cell_fp16 = cell_mx.astype(mx.float16)
            i0, f0, o0, g0 = layer_gates(
                sequence[0], hidden_fp16[0], wih0, whh0, b0
            )
            h0_native = mx.array(native_next_hidden[0].astype(np.float16))
            i1n, f1n, o1n, g1n = layer_gates(
                h0_native, hidden_fp16[1], wih1, whh1, b1
            )
            h0_cr = sigmoid_tanh(o0, g0)
            i1c, f1c, o1c, g1c = layer_gates(
                h0_cr, hidden_fp16[1], wih1, whh1, b1
            )
        for name, value in (
            ("hidden16_0", hidden_fp16[0]), ("cell16_0", cell_fp16[0]),
            ("hidden16_1", hidden_fp16[1]), ("cell16_1", cell_fp16[1]),
            ("i0", i0), ("f0", f0), ("o0", o0), ("g0", g0),
            ("i1n", i1n), ("f1n", f1n), ("o1n", o1n), ("g1n", g1n),
            ("i1c", i1c), ("f1c", f1c), ("o1c", o1c), ("g1c", g1c),
        ):
            dump[f"t{index:04d}_{name}"] = np.asarray(value)
    return dump


def main() -> int:
    traces = json.loads((CAPTURE / "tdt_tensors.json").read_text())["traces"]
    ran = [t for t in traces if t.get("ran_decoder")]
    started = time.monotonic()
    first = run_once(ran)
    second = run_once(ran)
    mismatch = [k for k in first if not np.array_equal(first[k], second[k])]
    if mismatch:
        raise SystemExit(f"GPU gate dump not deterministic: {mismatch}")
    np.savez(OUT, **first)
    print(
        json.dumps(
            {
                "host": platform.node(),
                "device": str(mx.device_info().get("device_name")),
                "mlx": mx.__version__,
                "traces": len(ran),
                "arrays": len(first),
                "deterministic": True,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

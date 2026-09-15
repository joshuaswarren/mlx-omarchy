# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Gate-dump v2: compare the shipped two-matmul gate form against the packed
single-matmul Core ML lstm form (concat(x, h) @ [wih | whh].T + b, fp32
accumulate, one fp16 rounding) for every ran_decoder trace, both layers,
with the native layer-0 output injected into layer 1.
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
OUT = Path("/tmp/vulkan-tdt-142/gate_dump2.npz")

_SPECS = {
    "embedding_weight_to_fp16": ((8193, 640), np.dtype("<f2")),
    "concat_1_to_fp16": ((2560, 640), np.dtype("<f2")),
    "concat_2_to_fp16": ((2560, 640), np.dtype("<f2")),
    "concat_0_to_fp16": ((2560,), np.dtype("<f2")),
    "concat_4_to_fp16": ((2560, 640), np.dtype("<f2")),
    "concat_5_to_fp16": ((2560, 640), np.dtype("<f2")),
    "concat_3_to_fp16": ((2560,), np.dtype("<f2")),
}


def split4(gates):
    with mx.stream(mx.gpu):
        i, f, o, g = mx.split(gates, 4, axis=-1)
    mx.eval(i, f, o, g)
    return i, f, o, g


def gates_two_matmul(x, h, wih, whh, bias):
    with mx.stream(mx.gpu):
        gates = x @ wih.T + h @ whh.T + bias
    mx.eval(gates)
    return gates


def gates_packed_fp32(x, h, wih, whh, bias):
    with mx.stream(mx.gpu):
        xh = mx.concatenate((x, h), axis=-1).astype(mx.float32)
        w = mx.concatenate((wih, whh), axis=1).astype(mx.float32)
        gates = (xh @ w.T + bias.astype(mx.float32)).astype(mx.float16)
    mx.eval(gates)
    return gates


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
            indices = input_ids.astype(mx.int16).astype(mx.int32)
            indices = mx.where(indices >= 0, indices, indices + 8193)
            indices = indices.astype(mx.int16)
            embedded = mx.take(embedding, indices, axis=0)
            sequence = mx.transpose(embedded, (1, 0, 2))
            hidden_fp16 = mx.array(hidden).astype(mx.float16)
            cell_fp16 = mx.array(cell).astype(mx.float16)
            h0_native = mx.array(native_next_hidden[0].astype(np.float16))
        for tag, x, h, wih, whh, bias, suffix in (
            ("0", sequence[0], hidden_fp16[0], wih0, whh0, b0, "0"),
            ("1", h0_native, hidden_fp16[1], wih1, whh1, b1, "1n"),
        ):
            a = gates_two_matmul(x, h, wih, whh, bias)
            b = gates_packed_fp32(x, h, wih, whh, bias)
            ia, fa, oa, ga = split4(a)
            ib, fb, ob, gb = split4(b)
            for name, value in (
                ("i", ia), ("f", fa), ("o", oa), ("g", ga),
            ):
                dump[f"t{index:04d}_{name}{suffix}"] = np.asarray(value)
            for name, value in (
                ("i", ib), ("f", fb), ("o", ob), ("g", gb),
            ):
                dump[f"t{index:04d}_{name}{suffix}p"] = np.asarray(value)
        for name, value in (
            ("hidden16_0", hidden_fp16[0]), ("cell16_0", cell_fp16[0]),
            ("hidden16_1", hidden_fp16[1]), ("cell16_1", cell_fp16[1]),
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

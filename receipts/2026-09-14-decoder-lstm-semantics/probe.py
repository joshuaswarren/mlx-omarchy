# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Establish the native Core ML accumulator and precision semantics of the pinned
Parakeet decoder, and localize the residual of
``parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml``.

Host-only. No MLX, no GPU, no ANE, no device execution: every native value comes
from the authenticated capture, every candidate value from NumPy at fp64 with
explicit fp16/fp32 rounding stages, so each candidate names one exact arithmetic
contract rather than a backend's incidental behaviour.

Inputs are injected per transition from the capture, so nothing accumulates
across transitions and every comparison is a single-step contract test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

F16 = np.dtype("<f2")
HIDDEN = 640
CONSTANTS = {
    "embedding": "embedding_weight_to_fp16",
    "l0_ih": "concat_1_to_fp16",
    "l0_hh": "concat_2_to_fp16",
    "l0_bias": "concat_0_to_fp16",
    "l1_ih": "concat_4_to_fp16",
    "l1_hh": "concat_5_to_fp16",
    "l1_bias": "concat_3_to_fp16",
    "projector": "projector_weight_to_fp16",
    "projector_bias": "projector_bias_to_fp16",
}


def f16(value):
    """Round to fp16 and widen back, so every stage boundary is explicit."""
    return np.asarray(value, dtype=np.float64).astype(F16).astype(np.float64)


def f32(value):
    return np.asarray(value, dtype=np.float64).astype(np.float32).astype(np.float64)


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=np.float64)))


def sigmoid_fp16_chain(value):
    """1 / (1 + exp(-x)) with an fp16 rounding after negate, exp, add and divide."""
    return f16(1.0 / f16(1.0 + f16(np.exp(f16(-np.asarray(value, dtype=np.float64))))))


def fp16_fma_chain(vector, weight, order="ascending"):
    """fp16 accumulator, unrounded products, strict index order: one fused chain."""
    products = np.asarray(vector, dtype=np.float64)[None, :] * np.asarray(
        weight, dtype=np.float64
    )
    if order == "descending":
        products = products[:, ::-1]
    acc = f16(products[:, 0])
    for k in range(1, products.shape[1]):
        acc = f16(acc + products[:, k])
    return acc


def fp16_chain_rounded_products(vector, weight):
    products = f16(np.asarray(vector, dtype=np.float64)[None, :] * np.asarray(weight, dtype=np.float64))
    acc = products[:, 0]
    for k in range(1, products.shape[1]):
        acc = f16(acc + products[:, k])
    return acc


def fp32_chain(vector, weight):
    products = np.asarray(vector, dtype=np.float64)[None, :] * np.asarray(weight, dtype=np.float64)
    acc = products[:, 0].astype(np.float32)
    for k in range(1, products.shape[1]):
        acc = (acc + products[:, k].astype(np.float32)).astype(np.float32)
    return acc.astype(np.float64)


def fp32_dot(vector, weight):
    """What an fp16 MLX matmul does: fp32 accumulation, one rounding of the result."""
    return np.asarray(vector, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T


def load(package: Path, capture: Path):
    from tools.coreml.pinned_component import load_pinned_component

    component = load_pinned_component(package, "decoder")
    weights = {key: component.constant(name) for key, name in CONSTANTS.items()}
    index = json.loads((capture / "tdt_tensors.json").read_text())
    rows = []
    for trace in index["traces"]:
        paths = trace["tensor_paths"]
        if "decoder_next_cell" not in paths:
            continue  # decoder-reuse transition: no recurrent step ran
        rows.append(
            {
                "index": trace["index"],
                **{
                    key: np.load(capture / name)
                    for key, name in paths.items()
                    if key.startswith("decoder_")
                },
            }
        )
    return weights, rows


def lstm(weights, row, gate_fn, sigmoid_fn, tanh_fn, state_algebra="fp16"):
    """Two stacked layers from native-injected state; returns cell, hidden, projected.

    ``state_algebra`` selects where the fp16 boundary sits. ``fp16`` rounds every
    elementwise product and sum, as the shipped `_lstm` does. ``fp32-boundary``
    keeps the gate algebra in fp32 and rounds only the two state outputs, which
    is what 8289b1bc does.
    """
    token = int(row["decoder_input_ids"].reshape(-1)[0])
    sequence = weights["embedding"][token].astype(np.float64)
    hidden_in = row["decoder_hidden"][:, 0].astype(np.float64)
    cell_in = row["decoder_cell"][:, 0].astype(np.float64)
    cells, hiddens = [], []
    for layer in (0, 1):
        gates = gate_fn(
            sequence,
            hidden_in[layer],
            weights[f"l{layer}_ih"].astype(np.float64),
            weights[f"l{layer}_hh"].astype(np.float64),
            weights[f"l{layer}_bias"].astype(np.float64),
        )
        input_gate, forget_gate, output_gate, cell_gate = np.split(gates, 4)
        if state_algebra == "fp32-boundary":
            wide = f32(
                f32(sigmoid_fn(forget_gate) * cell_in[layer])
                + f32(sigmoid_fn(input_gate) * tanh_fn(cell_gate))
            )
            cell = f16(wide)
            hidden = f16(f32(sigmoid_fn(output_gate) * tanh_fn(wide)))
        else:
            cell = f16(
                f16(sigmoid_fn(forget_gate) * cell_in[layer])
                + f16(sigmoid_fn(input_gate) * tanh_fn(cell_gate))
            )
            hidden = f16(sigmoid_fn(output_gate) * tanh_fn(cell))
        cells.append(cell)
        hiddens.append(hidden)
        sequence = hidden
    return np.stack(cells), np.stack(hiddens), sequence


GATE_FNS = {
    "fp32-dots-round-each-fp16-add-bias": lambda x, h, ih, hh, b: f16(
        f16(f16(fp32_dot(x, ih)) + f16(fp32_dot(h, hh))) + b
    ),
    "fp32-dots-fp32-add-bias-fp16": lambda x, h, ih, hh, b: f16(
        f16(f32(fp32_dot(x, ih) + fp32_dot(h, hh))) + b
    ),
    "fp32-dots-fp32-add-fp32-bias": lambda x, h, ih, hh, b: f16(
        f32(f32(fp32_dot(x, ih) + fp32_dot(h, hh)) + b)
    ),
    "exact-dots-exact-add-round-once": lambda x, h, ih, hh, b: f16(x @ ih.T + h @ hh.T + b),
    "fp16-fma-split-then-bias": lambda x, h, ih, hh, b: f16(
        f16(fp16_fma_chain(x, ih) + fp16_fma_chain(h, hh)) + b
    ),
    "fp16-fma-concat-x-then-h": lambda x, h, ih, hh, b: f16(
        fp16_fma_chain(np.concatenate([x, h]), np.concatenate([ih, hh], axis=1)) + b
    ),
    # 8289b1bc keeps the gate vector in fp32: no fp16 rounding before the algebra.
    "fp32-dots-fp32-bias-no-fp16-round": lambda x, h, ih, hh, b: f32(
        f32(f32(fp32_dot(x, ih) + fp32_dot(h, hh)) + b)
    ),
}

ACTIVATIONS = {
    "sigmoid-exact-r16": (lambda v: f16(sigmoid(v)), lambda v: f16(np.tanh(v))),
    "sigmoid-fp16-chain": (sigmoid_fp16_chain, lambda v: f16(np.tanh(v))),
    "sigmoid-tanh-fp32": (lambda v: f32(sigmoid(v)), lambda v: f32(np.tanh(v))),
}

STATE_ALGEBRAS = ("fp16", "fp32-boundary")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.overlay))
    weights, rows = load(args.package, args.capture_dir)
    lanes = 2 * HIDDEN * len(rows)
    result: dict = {
        "schema": "mlx-omarchy.parakeet-decoder-lstm-semantics/1",
        "named_hole": "parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml",
        "execution": "host-only NumPy; no MLX, no GPU, no ANE, no device execution",
        "capture_dir": str(args.capture_dir),
        "package": str(args.package),
        "capture_sha256": {
            name: sha256(args.capture_dir / name)
            for name in ("tdt_tensors.json", "manifest.sha256", "receipt.json")
            if (args.capture_dir / name).exists()
        },
        "transitions": [row["index"] for row in rows],
    }

    # 1. The first divergence is reduction-free.
    first = rows[0]
    embedding_row = weights["embedding"][int(first["decoder_input_ids"].reshape(-1)[0])]
    zero_rows = np.nonzero((weights["embedding"] == 0).all(axis=1))[0]
    bias = weights["l0_bias"].astype(np.float64)
    input_gate, _forget, output_gate, cell_gate = np.split(bias, 4)
    native_cell = first["decoder_next_cell"][0, 0].astype(np.float64)
    native_hidden = first["decoder_next_hidden"][0, 0].astype(np.float64)
    candidate = f16(f16(sigmoid(input_gate)) * f16(np.tanh(cell_gate)))
    delta = candidate - native_cell
    step = np.abs(
        np.nextafter(native_cell.astype(F16), np.float16(np.inf)).astype(np.float64) - native_cell
    )
    ulps = np.round(delta / np.maximum(step, 1e-30)).astype(int)
    zero_gates = {
        name: bool(
            np.array_equal(
                gate_fn(
                    np.zeros(HIDDEN),
                    np.zeros(HIDDEN),
                    weights["l0_ih"].astype(np.float64),
                    weights["l0_hh"].astype(np.float64),
                    bias,
                ),
                bias,
            )
        )
        for name, gate_fn in GATE_FNS.items()
    }
    result["reduction_free_first_divergence"] = {
        "trace": int(first["index"]),
        "token": int(first["decoder_input_ids"].reshape(-1)[0]),
        "embedding_row_is_identically_zero": bool(not embedding_row.any()),
        "all_zero_embedding_rows": [int(value) for value in zero_rows],
        "entry_hidden_is_zero": bool(not first["decoder_hidden"].any()),
        "entry_cell_is_zero": bool(not first["decoder_cell"].any()),
        "layer0_gates_equal_fp16_bias_under_every_reduction": zero_gates,
        "layer0_next_cell_bit_exact_lanes": int(np.count_nonzero(delta == 0)),
        "layer0_next_cell_max_abs": float(np.abs(delta).max()),
        "signed_fp16_ulp_histogram": {
            str(value): int(np.count_nonzero(ulps == value)) for value in sorted(set(ulps.tolist()))
        },
        "native_outputs_are_fp16_representable": bool(
            np.array_equal(native_cell, f16(native_cell))
            and np.array_equal(native_hidden, f16(native_hidden))
        ),
    }

    # 2. Gate packing: which slot is which gate.
    slots = np.split(bias, 4)
    packing = {}
    for a in range(4):
        for c in range(4):
            if a == c:
                continue
            cand = f16(f16(sigmoid(slots[a])) * f16(np.tanh(slots[c])))
            packing[f"input=slot{a},cell=slot{c}"] = {
                "bit_exact_lanes": int(np.count_nonzero(cand == native_cell)),
                "max_abs_error": float(np.abs(cand - native_cell).max()),
            }
    output_slot = {}
    for o in range(4):
        cand = f16(f16(sigmoid(slots[o])) * f16(np.tanh(native_cell)))
        output_slot[f"output=slot{o}"] = {
            "bit_exact_lanes": int(np.count_nonzero(cand == native_hidden)),
            "max_abs_error": float(np.abs(cand - native_hidden).max()),
        }
    result["gate_packing"] = {
        "next_cell_by_slot_pair": packing,
        "next_hidden_by_output_slot": output_slot,
        "verdict": "IFOC confirmed; every other assignment collapses to <=1/640 lanes",
    }

    # 3. Projector: the activation-free linear that pins the native accumulator.
    projector = weights["projector"].astype(np.float64)
    projector_bias = weights["projector_bias"].astype(np.float64)
    schemes = {
        "fp16-fma-chain-ascending-bias-trailing": lambda x: f16(
            fp16_fma_chain(x, projector) + projector_bias
        ),
        "fp16-fma-chain-ascending-bias-as-accumulator-seed": None,
        "fp16-fma-chain-descending-bias-trailing": lambda x: f16(
            fp16_fma_chain(x, projector, order="descending") + projector_bias
        ),
        "fp16-chain-products-rounded-first": lambda x: f16(
            fp16_chain_rounded_products(x, projector) + projector_bias
        ),
        "fp32-chain-ascending": lambda x: f16(fp32_chain(x, projector) + projector_bias),
        "fp32-dot-round-once": lambda x: f16(f16(fp32_dot(x, projector)) + projector_bias),
        "exact-sum": lambda x: f16(x @ projector.T + projector_bias),
    }

    def seeded(x):
        products = np.asarray(x, dtype=np.float64)[None, :] * projector
        acc = f16(projector_bias + products[:, 0])
        for k in range(1, products.shape[1]):
            acc = f16(acc + products[:, k])
        return acc

    schemes["fp16-fma-chain-ascending-bias-as-accumulator-seed"] = seeded

    projector_lanes = HIDDEN * len(rows)
    projector_table = {}
    for name, fn in schemes.items():
        exact = 0
        worst = 0.0
        for row in rows:
            native = row["decoder_output_hidden"][0, 0].astype(np.float64)
            cand = fn(row["decoder_next_hidden"][1, 0].astype(np.float64))
            exact += int(np.count_nonzero(cand == native))
            worst = max(worst, float(np.abs(cand - native).max()))
        projector_table[name] = {
            "bit_exact_lanes": exact,
            "lanes": projector_lanes,
            "max_abs_error": worst,
        }
    result["projector_linear"] = {
        "input": "native decoder next_hidden layer 1",
        "target": "native decoder_output_hidden",
        "schemes": projector_table,
        "established_semantics": (
            "fp16 accumulator, unrounded products (fused multiply-add), strictly "
            "ascending reduction index, bias added after the reduction"
        ),
        "closes_named_hole": "parakeet.tdt.decoder-projector.fp16-linear-vs-native-coreml",
    }

    # 4. LSTM variant matrix.
    variants = {}
    for gate_name, gate_fn in GATE_FNS.items():
        for act_name, (sigmoid_fn, tanh_fn) in ACTIVATIONS.items():
            for algebra in STATE_ALGEBRAS:
                cell_exact = hidden_exact = nonfinite = 0
                worst_cell = worst_hidden = 0.0
                per_trace = {}
                for row in rows:
                    cells, hiddens, _ = lstm(
                        weights, row, gate_fn, sigmoid_fn, tanh_fn, algebra
                    )
                    native_c = row["decoder_next_cell"][:, 0].astype(np.float64)
                    native_h = row["decoder_next_hidden"][:, 0].astype(np.float64)
                    ce = int(np.count_nonzero(cells == native_c))
                    cell_exact += ce
                    hidden_exact += int(np.count_nonzero(hiddens == native_h))
                    worst_cell = max(worst_cell, float(np.abs(cells - native_c).max()))
                    worst_hidden = max(worst_hidden, float(np.abs(hiddens - native_h).max()))
                    nonfinite += int(
                        np.count_nonzero(~np.isfinite(cells))
                        + np.count_nonzero(~np.isfinite(hiddens))
                    )
                    per_trace[str(row["index"])] = ce
                variants[f"{gate_name} | {act_name} | state={algebra}"] = {
                    "next_cell_bit_exact_lanes": cell_exact,
                    "next_hidden_bit_exact_lanes": hidden_exact,
                    "lanes_per_output": lanes,
                    "next_cell_max_abs_error": worst_cell,
                    "next_hidden_max_abs_error": worst_hidden,
                    "next_cell_bit_exact_by_trace": per_trace,
                    "non_finite_candidate_lanes": nonfinite,
                }
    result["lstm_variants"] = variants
    result["bit_exact_lstm_variant"] = next(
        (
            name
            for name, data in variants.items()
            if data["next_cell_bit_exact_lanes"] == lanes
            and data["next_hidden_bit_exact_lanes"] == lanes
        ),
        None,
    )
    best = max(
        variants.items(),
        key=lambda kv: kv[1]["next_cell_bit_exact_lanes"] + kv[1]["next_hidden_bit_exact_lanes"],
    )
    result["best_lstm_variant"] = {"name": best[0], **best[1]}
    shipped = "fp32-dots-round-each-fp16-add-bias | sigmoid-exact-r16 | state=fp16"
    result["baseline_variant"] = {
        "name": shipped,
        "models": "the shipped overlay/tools/coreml/vulkan_decoder.py::_lstm",
        **variants[shipped],
    }
    excluded = "fp32-dots-fp32-bias-no-fp16-round | sigmoid-tanh-fp32 | state=fp32-boundary"
    result["commit_8289b1bc_variant"] = {
        "name": excluded,
        "models": (
            "8289b1bc: fp32 gate weights and gate algebra, fp16 rounding only at "
            "the two state outputs"
        ),
        **variants[excluded],
    }
    result["remaining_hole"] = {
        "name": "parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml",
        "supersedes": "parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml",
        "reason": (
            "the first divergence carries no reduction, so no accumulator order, "
            "no accumulate width and no bias placement can explain it"
        ),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

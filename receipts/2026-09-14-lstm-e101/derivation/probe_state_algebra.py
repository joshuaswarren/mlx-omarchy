# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Isolate the divergent recurrent step of the pinned TDT decoder LSTM.

The elementwise ``sigmoid``/``tanh`` contract is closed: correctly-rounded
fp16 matches the captured native tensors 1280/1280 and the full Mac fixture
1536/1536. The remaining named hole
``parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml`` must therefore
live in the recurrence itself. This probe localises it step by step:

1. Transition 0 layer 0 is reduction-free (all-zero embedding row, zero
   entry state), so the gates are exactly the fp16 bias constants. The only
   remaining freedom is how the native consumes the unary outputs and rounds
   the cell update. Enumerate unary-materialisation x product-width x state
   rounding and score against the captured ``decoder_next_cell``.
2. Feed the winning per-step contract through all 15 comparable transitions
   under each gate-reduction candidate and find the first transition, layer
   and tensor where bit-exactness breaks.
3. Score the shipped ``_lstm`` contract on the same transitions to quantify
   the per-step residual the live decode accumulates into the emission-101
   miss (8029 vs native 7892).

Host-only NumPy; no MLX, no GPU, no ANE, no device execution. Every native
value comes from the authenticated capture; every candidate names one exact
arithmetic contract.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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


def sigmoid64(value):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=np.float64)))


def tanh64(value):
    return np.tanh(np.asarray(value, dtype=np.float64))


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


def score(candidate: np.ndarray, native: np.ndarray) -> dict:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(native, dtype=np.float64)
    delta = cand - ref
    step = np.abs(
        np.nextafter(ref.astype(F16), np.float16(np.inf)).astype(np.float64) - ref
    )
    ulps = np.round(delta / np.maximum(step, 1e-30)).astype(int)
    return {
        "bit_exact_lanes": int(np.count_nonzero(delta == 0)),
        "lanes": int(delta.size),
        "max_abs": float(np.abs(delta).max()),
        "signed_ulp_histogram": {
            str(v): int(np.count_nonzero(ulps == v)) for v in sorted(set(ulps.ravel().tolist()))
        },
    }


# --- stage 1: reduction-free cell update (transition 0, layer 0) -------------

UNARY = {
    "r16": lambda fn, x: f16(fn(x)),  # captured fp16 tensor contract
    "f32": lambda fn, x: f32(fn(x)),
    "f64": lambda fn, x: np.asarray(fn(x), dtype=np.float64),
}
PRODUCT = {
    "p32": lambda a, b: f32(a * b),
    "p64": lambda a, b: a * b,
}


def cell_update(forget_sig, cell_in, input_sig, tanh_sig, product, state_round=f16):
    return state_round(product(forget_sig, cell_in) + product(input_sig, tanh_sig))


def hidden_update(out_sig, cell, product, state_round=f16):
    return state_round(product(out_sig, cell))


# --- gate reductions ----------------------------------------------------------

def p32_dot(vector, weight):
    return np.asarray(vector, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T


GATE_FNS = {
    # the shipped GPU chain: fp32-accumulating fp16 matmuls, fp16 adds, fp16 bias
    "shipped-fp16-adds": lambda x, h, ih, hh, b: f16(
        f16(f16(p32_dot(x, ih)) + f16(p32_dot(h, hh))) + b
    ),
    "fp32-dots-round-each-fp16-add-bias": lambda x, h, ih, hh, b: f16(
        f16(f16(p32_dot(x, ih)) + f16(p32_dot(h, hh))) + b
    ),
    "fp32-dots-fp32-add-bias-fp16": lambda x, h, ih, hh, b: f16(
        f16(f32(p32_dot(x, ih) + p32_dot(h, hh))) + b
    ),
    "fp32-dots-fp32-add-fp32-bias": lambda x, h, ih, hh, b: f16(
        f32(f32(p32_dot(x, ih) + p32_dot(h, hh)) + b)
    ),
    "exact-dots-exact-add-round-once": lambda x, h, ih, hh, b: f16(
        x @ ih.T + h @ hh.T + b
    ),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.overlay))
    weights, rows = load(args.package, args.capture_dir)
    result: dict = {
        "schema": "mlx-omarchy.decoder-lstm-e101/1",
        "named_hole": "parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml",
        "execution": "host-only NumPy; no MLX, no GPU, no ANE, no device execution",
        "capture_dir": str(args.capture_dir),
        "capture_sha256": {
            name: hashlib.sha256((args.capture_dir / name).read_bytes()).hexdigest()
            for name in ("tdt_tensors.json", "manifest.sha256")
        },
        "transitions": [row["index"] for row in rows],
    }

    # ---- stage 1: transition 0, layer 0 ------------------------------------
    first = rows[0]
    token = int(first["decoder_input_ids"].reshape(-1)[0])
    zero_entry = not first["decoder_hidden"].any() and not first["decoder_cell"].any()
    bias0 = weights["l0_bias"].astype(np.float64)
    b_i, b_f, b_o, b_g = np.split(bias0, 4)
    native_cell0 = first["decoder_next_cell"][0, 0]
    native_hidden0 = first["decoder_next_hidden"][0, 0]

    grid = []
    seen = set()
    for s_mode, t_mode, prod in itertools.product(
        ("r16", "f32", "f64"), ("r16", "f32", "f64"), ("p32", "p64")
    ):
        key = (s_mode, t_mode, prod)
        if key in seen:
            continue
        seen.add(key)
        s_i = UNARY[s_mode](sigmoid64, b_i)
        s_f = UNARY[s_mode](sigmoid64, b_f)
        t_g = UNARY[t_mode](tanh64, b_g)
        zero = np.zeros(HIDDEN)
        cell = cell_update(s_f, zero, s_i, t_g, PRODUCT[prod])
        grid.append(
            {
                "sigmoid_feed": s_mode,
                "tanh_feed": t_mode,
                "product": prod,
                **score(cell, native_cell0),
            }
        )
    grid.sort(key=lambda g: (-g["bit_exact_lanes"], g["max_abs"]))
    result["reduction_free_cell_update"] = {
        "trace": int(first["index"]),
        "token": token,
        "embedding_row_is_identically_zero": bool(not weights["embedding"][token].any()),
        "zero_entry_state": zero_entry,
        "native_tensor": "decoder_next_cell[0,0]",
        "variants": grid,
    }

    best = grid[0]
    s_mode, t_mode, prod = best["sigmoid_feed"], best["tanh_feed"], best["product"]
    result["winner_cell_contract"] = {k: best[k] for k in ("sigmoid_feed", "tanh_feed", "product", "bit_exact_lanes")}

    # hidden update at transition 0: tanh input rounded vs wide
    s_o = UNARY[s_mode](sigmoid64, b_o)
    t_g_wide = UNARY[t_mode](tanh64, b_g)
    s_i_wide = UNARY[s_mode](sigmoid64, b_i)
    s_f_wide = UNARY[s_mode](sigmoid64, b_f)
    cell_rounded = cell_update(s_f_wide, np.zeros(HIDDEN), s_i_wide, t_g_wide, PRODUCT[prod])
    cell_wide = PRODUCT[prod](s_f_wide, np.zeros(HIDDEN)) + PRODUCT[prod](s_i_wide, t_g_wide)
    hidden_variants = []
    for cell_feed in ("r16", "wide"):
        for so_mode, t_mode_h, prod_h in itertools.product(
            (s_mode, "r16", "f32"), ("r16", "f32", "f64"), ("p32", "p64")
        ):
            cell_in = cell_rounded if cell_feed == "r16" else cell_wide
            h = hidden_update(
                UNARY[so_mode](sigmoid64, b_o),
                UNARY[t_mode_h](tanh64, cell_in),
                PRODUCT[prod_h],
            )
            hidden_variants.append(
                {
                    "tanh_input": cell_feed,
                    "sigmoid_feed": so_mode,
                    "tanh_feed": t_mode_h,
                    "product": prod_h,
                    **score(h, native_hidden0),
                }
            )
    hidden_variants.sort(key=lambda g: (-g["bit_exact_lanes"], g["max_abs"]))
    result["reduction_free_hidden_update"] = {
        "native_tensor": "decoder_next_hidden[0,0]",
        "variants": hidden_variants[:12],
    }
    best_h = hidden_variants[0]

    # ---- stage 2: recurrence over all captured transitions ------------------
    # per-step contract from stage 1, gate reductions enumerated; native-injected
    # entry state each transition (single-step contract test, no accumulation).
    def unary_pair(mode, *values):
        return tuple(UNARY[mode](fn, v) for fn, v in zip((sigmoid64, tanh64), values))

    recurrence = []
    for gate_name, gate_fn in GATE_FNS.items():
        cell_hits = 0
        hidden_hits = 0
        lanes = 0
        per_transition = []
        first_break = None
        for row in rows:
            x = weights["embedding"][int(row["decoder_input_ids"].reshape(-1)[0])].astype(np.float64)
            hidden_in = row["decoder_hidden"][:, 0].astype(np.float64)
            cell_in = row["decoder_cell"][:, 0].astype(np.float64)
            cells, hiddens = [], []
            for layer in (0, 1):
                gates = gate_fn(
                    x if layer == 0 else hiddens[-1],
                    hidden_in[layer],
                    weights[f"l{layer}_ih"].astype(np.float64),
                    weights[f"l{layer}_hh"].astype(np.float64),
                    weights[f"l{layer}_bias"].astype(np.float64),
                )
                g_i, g_f, g_o, g_g = np.split(gates, 4)
                sig = UNARY[s_mode]
                tan = UNARY[t_mode]
                cell = f16(
                    PRODUCT[prod](sig(sigmoid64, g_f), cell_in[layer])
                    + PRODUCT[prod](sig(sigmoid64, g_i), tan(tanh64, g_g))
                )
                t_in = f16(cell) if best_h["tanh_input"] == "r16" else cell
                hidden = f16(
                    PRODUCT[best_h["product"]](
                        UNARY[best_h["sigmoid_feed"]](sigmoid64, g_o),
                        UNARY[best_h["tanh_feed"]](tanh64, t_in),
                    )
                )
                cells.append(cell)
                hiddens.append(hidden)
            cell_rec = score(np.stack(cells), row["decoder_next_cell"][:, 0])
            hidden_rec = score(np.stack(hiddens), row["decoder_next_hidden"][:, 0])
            cell_hits += cell_rec["bit_exact_lanes"]
            hidden_hits += hidden_rec["bit_exact_lanes"]
            lanes += 2 * HIDDEN
            broke = cell_rec["bit_exact_lanes"] < 2 * HIDDEN or hidden_rec["bit_exact_lanes"] < 2 * HIDDEN
            if broke and first_break is None:
                first_break = {
                    "trace": int(row["index"]),
                    "cell": cell_rec,
                    "hidden": hidden_rec,
                }
            per_transition.append(
                {
                    "trace": int(row["index"]),
                    "cell_exact": cell_rec["bit_exact_lanes"],
                    "hidden_exact": hidden_rec["bit_exact_lanes"],
                }
            )
        recurrence.append(
            {
                "gate_reduction": gate_name,
                "cell_exact_lanes": cell_hits,
                "hidden_exact_lanes": hidden_hits,
                "lanes": lanes,
                "first_break": first_break,
                "per_transition": per_transition,
            }
        )
    recurrence.sort(key=lambda r: -(r["cell_exact_lanes"] + r["hidden_exact_lanes"]))
    result["recurrence_contracts"] = recurrence

    # ---- stage 3: the shipped contract on the same transitions --------------
    shipped = []
    for row in rows:
        x = weights["embedding"][int(row["decoder_input_ids"].reshape(-1)[0])].astype(np.float64)
        hidden_in = row["decoder_hidden"][:, 0].astype(np.float64)
        cell_in = row["decoder_cell"][:, 0].astype(np.float64)
        cells, hiddens = [], []
        for layer in (0, 1):
            gates = GATE_FNS["shipped-fp16-adds"](
                x if layer == 0 else hiddens[-1],
                hidden_in[layer],
                weights[f"l{layer}_ih"].astype(np.float64),
                weights[f"l{layer}_hh"].astype(np.float64),
                weights[f"l{layer}_bias"].astype(np.float64),
            )
            g_i, g_f, g_o, g_g = np.split(gates, 4)
            cell = f16(
                f16(f16(sigmoid64(g_f)) * cell_in[layer])
                + f16(f16(sigmoid64(g_i)) * f16(tanh64(g_g)))
            )
            hidden = f16(f16(sigmoid64(g_o)) * f16(tanh64(cell)))
            cells.append(cell)
            hiddens.append(hidden)
        shipped.append(
            {
                "trace": int(row["index"]),
                "cell": score(np.stack(cells), row["decoder_next_cell"][:, 0]),
                "hidden": score(np.stack(hiddens), row["decoder_next_hidden"][:, 0]),
            }
        )
    result["shipped_contract_per_transition"] = shipped

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {args.output}")
    print(json.dumps(result["winner_cell_contract"], indent=1))
    print("best hidden:", json.dumps(best_h, indent=1))
    top = recurrence[0]
    print(
        "best recurrence:",
        top["gate_reduction"],
        "cell",
        top["cell_exact_lanes"],
        "/",
        top["lanes"],
        "hidden",
        top["hidden_exact_lanes"],
    )
    if top["first_break"]:
        print("first break:", json.dumps(top["first_break"], indent=1)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

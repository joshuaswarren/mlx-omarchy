# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Establish the native Core ML elementwise ``sigmoid``/``tanh`` semantics of the
pinned Parakeet decoder, on the reduction-free transition where the LSTM collapses
to two elementwise function evaluations and one product.

Named hole: ``parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml``.

Host-only. No MLX, no GPU, no ANE, no device execution, no ``/tmp/m1-gpu.lock``.
Every native value comes from the authenticated capture; every candidate value is
NumPy at fp64 with explicit fp16/fp32 rounding stages, so each candidate names one
exact arithmetic contract rather than a backend's incidental behaviour.

Transition 0 decodes token 8192, whose embedding row is identically zero, and its
entry hidden and entry cell are zero. Layer 0 therefore carries no reduction: the
gates equal the pinned fp16 bias bit for bit, and

    next_cell   = sigmoid(bias_i) * tanh(bias_c)
    next_hidden = sigmoid(bias_o) * tanh(next_cell)

over known fp16 arguments. That is 1280 lanes of pure elementwise evidence, which
is what this probe measures.
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

# log2(e) at three widths: a hardware exp2 needs this constant, and its width is
# part of the arithmetic contract.
L2E64 = 1.4426950408889634
L2E32 = float(np.float32(L2E64))
L2E16 = float(np.float16(L2E64))


def f16(value):
    """Round to fp16 and widen back, so every stage boundary is explicit."""
    return np.asarray(value, dtype=np.float64).astype(F16).astype(np.float64)


def f32(value):
    return np.asarray(value, dtype=np.float64).astype(np.float32).astype(np.float64)


def f64(value):
    return np.asarray(value, dtype=np.float64)


ROUND = {16: f16, 32: f32, 64: f64}


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-f64(value)))


def ulp16(value):
    """Distance from |value| to the next larger fp16, i.e. one fp16 ulp there."""
    a = np.abs(f64(value))
    return np.abs(np.nextafter(a.astype(F16), np.float16(np.inf)).astype(np.float64) - a)


def round_interval(value):
    """Open interval of reals that round to the fp16 number ``value``."""
    v16 = f64(value).astype(F16)
    up = np.nextafter(v16, np.float16(np.inf)).astype(np.float64)
    dn = np.nextafter(v16, np.float16(-np.inf)).astype(np.float64)
    v = v16.astype(np.float64)
    return (v + dn) / 2.0, (v + up) / 2.0


def shift_ulp(value, steps):
    """Move fp16 ``value`` by ``steps`` fp16 ulps (signed, along the real line)."""
    out = f64(value).astype(F16)
    target = np.float16(np.inf) if steps > 0 else np.float16(-np.inf)
    for _ in range(abs(int(steps))):
        out = np.nextafter(out, target)
    return out.astype(np.float64)


# --------------------------------------------------------------------------- #
# candidate elementwise contracts
# --------------------------------------------------------------------------- #
def make_sigmoid(form, precisions, l2e=L2E64):
    """One exact sigmoid contract: a formula plus a rounding width per stage."""
    p_exp, p_add, p_out = (ROUND[p] for p in precisions)
    if form == "exact":
        return lambda x: p_out(sigmoid(x))
    if form == "recip":  # 1 / (1 + exp(-x))
        return lambda x: p_out(1.0 / p_add(1.0 + p_exp(np.exp(-f64(x)))))
    if form == "posexp":  # exp(x) / (1 + exp(x))
        return lambda x: (lambda e: p_out(e / p_add(1.0 + e)))(p_exp(np.exp(f64(x))))
    if form == "exp2":  # 1 / (1 + 2 ** (-x * log2e)), argument rounded with the stage
        return lambda x: p_out(
            1.0 / p_add(1.0 + p_exp(np.exp2(p_exp(-f64(x) * l2e))))
        )
    if form == "exp2wide":  # exp2 argument kept wide, result rounded
        return lambda x: p_out(1.0 / p_add(1.0 + p_exp(np.exp2(-f64(x) * l2e))))
    if form == "fromtanh":  # 0.5 * tanh(x / 2) + 0.5
        return lambda x: p_out(p_add(0.5 * p_exp(np.tanh(p_exp(0.5 * f64(x)))) + 0.5))
    raise ValueError(form)


def make_tanh(form, precisions, sigmoid_fn=None, l2e=L2E64):
    p_exp, p_add, p_out = (ROUND[p] for p in precisions)
    if form == "exact":
        return lambda x: p_out(np.tanh(f64(x)))
    if form == "negexp":  # (1 - e) / (1 + e), e = exp(-2x)
        return lambda x: (lambda e: p_out(p_add(1.0 - e) / p_add(1.0 + e)))(
            p_exp(np.exp(-2.0 * f64(x)))
        )
    if form == "posexp":  # (e - 1) / (e + 1), e = exp(2x)
        return lambda x: (lambda e: p_out(p_add(e - 1.0) / p_add(e + 1.0)))(
            p_exp(np.exp(2.0 * f64(x)))
        )
    if form == "oneminus":  # 1 - 2 / (exp(2x) + 1)
        return lambda x: (lambda e: p_out(1.0 - p_add(2.0 / p_add(e + 1.0))))(
            p_exp(np.exp(2.0 * f64(x)))
        )
    if form == "odd":  # sign(x) * (1 - 2 / (exp(2|x|) + 1)): magnitude path, sign applied
        return lambda x: (
            lambda s, e: p_out(s * p_out(1.0 - p_add(2.0 / p_add(e + 1.0))))
        )(np.sign(f64(x)), p_exp(np.exp(2.0 * np.abs(f64(x)))))
    if form == "oddexp2":
        return lambda x: (
            lambda s, e: p_out(s * p_out(1.0 - p_add(2.0 / p_add(e + 1.0))))
        )(np.sign(f64(x)), p_exp(np.exp2(p_exp(2.0 * np.abs(f64(x)) * l2e))))
    if form == "exp2":
        return lambda x: (lambda e: p_out(1.0 - p_add(2.0 / p_add(e + 1.0))))(
            p_exp(np.exp2(p_exp(2.0 * f64(x) * l2e)))
        )
    if form == "via_sig":  # 2 * sigmoid(2x) - 1
        return lambda x: p_out(2.0 * sigmoid_fn(p_exp(2.0 * f64(x))) - 1.0)
    if form == "via_sig_r":  # with the doubling rounded before the subtraction
        return lambda x: p_out(p_add(2.0 * sigmoid_fn(p_exp(2.0 * f64(x)))) - 1.0)
    if form == "diffsig":  # sigmoid(2x) - sigmoid(-2x): odd by construction
        return lambda x: p_out(
            sigmoid_fn(p_exp(2.0 * f64(x))) - sigmoid_fn(p_exp(-2.0 * f64(x)))
        )
    if form == "sigshift":  # 2 * (sigmoid(2x) - 1/2), the cancellation moved inward
        return lambda x: p_out(
            2.0 * p_add(sigmoid_fn(p_exp(2.0 * f64(x))) - 0.5)
        )
    if form == "oneminus2sig":  # 1 - 2 * sigmoid(-2x)
        return lambda x: p_out(1.0 - p_add(2.0 * sigmoid_fn(p_exp(-2.0 * f64(x)))))
    if form == "mulrcppos":  # (e - 1) * reciprocal(e + 1), two roundings on the divide
        return lambda x: (
            lambda e: p_out(p_add(e - 1.0) * p_out(1.0 / p_add(e + 1.0)))
        )(p_exp(np.exp(2.0 * f64(x))))
    if form == "mulrcpneg":
        return lambda x: (
            lambda e: p_out(p_add(1.0 - e) * p_out(1.0 / p_add(1.0 + e)))
        )(p_exp(np.exp(-2.0 * f64(x))))
    if form == "oddmulrcp":
        return lambda x: (
            lambda s, e: p_out(
                s * p_out(1.0 - p_add(2.0 * p_out(1.0 / p_add(e + 1.0))))
            )
        )(np.sign(f64(x)), p_exp(np.exp(2.0 * np.abs(f64(x)))))
    if form == "pade":  # the classic x(27 + x^2) / (27 + 9x^2) rational
        return lambda x: (
            lambda v: p_out(
                p_add(v * p_add(27.0 + p_exp(v * v))) / p_add(27.0 + 9.0 * p_exp(v * v))
            )
        )(f64(x))
    raise ValueError(form)


def sigmoid_catalogue():
    out = {}
    widths = list(itertools.product((16, 32), repeat=3))
    for form in ("exact", "recip", "posexp", "fromtanh"):
        for w in widths:
            out[f"sig:{form}:{w[0]}{w[1]}{w[2]}"] = make_sigmoid(form, w)
    for form in ("exp2", "exp2wide"):
        for l2e, tag in ((L2E64, "e64"), (L2E32, "e32"), (L2E16, "e16")):
            for w in widths:
                out[f"sig:{form}.{tag}:{w[0]}{w[1]}{w[2]}"] = make_sigmoid(form, w, l2e)
    return out


def tanh_catalogue(sigs):
    out = {}
    widths = list(itertools.product((16, 32), repeat=3))
    for form in (
        "exact",
        "negexp",
        "posexp",
        "oneminus",
        "odd",
        "pade",
        "mulrcppos",
        "mulrcpneg",
        "oddmulrcp",
    ):
        for w in widths:
            out[f"tanh:{form}:{w[0]}{w[1]}{w[2]}"] = make_tanh(form, w)
    for form in ("exp2", "oddexp2"):
        for l2e, tag in ((L2E64, "e64"), (L2E32, "e32"), (L2E16, "e16")):
            for w in widths:
                out[f"tanh:{form}.{tag}:{w[0]}{w[1]}{w[2]}"] = make_tanh(form, w, l2e=l2e)
    # tanh built from every sigmoid contract, which is how a shared hardware
    # nonlinearity unit would most plausibly serve both activations.
    for name, fn in sigs.items():
        for form in ("via_sig", "via_sig_r", "diffsig", "sigshift", "oneminus2sig"):
            for w in widths:
                out[f"tanh:{form}[{name}]:{w[0]}{w[1]}{w[2]}"] = make_tanh(
                    form, w, sigmoid_fn=fn
                )
    return out


def dedupe(catalogue, *argument_sets):
    """Collapse contracts that are numerically identical on the probe arguments."""
    seen, out = {}, {}
    for name, fn in catalogue.items():
        try:
            key = b"".join(np.ascontiguousarray(fn(a)).tobytes() for a in argument_sets)
        except (ValueError, TypeError, FloatingPointError):
            continue
        if key in seen:
            continue
        seen[key] = name
        out[name] = fn
    return out


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
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
    return component, weights, rows


def activation_attributes(component):
    """The declared MIL contract of both lstm ops: which activation sits where."""
    wanted = ("recurrent_activation", "cell_activation", "activation")
    names = {}
    for op in component.block.operations:
        if op.type != "lstm":
            continue
        for key in wanted:
            names[f"{op.outputs[0].name}.{key}"] = op.inputs[key].arguments[0].name
    values = {}
    for op in component.block.operations:
        if op.type != "const":
            continue
        out = op.outputs[0].name
        if out in names.values():
            val = op.attributes.get("val")
            values[out] = list(val.immediateValue.tensor.strings.values)[0]
    return {key: values.get(ref, "?") for key, ref in names.items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    sys.path.insert(0, str(args.overlay))
    component, weights, rows = load(args.package, args.capture_dir)

    result: dict = {
        "schema": "mlx-omarchy.parakeet-decoder-activations/1",
        "named_hole": "parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml",
        "execution": "host-only NumPy; no MLX, no GPU, no ANE, no device execution",
        "capture_dir": str(args.capture_dir),
        "package": str(args.package),
        "capture_sha256": {
            name: sha256(args.capture_dir / name)
            for name in ("tdt_tensors.json", "manifest.sha256", "receipt.json")
            if (args.capture_dir / name).exists()
        },
        "transitions": [row["index"] for row in rows],
        "declared_mil_activations": activation_attributes(component),
    }

    # ---- the reduction-free transition -------------------------------------- #
    first = rows[0]
    token = int(first["decoder_input_ids"].reshape(-1)[0])
    bias = weights["l0_bias"].astype(np.float64)
    b_i, b_f, b_o, b_c = np.split(bias, 4)
    native_cell = first["decoder_next_cell"][0, 0].astype(np.float64)
    native_hidden = first["decoder_next_hidden"][0, 0].astype(np.float64)

    baseline = f16(f16(sigmoid(b_i)) * f16(np.tanh(b_c)))
    delta = baseline - native_cell
    ulps = np.round(delta / np.maximum(ulp16(native_cell), 1e-30)).astype(int)
    result["reduction_free_lanes"] = {
        "trace": int(first["index"]),
        "token": token,
        "embedding_row_is_identically_zero": bool(not weights["embedding"][token].any()),
        "entry_hidden_is_zero": bool(not first["decoder_hidden"].any()),
        "entry_cell_is_zero": bool(not first["decoder_cell"].any()),
        "cell_lanes": HIDDEN,
        "hidden_lanes": HIDDEN,
        "pinned_baseline_bit_exact_cell_lanes": int(np.count_nonzero(delta == 0)),
        "pinned_baseline_max_abs": float(np.abs(delta).max()),
        "pinned_baseline_signed_ulp_histogram": {
            str(v): int(np.count_nonzero(ulps == v)) for v in sorted(set(ulps.tolist()))
        },
        "native_is_fp16_representable": bool(
            np.array_equal(native_cell, f16(native_cell))
            and np.array_equal(native_hidden, f16(native_hidden))
        ),
        "shared_sigmoid_arguments_cell_vs_hidden": int(len(np.intersect1d(b_i, b_o))),
        "shared_tanh_arguments_cell_vs_hidden": int(
            len(np.intersect1d(b_c, native_cell))
        ),
        "duplicate_sigmoid_arguments_within_cell": int(
            HIDDEN - len(np.unique(b_i))
        ),
        "duplicate_tanh_arguments_within_cell": int(HIDDEN - len(np.unique(b_c))),
    }

    # ---- catalogue sweep ----------------------------------------------------- #
    sigs = dedupe(sigmoid_catalogue(), b_i, b_o)
    tanhs = dedupe(tanh_catalogue(sigmoid_catalogue()), b_c, native_cell)
    sig_cell = {n: f(b_i) for n, f in sigs.items()}
    sig_hidden = {n: f(b_o) for n, f in sigs.items()}
    tanh_cell = {n: f(b_c) for n, f in tanhs.items()}
    tanh_hidden = {n: f(native_cell) for n, f in tanhs.items()}

    scored = []
    for sn in sigs:
        sc, sh = sig_cell[sn], sig_hidden[sn]
        for tn in tanhs:
            cell = f16(sc * tanh_cell[tn])
            hidden = f16(sh * tanh_hidden[tn])
            n_cell = int(np.count_nonzero(cell == native_cell))
            n_hidden = int(np.count_nonzero(hidden == native_hidden))
            scored.append(
                {
                    "sigmoid": sn,
                    "tanh": tn,
                    "cell_lanes": n_cell,
                    "hidden_lanes": n_hidden,
                    "joint_lanes": n_cell + n_hidden,
                    "cell_max_abs": float(np.abs(cell - native_cell).max()),
                    "hidden_max_abs": float(np.abs(hidden - native_hidden).max()),
                }
            )
    scored.sort(key=lambda r: (-r["joint_lanes"], -r["cell_lanes"], r["sigmoid"], r["tanh"]))
    pinned = next(
        r
        for r in scored
        if r["sigmoid"] == "sig:exact:161616" and r["tanh"] == "tanh:exact:161616"
    )
    result["catalogue"] = {
        "sigmoid_contracts": len(sigs),
        "tanh_contracts": len(tanhs),
        "combinations": len(scored),
        "pinned_baseline": pinned,
        "best_by_joint": scored[:20],
        "best_by_cell": sorted(scored, key=lambda r: -r["cell_lanes"])[:10],
        "best_by_hidden": sorted(scored, key=lambda r: -r["hidden_lanes"])[:10],
        "bit_exact_on_all_1280": [r for r in scored if r["joint_lanes"] == 2 * HIDDEN],
    }

    # ---- what the residual is, quantitatively -------------------------------- #
    s_exact, t_exact = sigmoid(b_i), np.tanh(b_c)
    product = s_exact * t_exact
    q = native_cell / product - 1.0  # relative error of the native product
    rounding_only = f16(product) / product - 1.0
    so_exact, to_exact = sigmoid(b_o), np.tanh(native_cell)
    q_hidden = native_hidden / (so_exact * to_exact) - 1.0
    excess = float(np.sqrt(max(q.var() - rounding_only.var(), 0.0)))
    result["residual"] = {
        "note": (
            "q = native / (exact sigmoid * exact tanh) - 1. Only the SUM of the two "
            "relative activation errors is identifiable from a product, so the mean "
            "below is the joint magnitude bias of the pair, not either one alone."
        ),
        "cell_mean_relative_bias": float(q.mean()),
        "cell_relative_bias_standard_error": float(q.std(ddof=1) / np.sqrt(q.size)),
        "cell_relative_bias_sigma": float(q.mean() / (q.std(ddof=1) / np.sqrt(q.size))),
        "cell_relative_std": float(q.std(ddof=1)),
        "hidden_mean_relative_bias": float(q_hidden.mean()),
        "hidden_relative_bias_sigma": float(
            q_hidden.mean() / (q_hidden.std(ddof=1) / np.sqrt(q_hidden.size))
        ),
        "product_rounding_only_relative_std": float(rounding_only.std(ddof=1)),
        "excess_relative_std_beyond_product_rounding": excess,
        "per_activation_rms_ulps_if_independent_and_equal": float(
            excess / np.sqrt(2.0) / np.mean(ulp16(product) / np.abs(product))
        ),
        "mean_bias_in_product_ulps": float(
            np.mean(q * np.abs(product) / ulp16(product))
        ),
    }

    # ---- additive decomposition of the two error curves ---------------------- #
    def bins(x, count):
        edges = np.quantile(x, np.linspace(0.0, 1.0, count + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        idx = np.digitize(x, edges[1:-1])
        design = np.zeros((x.size, count))
        design[np.arange(x.size), idx] = 1.0
        return design, edges

    n_bin = 20
    design_i, edges_i = bins(b_i, n_bin)
    design_c, edges_c = bins(b_c, n_bin)
    # native - exact = tanh * d_sigmoid(b_i) + sigmoid * d_tanh(b_c) + rounding
    design = np.concatenate(
        [t_exact[:, None] * design_i, s_exact[:, None] * design_c], axis=1
    )
    coef, *_ = np.linalg.lstsq(design, native_cell - product, rcond=None)
    d_sig, d_tanh = coef[:n_bin], coef[n_bin:]
    mids_i = (edges_i[:-1] + edges_i[1:]) / 2.0
    mids_c = (edges_c[:-1] + edges_c[1:]) / 2.0
    result["error_curves"] = {
        "note": (
            "Binned additive fit. A constant relative offset is degenerate between "
            "the two curves, so read the shapes, and read only the sum of the levels."
        ),
        "sigmoid": [
            {
                "b_i": float(m),
                "sigmoid": float(sigmoid(m)),
                "abs_error": float(d),
                "ulps": float(d / ulp16(sigmoid(m))),
                "relative": float(d / sigmoid(m)),
            }
            for m, d in zip(mids_i, d_sig)
        ],
        "tanh": [
            {
                "b_c": float(m),
                "tanh": float(np.tanh(m)),
                "abs_error": float(d),
                "ulps": float(d / ulp16(np.tanh(m))),
                "relative": float(d / np.tanh(m)),
            }
            for m, d in zip(mids_c, d_tanh)
        ],
        "tanh_magnitude_bias_sign_agrees_with_argument_sign": int(
            np.count_nonzero(np.sign(d_tanh) == np.sign(mids_c))
        ),
    }

    # ---- how far from correctly rounded can the native functions be? --------- #
    # Feasibility of "both activations within +-k fp16 ulps of correctly rounded",
    # evaluated exactly per lane on the product constraint.
    s_round, t_round = f16(s_exact), f16(t_exact)
    so_round, to_round = f16(so_exact), f16(to_exact)
    faithful = {}
    for k in (0, 1, 2, 3, 4):
        offsets = range(-k, k + 1)
        s_grid = np.stack([shift_ulp(s_round, a) for a in offsets])
        t_grid = np.stack([shift_ulp(t_round, b) for b in offsets])
        ok = np.zeros(HIDDEN, bool)
        for a in range(len(s_grid)):
            for b in range(len(t_grid)):
                ok |= f16(s_grid[a] * t_grid[b]) == native_cell
        so_grid = np.stack([shift_ulp(so_round, a) for a in offsets])
        to_grid = np.stack([shift_ulp(to_round, b) for b in offsets])
        ok_h = np.zeros(HIDDEN, bool)
        for a in range(len(so_grid)):
            for b in range(len(to_grid)):
                ok_h |= f16(so_grid[a] * to_grid[b]) == native_hidden
        faithful[str(k)] = {
            "cell_lanes_explainable": int(ok.sum()),
            "hidden_lanes_explainable": int(ok_h.sum()),
            "cell_lanes_unexplainable": int((~ok).sum()),
            "hidden_lanes_unexplainable": int((~ok_h).sum()),
        }
    result["faithfulness_bound"] = {
        "note": (
            "A lane is explainable at k if some pair of fp16 values within k ulps of "
            "the correctly rounded sigmoid and tanh multiplies and rounds to the "
            "native lane. Unexplainable lanes falsify that accuracy claim outright."
        ),
        "by_max_ulp_offset": faithful,
    }

    # ---- isolation: arguments that repeat ----------------------------------- #
    # Every lane is one product of two unknown function values, so a single lane
    # determines neither. Arguments that repeat break that: all lanes sharing a
    # sigmoid argument share ONE native sigmoid value, so the group is a constraint
    # on the tanh contract alone, with no sigmoid assumption at all, and vice versa.
    # Both equations read the SAME layer-0 gate vector, so every sigmoid argument
    # here belongs to one elementwise sigmoid op and the b_i/b_o groups are sound.
    # The two tanh instances are different ops (cell_activation on the gate, then
    # activation on the cell), so tanh groups are reported same-op and cross
    # separately, and the falsification verdict is taken on same-op groups only.
    def build_groups(keys, partners, natives):
        groups: dict[float, list[tuple[float, float]]] = {}
        for key, partner, native in zip(keys, partners, natives):
            groups.setdefault(float(key), []).append((float(partner), float(native)))
        return [v for v in groups.values() if len(v) > 1]

    sigmoid_groups = build_groups(
        np.concatenate([b_i, b_o]),
        np.concatenate([b_c, native_cell]),
        np.concatenate([native_cell, native_hidden]),
    )
    tanh_groups_same_op = build_groups(b_c, b_i, native_cell) + build_groups(
        native_cell, b_o, native_hidden
    )
    tanh_groups_all = build_groups(
        np.concatenate([b_c, native_cell]),
        np.concatenate([b_i, b_o]),
        np.concatenate([native_cell, native_hidden]),
    )

    def feasible_shared_value(observations, partner_fn, bounds):
        """Closed interval of shared values consistent with every observation.

        ``partner_fn`` supplies the other activation's value, so an empty interval
        falsifies that contract without assuming anything about this activation
        beyond it being a deterministic function of its argument.
        """
        low, high = bounds
        for partner_arg, native in observations:
            other = float(partner_fn(np.array([partner_arg]))[0])
            lo, hi = round_interval(np.array([native]))
            lo, hi = float(lo[0]), float(hi[0])
            if other == 0.0:
                if native != 0.0:
                    return None
                continue
            a, b = lo / other, hi / other
            if a > b:
                a, b = b, a
            low, high = max(low, a), min(high, b)
            if low > high:
                return None
        return low, high

    def contains_fp16(low, high):
        candidate = np.float16(low)
        if float(candidate) < low:
            candidate = np.nextafter(candidate, np.float16(np.inf))
        return float(candidate) <= high

    def falsify(groups, contracts, bounds):
        out = {}
        for name, fn in contracts.items():
            empty = fp16_empty = 0
            for observations in groups:
                span = feasible_shared_value(observations, fn, bounds)
                if span is None:
                    empty += 1
                    fp16_empty += 1
                elif not contains_fp16(*span):
                    fp16_empty += 1
            out[name] = {"groups_with_no_real_value": empty,
                         "groups_with_no_fp16_value": fp16_empty}
        return out

    tanh_verdicts = falsify(sigmoid_groups, tanhs, (0.0, 1.0))
    sigmoid_verdicts = falsify(tanh_groups_same_op, sigs, (-1.0, 1.0))
    sigmoid_verdicts_all = falsify(tanh_groups_all, sigs, (-1.0, 1.0))

    def survivors(verdicts):
        return sorted(
            n for n, v in verdicts.items() if v["groups_with_no_fp16_value"] == 0
        )

    surviving_tanh = survivors(tanh_verdicts)
    surviving_sigmoid = survivors(sigmoid_verdicts)
    result["isolated_constraints"] = {
        "note": (
            "A group is a set of lanes sharing one activation argument, so the shared "
            "activation value is one unknown. An empty feasible interval falsifies the "
            "OTHER activation's contract with no assumption about this one; an empty "
            "fp16 intersection falsifies it given only that this activation emits fp16."
        ),
        "sigmoid_argument_groups": len(sigmoid_groups),
        "sigmoid_argument_group_lanes": sum(len(v) for v in sigmoid_groups),
        "tanh_argument_groups_same_op": len(tanh_groups_same_op),
        "tanh_argument_group_lanes_same_op": sum(len(v) for v in tanh_groups_same_op),
        "tanh_argument_groups_including_cross_op": len(tanh_groups_all),
        "tanh_contracts_surviving_sigmoid_free_test": len(surviving_tanh),
        "tanh_contracts_tested": len(tanhs),
        "sigmoid_contracts_surviving_tanh_free_test": len(surviving_sigmoid),
        "sigmoid_contracts_surviving_with_cross_op_groups": len(
            survivors(sigmoid_verdicts_all)
        ),
        "sigmoid_contracts_tested": len(sigs),
        "exact_tanh_verdict": tanh_verdicts["tanh:exact:161616"],
        "exact_sigmoid_verdict": sigmoid_verdicts["sig:exact:161616"],
        "fp16_chain_sigmoid_verdict": sigmoid_verdicts["sig:recip:161616"],
        "surviving_sigmoid_contracts": surviving_sigmoid,
        "surviving_tanh_contracts": surviving_tanh[:40],
    }

    # ---- measure the native functions, rather than only exclude forms -------- #
    # Arc consistency over the bipartite constraint graph: nodes are the distinct
    # sigmoid and tanh arguments, edges are the 1280 lanes, and each lane says that
    # the product of its two node values rounds to the captured fp16 lane. Interval
    # narrowing is useless here, because one lane pins a product to half an ulp
    # while each factor is unknown by more than that, so the domains are kept
    # discrete: every node ranges over the fp16 numbers within ``radius`` ulps of
    # correct rounding, and a value survives only if some partner value supports it
    # on every lane the node touches. A node reduced to one value is a native
    # activation value read straight out of the capture.
    def scalar16(value):
        return float(np.float16(value))

    def ulp_domain(argument, exact_fn, radius):
        correct = f16(exact_fn(np.array([argument])))
        return [
            float(shift_ulp(correct, step)[0]) for step in range(-radius, radius + 1)
        ]

    def arc_consistency(lane_list, radius, monotone=False):
        """Prune fp16 domains until no lane, and optionally no ordering, is violated.

        ``monotone`` adds the structural assumption that the native activation is
        non-decreasing in its argument, which chains every argument together rather
        than leaving most nodes with a single lane. It is an assumption, so it is
        reported separately; ``lanes_unsatisfiable`` is how it gets tested.
        """
        domain = {}
        for lane in lane_list:
            domain.setdefault(("s", lane[0]), ulp_domain(lane[0], sigmoid, radius))
            domain.setdefault(("t", lane[1]), ulp_domain(lane[1], np.tanh, radius))
        edges = [(("s", a), ("t", b), c) for a, b, c in lane_list]
        chains = {
            kind: sorted(key for key in domain if key[0] == kind) for kind in ("s", "t")
        }
        emptied = set()  # distinct constraints the radius cannot satisfy
        for _ in range(256):
            changed = False
            for lane, (key_s, key_t, native) in enumerate(edges):
                values_s, values_t = domain[key_s], domain[key_t]
                keep_s = [
                    s for s in values_s if any(scalar16(s * t) == native for t in values_t)
                ]
                keep_t = [
                    t for t in values_t if any(scalar16(s * t) == native for s in values_s)
                ]
                if not keep_s or not keep_t:
                    emptied.add(("lane", lane))
                    continue  # this lane cannot be met inside the radius; leave it
                if len(keep_s) < len(values_s):
                    domain[key_s] = keep_s
                    changed = True
                if len(keep_t) < len(values_t):
                    domain[key_t] = keep_t
                    changed = True
            if monotone:
                for chain in chains.values():
                    floor = -np.inf
                    for key in chain:  # arguments ascend, so values may not fall
                        kept = [v for v in domain[key] if v >= floor]
                        if not kept:
                            emptied.add(("order-up", key))
                            continue
                        if len(kept) < len(domain[key]):
                            domain[key] = kept
                            changed = True
                        floor = min(kept)
                    ceiling = np.inf
                    for key in reversed(chain):
                        kept = [v for v in domain[key] if v <= ceiling]
                        if not kept:
                            emptied.add(("order-down", key))
                            continue
                        if len(kept) < len(domain[key]):
                            domain[key] = kept
                            changed = True
                        ceiling = max(kept)
            if not changed:
                break
        return domain, len(emptied)

    lane_list = [
        *zip(b_i.tolist(), b_c.tolist(), native_cell.tolist()),
        *zip(b_o.tolist(), native_cell.tolist(), native_hidden.tolist()),
    ]
    radii = (1, 2, 3)
    solved = {radius: arc_consistency(lane_list, radius) for radius in radii}
    solved_monotone = {
        radius: arc_consistency(lane_list, radius, monotone=True) for radius in radii
    }

    def pinned_table(domain, kind, exact_fn):
        out = {}
        for (node_kind, argument), values in domain.items():
            if node_kind != kind or len(values) != 1:
                continue
            correct = float(f16(exact_fn(np.array([argument])))[0])
            step = float(ulp16(np.array([correct]))[0]) or 1.0
            out[argument] = (values[0], correct, int(round((values[0] - correct) / step)))
        return out

    def summarise(runs):
        out = {}
        for kind, exact_fn in (("s", sigmoid), ("t", np.tanh)):
            tables = [pinned_table(runs[r][0], kind, exact_fn) for r in radii]
            sound = [
                table for table, r in zip(tables, radii) if runs[r][1] == 0
            ] or tables[-1:]
            shared = set(sound[0])
            for table in sound[1:]:
                shared &= set(table)
            pinned_rows, offsets = [], {}
            for argument in sorted(shared):
                if len({table[argument][0] for table in sound}) != 1:
                    continue
                value, correct, offset = sound[-1][argument]
                offsets[str(offset)] = offsets.get(str(offset), 0) + 1
                pinned_rows.append(
                    {
                        "argument": argument,
                        "native_value": value,
                        "correctly_rounded": correct,
                        "signed_ulp_offset": offset,
                    }
                )
            out[kind] = {
                "pinned": pinned_rows,
                "signed_ulp_offset_from_correct_rounding": offsets,
                "pinned_per_radius": [len(t) for t in tables],
                "pinned_count": len(pinned_rows),
            }
        return out

    plain, monotone = summarise(solved), summarise(solved_monotone)

    lane_counts = {(r["sigmoid"], r["tanh"]): r["joint_lanes"] for r in scored}

    def agreement(pinned_rows, contracts, partner):
        """How many pinned native values each contract reproduces exactly.

        ``partner`` is the other activation's leading contract, so a contract that
        matches every pinned value can be shown next to the lane count it actually
        earns: pointwise agreement on a handful of arguments is weak evidence.
        """
        if not pinned_rows:
            return {}
        arguments = np.array([row["argument"] for row in pinned_rows])
        native = np.array([row["native_value"] for row in pinned_rows])
        scores = {
            name: int(np.count_nonzero(fn(arguments) == native))
            for name, fn in contracts.items()
        }
        best = max(scores.values())

        def with_lanes(names):
            return [
                {
                    "contract": name,
                    "joint_lanes": lane_counts.get(
                        (partner, name), lane_counts.get((name, partner))
                    ),
                }
                for name in sorted(names)[:10]
            ]

        return {
            "pinned_values": len(pinned_rows),
            "best_agreement": best,
            "paired_with": partner,
            "reproducing_every_pinned_value": with_lanes(
                n for n, v in scores.items() if v == len(pinned_rows)
            ),
            "leaders": with_lanes(n for n, v in scores.items() if v == best),
        }

    result["measured_native_values"] = {
        "note": (
            "Discrete arc consistency over fp16 domains of the stated radius around "
            "correct rounding. Radii whose lane set is fully satisfiable are sound; a "
            "radius with unsatisfiable lanes is refuted as an accuracy claim and its "
            "pins are not used. Each listed entry is a native activation value read "
            "out of the capture with no candidate contract assumed."
        ),
        "domain_radii_in_ulps": list(radii),
        "lanes_unsatisfiable_within_radius": {str(r): solved[r][1] for r in radii},
        "lanes_unsatisfiable_within_radius_monotone": {
            str(r): solved_monotone[r][1] for r in radii
        },
        "lane_constraints": len(lane_list),
        "sigmoid_arguments": len({lane[0] for lane in lane_list}),
        "tanh_arguments": len({lane[1] for lane in lane_list}),
        "sigmoid": plain["s"],
        "tanh": plain["t"],
        "monotone_note": (
            "Same solver plus the structural assumption that each native activation "
            "is non-decreasing in its argument, which chains all arguments together "
            "instead of leaving most nodes with a single lane."
        ),
        "monotone_sigmoid": monotone["s"],
        "monotone_tanh": monotone["t"],
        "pointwise_contract_agreement": {
            "sigmoid": agreement(
                monotone["s"]["pinned"], sigs, "tanh:exact:161616"
            ),
            "tanh": agreement(monotone["t"]["pinned"], tanhs, "sig:recip:161616"),
        },
    }

    # ---- what the pinned values say about the native error ------------------- #
    # A pinned native value plus the exact function value bounds the native
    # evaluation error: the native emitted fp16 number is the rounding of some
    # internal value, so that internal value sits within half an ulp of it, and the
    # distance from the exact value follows. Errors are reported toward larger
    # magnitude, so both signs of the argument read on one scale.
    def error_bounds(pinned_rows, exact_fn):
        rows, window = [], (-np.inf, np.inf)
        for row in pinned_rows:
            argument = row["argument"]
            exact = float(exact_fn(np.array([argument]))[0])
            correct = row["correctly_rounded"]
            step = float(ulp16(np.array([correct]))[0]) or 1.0
            offset = row["signed_ulp_offset"]
            fraction = (exact - correct) / step
            low, high = offset - 0.5 - fraction, offset + 0.5 - fraction
            sign = 1.0 if exact >= 0.0 else -1.0
            magnitude = sorted((sign * low, sign * high))
            window = (max(window[0], magnitude[0]), min(window[1], magnitude[1]))
            rows.append(
                {
                    "argument": argument,
                    "exact": exact,
                    "exact_position_in_cell": fraction,
                    "native_minus_correctly_rounded_ulps": offset,
                    "magnitude_error_bound_ulps": magnitude,
                }
            )
        return {
            "per_argument": rows,
            "constant_magnitude_offset_window_ulps": (
                list(window) if window[0] <= window[1] else None
            ),
        }

    result["error_bounds_from_pins"] = {
        "note": (
            "Bounds on the native evaluation error at the pinned arguments, in fp16 "
            "ulps toward larger magnitude. A window means one constant magnitude "
            "offset satisfies every pin; null means the native error demonstrably "
            "varies with the argument."
        ),
        "sigmoid": error_bounds(monotone["s"]["pinned"], sigmoid),
        "tanh": error_bounds(monotone["t"]["pinned"], np.tanh),
    }

    # ---- does that bias, applied to exact tanh, actually recover lanes? ------ #
    # Not an implementable contract: a fitted correction, reported because it tests
    # the bias the pins predict against all 1280 lanes at once.
    best_sigmoid = sigs["sig:recip:161616"]
    s_cell, s_hidden = best_sigmoid(b_i), best_sigmoid(b_o)

    def bias_lanes(shift):
        cell = f16(s_cell * f16(shift(b_c)))
        hidden = f16(s_hidden * f16(shift(native_cell)))
        return int(np.count_nonzero(cell == native_cell)) + int(
            np.count_nonzero(hidden == native_hidden)
        )

    offsets = [
        {
            "magnitude_offset_ulps": float(round(m, 3)),
            "lanes": bias_lanes(
                lambda x, m=m: np.tanh(f64(x))
                + m * ulp16(np.tanh(f64(x))) * np.sign(f64(x))
            ),
        }
        for m in np.arange(-0.4, 0.81, 0.02)
    ]
    scales = [
        {
            "relative_scale": float(f"{c:.2e}"),
            "lanes": bias_lanes(lambda x, c=c: np.tanh(f64(x)) * (1.0 + c)),
        }
        for c in np.arange(-2e-4, 5.01e-4, 1e-5)
    ]
    result["tanh_bias_sweep"] = {
        "note": (
            "Exact tanh biased toward larger magnitude, paired with the fp16 chain "
            "sigmoid, scored on all 1280 reduction-free lanes. A fitted correction, "
            "not a contract: it measures the size of the native tanh bias."
        ),
        "unbiased_lanes": bias_lanes(lambda x: np.tanh(f64(x))),
        "best_magnitude_offset": max(offsets, key=lambda r: r["lanes"]),
        "best_relative_scale": max(scales, key=lambda r: r["lanes"]),
        "magnitude_offset_sweep": offsets,
        "relative_scale_sweep": scales,
    }

    # ---- does the best elementwise contract carry to all 15 transitions? ----- #
    def gate(x, h, w_ih, w_hh, b):
        """The fp32-class reduction the previous study measured as least-bad."""
        return f16(
            f16(
                f16(np.asarray(x, np.float32) @ np.asarray(w_ih, np.float32).T)
                + f16(np.asarray(h, np.float32) @ np.asarray(w_hh, np.float32).T)
            )
            + b
        )

    def run_lstm(row, sig_fn, tanh_fn):
        seq = weights["embedding"][int(row["decoder_input_ids"].reshape(-1)[0])].astype(
            np.float64
        )
        hidden_in = row["decoder_hidden"][:, 0].astype(np.float64)
        cell_in = row["decoder_cell"][:, 0].astype(np.float64)
        cells, hiddens = [], []
        for layer in (0, 1):
            gates = gate(
                seq,
                hidden_in[layer],
                weights[f"l{layer}_ih"].astype(np.float64),
                weights[f"l{layer}_hh"].astype(np.float64),
                weights[f"l{layer}_bias"].astype(np.float64),
            )
            g_i, g_f, g_o, g_c = np.split(gates, 4)
            cell = f16(
                f16(sig_fn(g_f) * cell_in[layer]) + f16(sig_fn(g_i) * tanh_fn(g_c))
            )
            hidden = f16(sig_fn(g_o) * tanh_fn(cell))
            cells.append(cell)
            hiddens.append(hidden)
            seq = hidden
        return np.stack(cells), np.stack(hiddens)

    def score_all(sig_fn, tanh_fn):
        cell_ok = hidden_ok = 0
        cell_max = hidden_max = 0.0
        for row in rows:
            cells, hiddens = run_lstm(row, sig_fn, tanh_fn)
            nc = row["decoder_next_cell"][:, 0].astype(np.float64)
            nh = row["decoder_next_hidden"][:, 0].astype(np.float64)
            cell_ok += int(np.count_nonzero(cells == nc))
            hidden_ok += int(np.count_nonzero(hiddens == nh))
            cell_max = max(cell_max, float(np.abs(cells - nc).max()))
            hidden_max = max(hidden_max, float(np.abs(hiddens - nh).max()))
        total = 2 * HIDDEN * len(rows)
        return {
            "next_cell_lanes": cell_ok,
            "next_hidden_lanes": hidden_ok,
            "lanes": total,
            "next_cell_max_abs": cell_max,
            "next_hidden_max_abs": hidden_max,
        }

    carry = []
    wanted = [(pinned["sigmoid"], pinned["tanh"])] + [
        (r["sigmoid"], r["tanh"]) for r in scored[:4]
    ]
    for sn, tn in dict.fromkeys(wanted):
        carry.append({"sigmoid": sn, "tanh": tn, **score_all(sigs[sn], tanhs[tn])})
    result["all_transitions"] = {
        "note": (
            "Full two-layer LSTM from native-injected state, with the fp32-class gate "
            "reduction the previous study measured as least-bad. The gate reduction is "
            "still unresolved, so these counts are not a clean activation measurement; "
            "they only show whether the elementwise gain survives a reduction."
        ),
        "variants": carry,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")

    rr = result["reduction_free_lanes"]
    print(f"reduction-free lanes: cell {rr['cell_lanes']} hidden {rr['hidden_lanes']}")
    print(
        "pinned baseline: "
        f"{rr['pinned_baseline_bit_exact_cell_lanes']}/{HIDDEN} cell lanes, "
        f"max abs {rr['pinned_baseline_max_abs']}"
    )
    print(f"declared MIL activations: {result['declared_mil_activations']}")
    print(
        f"catalogue: {len(sigs)} sigmoid x {len(tanhs)} tanh = {len(scored)} combinations"
    )
    for row in scored[:8]:
        print(
            f"  {row['joint_lanes']:4d}/1280  cell {row['cell_lanes']:3d} "
            f"hidden {row['hidden_lanes']:3d}  {row['sigmoid']} + {row['tanh']}"
        )
    print(f"bit-exact on all 1280: {len(result['catalogue']['bit_exact_on_all_1280'])}")
    res = result["residual"]
    print(
        f"joint magnitude bias {res['cell_mean_relative_bias']:.3e} "
        f"({res['cell_relative_bias_sigma']:.1f} sigma), "
        f"excess scatter {res['excess_relative_std_beyond_product_rounding']:.3e}"
    )
    print(f"faithfulness: {json.dumps(faithful)}")
    iso = result["isolated_constraints"]
    print(
        f"isolation: {iso['sigmoid_argument_groups']} sigmoid-argument groups "
        f"({iso['sigmoid_argument_group_lanes']} lanes), "
        f"{iso['tanh_argument_groups_same_op']} same-op tanh-argument groups "
        f"({iso['tanh_argument_group_lanes_same_op']} lanes)"
    )
    print(
        f"  sigmoid-free test: {iso['tanh_contracts_surviving_sigmoid_free_test']}"
        f"/{iso['tanh_contracts_tested']} tanh contracts survive; "
        f"exact tanh {iso['exact_tanh_verdict']}"
    )
    print(
        f"  tanh-free test: {iso['sigmoid_contracts_surviving_tanh_free_test']}"
        f"/{iso['sigmoid_contracts_tested']} sigmoid contracts survive; "
        f"exact sigmoid {iso['exact_sigmoid_verdict']}, "
        f"fp16 chain {iso['fp16_chain_sigmoid_verdict']}"
    )
    pins = result["measured_native_values"]
    print(
        f"lanes unsatisfiable by radius: "
        f"{json.dumps(pins['lanes_unsatisfiable_within_radius'])} plain, "
        f"{json.dumps(pins['lanes_unsatisfiable_within_radius_monotone'])} monotone"
    )
    for tag in ("sigmoid", "tanh"):
        flat, mono = pins[tag], pins[f"monotone_{tag}"]
        print(
            f"measured {tag}: {flat['pinned_count']} pinned of "
            f"{pins[tag + '_arguments']} arguments "
            f"(per radius {flat['pinned_per_radius']}); monotone "
            f"{mono['pinned_count']} (per radius {mono['pinned_per_radius']}), offsets "
            f"{json.dumps(mono['signed_ulp_offset_from_correct_rounding'])}"
        )
        print(f"  pointwise: {json.dumps(pins['pointwise_contract_agreement'][tag])}")
        bound = result["error_bounds_from_pins"][tag]
        print(
            f"  constant magnitude offset window (ulps): "
            f"{bound['constant_magnitude_offset_window_ulps']}"
        )
    sweep = result["tanh_bias_sweep"]
    print(
        f"tanh bias sweep: unbiased {sweep['unbiased_lanes']}/1280, best offset "
        f"{sweep['best_magnitude_offset']}, best scale {sweep['best_relative_scale']}"
    )
    for row in carry:
        print(
            f"  all-15: cell {row['next_cell_lanes']}/{row['lanes']} hidden "
            f"{row['next_hidden_lanes']}/{row['lanes']}  {row['sigmoid']} + {row['tanh']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Test the ANE staged-evaluation hypothesis for the native Parakeet decoder
``sigmoid``/``tanh``, against the pinned Apple unary lookup tables.

Named hole: ``parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml``.

Three things this adds to ``receipts/2026-09-14-decoder-activations``:

1. The Apple unary lookup tables are decoded from the checked-in mil-hwx-compiler
   oracle records and turned into runnable candidate activations, so the hardware's
   own table is tested rather than a guessed polynomial.
2. The ANE staged forms recorded in ane-linux-experiments ``THEORY.MD`` -- the
   activation exponential that divides by eight and squares three times, the
   stable positive/negative sigmoid halves, and the Newton reciprocal -- are each
   built with an explicit rounding width per stage and tested.
3. A refutation metric that needs no partner contract and no repeated arguments:
   for a candidate on one side, solve every lane for the *other* activation's fp16
   value and count arguments whose feasible set is empty. An empty set falsifies
   the candidate assuming only that the partner is a deterministic fp16 function
   and that the lane is one fp16 multiply. That scores 1172 or 1198 arguments
   instead of 42 groups.

Host-only. No MLX, no GPU, no ANE, no device execution, no ``/tmp/m1-gpu.lock``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

erf = np.vectorize(math.erf)

HIDDEN = 640
INF16 = np.float16(np.inf)
# The seven nonlinear unary tables are a 128-byte fp16 block: four header words
# (input clamp low, input clamp high, output below low, output above high) then 33
# knots. mil-hwx-compiler research/oracle-diff.md records the block as a
# target-independent prefix, identical for H13 and H14.
KNOT_LO, KNOT_COUNT = 4, 33
TABLE_GRID = {  # operation -> (first knot argument, knot step, odd in the argument)
    "sigmoid": (-8.0, 0.5, False),
    "tanh": (0.0, 0.125, True),
    "exp": (0.0, 1.0 / 32.0, False),  # 2**f mantissa table, not exp(x) directly
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# the pinned Apple unary tables
# --------------------------------------------------------------------------- #
def read_table(oracles: Path, target: str, operation: str):
    record = json.loads((oracles / target / f"unary_{operation}_c64.json").read_text())
    section = record["constant_section"]
    words = [0.0] * 64
    # Only sections of at most 256 bytes carry the word table, so the H13 record
    # (the same prefix padded to 2 KiB) contributes its hash, not its values.
    for entry in section["nonzero_fp16_words"] or ():
        value = entry["value"]
        words[entry["index"]] = (
            float("inf") if value == "+inf" else -float("inf") if value == "-inf" else value
        )
    return {
        "prefix_128_sha256": section["prefix_128_sha256"],
        "size": section["size"],
        "header": words[:KNOT_LO],
        "knots": words[KNOT_LO : KNOT_LO + KNOT_COUNT],
        "trailing": words[KNOT_LO + KNOT_COUNT : 43],
    }


def knot_arguments(operation: str):
    start, step, _ = TABLE_GRID[operation]
    return np.array([start + step * k for k in range(KNOT_COUNT)])


def verify_knots(table, operation, f16, exact):
    """Are the knots exactly the fp16 rounding of the function at the knot grid?"""
    knots = np.array(table["knots"])
    reference = f16(exact(knot_arguments(operation)))
    return {
        "knots_are_exact_fp16_of_the_function": bool(np.array_equal(knots, reference)),
        "knots_differing": int(np.count_nonzero(knots != reference)),
        "max_abs_difference": float(np.abs(knots - reference).max()),
    }


# --------------------------------------------------------------------------- #
# candidate activations
# --------------------------------------------------------------------------- #
def make_lut(table, operation, round_out):
    """Linear interpolation in the hardware table, with the header's clamps."""
    knots = np.array(table["knots"])
    start, step, odd = TABLE_GRID[operation]
    lo, hi, below, above = table["header"]

    def evaluate(x, r=round_out):
        x = np.asarray(x, np.float64)
        sign = np.sign(x) if odd else np.ones_like(x)
        value = np.abs(x) if odd else x
        index = (value - start) / step
        k = np.clip(np.floor(index).astype(int), 0, KNOT_COUNT - 2)
        t = r(index - k)
        a, b = knots[k], knots[k + 1]
        out = r(a + r(t * r(b - a)))
        out = np.where(value <= lo, below, out)
        out = np.where(value >= hi, above, out)
        return r(sign * out)

    return evaluate


def make_exp(kind, width, table, rounds, l2e, squarings):
    """One exp contract: table or correctly rounded, optionally div-by-8 and squared.

    ``squarings`` implements the ANE activation exponential recorded in THEORY.MD:
    divide the argument by eight and square three times.
    """
    r = rounds[width]
    knots = np.array(table["knots"])

    def table_exp(z):
        y = r(np.asarray(z, np.float64) * l2e)
        n = np.floor(y)
        index = r(r(y - n) * 32.0)
        k = np.clip(np.floor(index).astype(int), 0, 31)
        t = r(index - k)
        a, b = knots[k], knots[k + 1]
        return r(r(a + r(t * r(b - a))) * np.exp2(n))

    seed = table_exp if kind == "table" else (lambda z: r(np.exp(np.asarray(z, np.float64))))
    if not squarings:
        return seed

    scale = float(2**squarings)

    def staged(z):
        value = seed(np.asarray(z, np.float64) / scale)
        for _ in range(squarings):
            value = r(value * value)
        return value

    return staged


def make_reciprocal(kind, width, rounds, steps=12, seed=1.0 / 128.0):
    """Divide, or the ANE Newton reciprocal THEORY.MD records for softmax."""
    r = rounds[width]
    if kind == "div":
        return lambda a: r(1.0 / np.asarray(a, np.float64))

    def newton(a):
        a = np.asarray(a, np.float64)
        out = np.full(np.shape(a), seed, float)
        for _ in range(steps):
            out = r(out * r(2.0 - r(a * out)))
        return out

    return newton


def staged_catalogue(tables, rounds, l2e):
    """Staged sigmoid and tanh contracts built out of one exp and one reciprocal."""
    exps = {
        f"{kind}{width}{'.sq3' if squarings else ''}": make_exp(
            kind, width, tables["exp"], rounds, l2e, squarings
        )
        for kind in ("exact", "table")
        for width in (16, 32)
        for squarings in (0, 3)
    }
    reciprocals = {"div": "div", "newton12": "newton"}
    widths = list(itertools.product((16, 32), repeat=2))
    sigmoids, tanhs = {}, {}
    for exp_name, exp_fn in exps.items():
        for add_w, out_w in widths:
            for rec_name, rec_kind in reciprocals.items():
                ra, ro = rounds[add_w], rounds[out_w]
                rec = make_reciprocal(rec_kind, out_w, rounds)
                tag = f"{exp_name}|{add_w}{out_w}|{rec_name}"

                def plain(x, ra=ra, ro=ro, rec=rec, exp_fn=exp_fn):
                    return ro(rec(ra(1.0 + exp_fn(-np.asarray(x, np.float64)))))

                def halves(x, ra=ra, ro=ro, rec=rec, exp_fn=exp_fn):
                    x = np.asarray(x, np.float64)
                    e = exp_fn(-np.abs(x))
                    positive = ro(rec(ra(1.0 + e)))
                    negative = ro(e * rec(ra(1.0 + e)))
                    return np.where(x >= 0.0, positive, negative)

                sigmoids[f"S[{tag}|plain]"] = plain
                sigmoids[f"S[{tag}|halves]"] = halves
                for form in ("oneminus2", "ratio", "mulrcp", "negexp", "negmul"):
                    for odd in (True, False):

                        def tanh_fn(
                            x, ra=ra, ro=ro, rec=rec, exp_fn=exp_fn, form=form, odd=odd
                        ):
                            x = np.asarray(x, np.float64)
                            sign = np.sign(x)
                            a = np.abs(x) if odd else x
                            if form in ("negexp", "negmul"):
                                e = exp_fn(-2.0 * a)
                                value = (
                                    ro(ra(1.0 - e) / ra(1.0 + e))
                                    if form == "negexp"
                                    else ro(ra(1.0 - e) * rec(ra(1.0 + e)))
                                )
                            else:
                                big = exp_fn(2.0 * a)
                                if form == "oneminus2":
                                    value = ro(1.0 - ra(2.0 * rec(ra(big + 1.0))))
                                elif form == "ratio":
                                    value = ro(ra(big - 1.0) / ra(big + 1.0))
                                else:
                                    value = ro(ra(big - 1.0) * rec(ra(big + 1.0)))
                            return ro(sign * value) if odd else value

                        tanhs[f"T[{tag}|{form}|{'odd' if odd else 'dir'}]"] = tanh_fn
    return sigmoids, tanhs


# --------------------------------------------------------------------------- #
# feasibility: solve each lane for the partner activation's fp16 value
# --------------------------------------------------------------------------- #
def ordinal16(values):
    """Monotone integer ordering of fp16, so a value interval is an index range."""
    bits = np.asarray(values, dtype=np.float16).view(np.uint16).astype(np.int32)
    return np.where((bits >> 15).astype(bool), -(bits & 0x7FFF), bits)


def strict_above(x):
    """Smallest fp16 strictly greater than ``x`` (two steps cover the cast's rounding)."""
    out = np.asarray(x, np.float64).astype(np.float16)
    for _ in range(2):
        out = np.where(out.astype(np.float64) <= x, np.nextafter(out, INF16), out).astype(
            np.float16
        )
    return out


def strict_below(x):
    out = np.asarray(x, np.float64).astype(np.float16)
    for _ in range(2):
        out = np.where(out.astype(np.float64) >= x, np.nextafter(out, -INF16), out).astype(
            np.float16
        )
    return out


def round_interval(native):
    value = np.asarray(native, np.float64).astype(np.float16)
    up = np.nextafter(value, INF16).astype(np.float64)
    down = np.nextafter(value, -INF16).astype(np.float64)
    value = value.astype(np.float64)
    return (value + down) / 2.0, (value + up) / 2.0


def feasible_ranges(partner, group_index, group_count, native, clip):
    """Per argument, the fp16 index range of partner values consistent with every lane."""
    low, high = round_interval(native)
    partner = np.asarray(partner, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = low / partner, high / partner
    lo = np.where(partner == 0.0, -np.inf, np.minimum(a, b))
    hi = np.where(partner == 0.0, np.inf, np.maximum(a, b))
    lo, hi = np.maximum(lo, clip[0]), np.minimum(hi, clip[1])
    first, last = strict_above(lo), strict_below(hi)
    index_lo, index_hi = ordinal16(first), ordinal16(last)
    # No fp16 number inside the open interval: force an impossible range.
    vacant = np.asarray(first, np.float64) >= hi
    index_lo = np.where(vacant, 1 << 20, index_lo)
    out_lo = np.full(group_count, -(1 << 30), np.int64)
    out_hi = np.full(group_count, 1 << 30, np.int64)
    np.maximum.at(out_lo, group_index, index_lo)
    np.minimum.at(out_hi, group_index, index_hi)
    return out_lo, out_hi


def from_ordinal(order):
    order = np.asarray(order, np.int32)
    bits = np.where(order < 0, (-order) | 0x8000, order).astype(np.uint16)
    return bits.view(np.float16).astype(np.float64)


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument(
        "--pins", type=Path, default=here.parent / "2026-09-14-decoder-activations/result.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    sys.path.insert(0, str(args.overlay))
    sys.path.insert(0, str(args.pins.parent))
    import probe as base  # the previous receipt's probe: capture loader and catalogue

    f16, f32, f64 = base.f16, base.f32, base.f64
    rounds = {16: f16, 32: f32, 64: f64}
    sigmoid_exact, tanh_exact = base.sigmoid, (lambda x: np.tanh(f64(x)))

    component, weights, traces = base.load(args.package, args.capture_dir)
    first = traces[0]
    b_i, _b_f, b_o, b_c = np.split(weights["l0_bias"].astype(np.float64), 4)
    native_cell = first["decoder_next_cell"][0, 0].astype(np.float64)
    native_hidden = first["decoder_next_hidden"][0, 0].astype(np.float64)

    result: dict = {
        "schema": "mlx-omarchy.parakeet-decoder-activation-stages/1",
        "named_hole": "parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml",
        "execution": "host-only NumPy; no MLX, no GPU, no ANE, no device execution",
        "capture_dir": str(args.capture_dir),
        "capture_sha256": {
            name: sha256(args.capture_dir / name)
            for name in ("tdt_tensors.json", "manifest.sha256", "receipt.json")
            if (args.capture_dir / name).exists()
        },
        "pins_source": {"path": str(args.pins), "sha256": sha256(args.pins)},
        "transitions": [row["index"] for row in traces],
    }

    # ---- the hardware tables ------------------------------------------------- #
    tables, table_report = {}, {}
    for operation, exact in (
        ("sigmoid", sigmoid_exact),
        ("tanh", tanh_exact),
        ("exp", lambda x: np.exp2(f64(x))),  # the exp table stores 2**f, not exp(x)
    ):
        h13 = read_table(args.oracles, "h13", operation)
        h14 = read_table(args.oracles, "h14", operation)
        tables[operation] = h14
        table_report[operation] = {
            "record_sha256": {
                target: sha256(args.oracles / target / f"unary_{operation}_c64.json")
                for target in ("h13", "h14")
            },
            "prefix_128_sha256": h14["prefix_128_sha256"],
            "h13_and_h14_prefix_identical": h13["prefix_128_sha256"]
            == h14["prefix_128_sha256"],
            "header_clamp_low_high_out_below_out_above": h14["header"],
            "knot_arguments": [float(v) for v in knot_arguments(operation)[[0, 1, -1]]],
            **verify_knots(h14, operation, f16, exact),
        }
    # silu and gelu are the control: same block, knots NOT exact function values, so
    # the family is a fitted piecewise-linear contract rather than a sampling of f.
    for operation, exact in (
        ("silu", lambda x: f64(x) * sigmoid_exact(x)),
        ("gelu", lambda x: 0.5 * f64(x) * (1.0 + erf(f64(x) / np.sqrt(2.0)))),
    ):
        h14 = read_table(args.oracles, "h14", operation)
        grid = np.array([-8.0 + 0.5 * k for k in range(KNOT_COUNT)])
        if operation == "gelu":
            grid = np.array([-4.0 + 0.25 * k for k in range(KNOT_COUNT)])
        knots = np.array(h14["knots"])
        table_report[operation] = {
            "record_sha256": sha256(args.oracles / "h14" / f"unary_{operation}_c64.json"),
            "note": "control: same 128-byte block, knots are a fit, not exact samples",
            "knots_are_exact_fp16_of_the_function": bool(
                np.array_equal(knots, f16(exact(grid)))
            ),
            "knots_differing": int(np.count_nonzero(knots != f16(exact(grid)))),
        }
    result["hardware_unary_tables"] = table_report

    # ---- candidates ---------------------------------------------------------- #
    sigmoids, tanhs = staged_catalogue(tables, rounds, base.L2E64)
    sigmoids["S[lut.lerp16]"] = make_lut(tables["sigmoid"], "sigmoid", f16)
    sigmoids["S[lut.lerp32]"] = make_lut(tables["sigmoid"], "sigmoid", f32)
    sigmoids["S[correctly_rounded]"] = lambda x: f16(sigmoid_exact(x))
    tanhs["T[lut.lerp16]"] = make_lut(tables["tanh"], "tanh", f16)
    tanhs["T[lut.lerp32]"] = make_lut(tables["tanh"], "tanh", f32)
    tanhs["T[correctly_rounded]"] = lambda x: f16(tanh_exact(x))
    sigmoids.update(base.dedupe(base.sigmoid_catalogue(), b_i, b_o))
    tanhs.update(base.dedupe(base.tanh_catalogue(base.sigmoid_catalogue()), b_c, native_cell))
    sigmoid_values = {n: (f16(f(b_i)), f16(f(b_o))) for n, f in sigmoids.items()}
    tanh_values = {n: (f16(f(b_c)), f16(f(native_cell))) for n, f in tanhs.items()}

    # ---- the pins the previous receipt measured ------------------------------ #
    pins = json.loads(args.pins.read_text())["measured_native_values"]
    pin = {
        "sigmoid": (
            np.array([r["argument"] for r in pins["monotone_sigmoid"]["pinned"]]),
            np.array([r["native_value"] for r in pins["monotone_sigmoid"]["pinned"]]),
        ),
        "tanh": (
            np.array([r["argument"] for r in pins["monotone_tanh"]["pinned"]]),
            np.array([r["native_value"] for r in pins["monotone_tanh"]["pinned"]]),
        ),
    }
    pin_score = {
        "sigmoid": {
            n: int(np.count_nonzero(f16(f(pin["sigmoid"][0])) == pin["sigmoid"][1]))
            for n, f in sigmoids.items()
        },
        "tanh": {
            n: int(np.count_nonzero(f16(f(pin["tanh"][0])) == pin["tanh"][1]))
            for n, f in tanhs.items()
        },
    }

    # ---- refutation without a partner contract ------------------------------ #
    tanh_arguments = np.concatenate([b_c, native_cell])
    sigmoid_arguments = np.concatenate([b_i, b_o])
    natives = np.concatenate([native_cell, native_hidden])
    t_unique, t_index = np.unique(tanh_arguments, return_inverse=True)
    s_unique, s_index = np.unique(sigmoid_arguments, return_inverse=True)

    def refuted_by(values, unique, index, clip):
        lo, hi = feasible_ranges(
            np.concatenate(values), index, unique.size, natives, clip
        )
        return int(np.count_nonzero(lo > hi)), lo, hi

    sigmoid_refuted = {
        n: refuted_by(v, t_unique, t_index, (-1.0, 1.0))[0]
        for n, v in sigmoid_values.items()
    }
    tanh_refuted = {
        n: refuted_by(v, s_unique, s_index, (0.0, 1.0))[0] for n, v in tanh_values.items()
    }

    # ---- lanes --------------------------------------------------------------- #
    def lanes(sigmoid_name, tanh_name):
        sc, sh = sigmoid_values[sigmoid_name]
        tc, th = tanh_values[tanh_name]
        cell = int(np.count_nonzero(f16(sc * tc) == native_cell))
        hidden = int(np.count_nonzero(f16(sh * th) == native_hidden))
        return cell, hidden

    def row(sigmoid_name, tanh_name):
        cell, hidden = lanes(sigmoid_name, tanh_name)
        return {
            "sigmoid": sigmoid_name,
            "tanh": tanh_name,
            "cell_lanes": cell,
            "hidden_lanes": hidden,
            "joint_lanes": cell + hidden,
            "sigmoid_pins": pin_score["sigmoid"][sigmoid_name],
            "tanh_pins": pin_score["tanh"][tanh_name],
            "sigmoid_refuting_arguments": sigmoid_refuted[sigmoid_name],
            "tanh_refuting_arguments": tanh_refuted[tanh_name],
        }

    best_sigmoid = min(sigmoid_refuted, key=lambda n: (sigmoid_refuted[n], n))
    best_tanh = min(tanh_refuted, key=lambda n: (tanh_refuted[n], n))
    named = [
        ("S[exact16|1616|div|plain]", "T[correctly_rounded]"),
        ("S[exact16|1616|newton12|plain]", "T[correctly_rounded]"),
        ("S[exact16|1632|newton12|plain]", "T[correctly_rounded]"),
        ("S[table16|1616|div|plain]", "T[correctly_rounded]"),
        ("S[exact16.sq3|1616|div|plain]", "T[correctly_rounded]"),
        ("S[exact16|1616|div|halves]", "T[correctly_rounded]"),
        ("S[correctly_rounded]", "T[correctly_rounded]"),
        ("S[lut.lerp16]", "T[lut.lerp16]"),
        ("S[exact16|1616|div|plain]", "T[lut.lerp16]"),
        ("S[exact16|1616|div|plain]", "tanh:mulrcppos:321616"),
        (best_sigmoid, best_tanh),
    ]
    named = [
        pair
        for pair in dict.fromkeys(named)
        if pair[0] in sigmoid_values and pair[1] in tanh_values
    ]
    sweep = [lanes(s, t) + (s, t) for s in sigmoid_values for t in tanh_values]
    result["candidates"] = {
        "sigmoid_contracts": len(sigmoids),
        "tanh_contracts": len(tanhs),
        "note": (
            "refuting_arguments is the count of the OTHER activation's arguments with "
            "no feasible fp16 value, assuming only that the other activation is a "
            "deterministic fp16 function and the lane is one fp16 multiply. Zero is "
            "necessary, not sufficient."
        ),
        "sigmoid_argument_count": int(s_unique.size),
        "tanh_argument_count": int(t_unique.size),
        "named": [row(s, t) for s, t in named],
        "sigmoid_least_refuted": sorted(sigmoid_refuted.items(), key=lambda kv: kv[1])[:8],
        "tanh_least_refuted": sorted(tanh_refuted.items(), key=lambda kv: kv[1])[:8],
        "sigmoid_refuting_minimum": min(sigmoid_refuted.values()),
        "sigmoid_contracts_at_minimum": sum(
            v == min(sigmoid_refuted.values()) for v in sigmoid_refuted.values()
        ),
        "tanh_refuting_minimum": min(tanh_refuted.values()),
        "tanh_contracts_at_minimum": sum(
            v == min(tanh_refuted.values()) for v in tanh_refuted.values()
        ),
        "combinations": len(sweep),
        "best_joint_lanes": row(*max(sweep, key=lambda e: e[0] + e[1])[2:]),
        "bit_exact_on_1280": [
            {"sigmoid": s, "tanh": t}
            for cell, hidden, s, t in sweep
            if cell + hidden == 2 * HIDDEN
        ],
    }

    # ---- conditional measurement of the native tanh ------------------------- #
    # Fixing the leading sigmoid contract turns nearly every lane into a solved
    # equation for one native tanh value, which is 949 measurements where the
    # partner-free method of the previous receipt pinned 8.
    lead = "S[exact16|1616|div|plain]"
    empty, lo, hi = refuted_by(sigmoid_values[lead], t_unique, t_index, (-1.0, 1.0))
    unique_mask = lo == hi
    measured_arguments = t_unique[unique_mask]
    measured_values = from_ordinal(lo[unique_mask])
    exact = tanh_exact(measured_arguments)
    step = base.ulp16(f16(exact))
    step = np.where(step == 0.0, 1.0, step)
    magnitude_bias = (measured_values - exact) / step * np.sign(measured_arguments)
    signed = np.round((measured_values - f16(exact)) / step)
    offsets = collections.Counter(int(v) for v in signed)
    split = {
        str(int(o)): {
            "negative_arguments": int(
                np.count_nonzero((signed == o) & (measured_arguments < 0.0))
            ),
            "positive_arguments": int(
                np.count_nonzero((signed == o) & (measured_arguments > 0.0))
            ),
        }
        for o in sorted(set(signed.tolist()))
    }
    grid = 0.0625
    binned = []
    absolute = np.abs(measured_arguments)
    for k in range(int(1.0 / grid)):
        sel = (absolute >= k * grid) & (absolute < (k + 1) * grid)
        if sel.sum() < 8:
            continue
        binned.append(
            {
                "abs_argument_low": round(k * grid, 6),
                "count": int(sel.sum()),
                "mean_magnitude_bias_ulps": float(magnitude_bias[sel].mean()),
                "standard_error": float(
                    magnitude_bias[sel].std(ddof=1) / np.sqrt(sel.sum())
                ),
            }
        )
    agreement = {
        n: int(np.count_nonzero(f16(f(measured_arguments)) == measured_values))
        for n, f in tanhs.items()
    }
    result["conditional_native_tanh"] = {
        "note": (
            "Native tanh values solved lane by lane with the leading sigmoid contract "
            "held fixed. Conditional on that contract: the level of the bias is "
            "degenerate with the sigmoid's own mean bias, the shape is not."
        ),
        "conditioning_sigmoid": lead,
        "arguments": int(t_unique.size),
        "arguments_with_no_feasible_value": empty,
        "arguments_uniquely_determined": int(unique_mask.sum()),
        "signed_ulp_offset_from_correct_rounding": dict(sorted(offsets.items())),
        "signed_ulp_offset_by_argument_sign": split,
        "mean_magnitude_bias_ulps": float(magnitude_bias.mean()),
        "standard_error": float(magnitude_bias.std(ddof=1) / np.sqrt(magnitude_bias.size)),
        "binned_magnitude_bias": binned,
        "best_agreement": max(agreement.values()),
        "correctly_rounded_agreement": agreement["T[correctly_rounded]"],
        "contracts_reaching_best": sorted(
            n for n, v in agreement.items() if v == max(agreement.values())
        )[:6],
    }

    # ---- structural tests on the measured values ---------------------------- #
    # tanh = 1 - 2*r with an fp16 reciprocal r quantizes the result to every other
    # fp16 number where |tanh| < 0.5, and tanh = 2*s - 1 with fp16 s to every fourth.
    def is_fp16(values):
        return np.array([float(np.float16(v)) == v for v in values])

    window = (np.abs(measured_values) >= 0.25) & (np.abs(measured_values) < 0.5)
    result["structural_quantization"] = {
        "note": (
            "Tested where |tanh| is in [0.25,0.5), the binade in which each form's "
            "output grid is coarser than fp16. A value off the grid is a counterexample."
        ),
        "measured_values_in_window": int(window.sum()),
        "consistent_with_one_minus_two_reciprocal": int(
            is_fp16((1.0 - np.abs(measured_values[window])) / 2.0).sum()
        ),
        "consistent_with_two_sigmoid_minus_one": int(
            is_fp16((1.0 + np.abs(measured_values[window])) / 4.0).sum()
        ),
        "control_value_is_fp16": int(is_fp16(np.abs(measured_values[window])).sum()),
    }

    # ---- which argument feeds the second tanh ------------------------------- #
    sc, sh = sigmoid_values[lead]
    internal_cell = sc * f16(tanh_exact(b_c))
    result["second_tanh_argument"] = {
        "note": "the hidden lane's tanh argument: the captured fp16 cell, or an internal one",
        "variants": {
            name: int(np.count_nonzero(f16(sh * f16(tanh_exact(argument))) == native_hidden))
            for name, argument in (
                ("native_fp16_cell", native_cell),
                ("unrounded_internal_cell", internal_cell),
                ("fp32_internal_cell", f32(internal_cell)),
                ("fp16_internal_cell", f16(internal_cell)),
            )
        },
        "lanes": HIDDEN,
    }

    # ---- all 15 transitions -------------------------------------------------- #
    def gate(x, h, w_ih, w_hh, b):
        """The fp32-class reduction the previous study measured as least-bad."""
        return f16(
            f16(
                f16(np.asarray(x, np.float32) @ np.asarray(w_ih, np.float32).T)
                + f16(np.asarray(h, np.float32) @ np.asarray(w_hh, np.float32).T)
            )
            + b
        )

    def run(row_, sigmoid_fn, tanh_fn):
        seq = weights["embedding"][int(row_["decoder_input_ids"].reshape(-1)[0])].astype(
            np.float64
        )
        hidden_in = row_["decoder_hidden"][:, 0].astype(np.float64)
        cell_in = row_["decoder_cell"][:, 0].astype(np.float64)
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
                f16(sigmoid_fn(g_f) * cell_in[layer]) + f16(sigmoid_fn(g_i) * tanh_fn(g_c))
            )
            hidden = f16(sigmoid_fn(g_o) * tanh_fn(cell))
            cells.append(cell)
            hiddens.append(hidden)
            seq = hidden
        return np.stack(cells), np.stack(hiddens)

    def score_all(sigmoid_name, tanh_name):
        sigmoid_fn = lambda x: f16(sigmoids[sigmoid_name](x))  # noqa: E731
        tanh_fn = lambda x: f16(tanhs[tanh_name](x))  # noqa: E731
        cell_ok = hidden_ok = nonfinite = 0
        cell_max = hidden_max = 0.0
        for row_ in traces:
            cells, hiddens = run(row_, sigmoid_fn, tanh_fn)
            nc = row_["decoder_next_cell"][:, 0].astype(np.float64)
            nh = row_["decoder_next_hidden"][:, 0].astype(np.float64)
            cell_ok += int(np.count_nonzero(cells == nc))
            hidden_ok += int(np.count_nonzero(hiddens == nh))
            nonfinite += int(np.count_nonzero(~np.isfinite(cells)))
            nonfinite += int(np.count_nonzero(~np.isfinite(hiddens)))
            # np.maximum propagates a nan, where the builtin max silently drops it.
            cell_max = float(np.maximum(cell_max, np.abs(cells - nc).max()))
            hidden_max = float(np.maximum(hidden_max, np.abs(hiddens - nh).max()))
        return {
            "sigmoid": sigmoid_name,
            "tanh": tanh_name,
            "next_cell_lanes": cell_ok,
            "next_hidden_lanes": hidden_ok,
            "lanes": 2 * HIDDEN * len(traces),
            "nonfinite_lanes": nonfinite,
            "next_cell_max_abs": cell_max,
            "next_hidden_max_abs": hidden_max,
        }

    result["all_transitions"] = {
        "note": (
            "Full two-layer LSTM from native-injected state with the fp32-class gate "
            "reduction. The reduction is still an open hole, so these counts are not a "
            "clean activation measurement."
        ),
        "variants": [score_all(s, t) for s, t in dict.fromkeys(named)],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")

    # ---- summary ------------------------------------------------------------- #
    for operation, report in result["hardware_unary_tables"].items():
        print(
            f"table {operation:8s} exact-knot={report['knots_are_exact_fp16_of_the_function']} "
            f"differing={report['knots_differing']} "
            f"header={report.get('header_clamp_low_high_out_below_out_above')}"
        )
    print(
        f"candidates: {len(sigmoids)} sigmoid x {len(tanhs)} tanh; "
        f"{s_unique.size} sigmoid arguments, {t_unique.size} tanh arguments"
    )
    for entry in result["candidates"]["named"]:
        print(
            f"  {entry['joint_lanes']:4d}/1280  cell {entry['cell_lanes']:3d} "
            f"hidden {entry['hidden_lanes']:3d}  pins {entry['sigmoid_pins']}/11+"
            f"{entry['tanh_pins']}/8  refuted s={entry['sigmoid_refuting_arguments']} "
            f"t={entry['tanh_refuting_arguments']}  {entry['sigmoid']} + {entry['tanh']}"
        )
    print(f"bit-exact on 1280: {len(result['candidates']['bit_exact_on_1280'])}")
    print(f"least-refuted sigmoid: {result['candidates']['sigmoid_least_refuted'][:3]}")
    print(f"least-refuted tanh:    {result['candidates']['tanh_least_refuted'][:3]}")
    measured = result["conditional_native_tanh"]
    print(
        f"conditional tanh: {measured['arguments_uniquely_determined']} measured, "
        f"{measured['arguments_with_no_feasible_value']} refuting, mean magnitude bias "
        f"{measured['mean_magnitude_bias_ulps']:+.3f} +- {measured['standard_error']:.3f} ulps"
    )
    print(f"  offsets {measured['signed_ulp_offset_from_correct_rounding']}")
    print(
        f"  best agreement {measured['best_agreement']}/"
        f"{measured['arguments_uniquely_determined']}, correct rounding "
        f"{measured['correctly_rounded_agreement']}"
    )
    print(f"structural: {json.dumps(result['structural_quantization'])}")
    print(f"second tanh argument: {json.dumps(result['second_tanh_argument']['variants'])}")
    for entry in result["all_transitions"]["variants"]:
        print(
            f"  all-15: cell {entry['next_cell_lanes']:5d}/{entry['lanes']} hidden "
            f"{entry['next_hidden_lanes']:5d}  {entry['sigmoid']} + {entry['tanh']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

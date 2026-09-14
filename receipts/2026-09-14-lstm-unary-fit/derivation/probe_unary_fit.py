#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Search fp16 unary contracts for the TDT decoder LSTM reduction-free lanes.

The emission-101 receipt (2026-09-14-lstm-e101) named the divergent step as the
unary evaluation inside the ``lstm`` op.  This probe searches a small space of
fp16 ``sigmoid``/``tanh`` contracts -- correctly rounded, staged fp16 chains,
macOS CPU (= correctly rounded, measured), capture pins, and piecewise
combinations -- for the pair that maximizes bit-exact ``next_cell`` /
``next_hidden`` lanes on the reduction-free step (transition 0, layer 0, where
the gates equal the fp16 bias constants bit for bit).

Every native value comes from the authenticated capture; every candidate is
NumPy at fp64 with explicit fp16 rounding stages, so each candidate names one
exact arithmetic contract.  Host-only: no MLX, no GPU, no ANE, no lock.
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
}

# Native activation values pinned out of the capture by the arc-consistency
# solve in receipts/2026-09-14-decoder-activations.md ("Measured native
# values").  Arguments are fp16 bias values; values are fp16.  Decimals are the
# receipt's printouts, snapped to the exact fp16 grid below and verified
# against the capture biases.
SIGMOID_PINS = [
    (-0.288086, 0.428466797),
    (+0.090027, 0.522460938),
    (+0.154419, 0.538574219),
    (+0.154663, 0.538574219),
    (+0.261230, 0.564941406),
    (+0.314453, 0.577636719),
    (+0.438965, 0.607910156),
    (+0.529785, 0.629394531),
    (+0.566406, 0.638183594),
    (+0.566895, 0.638183594),
    (+0.567871, 0.638183594),
]
TANH_PINS = [
    (-1.099609, -0.800292969),
    (-0.681641, -0.592773438),
    (-0.560547, -0.508300781),
    (-0.113953, -0.113464355),
    (+0.282715, +0.275390625),
    (+0.319092, +0.308837891),
    (+0.458984, +0.429199219),
    (+0.622070, +0.552734375),
]


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


def snap(pairs):
    return [
        (float(np.float16(a)), float(np.float16(v)))
        for a, v in pairs
    ]


def load(package: Path, capture: Path):
    from tools.coreml.pinned_component import load_pinned_component

    component = load_pinned_component(package, "decoder")
    weights = {key: component.constant(name) for key, name in CONSTANTS.items()}
    index = json.loads((capture / "tdt_tensors.json").read_text())
    rows = []
    for trace in index["traces"]:
        paths = trace["tensor_paths"]
        if "decoder_next_cell" not in paths:
            continue
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


# --- candidate unaries -------------------------------------------------------
# Each is (name, fn): fn maps fp64 arguments to fp64 values on the fp16 grid.


def sig_chain(round_neg: bool, round_exp: bool, round_add: bool):
    def fn(x):
        t = -np.asarray(x, dtype=np.float64)
        if round_neg:
            t = f16(t)
        with np.errstate(over="ignore"):
            e = np.exp(t)
        if round_exp:
            e = f16(e)
        s = 1.0 + e
        if round_add:
            s = f16(s)
        return f16(1.0 / s)

    return fn


def sig_exp2(log2e_width: str):
    l2e = {
        "16": float(np.float16(np.log2(np.e))),
        "32": float(np.float32(np.log2(np.e))),
        "64": float(np.log2(np.e)),
    }[log2e_width]

    def fn(x):
        with np.errstate(over="ignore"):
            return f16(1.0 / (1.0 + np.exp2(-np.asarray(x, dtype=np.float64) * l2e)))

    return fn


def sig_tanh_half(round_stages: bool):
    def fn(x):
        h = np.asarray(x, dtype=np.float64) * 0.5
        t = tanh64(h)
        if round_stages:
            return f16(f16(f16(0.5) * f16(t)) + f16(0.5))
        return f16(0.5 * t + 0.5)

    return fn


def tanh_chain(round_2x: bool, round_e: bool, round_sub: bool):
    def fn(x):
        t = 2.0 * np.asarray(x, dtype=np.float64)
        if round_2x:
            t = f16(t)
        e = np.exp(t)
        if round_e:
            e = f16(e)
        n = e - 1.0
        d = e + 1.0
        if round_sub:
            n = f16(n)
            d = f16(d)
        return f16(n / d)

    return fn


def tanh_one_minus(round_2x: bool, round_e: bool, width: str):
    round_out = f16 if width == "16" else (f32 if width == "32" else (lambda v: np.asarray(v, dtype=np.float64)))

    def fn(x):
        t = 2.0 * np.asarray(x, dtype=np.float64)
        if round_2x:
            t = f16(t)
        e = np.exp(t)
        if round_e:
            e = f16(e)
        return f16(1.0 - round_out(2.0 / (e + 1.0)))

    return fn


def tanh_mulrcp(inter_width: str):
    """Magnitude/sign variant: (e-1) * recip(e+1) on |x|, sign applied last."""
    rnd = f16 if inter_width == "16" else f32

    def fn(x):
        ax = np.abs(np.asarray(x, dtype=np.float64))
        e = rnd(np.exp(rnd(2.0 * ax)))
        n = rnd(e - 1.0)
        d = rnd(e + 1.0)
        r = rnd(1.0 / d)
        return f16(np.sign(x) * rnd(n * r))

    return fn


def tanh_scaled(k: float):
    def fn(x):
        return f16(tanh64(x) * (1.0 + k))

    return fn


def tanh_from_sigmoid(sigmoid, mode: str, round_2x: bool):
    def fn(x):
        t = 2.0 * np.asarray(x, dtype=np.float64)
        if round_2x:
            t = f16(t)
        if mode == "2s-1":
            return f16(f16(2.0 * sigmoid(t)) - 1.0)
        return f16(sigmoid(t) - sigmoid(-t))

    return fn


def with_pins(fn, pins):
    table = [(np.float16(a), np.float16(v)) for a, v in pins]

    def wrapped(x):
        y = fn(x)
        x16 = np.asarray(x, dtype=F16)
        for a, v in table:
            y = np.where(x16 == a, np.float64(v), y)
        return y

    return wrapped


def build_sigmoids():
    out = [("cr", lambda x: f16(sigmoid64(x)))]
    for rneg, rexp, radd in itertools.product((False, True), repeat=3):
        name = "chain" + "".join(
            "1" if r else "0" for r in (rneg, rexp, radd, True)
        )
        out.append((name, sig_chain(rneg, rexp, radd)))
    for w in ("16", "32", "64"):
        out.append((f"exp2-l2e{w}", sig_exp2(w)))
    for staged in (True, False):
        out.append((f"tanhhalf{'-staged' if staged else ''}", sig_tanh_half(staged)))
    return out


def build_tanhs(sigmoids):
    out = [("cr", lambda x: f16(tanh64(x)))]
    for r2x, re, rsub in itertools.product((False, True), repeat=3):
        name = "chain" + "".join("1" if r else "0" for r in (r2x, re, rsub))
        out.append((name, tanh_chain(r2x, re, rsub)))
    for r2x, re, w in itertools.product((False, True), ("16", "32"), ("16", "32", "64")):
        out.append((f"1m2o-{r2x:d}{re}-{w}", tanh_one_minus(r2x, re, w)))
    for w in ("16", "32"):
        out.append((f"mulrcp{w}", tanh_mulrcp(w)))
    for k in (2.5e-5, 5e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4):
        out.append((f"cr*(1+{k:g})", tanh_scaled(k)))
    for sname, sfn in sigmoids:
        for mode in ("2s-1", "sub"):
            for r2x in (True, False):
                out.append(
                    (
                        f"{mode}({sname}){'-r2x' if r2x else ''}",
                        tanh_from_sigmoid(sfn, mode, r2x),
                    )
                )
    return out


# --- contracts ---------------------------------------------------------------


def score(candidate, native):
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(native, dtype=np.float64)
    delta = cand - ref
    return int(np.count_nonzero(delta == 0)), float(np.abs(delta).max())


def pin_hits(fn, pins):
    args = np.array([a for a, _ in pins], dtype=np.float64)
    want = [np.float16(v) for _, v in pins]
    got = np.asarray(fn(args), dtype=F16)
    return int(sum(int(g == w) for g, w in zip(got, want)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--live-gates", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.overlay))
    weights, rows = load(args.package, args.capture_dir)

    first = rows[0]
    assert not first["decoder_hidden"].any() and not first["decoder_cell"].any()
    bias0 = weights["l0_bias"].astype(np.float64)
    b_i, b_f, b_o, b_g = (part.reshape(-1) for part in np.split(bias0, 4))
    native_cell0 = first["decoder_next_cell"][0, 0].reshape(-1).astype(np.float64)
    native_hidden0 = first["decoder_next_hidden"][0, 0].reshape(-1).astype(np.float64)

    sig_pins = snap(SIGMOID_PINS)
    tan_pins = snap(TANH_PINS)

    # Pin provenance: every pinned argument must appear in the capture biases
    # (sigmoid: input/output gate slots; tanh: cell gate slot or native cell).
    sig_args = np.concatenate([b_i, b_o]).astype(F16)
    tan_args = np.concatenate([b_g, native_cell0.astype(F16)]).astype(F16)
    missing_sig = [a for a, _ in sig_pins if np.float16(a) not in sig_args]
    missing_tan = [a for a, _ in tan_pins if np.float16(a) not in tan_args]
    if missing_sig or missing_tan:
        raise SystemExit(
            f"pinned arguments absent from capture: sigmoid {missing_sig} tanh {missing_tan}"
        )

    sigmoids = build_sigmoids()
    tanhs = build_tanhs(sigmoids)
    sigmoids += [
        (f"{n}+pins", with_pins(fn, sig_pins)) for n, fn in list(sigmoids)
    ]
    tanhs += [(f"{n}+pins", with_pins(fn, tan_pins)) for n, fn in list(tanhs)]

    sig_pin_counts = {n: pin_hits(fn, sig_pins) for n, fn in sigmoids}
    tan_pin_counts = {n: pin_hits(fn, tan_pins) for n, fn in tanhs}

    b_i16, b_g16, b_o16 = (np.asarray(v, dtype=F16) for v in (b_i, b_g, b_o))
    results = []
    for (sname, sfn), (tname, tfn) in itertools.product(sigmoids, tanhs):
        s_i, s_o = sfn(b_i), sfn(b_o)
        t_g = tfn(b_g)
        cell = f16(s_i * t_g)
        cell_exact, cell_max = score(cell, native_cell0)
        hidden = f16(s_o * tfn(cell))
        hidden_exact, hidden_max = score(hidden, native_hidden0)
        hidden_id = f16(s_o * tfn(native_cell0))
        hidden_id_exact, _ = score(hidden_id, native_hidden0)
        results.append(
            {
                "sigmoid": sname,
                "tanh": tname,
                "cell": cell_exact,
                "hidden": hidden_exact,
                "hidden_identity": hidden_id_exact,
                "joint": cell_exact + hidden_exact,
                "cell_max_abs": cell_max,
                "hidden_max_abs": hidden_max,
            }
        )
    results.sort(key=lambda r: (-r["joint"], -r["cell"], r["cell_max_abs"]))
    perfect = [r for r in results if r["cell"] == HIDDEN and r["hidden"] == HIDDEN]

    # Recurrence generalization for the top pairs plus named references.
    def gates_shipped(x, h, ih, hh, b):
        dot = lambda v, w: np.asarray(v, dtype=np.float32) @ np.asarray(w, dtype=np.float32).T
        return f16(f16(f16(dot(x, ih)) + f16(dot(h, hh))) + b)

    recurrence = {}
    chosen = []
    for r in results[:8]:
        key = (r["sigmoid"], r["tanh"])
        if key not in chosen:
            chosen.append(key)
    for key in (("cr", "cr"), ("chain1111", "cr")):
        if key not in chosen:
            chosen.append(key)
    smap, tmap = dict(sigmoids), dict(tanhs)
    for sname, tname in chosen:
        sfn, tfn = smap[sname], tmap[tname]
        cell_hits = hidden_hits = lanes = 0
        for row in rows:
            x = weights["embedding"][
                int(row["decoder_input_ids"].reshape(-1)[0])
            ].astype(np.float64)
            hidden_in = row["decoder_hidden"][:, 0].astype(np.float64)
            cell_in = row["decoder_cell"][:, 0].astype(np.float64)
            cells, hiddens = [], []
            for layer in (0, 1):
                gates = gates_shipped(
                    x if layer == 0 else hiddens[-1],
                    hidden_in[layer],
                    weights[f"l{layer}_ih"].astype(np.float64),
                    weights[f"l{layer}_hh"].astype(np.float64),
                    weights[f"l{layer}_bias"].astype(np.float64),
                )
                g_i, g_f, g_o, g_g = (p.reshape(-1) for p in np.split(gates, 4))
                cell = f16(
                    f16(sfn(g_f) * cell_in[layer]) + f16(sfn(g_i) * tfn(g_g))
                )
                hidden = f16(sfn(g_o) * tfn(cell))
                cells.append(cell)
                hiddens.append(hidden)
            cell_hits += score(np.stack(cells), row["decoder_next_cell"][:, 0])[0]
            hidden_hits += score(np.stack(hiddens), row["decoder_next_hidden"][:, 0])[0]
            lanes += 2 * HIDDEN
        recurrence[f"{sname}|{tname}"] = {
            "cell": cell_hits,
            "hidden": hidden_hits,
            "lanes": lanes,
        }

    # Live footprint: lanes changed per call-layer versus the shipped pair on
    # the recorded GPU gate args of the emission-101 shipped arm.
    footprint = {}
    if args.live_gates and Path(args.live_gates).exists():
        gates = np.load(args.live_gates)
        base_s, base_t = smap["cr"], tmap["cr"]
        for sname, tname in chosen:
            sfn, tfn = smap[sname], tmap[tname]
            d_cell, d_hidden, n = [], [], 0
            for key in ("l0", "l1"):
                for row in gates[key]:
                    g_i, g_f, g_o, g_g = (p.reshape(-1) for p in np.split(row.astype(np.float64), 4))
                    cell_in = np.zeros(HIDDEN)  # footprint is unary-driven; entry state cancels
                    cell0 = f16(f16(base_s(g_f) * cell_in) + f16(base_s(g_i) * base_t(g_g)))
                    hid0 = f16(base_s(g_o) * base_t(cell0))
                    cell1 = f16(f16(sfn(g_f) * cell_in) + f16(sfn(g_i) * tfn(g_g)))
                    hid1 = f16(sfn(g_o) * tfn(cell1))
                    d_cell.append(int(np.count_nonzero(cell1 != cell0)))
                    d_hidden.append(int(np.count_nonzero(hid1 != hid0)))
                    n += 1
            footprint[f"{sname}|{tname}"] = {
                "cell_changed_mean": float(np.mean(d_cell)),
                "hidden_changed_mean": float(np.mean(d_hidden)),
                "cell_changed_max": int(np.max(d_cell)),
                "hidden_changed_max": int(np.max(d_hidden)),
                "gate_rows": n,
            }

    result = {
        "schema": "mlx-omarchy.decoder-lstm-unary-fit/1",
        "named_hole": "parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml",
        "execution": "host-only NumPy; no MLX, no GPU, no ANE, no device execution",
        "capture_sha256": hashlib.sha256(
            (args.capture_dir / "tdt_tensors.json").read_bytes()
        ).hexdigest(),
        "candidates": {"sigmoids": len(sigmoids), "tanhs": len(tanhs), "pairs": len(results)},
        "pin_provenance": {"sigmoid_pins": len(sig_pins), "tanh_pins": len(tan_pins)},
        "sigmoid_pin_counts": sig_pin_counts,
        "tanh_pin_counts": tan_pin_counts,
        "leaderboard": results[:24],
        "perfect_pairs": perfect,
        "recurrence_15_transitions": recurrence,
        "live_footprint_vs_shipped": footprint,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {args.output}")

    print("pairs scored:", len(results))
    print("perfect (640/640 both):", len(perfect))
    for r in results[:6]:
        print(
            f"  {r['sigmoid']:>24} | {r['tanh']:>24} | cell {r['cell']:3d} hidden {r['hidden']:3d}"
            f" hid_id {r['hidden_identity']:3d} | pins {sig_pin_counts[r['sigmoid']]}/11 {tan_pin_counts[r['tanh']]}/8"
        )
    print("recurrence:", json.dumps(recurrence, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Score the jwm1 H13 unary ``sigmoid``/``tanh`` device outputs against the native
Core ML decoder capture, and identify the hardware interpolator from the samples.

Named hole: ``parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml``.

Host-only scoring of device outputs already captured in ``device/``. The device run
itself is ``run_remote.sh`` on jwm1; this script never opens ``accel0``.

Layout of each 1536-lane fixture (``device/layout.json``): the 1280 decoder
arguments of transition 0 (``sigmoid``: the pinned layer-0 ``b_i`` then ``b_o``;
``tanh``: ``b_c`` then the captured ``next_cell``), then the 33 table knots, the
32 segment midpoints, 6 clamp-boundary probes, and a 185-lane sweep through the
argument range on a fine fp16 grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F16 = np.dtype("<f2")
GRID = {"sigmoid": (-8.0, 0.5), "tanh": (0.0, 0.125)}
TABLE_OFFSET = 4736  # the 128-byte fp16 block inside program-0.anec


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(anec: Path):
    words = np.frombuffer(anec.read_bytes()[TABLE_OFFSET : TABLE_OFFSET + 128], F16)
    words = words.astype(np.float64)
    return {
        "prefix_128_sha256": hashlib.sha256(anec.read_bytes()[TABLE_OFFSET : TABLE_OFFSET + 128]).hexdigest(),
        "header": words[:4],
        "knots": words[4:37],
    }


def raz16(value):
    """Round to fp16 with ties away from zero (round-half-up in magnitude)."""
    y = np.asarray(value, np.float64)
    y16 = y.astype(F16).astype(np.float64)
    # nearest-even may have chosen the lower magnitude on an exact tie; detect ties
    up = np.nextafter(y16.astype(F16), np.float16(np.inf)).astype(np.float64)
    dn = np.nextafter(y16.astype(F16), np.float16(-np.inf)).astype(np.float64)
    tie_up = np.isclose(y, (y16 + up) / 2.0, rtol=0, atol=0) & (np.abs(up) > np.abs(y16))
    tie_dn = np.isclose(y, (y16 + dn) / 2.0, rtol=0, atol=0) & (np.abs(dn) > np.abs(y16))
    return np.where(tie_up, up, np.where(tie_dn, dn, y16))


def h13_lut(table, op, x):
    """The identified H13 interpolator.

    Index by magnitude and anchor at the knot nearer zero; ``s`` is the fraction
    from that knot outward.  ``p = raz16(s * (a1 - a0))``, ``y = raz16(a0 + p)``.
    The farthest knot is evaluated as ``s = 1`` of the final interval.  Header
    clamps apply at and beyond the clamp words.
    """
    start, step = GRID[op]
    kn, (lo, hi, below, above) = table["knots"], table["header"]
    x = np.asarray(x, np.float64)
    if op == "tanh":
        u, sign = np.abs(x), np.sign(x)
        i = np.minimum(np.floor(u / step).astype(int), 31)
        s = u / step - i
        a0, a1 = kn[i], kn[i + 1]
        out_lo, out_hi = u <= lo, u >= hi
    else:
        idx = (x - start) / step
        pos = x >= 0
        i_lo = np.clip(np.floor(idx).astype(int), 0, 31)
        i_hi = np.clip(np.ceil(idx).astype(int), 1, 32)
        a0 = np.where(pos, kn[i_lo], kn[i_hi])
        a1 = np.where(pos, kn[np.minimum(i_lo + 1, 32)], kn[i_hi - 1])
        s = np.where(pos, idx - i_lo, i_hi - idx)
        sign = np.ones_like(x)
        out_lo, out_hi = x <= lo, x >= hi
    y = raz16(a0 + raz16(s * (a1 - a0)))
    y = np.where(out_lo, below, y)
    y = np.where(out_hi, above, y)
    return (sign * y).astype(F16).astype(np.float64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--stages", type=Path, default=HERE.parent / "2026-09-14-decoder-activation-stages")
    parser.add_argument("--pins", type=Path, default=HERE.parent / "2026-09-14-decoder-activations")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    for p in (args.overlay, args.pins, args.stages):
        sys.path.insert(0, str(p))
    import probe as base  # capture loader, fp16 helpers
    import stages  # partner-free feasibility solver

    f16 = base.f16
    component, weights, traces = base.load(args.package, args.capture_dir)
    first = traces[0]
    b_i, _b_f, b_o, b_c = np.split(weights["l0_bias"].astype(np.float64), 4)
    native_cell = first["decoder_next_cell"][0, 0].astype(np.float64)
    native_hidden = first["decoder_next_hidden"][0, 0].astype(np.float64)
    natives = np.concatenate([native_cell, native_hidden])
    sig_args = np.concatenate([b_i, b_o])
    tanh_args = np.concatenate([b_c, native_cell])

    layout = json.loads((HERE / "device/layout.json").read_text())
    X, Y, table, dev = {}, {}, {}, {}
    for op in ("sigmoid", "tanh"):
        X[op] = np.concatenate([np.fromfile(HERE / f"device/x_{op}_{p}.bin", F16) for p in range(3)]).astype(np.float64)
        Y[op] = np.concatenate([np.fromfile(HERE / f"device/y_out_{op}_{p}.bin", F16) for p in range(3)]).astype(np.float64)
        table[op] = read_table(HERE / f"bundle-{op}/program-0.anec")
        lo, hi = layout[op]["args"]
        expect = sig_args if op == "sigmoid" else tanh_args
        assert np.array_equal(X[op][lo:hi], expect), f"{op} fixture does not carry the decoder arguments"
        d = {}
        for x, y in zip(X[op], Y[op]):
            assert d.setdefault(x, y) == y, "device returned two values for one argument"
        dev[op] = d

    result: dict = {
        "schema": "mlx-omarchy.parakeet-decoder-activations-on-ane/1",
        "named_hole": "parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml",
        "capture_sha256": {n: sha256(args.capture_dir / n) for n in ("tdt_tensors.json", "manifest.sha256")},
        "capture_receipt_sha256": sha256(args.capture_dir.parent / "receipt.json"),
        "capture_compute_units": {
            "requested": json.loads((args.capture_dir / "environment.json").read_text())["compute_units"],
            "receipt_backend_claim": json.loads((args.capture_dir.parent / "receipt.json").read_text())["backend_claim"],
        },
        "device_files_sha256": {p.name: sha256(p) for p in sorted((HERE / "device").glob("*.bin"))},
        "bundles": {
            op: {
                "manifest_sha256": sha256(HERE / f"bundle-{op}/manifest.json"),
                "program_sha256": sha256(HERE / f"bundle-{op}/program-0.anec"),
                "table_prefix_128_sha256": table[op]["prefix_128_sha256"],
                "header": table[op]["header"].tolist(),
            }
            for op in ("sigmoid", "tanh")
        },
        "transition": int(first["index"]),
    }

    # ---- pins ---------------------------------------------------------------- #
    pins = json.loads((args.pins / "result.json").read_text())["measured_native_values"]
    pin_report = {}
    for op, key in (("sigmoid", "monotone_sigmoid"), ("tanh", "monotone_tanh")):
        rows = []
        for r in pins[key]["pinned"]:
            a, v = r["argument"], r["native_value"]
            rows.append({"argument": a, "native": v, "device": float(dev[op][a]), "device_matches": bool(dev[op][a] == v)})
        pin_report[op] = {"reproduced": int(sum(r["device_matches"] for r in rows)), "of": len(rows), "rows": rows}
    result["pins"] = pin_report

    # ---- lanes --------------------------------------------------------------- #
    ds_i = np.array([dev["sigmoid"][a] for a in b_i])
    ds_o = np.array([dev["sigmoid"][a] for a in b_o])
    dt_c = np.array([dev["tanh"][a] for a in b_c])
    dt_h = np.array([dev["tanh"][a] for a in native_cell])
    chain = lambda x: f16(1.0 / f16(1.0 + f16(np.exp(-x))))  # noqa: E731
    cr_t_c, cr_t_h = f16(np.tanh(b_c)), f16(np.tanh(native_cell))
    lanes = lambda sc, tc, sh, th: (  # noqa: E731
        int(np.count_nonzero(f16(sc * tc) == native_cell)),
        int(np.count_nonzero(f16(sh * th) == native_hidden)),
    )
    result["lanes_of_1280"] = {
        "device_sigmoid_device_tanh": lanes(ds_i, dt_c, ds_o, dt_h),
        "device_sigmoid_correctly_rounded_tanh": lanes(ds_i, cr_t_c, ds_o, cr_t_h),
        "fp16_chain_sigmoid_device_tanh": lanes(chain(b_i), dt_c, chain(b_o), dt_h),
        "fp16_chain_sigmoid_correctly_rounded_tanh_reference": lanes(chain(b_i), cr_t_c, chain(b_o), cr_t_h),
        "note": "pairs are (next_cell lanes, next_hidden lanes), each of 640",
    }

    # ---- partner-free refutation ---------------------------------------------- #
    t_unique, t_index = np.unique(tanh_args, return_inverse=True)
    s_unique, s_index = np.unique(sig_args, return_inverse=True)
    lo, hi = stages.feasible_ranges(np.concatenate([ds_i, ds_o]), t_index, t_unique.size, natives, (-1.0, 1.0))
    lo2, hi2 = stages.feasible_ranges(np.concatenate([dt_c, dt_h]), s_index, s_unique.size, natives, (0.0, 1.0))
    result["partner_free_refutation"] = {
        "device_sigmoid_refuting_tanh_arguments": [int(np.count_nonzero(lo > hi)), int(t_unique.size)],
        "device_tanh_refuting_sigmoid_arguments": [int(np.count_nonzero(lo2 > hi2)), int(s_unique.size)],
    }

    # ---- 949 conditional native tanh values ------------------------------------ #
    lo_c, hi_c = stages.feasible_ranges(np.concatenate([chain(b_i), chain(b_o)]), t_index, t_unique.size, natives, (-1.0, 1.0))
    uniq = lo_c == hi_c
    meas_args, meas_vals = t_unique[uniq], stages.from_ordinal(lo_c[uniq])
    dev_t = np.array([dev["tanh"][a] for a in meas_args])
    off = np.round((dev_t - meas_vals) / base.ulp16(meas_vals)).astype(int)
    result["conditional_native_tanh"] = {
        "measured": int(uniq.sum()),
        "device_equal": int(np.count_nonzero(dev_t == meas_vals)),
        "correctly_rounded_equal": int(np.count_nonzero(f16(np.tanh(meas_args)) == meas_vals)),
        "device_minus_native_ulps": {str(u): int(np.count_nonzero(off == u)) for u in np.unique(off)},
    }

    # ---- device versus correct rounding on the decoder arguments --------------- #
    for op, a, fn in (("sigmoid", sig_args, lambda x: 1 / (1 + np.exp(-x))), ("tanh", tanh_args, np.tanh)):
        dv = np.array([dev[op][v] for v in a])
        o = np.round((dv - f16(fn(a))) / base.ulp16(f16(fn(a)))).astype(int)
        result[f"device_{op}_minus_correctly_rounded_ulps"] = {
            "histogram": {str(u): int(np.count_nonzero(o == u)) for u in np.unique(o)},
            "max_abs": int(np.abs(o).max()),
            "bit_exact": int(np.count_nonzero(o == 0)),
        }

    # ---- table facts from the device ------------------------------------------- #
    rule = {}
    for op in ("sigmoid", "tanh"):
        L = layout[op]
        kn = table[op]["knots"]
        s = slice(*L["knots"])
        knots_eq = Y[op][s] == kn
        s = slice(*L["mids"])
        avg = f16((kn[:-1] + kn[1:]) / 2.0)
        mids_eq = Y[op][s] == avg
        n = L["pad"][0]
        model = h13_lut(table[op], op, X[op][:n])
        ok = model == Y[op][:n]
        rule[op] = {
            "knots_equal_table_entry": [int(knots_eq.sum()), 33],
            "knot_exceptions": [(float(X[op][slice(*L['knots'])][i]), float(Y[op][slice(*L['knots'])][i]), float(kn[i])) for i in np.nonzero(~knots_eq)[0]],
            "midpoints_equal_fp16_average": [int(mids_eq.sum()), 32],
            "clamp_probes": [(float(x), float(y)) for x, y in zip(X[op][slice(*L["clamps"])], Y[op][slice(*L["clamps"])])],
            "identified_rule_matches": [int(ok.sum()), int(n)],
            "identified_rule_matches_by_group": {g: [int(ok[slice(*L[g])].sum()), L[g][1] - L[g][0]] for g in ("args", "knots", "mids", "clamps", "sweep")},
            "identified_rule_exceptions": [(float(X[op][i]), float(Y[op][i]), float(model[i])) for i in np.nonzero(~ok)[0]],
        }
        # the alternatives the rule was chosen against, on interior lanes
        start, step = GRID[op]
        u = np.abs(X[op][:n]) if op == "tanh" else X[op][:n]
        interior = (u > start) & (u < start + 32 * step)
        ax, ay = X[op][:n][interior], Y[op][:n][interior]
        sign = np.sign(ax) if op == "tanh" else np.ones_like(ax)
        v = np.abs(ax) if op == "tanh" else ax
        idx = (v - start) / step
        k = np.clip(np.floor(idx).astype(int), 0, 31)
        t = idx - k
        a, b = kn[k], kn[k + 1]
        alts = {
            "rne(a + t*d), exact lerp": f16(a + t * (b - a)),
            "rne(a + rne16(t*d))": f16(a + f16(t * (b - a))),
            "raz(a + raz16(t*d)), lower-knot anchor": raz16(a + raz16(t * (b - a))),
            "raz(b - raz16((1-t)*d)), upper-knot anchor": raz16(b - raz16((1 - t) * (b - a))),
        }
        rule[op]["interior_lanes"] = int(interior.sum())
        rule[op]["alternatives_interior_matches"] = {k_: int(np.count_nonzero(sign * m == ay)) for k_, m in alts.items()}
        rule[op]["alternatives_interior_matches_negative_arguments"] = {
            k_: [int(np.count_nonzero((sign * m == ay) & (ax < 0))), int(np.count_nonzero(ax < 0))] for k_, m in alts.items()
        }
    result["interpolator"] = rule

    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(f"pins: sigmoid {pin_report['sigmoid']['reproduced']}/11 tanh {pin_report['tanh']['reproduced']}/8")
    print(f"lanes device/device: {result['lanes_of_1280']['device_sigmoid_device_tanh']}")
    print(f"refutation: {result['partner_free_refutation']}")
    print(f"conditional tanh: {result['conditional_native_tanh']['device_equal']}/{result['conditional_native_tanh']['measured']}")
    for op in ("sigmoid", "tanh"):
        print(f"{op} rule matches {rule[op]['identified_rule_matches']} exceptions {rule[op]['identified_rule_exceptions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

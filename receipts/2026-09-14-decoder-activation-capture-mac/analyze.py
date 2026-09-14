# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Score the macOS per-compute-unit ``sigmoid``/``tanh`` outputs (``mac/``) against
the native decoder capture, and read the decoder compute plan.

Named hole: ``parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml``.

Host-only. The Mac run is ``capture_mac.py`` on macstudio; this script reads its
outputs and the same capture, package, pins and solver the earlier receipts use.
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
UNITS = ("cpu_only", "cpu_and_gpu", "cpu_and_ne", "all")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--stages", type=Path, default=HERE.parent / "2026-09-14-decoder-activation-stages")
    parser.add_argument("--pins", type=Path, default=HERE.parent / "2026-09-14-decoder-activations")
    parser.add_argument("--jwm1", type=Path, default=HERE.parent / "2026-09-14-decoder-activations-on-ane")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    for p in (args.overlay, args.pins, args.stages):
        sys.path.insert(0, str(p))
    import probe as base
    import stages

    f16 = base.f16
    component, weights, traces = base.load(args.package, args.capture_dir)
    first = traces[0]
    b_i, _b_f, b_o, b_c = np.split(weights["l0_bias"].astype(np.float64), 4)
    native_cell = first["decoder_next_cell"][0, 0].astype(np.float64)
    native_hidden = first["decoder_next_hidden"][0, 0].astype(np.float64)
    natives = np.concatenate([native_cell, native_hidden])
    sig_args = np.concatenate([b_i, b_o])
    tanh_args = np.concatenate([b_c, native_cell])

    mac = json.loads((HERE / "mac/mac_result.json").read_text())
    layout = json.loads((args.jwm1 / "device/layout.json").read_text())
    X, Y, dev = {}, {}, {}
    for op in ("sigmoid", "tanh"):
        X[op] = np.concatenate([np.fromfile(args.jwm1 / f"device/x_{op}_{p}.bin", F16) for p in range(3)]).astype(np.float64)
        lo, hi = layout[op]["args"]
        assert np.array_equal(X[op][lo:hi], sig_args if op == "sigmoid" else tanh_args)
        Y[op] = {u: np.fromfile(HERE / f"mac/y_{u}_{op}.bin", F16).astype(np.float64) for u in UNITS}
        Y[op]["jwm1_h13_lut"] = np.concatenate([np.fromfile(args.jwm1 / f"device/y_out_{op}_{p}.bin", F16) for p in range(3)]).astype(np.float64)
        for u, y in Y[op].items():
            assert y.size == X[op].size, (op, u)
            d = {}
            for x, v in zip(X[op], y):
                assert d.setdefault(x, v) == v, (op, u, "two values for one argument")
            dev[(op, u)] = d

    result: dict = {
        "schema": "mlx-omarchy.parakeet-decoder-activation-capture-mac/1",
        "named_hole": "parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml",
        "capture_sha256": {n: sha256(args.capture_dir / n) for n in ("tdt_tensors.json", "manifest.sha256")},
        "mac_host": mac["host"],
        "mac_decoder_sha256": mac["decoder_sha256"],
        "mac_fixture_sha256": mac["fixture_sha256"],
        "mac_files_sha256": {p.name: sha256(p) for p in sorted((HERE / "mac").iterdir())},
        "transition": int(first["index"]),
        "units_identical_outputs": {
            op: len({hashlib.sha256(Y[op][u].astype(F16).tobytes()).hexdigest() for u in UNITS}) == 1 for op in ("sigmoid", "tanh")
        },
    }

    # ---- compute plans --------------------------------------------------------- #
    def plan_summary(rows):
        return {
            "ops": len(rows),
            "preferred_devices": sorted({r["preferred"] for r in rows if r["preferred"]}),
            "lstm": [{"output": r["outputs"][0], "preferred": r["preferred"], "supported": r["supported"], "weight": r["weight"]} for r in rows if r["op"].endswith("lstm")],
            "linear": [{"output": r["outputs"][0], "preferred": r["preferred"], "supported": r["supported"]} for r in rows if r["op"].endswith("linear")],
            "ops_with_ne_supported": [r["outputs"][0] for r in rows if r["supported"] and "MLNeuralEngineComputeDevice" in r["supported"]],
            "ops_preferred_non_cpu": [r["outputs"][0] for r in rows if r["preferred"] and r["preferred"] != "MLCPUComputeDevice"],
        }

    result["decoder_compute_plan"] = {u: plan_summary(mac["decoder_compute_plan"][u]) for u in UNITS}
    result["oneop_compute_plan"] = {u: [(r["op"], r["preferred"], r["supported"]) for r in mac["oneop"]["compute_plan"][u]] for u in UNITS}
    result["decoder_rerun_equal_to_capture"] = {
        u: {k: [v["equal_to_capture"], v["of"]] for k, v in mac["decoder_runs"][u].items()} for u in UNITS
    }

    # ---- pins ------------------------------------------------------------------ #
    pins = json.loads((args.pins / "result.json").read_text())["measured_native_values"]
    result["pins"] = {}
    for op, key in (("sigmoid", "monotone_sigmoid"), ("tanh", "monotone_tanh")):
        rows = pins[key]["pinned"]
        result["pins"][op] = {
            u: [int(sum(dev[(op, u)][r["argument"]] == r["native_value"] for r in rows)), len(rows)] for u in list(UNITS) + ["jwm1_h13_lut"]
        }
        result["pins"][op]["rows"] = [
            {"argument": r["argument"], "native": r["native_value"], "mac_cpu": float(dev[(op, "cpu_only")][r["argument"]])} for r in rows
        ]

    # ---- lanes and refutation, per unit ---------------------------------------- #
    chain = lambda x: f16(1.0 / f16(1.0 + f16(np.exp(-x))))  # noqa: E731
    t_unique, t_index = np.unique(tanh_args, return_inverse=True)
    s_unique, s_index = np.unique(sig_args, return_inverse=True)
    lo_c, hi_c = stages.feasible_ranges(np.concatenate([chain(b_i), chain(b_o)]), t_index, t_unique.size, natives, (-1.0, 1.0))
    uniq = lo_c == hi_c
    meas_args, meas_vals = t_unique[uniq], stages.from_ordinal(lo_c[uniq])
    per_unit = {}
    for u in list(UNITS) + ["jwm1_h13_lut"]:
        s_i = np.array([dev[("sigmoid", u)][a] for a in b_i])
        s_o = np.array([dev[("sigmoid", u)][a] for a in b_o])
        t_c = np.array([dev[("tanh", u)][a] for a in b_c])
        t_h = np.array([dev[("tanh", u)][a] for a in native_cell])
        lo, hi = stages.feasible_ranges(np.concatenate([s_i, s_o]), t_index, t_unique.size, natives, (-1.0, 1.0))
        lo2, hi2 = stages.feasible_ranges(np.concatenate([t_c, t_h]), s_index, s_unique.size, natives, (0.0, 1.0))
        t_meas = np.array([dev[("tanh", u)][a] for a in meas_args])
        s_all = np.array([dev[("sigmoid", u)][a] for a in sig_args])
        t_all = np.array([dev[("tanh", u)][a] for a in tanh_args])
        per_unit[u] = {
            "next_cell_lanes_of_640": int(np.count_nonzero(f16(s_i * t_c) == native_cell)),
            "next_hidden_lanes_of_640": int(np.count_nonzero(f16(s_o * t_h) == native_hidden)),
            "sigmoid_refuting_tanh_arguments": [int(np.count_nonzero(lo > hi)), int(t_unique.size)],
            "tanh_refuting_sigmoid_arguments": [int(np.count_nonzero(lo2 > hi2)), int(s_unique.size)],
            "tanh_equal_to_949_conditional_native": [int(np.count_nonzero(t_meas == meas_vals)), int(uniq.sum())],
            "sigmoid_equal_fp16_chain_on_1280_args": int(np.count_nonzero(s_all == chain(sig_args))),
            "sigmoid_equal_correctly_rounded_on_1280_args": int(np.count_nonzero(s_all == f16(1 / (1 + np.exp(-sig_args))))),
            "tanh_equal_correctly_rounded_on_1280_args": int(np.count_nonzero(t_all == f16(np.tanh(tanh_args)))),
        }
    result["per_unit"] = per_unit
    result["fp16_chain_sigmoid_correctly_rounded_tanh_reference"] = {
        "next_cell_lanes_of_640": int(np.count_nonzero(f16(chain(b_i) * f16(np.tanh(b_c))) == native_cell)),
        "next_hidden_lanes_of_640": int(np.count_nonzero(f16(chain(b_o) * f16(np.tanh(native_cell))) == native_hidden)),
    }

    # ---- identify the CPU contract on all 1536 lanes ---------------------------- #
    xs, xt = X["sigmoid"], X["tanh"]
    ys, yt = Y["sigmoid"]["cpu_only"], Y["tanh"]["cpu_only"]
    sigs = base.sigmoid_catalogue()
    tanhs = base.tanh_catalogue(sigs)
    score = lambda cat, x, y: {n: int(np.count_nonzero(np.asarray(fn(x)) == y)) for n, fn in cat.items()}  # noqa: E731

    def top(scores, n=12):
        best = sorted(scores.items(), key=lambda kv: -kv[1])
        return [[k, v] for k, v in best[:n]]

    ss = score(sigs, xs, ys)
    ts = score(tanhs, xt, yt)
    # extra forms the catalogue does not carry: fp32 evaluation rounded once to fp16
    extra_s = {
        "sig:f32exact->f16": f16(np.float32(1) / (np.float32(1) + np.exp(-xs.astype(np.float32)))),
        "sig:f32recip->f16": f16((np.float32(1) / (np.float32(1) + np.exp(-xs.astype(np.float32)).astype(np.float32))).astype(np.float32)),
        "sig:f64exact->f16": f16(1 / (1 + np.exp(-xs))),
        "sig:f32tanhform->f16": f16((np.float32(0.5) * np.tanh(np.float32(0.5) * xs.astype(np.float32)) + np.float32(0.5)).astype(np.float32)),
        "sig:f16chain": chain(xs),
    }
    extra_t = {
        "tanh:f32exact->f16": f16(np.tanh(xs.astype(np.float32) * 0 + xt.astype(np.float32)).astype(np.float32)),
        "tanh:f64exact->f16": f16(np.tanh(xt)),
    }
    for k, v in extra_s.items():
        ss[k] = int(np.count_nonzero(v == ys))
    for k, v in extra_t.items():
        ts[k] = int(np.count_nonzero(v == yt))
    result["cpu_contract_search"] = {
        "sigmoid_lanes": int(xs.size),
        "sigmoid_top": top(ss),
        "sigmoid_exact_matches": [k for k, v in ss.items() if v == xs.size],
        "tanh_lanes": int(xt.size),
        "tanh_top": top(ts),
        "tanh_exact_matches": [k for k, v in ts.items() if v == xt.size],
    }
    for op, x, y, fn in (("sigmoid", xs, ys, lambda v: 1 / (1 + np.exp(-v))), ("tanh", xt, yt, np.tanh)):
        cr = f16(fn(x))
        o = np.round((y - cr) / base.ulp16(cr)).astype(int)
        result[f"mac_cpu_{op}_minus_correctly_rounded_ulps_1536"] = {
            "histogram": {str(u): int(np.count_nonzero(o == u)) for u in np.unique(o)},
            "max_abs": int(np.abs(o).max()),
            "bit_exact": int(np.count_nonzero(o == 0)),
            "exceptions": [(float(x[i]), float(y[i]), float(cr[i])) for i in np.nonzero(o != 0)[0]][:40],
        }
    result["mac_cpu_vs_jwm1_lut_equal_1536"] = {
        op: int(np.count_nonzero(Y[op]["cpu_only"] == Y[op]["jwm1_h13_lut"])) for op in ("sigmoid", "tanh")
    }

    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print("units identical:", result["units_identical_outputs"])
    for u in UNITS:
        p = result["decoder_compute_plan"][u]
        print(u, "preferred:", p["preferred_devices"], "lstm:", [(r["preferred"], r["supported"]) for r in p["lstm"]], "rerun:", result["decoder_rerun_equal_to_capture"][u])
    print("pins:", {op: {u: v for u, v in result["pins"][op].items() if u != "rows"} for op in ("sigmoid", "tanh")})
    for u, r in per_unit.items():
        print(u, json.dumps(r))
    print("reference:", result["fp16_chain_sigmoid_correctly_rounded_tanh_reference"])
    print("cpu contract:", json.dumps(result["cpu_contract_search"], indent=1))
    for op in ("sigmoid", "tanh"):
        print(op, "vs correctly rounded:", result[f"mac_cpu_{op}_minus_correctly_rounded_ulps_1536"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

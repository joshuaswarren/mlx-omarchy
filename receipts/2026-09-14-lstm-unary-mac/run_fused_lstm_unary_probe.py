# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Fused-LSTM sigmoid/tanh extraction on the macOS reference host (macstudio).

Runs the LstmUnaryFit reduction-free arguments THROUGH THE FUSED ios18.lstm op
(not a standalone sigmoid/tanh op) and reads the per-gate unary values out of
the LSTM's own outputs, plus MLComputePlan rows for every LSTM op, plus a
bit-exact re-run of the pinned decoder against the original capture golden.

Isolation scheme (single timestep, W = R = 0, h0 = 0):
  c1 = f*c0 + i*g   with   b_f = +20 (f = 1), b_i = -30 (i = 0), b_g = 0 (g = 0)
  h1 = o * tanh(c1) with   b_o = probe arguments per lane (o = sigmoid probe)
Sigmoid run:  c0 = C (several constants); lane with b_o = +20 yields
  h = sigma(20)*tanh(c1) = u exactly once sigma(20) = 1.0 is verified (a
  b_o = 0 lane cross-checks sigma(0) = 0.5), so every h_j = round16(sigma_j*u)
  and sigma_j is recovered by intersecting fp16 preimages of the multiply
  across u values (the c0 = 20 run has u = 1.0 and reads sigma directly).
Tanh run:     b_o = +20 everywhere, c0 = tanh arguments; c1 = 1.0*c0 + 0 = c0
  exactly, so h_j = sigma(20)*tanh(c0_j) = tanh(c0_j) read directly.

Writes only into ./out/. Reads lstm_unary_fit_args.npz and the pinned decoder
package; changes nothing else on the host.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models import MLModel
from coremltools.models.compute_plan import MLComputePlan

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
F16 = np.dtype("<f2")
H = 1282          # 1280 sigmoid argument lanes + 2 calibration lanes
I_DIM = 640       # mimic decoder LSTM input width
SIG_I = slice(0, 640)
SIG_O = slice(640, 1280)
CAL20 = 1280
CAL0 = 1281
TANH_PADS = np.array(
    [0.0, -0.0, 20.0, -20.0, 10.0, -10.0, 5.0, -5.0, 2.0, -2.0,
     1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.01, -0.01], dtype=np.float32)
C0_VALUES = (20.0, 3.0, 2.5, 2.0, 1.5, 1.0)
UNITS = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def fp16_exact(a: np.ndarray) -> bool:
    a32 = np.asarray(a, np.float32)
    return bool(np.array_equal(a32.astype(F16).astype(np.float32), a32))


def cr_sigmoid(a: np.ndarray, w: int) -> np.ndarray:
    x = a.astype(np.float32 if w == 32 else np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float16)


def cr_tanh(a: np.ndarray, w: int) -> np.ndarray:
    x = a.astype(np.float32 if w == 32 else np.float64)
    return np.tanh(x).astype(np.float16)


def build_probe(bias_o: np.ndarray, tag: str):
    bias = np.concatenate([
        np.full(H, -30.0), np.full(H, 20.0), bias_o, np.zeros(H),
    ]).astype(np.float16)
    assert bias.shape == (4 * H,)
    wi = np.zeros((4 * H, I_DIM), np.float16)
    wh = np.zeros((4 * H, H), np.float16)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 1, I_DIM), dtype=types.fp16),
            mb.TensorSpec(shape=(1, H), dtype=types.fp16),
            mb.TensorSpec(shape=(1, H), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x, initial_h, initial_c):
        r = mb.lstm(
            x=x, initial_h=initial_h, initial_c=initial_c,
            weight_ih=wi, weight_hh=wh, bias=bias,
            direction="forward", output_sequence=False,
            recurrent_activation="sigmoid", cell_activation="tanh",
            activation="tanh", name="probe_lstm",
        )
        return r[1], r[2]  # mb.lstm returns [out_seq, h, c]; keep h and c
    pkg = OUT / f"probe_{tag}.mlpackage"
    model = ct.convert(
        prog, convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    model.save(str(pkg))
    mil = model.get_spec().mlProgram.functions["main"].block_specializations
    ops = [op.type for op in next(iter(mil.values())).operations if op.type != "const"]
    compiled = ct.models.utils.compile_model(str(pkg))
    return pkg, compiled, int(model.get_spec().specificationVersion), ops


def plan_rows(compiled: str, unit) -> list[dict]:
    plan = MLComputePlan.load_from_path(compiled, compute_units=unit)
    rows = []
    for fn_name, fn in plan.model_structure.program.functions.items():
        for op in fn.block.operations:
            op_name = getattr(op, "operator_name", None) or op.type
            if op_name == "const":
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            rows.append({
                "function": fn_name, "op": op_name,
                "outputs": [o.name for o in op.outputs],
                "preferred": type(usage.preferred_compute_device).__name__ if usage else None,
                "supported": [type(d).__name__ for d in usage.supported_compute_devices] if usage else None,
                "weight": cost.weight if cost else None,
            })
    return rows


def fp16_grid() -> np.ndarray:
    bits = np.arange(65536, dtype=np.uint16)
    g = bits.view(np.float16)
    return g[np.isfinite(g)]


def main() -> int:
    t_all = time.perf_counter()
    args_npz = np.load(HERE / "lstm_unary_fit_args.npz")
    sig_i = args_npz["sigmoid_input_gate"].astype(np.float16)
    sig_o = args_npz["sigmoid_output_gate"].astype(np.float16)
    tan_g = args_npz["tanh_cell_gate"].astype(np.float16)
    bias_full = args_npz["layer0_bias_full"].astype(np.float16)
    native_c = np.asarray(args_npz["native_next_cell"], np.float32)
    native_h = np.asarray(args_npz["native_next_hidden"], np.float32)
    for name, a in (("sigmoid_input_gate", sig_i), ("sigmoid_output_gate", sig_o),
                    ("tanh_cell_gate", tan_g), ("layer0_bias_full", bias_full)):
        assert np.isfinite(np.asarray(a, np.float32)).all(), f"{name} not finite"
    assert bias_full.shape == (2560,)
    assert native_c.shape in ((2, 1, 640), (640,)) and native_h.shape == native_c.shape, (native_c.shape, native_h.shape)
    golden_flat = native_c.ndim == 1

    sig_args = np.concatenate([sig_i, sig_o]).astype(np.float16)
    out = {
        "schema": "mlx-omarchy.fused-lstm-unary-mac/1",
        "host": {
            "hostname": platform.node(),
            "hw_model": sysctl("hw.model"),
            "chip": sysctl("machdep.cpu.brand_string"),
            "macos": subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip(),
            "build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
            "coreml_framework": subprocess.run(
                ["plutil", "-extract", "CFBundleVersion", "raw",
                 "/System/Library/Frameworks/CoreML.framework/Versions/A/Resources/Info.plist"],
                capture_output=True, text=True).stdout.strip(),
            "coremltools": ct.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "fixture": {
            "path": str(HERE / "lstm_unary_fit_args.npz"),
            "sha256": sha256(HERE / "lstm_unary_fit_args.npz"),
            "sigmoid_lanes": int(sig_args.size),
            "tanh_lanes": int(tan_g.size),
        },
        "decoder_package": "/Users/joshuawarren/.cache/mlx-omarchy/parakeet-reference/tdt-staging/b650695c-75aec2a-ane.dpiNlU/models/decoder.mlpackage",
    }
    out["decoder_package_sha256"] = {
        n: sha256(Path(out["decoder_package"]) / n)
        for n in ("Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin")
    }
    print("host", json.dumps(out["host"]), flush=True)

    # ---- 1. sigmoid probe -------------------------------------------------- #
    b_o_probe = np.concatenate([sig_args, [np.float16(20.0), np.float16(0.0)]]).astype(np.float16)
    assert b_o_probe.size == H
    pkg_s, comp_s, spec_s, ops_s = build_probe(b_o_probe, "sig")
    out["sigmoid_probe"] = {
        "package": str(pkg_s),
        "package_sha256": sha256(pkg_s / "Data/com.apple.CoreML/model.mlmodel"),
        "specificationVersion": spec_s, "nonconst_ops": ops_s,
    }
    print("sigmoid probe built spec", spec_s, ops_s, flush=True)
    out["sigmoid_probe"]["compute_plan"] = {n: plan_rows(comp_s, u) for n, u in UNITS.items()}
    m = MLModel(str(pkg_s), compute_units=ct.ComputeUnit.CPU_ONLY)
    zero_x = np.zeros((1, 1, I_DIM), np.float16)
    zero_h = np.zeros((1, H), np.float16)
    h_runs, u_runs, run_meta = {}, {}, {}
    for c0v in C0_VALUES:
        feed = {"x": zero_x, "initial_h": zero_h,
                "initial_c": np.full((1, H), c0v, np.float32).astype(np.float16)}
        y = m.predict(feed)
        y2 = m.predict(feed)
        hv = np.asarray(list(y.values())[0], np.float32).astype(np.float16).ravel()
        assert np.array_equal(hv, np.asarray(list(y2.values())[0], np.float32).astype(np.float16).ravel()), "nondeterministic"
        h_runs[c0v] = hv
        run_meta[c0v] = {"h_cal20": float(hv[CAL20]), "h_cal0": float(hv[CAL0])}
        print(f"sigmoid run c0={c0v}: h[+20]={hv[CAL20]!r} h[0]={hv[CAL0]!r}", flush=True)
    out["sigmoid_probe"]["runs"] = {str(k): v for k, v in run_meta.items()}
    # sigma(20) must read exactly 1.0 in the c0=20 run for direct extraction.
    sigma20 = h_runs[20.0][CAL20]
    out["sigmoid_probe"]["sigma20_read"] = float(sigma20)
    direct_ok = sigma20 == np.float16(1.0)
    out["sigmoid_probe"]["sigma20_is_one"] = bool(direct_ok)
    if not direct_ok:
        out["sigmoid_probe"]["verdict_note"] = f"sigma(20)={sigma20!r} != 1.0; falling back to candidate sets only"
    else:
        for c0v in C0_VALUES[1:]:
            u_runs[c0v] = h_runs[c0v][CAL20]      # = tanh(c1), read exactly
        out["sigmoid_probe"]["sigma0_consistent"] = bool(
            h_runs[20.0][CAL0] == np.float16(h_runs[20.0][CAL20] * np.float16(0.5)))
    grid = fp16_grid()
    lanes = np.arange(sig_args.size)
    assert np.array_equal(sig_args, b_o_probe[lanes])
    sigma_out = np.full(sig_args.size, np.nan, np.float32)
    ambiguous, mism32, mism64 = [], [], []
    chunk = 128
    for s0 in range(0, sig_args.size, chunk):
        sl = lanes[s0:s0 + chunk]
        cand_mask = np.ones((grid.size, sl.size), bool)
        for k, u in u_runs.items():
            prod = grid[:, None] * np.float16(u)
            cand_mask &= prod == h_runs[k][sl][None, :]
        for jj, lane in enumerate(sl):
            cs = grid[cand_mask[:, jj]]
            arg = sig_args[s0 + jj]
            if cs.size == 0:
                ambiguous.append((s0 + jj, float(arg), "no-candidate"))
                continue
            direct = np.float16(h_runs[20.0][lane])
            if cs.size == 1:
                val = cs[0]
            elif direct in cs:
                val = direct
            else:
                vals32 = sorted({float(cr_sigmoid(np.array([c], np.float16), 32)[0]) for c in cs})
                if len(vals32) == 1:
                    val = cs[0]
                else:
                    ambiguous.append((s0 + jj, float(arg), vals32))
                    continue
            sigma_out[s0 + jj] = float(val)
            if val != cr_sigmoid(np.array([arg]), 32)[0]:
                mism32.append((s0 + jj, float(arg), float(val), float(cr_sigmoid(np.array([arg]), 32)[0])))
            if val != cr_sigmoid(np.array([arg]), 64)[0]:
                mism64.append((s0 + jj, float(arg), float(val), float(cr_sigmoid(np.array([arg]), 64)[0])))
    out["sigmoid_probe"]["extraction"] = {
        "lanes": int(sig_args.size),
        "resolved": int(np.isfinite(sigma_out).sum()),
        "ambiguous": ambiguous[:40],
        "ambiguous_count": len(ambiguous),
        "mismatch_vs_cr_fp32": mism32[:40],
        "mismatch_count_vs_cr_fp32": len(mism32),
        "mismatch_vs_cr_fp64": mism64[:40],
        "mismatch_count_vs_cr_fp64": len(mism64),
        "arg_range": [float(sig_args.min()), float(sig_args.max())],
    }
    print("sigmoid extraction:", json.dumps(
        {k: v for k, v in out["sigmoid_probe"]["extraction"].items()
         if k in ("resolved", "ambiguous_count", "mismatch_count_vs_cr_fp32",
                  "mismatch_count_vs_cr_fp64", "arg_range")}), flush=True)
    np.savez(OUT / "sigma_extracted.npz", args=sig_args, sigma=sigma_out,
             sigma_input_gate=sigma_out[:640], sigma_output_gate=sigma_out[640:])

    # ---- 2. tanh probe ----------------------------------------------------- #
    c0_tanh = np.concatenate([tan_g, TANH_PADS]).astype(np.float32)
    pad = np.linspace(-8.0, 8.0, H - c0_tanh.size, dtype=np.float32)
    c0_tanh = np.concatenate([c0_tanh, pad]).astype(np.float16)
    assert c0_tanh.size == H
    pkg_t, comp_t, spec_t, ops_t = build_probe(np.full(H, 20.0, np.float16), "tanh")
    out["tanh_probe"] = {
        "package": str(pkg_t),
        "package_sha256": sha256(pkg_t / "Data/com.apple.CoreML/model.mlmodel"),
        "specificationVersion": spec_t, "nonconst_ops": ops_t,
    }
    print("tanh probe built spec", spec_t, flush=True)
    out["tanh_probe"]["compute_plan"] = {n: plan_rows(comp_t, u) for n, u in UNITS.items()}
    mt = MLModel(str(pkg_t), compute_units=ct.ComputeUnit.CPU_ONLY)
    feed = {"x": zero_x, "initial_h": zero_h, "initial_c": c0_tanh.reshape(1, H)}
    y = mt.predict(feed)
    y2 = mt.predict(feed)
    thv = np.asarray(list(y.values())[0], np.float32).astype(np.float16).ravel()
    assert np.array_equal(thv, np.asarray(list(y2.values())[0], np.float32).astype(np.float16).ravel()), "nondeterministic"
    tanh_args, tanh_out = c0_tanh, thv.copy()
    mismt32 = [(int(i), float(a), float(v), float(cr_tanh(np.array([a]), 32)[0]))
               for i, (a, v) in enumerate(zip(tanh_args, tanh_out)) if v != cr_tanh(np.array([a]), 32)[0]]
    mismt64 = [(int(i), float(a), float(v), float(cr_tanh(np.array([a]), 64)[0]))
               for i, (a, v) in enumerate(zip(tanh_args, tanh_out)) if v != cr_tanh(np.array([a]), 64)[0]]
    zer0 = int(np.flatnonzero(tanh_args == 0)[0])
    out["tanh_probe"]["extraction"] = {
        "lanes": int(H),
        "arg_lanes": int(tan_g.size),
        "mismatch_vs_cr_fp32": mismt32[:40],
        "mismatch_count_vs_cr_fp32": len(mismt32),
        "mismatch_vs_cr_fp64": mismt64[:40],
        "mismatch_count_vs_cr_fp64": len(mismt64),
        "negative_zero_preserved": bool(tanh_args[1] < 0 and np.signbit(tanh_out[1])),
        "zero_lane_value": float(tanh_out[zer0]),
        "saturated_lane_h": float(tanh_out[CAL20]),
    }
    print("tanh extraction:", json.dumps(
        {k: v for k, v in out["tanh_probe"]["extraction"].items()
         if "count" in k or "zero" in k or k == "saturated_lane_h"}), flush=True)
    np.savez(OUT / "tanh_extracted.npz", args=tanh_args, tanh=tanh_out,
             tanh_cell_gate=tanh_out[:640], tanh_pads=tanh_out[640:])

    # ---- 3. decoder re-run vs capture golden ------------------------------- #
    dec_pkg = Path(out["decoder_package"])
    ids = np.load(HERE / "tdt_trace_0000_decoder_input_ids.npy")
    hidden = np.load(HERE / "tdt_trace_0000_decoder_hidden.npy")
    cell = np.load(HERE / "tdt_trace_0000_decoder_cell.npy")
    out["decoder_inputs"] = {
        "ids": np.asarray(ids).ravel().tolist(),
        "hidden_abs_max": float(np.abs(hidden).max()),
        "cell_abs_max": float(np.abs(cell).max()),
        "hidden_all_zero": bool(np.all(hidden == 0)),
        "cell_all_zero": bool(np.all(cell == 0)),
    }
    comp_d = ct.models.utils.compile_model(str(dec_pkg))
    dec_plan = {}
    for n, u in UNITS.items():
        rows = plan_rows(comp_d, u)
        dec_plan[n] = [r for r in rows if "lstm" in r["op"]]
    out["decoder_compute_plan_lstm"] = dec_plan
    dec_runs = {}
    for n, u in UNITS.items():
        md = MLModel(str(dec_pkg), compute_units=u)
        feed = {"input_ids": ids, "hidden": hidden, "cell": cell}
        y = md.predict(feed)
        y2 = md.predict(feed)
        rec = {}
        for oname, ref in (("next_cell", native_c), ("next_hidden", native_h)):
            a = np.asarray(y[oname], np.float32)
            a2 = a.reshape((2, 1, 640))
            ref_l0 = ref.reshape((2, 1, 640))[0] if ref.ndim == 3 else ref
            rep = np.array_equal(a, np.asarray(y2[oname], np.float32))
            eq = a2[0] == ref_l0
            rec[oname] = {
                "shape": list(a.shape), "fp16_exact": fp16_exact(a), "repeat_identical": rep,
                "golden_scope": "layer0" if ref.ndim == 1 else "both",
                "equal_to_golden_layer0": int(eq.sum()), "of": int(eq.size),
                "layer0_equal": bool(eq.all()),
                "max_abs_diff_layer0": float(np.abs(a2[0] - ref_l0).max()),
            }
        dec_runs[n] = rec
        print("decoder", n, json.dumps(rec), flush=True)
    out["decoder_runs"] = dec_runs
    out["golden_scope_note"] = "fixture golden is flat 640 = layer-0 next state" if golden_flat else "fixture golden covers both layers"

    out["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("WROTE", OUT / "results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

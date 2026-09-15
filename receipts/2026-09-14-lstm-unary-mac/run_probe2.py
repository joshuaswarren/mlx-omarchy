# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Fused-LSTM sigmoid/tanh extraction v2 on macstudio.

v1 established: the fused ios18.lstm gate sigmoid is NOT correctly rounded
(sigma(20) = 1 - 2^-10 exactly, sigma(0) = 0.5 - 1 ulp), while the decoder
package reproduces the original capture-host golden bit-exact on layer 0.

v2 pins the full contract:
  1. P_sig  (b_o = 1280 fit args + calib lanes, c0 = 20): direct sigma reads;
     the c0 = 20 calibration proves the output multiplier U20 = 1.0 uniquely
     (the only fp16 u with round16(K*u) == K is 1.0 itself).
  2. P_grid (b_o = dense fp16 grid, c0 = 20): sigma curve over [-8, 8] plus
     saturation probes, all read exactly.
  3. P_t20  (b_o = +20 everywhere, c0 = tanh args and preimage-compensated
     args): h = round16(K * tanh_impl(c1)) with K = sigma(20); the second
     model output exposes c1 directly, verifying every argument.
  4. P_m075 (b_o = 0.75, c0 = compensated args and +20): second multiplier
     m* = sigma(0.75) (read exactly from its own c0 = 20 run); intersecting
     the K-preimage and m*-preimage sets resolves tanh_impl(t) uniquely.

Writes only into ./out2/. Reads lstm_unary_fit_args.npz; changes nothing else.
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
OUT = HERE / "out2"
OUT.mkdir(exist_ok=True)
F16 = np.dtype("<f2")
H = 1282
I_DIM = 640
N_ARGS = 1280
CAL20 = 1280
CAL0 = 1281
TANH_PADS = np.array(
    [0.0, -0.0, 20.0, -20.0, 10.0, -10.0, 5.0, -5.0, 2.0, -2.0,
     1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.01, -0.01], dtype=np.float32)
GRID_SPECIALS = np.array(
    [-30.0, -27.0, -25.0, -20.0, -16.0, -12.0, 12.0, 16.0, 20.0, 25.0, 27.0, 30.0],
    dtype=np.float32)


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def cr(a: np.ndarray, kind: str, w: int) -> np.ndarray:
    x = a.astype(np.float32 if w == 32 else np.float64)
    y = 1.0 / (1.0 + np.exp(-x)) if kind == "sigmoid" else np.tanh(x)
    return y.astype(np.float16)


def build(bias_o: np.ndarray, tag: str):
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
        return r[1], r[2]  # h, c

    pkg = OUT / f"probe_{tag}.mlpackage"
    model = ct.convert(
        prog, convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    model.save(str(pkg))
    mil = model.get_spec().mlProgram.functions["main"].block_specializations
    ops = [op.type for op in next(iter(mil.values())).operations if op.type != "const"]
    compiled = ct.models.utils.compile_model(str(pkg))
    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_ONLY)
    rows = []
    for fn_name, fn in plan.model_structure.program.functions.items():
        for op in fn.block.operations:
            name = getattr(op, "operator_name", None) or op.type
            if name == "const":
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            rows.append({
                "op": name,
                "preferred": type(usage.preferred_compute_device).__name__ if usage else None,
                "supported": [type(d).__name__ for d in usage.supported_compute_devices] if usage else None,
                "weight": cost.weight if cost else None,
            })
    return pkg, compiled, int(model.get_spec().specificationVersion), ops, rows


def run_model(path: str, c0: np.ndarray, tag: str):
    m = MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    feed = {"x": np.zeros((1, 1, I_DIM), np.float16),
            "initial_h": np.zeros((1, H), np.float16),
            "initial_c": c0.reshape(1, H).astype(np.float16)}
    y = m.predict(feed)
    y2 = m.predict(feed)
    vals = list(y.values())
    h = np.asarray(vals[0], np.float32).astype(np.float16).ravel()
    c = np.asarray(vals[1], np.float32).astype(np.float16).ravel()
    h2 = np.asarray(list(y2.values())[0], np.float32).astype(np.float16).ravel()
    assert np.array_equal(h, h2), f"{tag}: nondeterministic h"
    return h, c


def fp16_grid() -> np.ndarray:
    bits = np.arange(65536, dtype=np.uint16)
    g = bits.view(np.float16)
    return np.ascontiguousarray(g[np.isfinite(g)])


def preimages(grid: np.ndarray, k: np.float16, target: np.ndarray):
    """Smallest-|x| fp16 x with round16(k*x) == target (per lane)."""
    prod = grid * k
    buckets = {}
    for i, p in enumerate(prod):
        buckets.setdefault(p, []).append(i)
    out = np.empty(target.shape, np.float16)
    ok = np.zeros(target.shape, bool)
    for j, t in enumerate(target):
        cands = buckets.get(t, [])
        if cands:
            best = min(cands, key=lambda i: abs(float(grid[i])))
            out[j] = grid[best]
            ok[j] = True
    return out, ok


def main() -> int:
    t_all = time.perf_counter()
    args_npz = np.load(HERE / "lstm_unary_fit_args.npz")
    sig_i = args_npz["sigmoid_input_gate"].astype(np.float16)
    sig_o = args_npz["sigmoid_output_gate"].astype(np.float16)
    tan_g = args_npz["tanh_cell_gate"].astype(np.float16)
    sig_args = np.concatenate([sig_i, sig_o]).astype(np.float16)
    assert sig_args.size == N_ARGS

    out = {
        "schema": "mlx-omarchy.fused-lstm-unary-mac/2",
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
            "coremltools": ct.__version__, "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "fixture_sha256": sha256(HERE / "lstm_unary_fit_args.npz"),
        "decoder_package": "/Users/joshuawarren/.cache/mlx-omarchy/parakeet-reference/tdt-staging/b650695c-75aec2a-ane.dpiNlU/models/decoder.mlpackage",
    }
    print("host", json.dumps(out["host"]), flush=True)
    grid = fp16_grid()

    # ---- 1. P_sig: direct sigma on the 1280 fit arguments ------------------ #
    b_o_sig = np.concatenate([sig_args, [np.float16(20.0), np.float16(0.0)]]).astype(np.float16)
    pkg_s, comp_s, spec_s, ops_s, rows_s = build(b_o_sig, "sig")
    out["P_sig"] = {"package_sha256": sha256(pkg_s / "Data/com.apple.CoreML/model.mlmodel"),
                    "spec": spec_s, "ops": ops_s, "plan_cpu_only": rows_s}
    print("P_sig built", spec_s, ops_s, flush=True)
    h_s, c_s = run_model(str(pkg_s), np.full(H, 20.0, np.float32), "P_sig c0=20")
    K = h_s[CAL20]                       # sigma(20), exact read
    u_cands = grid[(grid * K) == K]      # U20 = 1.0 iff u_cands == {1.0}
    sigma_args = h_s[:N_ARGS].copy()
    sigma0 = h_s[CAL0]
    out["P_sig"]["sigma20"] = float(K)
    out["P_sig"]["sigma0"] = float(sigma0)
    out["P_sig"]["u20_candidates"] = [float(v) for v in u_cands]
    out["P_sig"]["u20_is_one"] = bool(u_cands.size == 1 and u_cands[0] == np.float16(1.0))
    m32 = [(int(i), float(a), float(v), float(cr(np.array([a]), "sigmoid", 32)[0]))
           for i, (a, v) in enumerate(zip(sig_args, sigma_args))
           if v != cr(np.array([a]), "sigmoid", 32)[0]]
    m64 = [(int(i), float(a), float(v), float(cr(np.array([a]), "sigmoid", 64)[0]))
           for i, (a, v) in enumerate(zip(sig_args, sigma_args))
           if v != cr(np.array([a]), "sigmoid", 64)[0]]
    out["P_sig"]["extraction"] = {
        "lanes": N_ARGS,
        "mismatch_count_vs_cr_fp32": len(m32), "mismatch_vs_cr_fp32": m32[:40],
        "mismatch_count_vs_cr_fp64": len(m64), "mismatch_vs_cr_fp64": m64[:40],
        "arg_range": [float(sig_args.min()), float(sig_args.max())],
    }
    print("P_sig: sigma20", float(K), "sigma0", float(sigma0), "u20_is_one",
          out["P_sig"]["u20_is_one"], "mm32", len(m32), "mm64", len(m64), flush=True)
    np.savez(OUT / "sigma_args.npz", args=sig_args, sigma=sigma_args)

    # ---- 2. P_grid: sigma curve -------------------------------------------- #
    dens = np.linspace(-8.0, 8.0, H - GRID_SPECIALS.size, dtype=np.float32)
    b_o_grid = np.concatenate([dens, GRID_SPECIALS]).astype(np.float16)
    pkg_g, _, spec_g, ops_g, rows_g = build(b_o_grid, "grid")
    out["P_grid"] = {"package_sha256": sha256(pkg_g / "Data/com.apple.CoreML/model.mlmodel"),
                     "spec": spec_g, "plan_cpu_only": rows_g}
    h_g, c_g = run_model(str(pkg_g), np.full(H, 20.0, np.float32), "P_grid c0=20")
    out["P_grid"]["sigma30"] = float(h_g[H - 1])
    out["P_grid"]["sigma_minus30"] = float(h_g[H - GRID_SPECIALS.size])
    gm = int(np.count_nonzero(h_g != cr(b_o_grid, "sigmoid", 32)))
    out["P_grid"]["mismatch_vs_cr_fp32"] = [gm, int(H)]
    np.savez(OUT / "sigma_grid.npz", args=b_o_grid, sigma=h_g)
    print("P_grid: sigma30", float(h_g[H - 1]), "sigma(-30)",
          float(h_g[H - GRID_SPECIALS.size]), "grid mismatches", gm, "/", H, flush=True)

    # ---- 3. compensated tanh ------------------------------------------------ #
    tan_args = np.concatenate(
        [tan_g, TANH_PADS, dens[:H - 640 - TANH_PADS.size].astype(np.float16)]
    ).astype(np.float16)
    assert tan_args.size == H
    t_comp, ok_comp = preimages(grid, K, tan_args)
    out["preimage_gaps"] = int((~ok_comp).sum())
    print("preimage gaps:", int((~ok_comp).sum()), flush=True)

    pkg_t, _, spec_t, ops_t, rows_t = build(np.full(H, 20.0, np.float16), "t20")
    out["P_t20"] = {"package_sha256": sha256(pkg_t / "Data/com.apple.CoreML/model.mlmodel"),
                    "spec": spec_t, "plan_cpu_only": rows_t}
    pkg_m, _, spec_m, ops_m, rows_m = build(np.full(H, 0.75, np.float16), "m075")
    out["P_m075"] = {"package_sha256": sha256(pkg_m / "Data/com.apple.CoreML/model.mlmodel"),
                     "spec": spec_m, "plan_cpu_only": rows_m}

    h_t20cal, c_t20cal = run_model(str(pkg_t), np.full(H, 20.0, np.float32), "P_t20 cal")
    h_m20cal, c_m20cal = run_model(str(pkg_m), np.full(H, 20.0, np.float32), "P_m075 cal")
    mstar = h_m20cal[CAL20]
    out["multipliers"] = {"K_sigma20": float(K), "mstar_sigma075": float(mstar),
                          "tanh_model_cal": float(h_t20cal[CAL20])}
    print("multipliers: K", float(K), "m*", float(mstar), flush=True)

    h_A, c_A = run_model(str(pkg_t), tan_args, "P_t20 raw")
    c0_comp = np.where(ok_comp, t_comp, tan_args)
    h_B, c_B = run_model(str(pkg_t), c0_comp, "P_t20 comp")
    h_C, c_C = run_model(str(pkg_m), c0_comp, "P_m075 comp")
    c_expected = np.where(ok_comp, tan_args, np.float16(np.nan))
    c_match = (c_B == c_expected) | ~ok_comp
    out["c_argument_verification"] = {
        "cB_matches_target": int(c_match.sum()), "of": int(H),
        "cA_equals_round16_Kt": int((c_A == (tan_args * K)).sum()),
    }
    print("c verify:", json.dumps(out["c_argument_verification"]), flush=True)

    # candidate intersection: {u: round16(K*u)==hB} ∩ {u: round16(m*u)==hC}
    prodK = grid * K
    prodM = grid * mstar
    t_val = np.full(H, np.nan, np.float32)
    amb, m32t, m64t = [], [], []
    for j in range(H):
        if not ok_comp[j]:
            amb.append((j, float(tan_args[j]), "no-preimage"))
            continue
        sB = grid[prodK == h_B[j]]
        sC = grid[prodM == h_C[j]]
        inter = np.intersect1d(sB, sC)
        if inter.size == 0:
            amb.append((j, float(tan_args[j]), "empty-intersection"))
            continue
        if inter.size > 1:
            vals = sorted({float(cr(np.array([u]), "tanh", 32)[0]) for u in inter})
            if len(vals) > 1:
                amb.append((j, float(tan_args[j]), vals))
                continue
        u = inter[0]
        t_val[j] = float(u)
        a = tan_args[j]
        if u != cr(np.array([a]), "tanh", 32)[0]:
            m32t.append((int(j), float(a), float(u), float(cr(np.array([a]), "tanh", 32)[0])))
        if u != cr(np.array([a]), "tanh", 64)[0]:
            m64t.append((int(j), float(a), float(u), float(cr(np.array([a]), "tanh", 64)[0])))
    zlane = int(np.flatnonzero(tan_args == 0.0)[0])
    nzlane = int(np.flatnonzero(np.signbit(tan_args) & (tan_args == 0.0))[0])
    out["tanh_extraction"] = {
        "lanes": H,
        "resolved_unique": int(np.isfinite(t_val).sum()),
        "ambiguous": amb[:40], "ambiguous_count": len(amb),
        "mismatch_count_vs_cr_fp32": len(m32t), "mismatch_vs_cr_fp32": m32t[:40],
        "mismatch_count_vs_cr_fp64": len(m64t), "mismatch_vs_cr_fp64": m64t[:40],
        "zero_lane_value": float(t_val[zlane]),
        "negative_zero_lane_value": float(t_val[nzlane]),
        "negative_zero_signbit": bool(np.signbit(np.float16(t_val[nzlane]))) if np.isfinite(t_val[nzlane]) else None,
    }
    print("tanh v2:", json.dumps({k: v for k, v in out["tanh_extraction"].items()
                                  if "count" in k or "zero" in k or k == "resolved_unique"}), flush=True)
    np.savez(OUT / "tanh_extracted.npz", args=tan_args, tanh=t_val,
             tanh_cell_gate=t_val[:640], pads=t_val[640:660],
             grid=t_val[660:], ok_comp=ok_comp)
    np.savez(OUT / "tanh_raw_runs.npz", h_A=h_A, h_B=h_B, h_C=h_C,
             c_A=c_A, c_B=c_B, c_C=c_C, c0_comp=c0_comp, tan_args=tan_args)

    out["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    (OUT / "results_v2.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("WROTE", OUT / "results_v2.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

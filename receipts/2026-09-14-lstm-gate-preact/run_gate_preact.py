# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Gate-preact capture probe v6 (macstudio, CPU_ONLY = BNNS).

Question: does any Core ML decomposition expose the fused ios18.lstm
pre-gate adds bit-exactly? Method:
  1. fused probe: real decoder weights + native trace inputs through
     mb.lstm; PIN against the capture's native next_hidden/next_cell.
  2. decomposition twins (wide fp32 body, single-rounding exposure forms
     from receipts/2026-09-14-lstm-fused-impl2.md items 3-4): expose the
     pre-gate adds zx/zh/z while also computing h1/c1 with the pinned
     algebra. A twin reproducing the fused op's h1/c1 bit-exactly on
     every lane validates its exposed z as the native preact.
Twins:
  A sep_biafter : z = (x@wihT + h@whhT) + b          (decoder op order)
  B_packed      : z = [x h]@[wih|whh]T + b
  C_biasingemm  : z = [x h 1]@[wih|whh|b]T
  D_hidfirst    : z = (h@whhT + x@wihT) + b
  E_fp16_perop  : decoder order in pure fp16 ops (two-rounding cell,
                  negative control)
Writes only into ./out6/.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
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
OUT = HERE / "out6"
OUT.mkdir(exist_ok=True)
HID = 640


def sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def load_case_data():
    w = np.load(HERE / "weights.npz")
    tr = np.load(HERE / "traces.npz")
    idxs = sorted({int(k[1:5]) for k in tr.files})
    cases = []
    for i in idxs:
        x0 = tr[f"t{i:04d}_x0"]
        h2, c2 = tr[f"t{i:04d}_h"], tr[f"t{i:04d}_c"]
        nh2, nc2 = tr[f"t{i:04d}_nh"], tr[f"t{i:04d}_nc"]
        cases.append({
            "index": i,
            "L0": {"x": x0, "h": h2[0:1], "c": c2[0:1], "nh": nh2[0:1], "nc": nc2[0:1]},
            "L1n": {"x": nh2[0:1], "h": h2[1:2], "c": c2[1:2], "nh": nh2[1:2], "nc": nc2[1:2]},
        })
    weights = {
        "L0": (w["concat_1_to_fp16"], w["concat_2_to_fp16"], w["concat_0_to_fp16"]),
        "L1": (w["concat_4_to_fp16"], w["concat_5_to_fp16"], w["concat_3_to_fp16"]),
    }
    return cases, weights


def build_fused(wih, whh, b, tag):
    wih = np.ascontiguousarray(wih)
    whh = np.ascontiguousarray(whh)
    b = np.ascontiguousarray(b)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 1, HID), dtype=types.fp16),
            mb.TensorSpec(shape=(1, HID), dtype=types.fp16),
            mb.TensorSpec(shape=(1, HID), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x, initial_h, initial_c):
        r = mb.lstm(
            x=x, initial_h=initial_h, initial_c=initial_c,
            weight_ih=wih, weight_hh=whh, bias=b,
            direction="forward", output_sequence=False,
            recurrent_activation="sigmoid", cell_activation="tanh",
            activation="tanh", name="probe_lstm",
        )
        return r[1], r[2]

    return convert_pkg(prog, f"fused_{tag}")


def _wide_body(x, h, c, wih_t, whh_t, bias, order):
    """fp32 body; single-rounding exposures per impl2 items 3-4."""
    wih_t = np.asarray(wih_t, np.float32)
    whh_t = np.asarray(whh_t, np.float32)
    bias32 = np.asarray(bias, np.float32)
    zx = mb.matmul(x=x, y=wih_t, name="zx")
    zh = mb.matmul(x=h, y=whh_t, name="zh")
    if order == "hidfirst":
        z = mb.add(x=mb.add(x=zh, y=zx, name="z_sum"), y=bias32, name="z")
    elif order == "biasingemm":
        ones = np.ones((1, 1), np.float32)
        s = mb.concat(values=[x, h, ones], axis=1, name="s")
        wcat = np.concatenate([wih_t, whh_t, bias32[None, :]], axis=0).astype(np.float32)
        z = mb.matmul(x=s, y=wcat, name="z")
    elif order == "packed":
        wcat = np.concatenate([wih_t, whh_t], axis=0).astype(np.float32)
        s = mb.concat(values=[x, h], axis=1, name="s")
        z = mb.add(x=mb.matmul(x=s, y=wcat, name="z_nb"), y=bias32, name="z")
    else:  # sep (biafter)
        z = mb.add(x=mb.add(x=zx, y=zh, name="z_sum"), y=bias32, name="z")
    z16 = mb.cast(x=z, dtype="fp16", name="z16")
    zi, zf, zo, zg = mb.split(num_splits=4, axis=1, x=z, name="z_split")
    gi = mb.cast(x=mb.sigmoid(x=zi, name="sig_i"), dtype="fp16", name="gi")
    gf = mb.cast(x=mb.sigmoid(x=zf, name="sig_f"), dtype="fp16", name="gf")
    go = mb.cast(x=mb.sigmoid(x=zo, name="sig_o"), dtype="fp16", name="go")
    gg = mb.cast(x=mb.tanh(x=zg, name="tanh_g"), dtype="fp16", name="gg")
    gi_w = mb.cast(x=gi, dtype="fp32", name="gi_w")
    gf_w = mb.cast(x=gf, dtype="fp32", name="gf_w")
    go_w = mb.cast(x=go, dtype="fp32", name="go_w")
    gg_w = mb.cast(x=gg, dtype="fp32", name="gg_w")
    c32 = mb.add(x=mb.mul(x=gf_w, y=c, name="fc"),
                 y=mb.mul(x=gi_w, y=gg_w, name="ig"), name="c32")
    c1 = mb.cast(x=c32, dtype="fp16", name="c1")
    t16 = mb.cast(x=mb.tanh(x=mb.cast(x=c1, dtype="fp32", name="c1_32"), name="tanh_c1"),
                  dtype="fp16", name="t_c1")
    h1 = mb.cast(x=mb.mul(x=go_w, y=mb.cast(x=t16, dtype="fp32", name="t_c1_32"),
                          name="h32"), dtype="fp16", name="h1")
    return zx, zh, z, z16, gi, gf, go, gg, h1, c1


def build_twin(wih, whh, b, order, tag):
    wih_t = np.ascontiguousarray(np.asarray(wih, np.float16).T)
    whh_t = np.ascontiguousarray(np.asarray(whh, np.float16).T)
    bias = np.asarray(b, np.float16)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, HID), dtype=types.fp32),
            mb.TensorSpec(shape=(1, HID), dtype=types.fp32),
            mb.TensorSpec(shape=(1, HID), dtype=types.fp32),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x, h, c):
        return _wide_body(x, h, c, wih_t, whh_t, bias, order)

    return convert_pkg(prog, f"twin_{tag}", precision=ct.precision.FLOAT32)


def build_fp16_twin(wih, whh, b, tag):
    wih_t = np.ascontiguousarray(np.asarray(wih, np.float16).T)
    whh_t = np.ascontiguousarray(np.asarray(whh, np.float16).T)
    bias = np.asarray(b, np.float16)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, HID), dtype=types.fp16),
            mb.TensorSpec(shape=(1, HID), dtype=types.fp16),
            mb.TensorSpec(shape=(1, HID), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x, h, c):
        zx = mb.matmul(x=x, y=wih_t, name="zx")
        zh = mb.matmul(x=h, y=whh_t, name="zh")
        z = mb.add(x=mb.add(x=zx, y=zh, name="z_sum"), y=bias, name="z")
        zi, zf, zo, zg = mb.split(num_splits=4, axis=1, x=z, name="z_split")
        gi, gf, go = mb.sigmoid(x=zi), mb.sigmoid(x=zf), mb.sigmoid(x=zo)
        gg = mb.tanh(x=zg)
        fc = mb.mul(x=gf, y=c, name="fc")
        ig = mb.mul(x=gi, y=gg, name="ig")
        c1 = mb.add(x=fc, y=ig, name="c1")
        h1 = mb.mul(x=go, y=mb.tanh(x=c1, name="tanh_c1"), name="h1")
        return z, zi, zf, zo, zg, h1, c1

    return convert_pkg(prog, f"twin_{tag}")


def convert_pkg(prog, tag, precision=ct.precision.FLOAT16):
    pkg = OUT / f"{tag}.mlpackage"
    model = ct.convert(
        prog, convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    model.save(str(pkg))
    return pkg


def plan_rows(pkg: Path):
    compiled = ct.models.utils.compile_model(str(pkg))
    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_ONLY)
    rows = []
    for fn_name, fn in plan.model_structure.program.functions.items():
        for op in fn.block.operations:
            name = getattr(op, "operator_name", None) or op.type
            if name == "const":
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            rows.append({"op": name,
                         "preferred": type(usage.preferred_compute_device).__name__ if usage else None})
    shutil.rmtree(compiled, ignore_errors=True)
    return rows


def main() -> int:
    t0 = time.perf_counter()
    cases, weights = load_case_data()
    out = {
        "schema": "mlx-omarchy.fused-lstm-gate-preact/6",
        "host": {
            "hostname": platform.node(),
            "hw_model": sysctl("hw.model"),
            "chip": sysctl("machdep.cpu.brand_string"),
            "macos": subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip(),
            "coremltools": ct.__version__, "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }
    print("host", json.dumps(out["host"]), flush=True)

    layers = {"L0": "L0", "L1n": "L1"}

    # ---- fused pin -------------------------------------------------------- #
    fused_pkg = {lay: build_fused(*weights[wl], lay) for lay, wl in layers.items()}
    out["fused_plan"] = plan_rows(fused_pkg["L0"])
    print("fused plan:", json.dumps(out["fused_plan"]), flush=True)

    results = {}
    pin = {}
    for lay in layers:
        okh = okc = 0
        fh_all, fc_all = [], []
        m = MLModel(str(fused_pkg[lay]), compute_units=ct.ComputeUnit.CPU_ONLY)
        for case in cases:
            cc = case[lay]
            feed = {"x": cc["x"].reshape(1, 1, HID),
                    "initial_h": cc["h"], "initial_c": cc["c"]}
            y = m.predict(feed)
            y2 = m.predict(feed)
            keys = list(y.keys())
            h1 = np.asarray(y[keys[0]], np.float32).astype(np.float16).ravel()
            c1 = np.asarray(y[keys[1]], np.float32).astype(np.float16).ravel()
            h2 = np.asarray(y2[keys[0]], np.float32).astype(np.float16).ravel()
            c2 = np.asarray(y2[keys[1]], np.float32).astype(np.float16).ravel()
            assert np.array_equal(h1, h2) and np.array_equal(c1, c2), "fused nondeterministic"
            okh += int(np.array_equal(h1, cc["nh"].ravel()))
            okc += int(np.array_equal(c1, cc["nc"].ravel()))
            fh_all.append(h1)
            fc_all.append(c1)
        results[lay] = {"h1": fh_all, "c1": fc_all}
        pin[lay] = {"hidden_bitexact": okh, "cell_bitexact": okc, "cases": len(cases)}
        print(f"pin {lay}:", json.dumps(pin[lay]), flush=True)
    out["pin"] = pin

    # ---- twins ------------------------------------------------------------ #
    twin_defs = [
        ("A_sep_biafter", "sep"),
        ("B_packed", "packed"),
        ("C_biasingemm", "biasingemm"),
        ("D_hidfirst", "hidfirst"),
    ]
    for vname, order in twin_defs:
        match_h = match_c = lanes = 0
        zb = {k: [] for k in ("z", "z16")}
        for lay, wl in layers.items():
            pkg = build_twin(*weights[wl], order, f"{vname}_{lay}")
            for ci, case in enumerate(cases):
                cc = case[lay]
                feed = {"x": cc["x"].astype(np.float32),
                        "h": cc["h"].astype(np.float32),
                        "c": cc["c"].astype(np.float32)}
                y = MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY).predict(feed)
                th = np.asarray(y["h1"], np.float32).astype(np.float16).ravel()
                tc = np.asarray(y["c1"], np.float32).astype(np.float16).ravel()
                match_h += int(np.array_equal(th, results[lay]["h1"][ci]))
                match_c += int(np.array_equal(tc, results[lay]["c1"][ci]))
                lanes += th.size
                for k in ("z", "z16"):
                    zb[k].append(np.asarray(y[k], np.float32))
            shutil.rmtree(pkg, ignore_errors=True)
        out[vname] = {"hidden_bitexact": match_h, "cell_bitexact": match_c, "lanes": lanes}
        print(vname, json.dumps(out[vname]), flush=True)
        np.savez(OUT / f"z_{vname}.npz",
                 **{k: np.concatenate(v, axis=0) for k, v in zb.items()})
    # fp16 per-op negative control
    vname = "E_fp16_perop"
    match_h = match_c = lanes = 0
    for lay, wl in layers.items():
        pkg = build_fp16_twin(*weights[wl], f"{vname}_{lay}")
        for ci, case in enumerate(cases):
            cc = case[lay]
            feed = {"x": cc["x"], "h": cc["h"], "c": cc["c"]}
            y = MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY).predict(feed)
            th = np.asarray(y["h1"], np.float32).astype(np.float16).ravel()
            tc = np.asarray(y["c1"], np.float32).astype(np.float16).ravel()
            match_h += int(np.array_equal(th, results[lay]["h1"][ci]))
            match_c += int(np.array_equal(tc, results[lay]["c1"][ci]))
            lanes += th.size
        shutil.rmtree(pkg, ignore_errors=True)
    out[vname] = {"hidden_bitexact": match_h, "cell_bitexact": match_c, "lanes": lanes}
    print(vname, json.dumps(out[vname]), flush=True)

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "results_v6.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("WROTE", OUT / "results_v6.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

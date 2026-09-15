# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Gate-preact capture probe v7 (macstudio, CPU_ONLY = BNNS).

Captures native fused ios18.lstm gate values through the fused op itself
(no decomposition). For each target gate T in {i,f,o,g}, one gate at a
time carries the real native-input trace (zeroed rows elsewhere, other
gates bias-saturated to known constants); the target gate value is read
exactly from the directly exposed cell output (i/f/g) or the hidden
product (o):
  o-read: h = round16(sigma16(z_o) * tanh16(c0)),  c1 == c0 asserted
  f-read: c1 = round16(sigma16(z_f) * c0)
  i-read: c1 = round16(sigma16(z_i) * K),  K = tanh16(b_g)
  g-read: c1 = round16(iK * tanh16(z_g)),  iK = sigma16(b_i)
Multiplier sweeps (3 settings per target) disambiguate candidates.
Writes only into ./out7/.
"""
from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
OUT = HERE / "out7"
OUT.mkdir(exist_ok=True)
HID = 640
N_BATCH = 16
GATE_OFF = {"i": 0, "f": HID, "o": 2 * HID, "g": 3 * HID}

# saturation biases and multiplier settings (fp16-exact values)
SAT = {"i0": -40.0, "f1": 20.0, "o1": 20.0, "gn": -20.0}
O_C0 = [0.25, 0.5, 1.0]          # tanh16 multipliers for o-read
F_C0 = [0.25, 0.5, 1.0]          # c0 multipliers for f-read
I_BG = [0.5, 1.0, 2.0]           # K = tanh16(b_g) for i-read (c0 = 0)
G_BI = [1.0, 2.0, 4.0]           # iK = sigma16(b_i) for g-read (c0 = 0)


def sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def load_cases():
    w = np.load(HERE / "weights.npz")
    tr = np.load(HERE / "traces.npz")
    idxs = sorted({int(k[1:5]) for k in tr.files})
    layers = {}
    for lay, names in (("L0", ("concat_1_to_fp16", "concat_2_to_fp16", "concat_0_to_fp16")),
                       ("L1n", ("concat_4_to_fp16", "concat_5_to_fp16", "concat_3_to_fp16"))):
        wih, whh, b = (np.asarray(w[n], np.float16) for n in names)
        rows = []
        for i in idxs:
            x = tr[f"t{i:04d}_x0"]
            h2, c2 = tr[f"t{i:04d}_h"], tr[f"t{i:04d}_c"]
            r = {"index": i}
            if lay == "L0":
                r.update(x=x, h=h2[0:1], c=c2[0:1])
            else:
                r.update(x=tr[f"t{i:04d}_nh"][0:1], h=h2[1:2], c=c2[1:2])
            rows.append(r)
        layers[lay] = {"wih": wih, "whh": whh, "b": b, "cases": rows}
    return layers


def build_model(wih, whh, b, target, setting, tag):
    wih = np.ascontiguousarray(wih, dtype=np.float16)
    whh = np.ascontiguousarray(whh, dtype=np.float16)
    bias = np.empty(4 * HID, np.float16)
    wi = np.zeros_like(wih)
    wh = np.zeros_like(whh)
    t0 = GATE_OFF[target]
    wi[t0:t0 + HID] = wih[t0:t0 + HID]
    wh[t0:t0 + HID] = whh[t0:t0 + HID]
    bias[:] = 0.0
    bias[GATE_OFF["i"]:GATE_OFF["i"] + HID] = SAT["i0"]
    bias[GATE_OFF["f"]:GATE_OFF["f"] + HID] = SAT["f1"]
    bias[GATE_OFF["o"]:GATE_OFF["o"] + HID] = SAT["o1"]
    bias[GATE_OFF["g"]:GATE_OFF["g"] + HID] = SAT["gn"]
    if target == "o":
        c0v = setting
        bias[t0:t0 + HID] = b[t0:t0 + HID]
    elif target == "f":
        c0v = setting
        bias[t0:t0 + HID] = b[t0:t0 + HID]
    elif target == "i":
        c0v = 0.0
        bias[GATE_OFF["g"]:GATE_OFF["g"] + HID] = setting   # K = tanh16(b_g)
        bias[t0:t0 + HID] = b[t0:t0 + HID]
    else:  # g
        c0v = 0.0
        bias[GATE_OFF["i"]:GATE_OFF["i"] + HID] = setting   # iK = sigma16(b_i)
        bias[t0:t0 + HID] = b[t0:t0 + HID]

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
            weight_ih=wi, weight_hh=wh, bias=bias,
            direction="forward", output_sequence=False,
            recurrent_activation="sigmoid", cell_activation="tanh",
            activation="tanh", name="probe_lstm",
        )
        return r[1], r[2]

    pkg = OUT / f"m_{tag}.mlpackage"
    model = ct.convert(
        prog, convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    model.save(str(pkg))
    return pkg, c0v


def main() -> int:
    t0w = time.perf_counter()
    layers = load_cases()
    out = {
        "schema": "mlx-omarchy.fused-lstm-gate-preact/7",
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

    plans = {lay: {} for lay in layers}
    for lay, L in layers.items():
        n = len(L["cases"])
        xs = np.concatenate([c["x"] for c in L["cases"]] +
                            [np.zeros((1, HID), np.float16)]).reshape(n + 1, 1, HID)
        hs = np.concatenate([c["h"] for c in L["cases"]] +
                            [np.zeros((1, HID), np.float16)]).reshape(n + 1, HID)
        for target, settings in (("o", O_C0), ("f", F_C0), ("i", I_BG), ("g", G_BI)):
            for setting in settings:
                tag = f"{lay}_{target}_{setting}"
                pkg, c0v = build_model(L["wih"], L["whh"], L["b"], target, setting, tag)
                cs = np.full((n + 1, HID), c0v, np.float16)
                m = MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
                keys = None
                h = np.zeros((n + 1, HID), np.float16)
                c = np.zeros((n + 1, HID), np.float16)
                for r in range(n + 1):
                    feed = {"x": xs[r:r + 1], "initial_h": hs[r:r + 1],
                            "initial_c": cs[r:r + 1]}
                    y = m.predict(feed)
                    y2 = m.predict(feed)
                    if keys is None:
                        keys = list(y.keys())
                    h[r] = np.asarray(y[keys[0]], np.float32).astype(np.float16).ravel()
                    c[r] = np.asarray(y[keys[1]], np.float32).astype(np.float16).ravel()
                    h2v = np.asarray(y2[keys[0]], np.float32).astype(np.float16).ravel()
                    c2v = np.asarray(y2[keys[1]], np.float32).astype(np.float16).ravel()
                    assert np.array_equal(h[r], h2v) and np.array_equal(c[r], c2v), f"{tag} nondet r{r}"
                if target == "o":
                    assert np.array_equal(c, cs), f"{tag}: cell identity broken"
                np.savez(OUT / f"obs_{tag}.npz", h=h, c=c, c0=cs, setting=c0v)
                shutil.rmtree(pkg, ignore_errors=True)
                plans[lay][f"{target}_{setting}"] = {"c0": float(c0v), "cases": n + 1}
                print(f"ran {tag}", flush=True)
    out["plans"] = plans
    out["elapsed_s"] = round(time.perf_counter() - t0w, 1)
    (OUT / "results_v7.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("WROTE", OUT / "results_v7.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

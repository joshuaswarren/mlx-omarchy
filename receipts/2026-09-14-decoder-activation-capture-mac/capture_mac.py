# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Run on the macOS reference host (macstudio, Apple M1 Ultra, macOS 26.6.2).

1. Build one ML Program with a single ``sigmoid`` and a single ``tanh`` over an
   fp16 input of 1536 lanes, run it under every Core ML compute-unit setting on the
   exact fp16 fixture the jwm1 H13 run used (``x_{sigmoid,tanh}_{0,1,2}.bin``), and
   save each unit's fp16 outputs.
2. Load the pinned decoder.mlpackage and record the MLComputePlan device selection
   per operation under every compute-unit setting.
3. Run the pinned decoder on the captured transition-0 inputs under every
   compute-unit setting and save the outputs, so the unit that reproduces the
   authenticated capture is measured rather than inferred.

Writes only into the directory it is run from (a temp dir). Reads the decoder
package and the copied fixtures; changes nothing else on the host.
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
from coremltools.models.compute_plan import MLComputePlan

HERE = Path.cwd()
F16 = np.dtype("<f2")
UNITS = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}
LANES = 1536


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def compute_plan(compiled: str, unit: ct.ComputeUnit) -> list[dict]:
    plan = MLComputePlan.load_from_path(compiled, compute_units=unit)
    rows = []
    for fn_name, fn in plan.model_structure.program.functions.items():
        for op in fn.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            rows.append(
                {
                    "function": fn_name,
                    "op": op.operator_name,
                    "outputs": [o.name for o in op.outputs],
                    "preferred": type(usage.preferred_compute_device).__name__ if usage else None,
                    "supported": [type(d).__name__ for d in usage.supported_compute_devices] if usage else None,
                    "weight": cost.weight if cost else None,
                }
            )
    return rows


def fp16_exact(arr: np.ndarray) -> bool:
    a = np.asarray(arr, np.float32)
    return bool(np.array_equal(a.astype(F16).astype(np.float32), a))


def main() -> int:
    decoder = Path(sys.argv[1]).expanduser()
    out = {
        "schema": "mlx-omarchy.parakeet-decoder-activation-capture-mac/1",
        "host": {
            "hostname": platform.node(),
            "hw_model": sysctl("hw.model"),
            "chip": sysctl("machdep.cpu.brand_string"),
            "macos": subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip(),
            "build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
            "coreml_framework": subprocess.run(
                ["plutil", "-extract", "CFBundleVersion", "raw", "/System/Library/Frameworks/CoreML.framework/Versions/A/Resources/Info.plist"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "coremltools": ct.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "decoder_package": str(decoder),
        "decoder_sha256": {
            n: sha256(decoder / n)
            for n in ("Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin", "Manifest.json")
        },
        "fixture_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("x_*.bin")) + sorted(HERE.glob("tdt_trace_0000_decoder_*.npy"))},
    }

    # ---- 1. one-op sigmoid and tanh per compute unit ---------------------------- #
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, LANES), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def prog(x):
        s = mb.sigmoid(x=x, name="sigmoid_out")
        t = mb.tanh(x=x, name="tanh_out")
        return s, t

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    oneop = HERE / "oneop.mlpackage"
    model.save(str(oneop))
    mil_text = model.get_spec().mlProgram.functions["main"].block_specializations
    out["oneop"] = {
        "spec_version": model.get_spec().specificationVersion,
        "block_specializations": list(mil_text.keys()),
        "ops": [op.type for op in next(iter(mil_text.values())).operations],
        "package_sha256": sha256(oneop / "Data/com.apple.CoreML/model.mlmodel"),
    }
    compiled_oneop = ct.models.utils.compile_model(str(oneop))
    out["oneop"]["compute_plan"] = {name: compute_plan(compiled_oneop, unit) for name, unit in UNITS.items()}

    X = {op: np.concatenate([np.fromfile(HERE / f"x_{op}_{p}.bin", F16) for p in range(3)]) for op in ("sigmoid", "tanh")}
    assert X["sigmoid"].size == LANES and X["tanh"].size == LANES
    out["oneop"]["runs"] = {}
    for name, unit in UNITS.items():
        m = ct.models.MLModel(str(oneop), compute_units=unit)
        rec = {}
        for op in ("sigmoid", "tanh"):
            x = X[op].reshape(1, LANES)
            t0 = time.perf_counter()
            y = m.predict({"x": x})
            dt = time.perf_counter() - t0
            y16 = np.asarray(y[f"{op}_out"], np.float32)
            (HERE / f"y_{name}_{op}.bin").write_bytes(y16.astype(F16).tobytes())
            # second run: determinism
            y2 = np.asarray(m.predict({"x": x})[f"{op}_out"], np.float32)
            rec[op] = {
                "output_dtype": str(np.asarray(y[f"{op}_out"]).dtype),
                "fp16_exact": fp16_exact(y16),
                "finite": bool(np.isfinite(y16).all()),
                "repeat_identical": bool(np.array_equal(y16, y2)),
                "wall_s": round(dt, 4),
                "sha256": hashlib.sha256(y16.astype(F16).tobytes()).hexdigest(),
            }
        out["oneop"]["runs"][name] = rec
        print(name, json.dumps(rec), flush=True)

    # ---- 2. decoder compute plan ------------------------------------------------- #
    compiled_dec = ct.models.utils.compile_model(str(decoder))
    out["decoder_compute_plan"] = {name: compute_plan(compiled_dec, unit) for name, unit in UNITS.items()}

    # ---- 3. decoder on the captured transition-0 inputs per compute unit -------- #
    ids = np.load(HERE / "tdt_trace_0000_decoder_input_ids.npy").astype(np.int32)
    hidden = np.load(HERE / "tdt_trace_0000_decoder_hidden.npy").astype(np.float32)
    cell = np.load(HERE / "tdt_trace_0000_decoder_cell.npy").astype(np.float32)
    native = {k: np.load(HERE / f"tdt_trace_0000_decoder_{k}.npy").astype(np.float32) for k in ("next_cell", "next_hidden", "output_hidden")}
    spec = ct.models.MLModel(str(decoder), compute_units=ct.ComputeUnit.CPU_ONLY).get_spec()
    in_names = [i.name for i in spec.description.input]
    out_names = [o.name for o in spec.description.output]
    out["decoder_io"] = {"inputs": in_names, "outputs": out_names}
    feed = {in_names[0]: ids, in_names[1]: hidden, in_names[2]: cell}
    out["decoder_runs"] = {}
    for name, unit in UNITS.items():
        m = ct.models.MLModel(str(decoder), compute_units=unit)
        y = m.predict(feed)
        y2 = m.predict(feed)
        rec = {}
        for oname in out_names:
            a = np.asarray(y[oname], np.float32)
            np.save(HERE / f"dec_{name}_{oname}.npy", a)
            key = {"next_cell": "next_cell", "next_hidden": "next_hidden"}.get(oname.replace("decoder_", ""), None)
            if key is None:
                key = "output_hidden" if "hidden" in oname and a.shape[0] == 1 else oname
            ref = native.get(key)
            rec[oname] = {
                "shape": list(a.shape),
                "fp16_exact": fp16_exact(a),
                "repeat_identical": bool(np.array_equal(a, np.asarray(y2[oname], np.float32))),
                "equal_to_capture": int(np.count_nonzero(a == ref)) if ref is not None and ref.shape == a.shape else None,
                "of": int(a.size),
                "max_abs_diff": float(np.abs(a - ref).max()) if ref is not None and ref.shape == a.shape else None,
                "compared_against": key,
            }
        out["decoder_runs"][name] = rec
        print(name, json.dumps(rec), flush=True)

    (HERE / "mac_result.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

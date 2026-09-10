#!/usr/bin/env python3
"""GPU probe (bounded, seconds): inject captured decode/prefill q/k/v inputs
into layer-0 Linears on the Linux Vulkan wheel; compare against
(a) native Metal capture bits, (b) RNE(f64 truth), (c) CPU mlx result.
Records wheel provenance. No source changes."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLX_DISABLE_COMPILE"] = "1"
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load

mx.set_default_device(mx.gpu)
MODEL = Path.home() / "models/Qwen2.5-0.5B-Instruct-bf16-mlx"
CAPTURE = Path("/tmp/bf16chain3-native-fixed/import-2/capture")


def to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def rne_bf16(a):
    b = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def tensor(name):
    meta = json.loads((CAPTURE / "capture.json").read_text())["tensors"][name]
    data = np.load(CAPTURE / meta["path"])
    return np.asarray(data, dtype=np.uint16)


model, _ = load(str(MODEL))
layer = model.model.layers[0]

wheel = subprocess.run(
    [str(Path.home() / "venv-bf16chain3/bin/python"), "-m", "pip", "show", "mlx-omarchy"],
    capture_output=True, text=True).stdout
version = next((l.split(": ", 1)[1].strip() for l in wheel.splitlines() if l.startswith("Version")), "?")

print(json.dumps({
    "mlx_version": mx.__version__,
    "wheel_version": version,
    "device": mx.device_info().get("device_name"),
    "cooperative_matrix_f32_8": mx.device_info().get("cooperative_matrix_f32_8"),
}), flush=True)

results = []
for phase in ("decode", "prefill"):
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        name = f"{phase}.{proj}"
        xb = tensor(name + ".input")
        native = tensor(name + ".output").reshape(-1)
        lin = getattr(layer.self_attn, proj)
        Wb = np.asarray(lin.weight.view(mx.uint16), dtype=np.uint16)
        xf = to_f32(xb).astype(np.float64).reshape(-1, xb.shape[-1])
        Wf = to_f32(Wb).astype(np.float64)
        acc = xf @ Wf.T
        if getattr(lin, "bias", None) is not None:
            bb = np.asarray(lin.bias.astype(mx.bfloat16).view(mx.uint16), dtype=np.uint16)
            acc = acc + to_f32(bb).astype(np.float64)
        truth = rne_bf16(acc).reshape(-1)

        # CPU reference in the same model object
        cpu_lin = lin
        cpu_model = model  # device switch per-call not supported; use separate small calc
        # Run on CPU via a throwaway copy of the op graph
        import mlx.core as mxc
        saved_default = mx.default_device()

        def cpu_bits():
            x = mx.array(np.uint16(xb)).view(mx.bfloat16)
            w = lin.weight
            b = lin.bias if getattr(lin, "bias", None) is not None else None
            # bf16 matmul on CPU via mx on cpu stream
            with mx.stream(mx.cpu):
                out = x @ w.T
                if b is not None:
                    out = out + b
                mx.eval(out)
            return np.asarray(out.view(mx.uint16), dtype=np.uint16).reshape(-1)

        cpu = cpu_bits()

        xg = mx.array(np.uint16(xb)).view(mx.bfloat16)
        out = lin(xg)
        mx.eval(out)
        gpu = np.asarray(out.view(mx.uint16), dtype=np.uint16).reshape(-1)

        rec = {
            "op": name,
            "gpu_vs_native_mismatch": int((gpu != native).sum()),
            "gpu_vs_native_max_bit_delta": int(np.abs(gpu.astype(np.int64) - native.astype(np.int64)).max()),
            "gpu_vs_truth_mismatch": int((gpu != truth).sum()),
            "gpu_vs_truth_max_bit_delta": int(np.abs(gpu.astype(np.int64) - truth.astype(np.int64)).max()),
            "gpu_vs_cpu_mismatch": int((gpu != cpu).sum()),
            "gpu_vs_cpu_max_bit_delta": int(np.abs(gpu.astype(np.int64) - cpu.astype(np.int64)).max()),
            "cpu_vs_truth_mismatch": int((cpu != truth).sum()),
            "native_vs_truth_max_bit_delta": int(np.abs(native.astype(np.int64) - truth.astype(np.int64)).max()),
            "elements": int(gpu.size),
        }
        print(json.dumps(rec), flush=True)
        results.append(rec)

Path("/tmp/bf16-rootcause/gpu_probe.json").write_text(json.dumps(results, indent=2) + "\n")
print("DONE")

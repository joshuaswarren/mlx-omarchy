#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLX_DISABLE_COMPILE"] = "1"

import mlx.core as mx
import numpy as np
from mlx_lm.models.qwen2 import swiglu
from mlx_lm.utils import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model_path = Path(args.model).resolve()
    native = Path(args.native)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capture = json.loads((native / "capture.json").read_text())
    model_config_sha256 = hashlib.sha256(
        (model_path / "config.json").read_bytes()
    ).hexdigest()
    if capture["model_config_sha256"] != model_config_sha256:
        raise RuntimeError(
            "model config mismatch: "
            f"{model_config_sha256} != {capture['model_config_sha256']}"
        )
    model, _ = load(str(model_path))
    layer = model.model.layers[0]

    def tensor(name):
        meta = capture["tensors"][name]
        data = np.load(native / meta["path"])
        value = mx.array(data)
        if meta["encoding"] == "raw-bfloat16-bits":
            value = value.view(mx.bfloat16)
        elif meta["encoding"] == "raw-float16-bits":
            value = value.view(mx.float16)
        return value

    def compare(name, result):
        expected_meta = capture["tensors"][f"{name}.output"]
        expected = np.load(native / expected_meta["path"])
        mx.eval(result)
        if expected_meta["encoding"] == "raw-bfloat16-bits":
            actual = np.array(result.view(mx.uint16), copy=True)
            delta = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
        elif expected_meta["encoding"] == "raw-float16-bits":
            actual = np.array(result.view(mx.uint16), copy=True)
            delta = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
        else:
            actual = np.array(result, copy=True)
            delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        if actual.shape != expected.shape:
            raise RuntimeError(f"{name}: shape {actual.shape} != {expected.shape}")
        mismatches = int(np.count_nonzero(actual != expected))
        record = {
            "op": name,
            "mismatches": mismatches,
            "elements": int(actual.size),
            "inputs": "native-injected",
            "max_raw_delta": float(delta.max(initial=0)),
        }
        print(json.dumps(record), flush=True)
        return record

    results = []
    for phase in ("prefill", "decode"):
        prefix = f"{phase}."
        results.append(
            compare(
                prefix + "input_norm",
                layer.input_layernorm(tensor(prefix + "input_norm.input")),
            )
        )
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = getattr(layer.self_attn, name)
            results.append(
                compare(prefix + name, projection(tensor(prefix + name + ".input")))
            )
        for name in ("rope_q", "rope_k"):
            offset = capture["operations"][prefix + name]["offset"]
            results.append(
                compare(
                    prefix + name,
                    layer.self_attn.rope(tensor(prefix + name + ".input"), offset=offset),
                )
            )
        mask_kind = capture["operations"][prefix + "attention"]["mask"]
        mask = (
            tensor(prefix + "attention.mask")
            if mask_kind == "tensor"
            else None if mask_kind == "none" else mask_kind
        )
        attention = mx.fast.scaled_dot_product_attention(
            tensor(prefix + "attention.query"),
            tensor(prefix + "attention.key"),
            tensor(prefix + "attention.value"),
            scale=capture["operations"][prefix + "attention"]["scale"],
            mask=mask,
        )
        results.append(compare(prefix + "attention", attention))
        results.append(
            compare(
                prefix + "o_proj",
                layer.self_attn.o_proj(tensor(prefix + "o_proj.input")),
            )
        )
        results.append(
            compare(
                prefix + "post_attention_norm",
                layer.post_attention_layernorm(
                    tensor(prefix + "post_attention_norm.input")
                ),
            )
        )
        for name in ("gate_proj", "up_proj"):
            projection = getattr(layer.mlp, name)
            results.append(
                compare(prefix + name, projection(tensor(prefix + name + ".input")))
            )
        results.append(
            compare(prefix + "sigmoid", mx.sigmoid(tensor(prefix + "sigmoid.input")))
        )
        results.append(
            compare(
                prefix + "swiglu",
                swiglu(
                    tensor(prefix + "swiglu.gate"), tensor(prefix + "swiglu.up")
                ),
            )
        )
        results.append(
            compare(
                prefix + "down_proj",
                layer.mlp.down_proj(tensor(prefix + "down_proj.input")),
            )
        )
        results.append(
            compare(
                prefix + "final_norm",
                model.model.norm(tensor(prefix + "final_norm.input")),
            )
        )
        results.append(
            compare(
                prefix + "lm_head",
                model.model.embed_tokens.as_linear(tensor(prefix + "lm_head.input")),
            )
        )

    provenance = {
        "source_commit": args.source_commit,
        "wheel_sha256": args.wheel_sha256,
        "wheel_version": mx.__version__,
        "device_name": mx.device_info().get("device_name"),
        "cooperative_matrix_f32_8": mx.device_info().get(
            "cooperative_matrix_f32_8"
        ),
        "VK_DRIVER_FILES": os.environ.get("VK_DRIVER_FILES"),
        "model_revision": model_path.name,
        "model_config_sha256": model_config_sha256,
        "native_capture": capture,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )
    print(json.dumps({"provenance": provenance, "results": results}, default=str))


if __name__ == "__main__":
    main()

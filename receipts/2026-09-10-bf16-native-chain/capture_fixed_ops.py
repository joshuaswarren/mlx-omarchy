#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLX_DISABLE_COMPILE"] = "1"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import mlx_lm.models.qwen2 as qwen2
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load


def prompt_tokens(tokenizer, text):
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True
    )
    if isinstance(tokens, str):
        add_special = not getattr(tokenizer, "bos_token", None) or not tokens.startswith(
            tokenizer.bos_token
        )
        tokens = tokenizer.encode(tokens, add_special_tokens=add_special)
    while tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model_path = Path(args.model).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    model, tokenizer = load(str(model_path))
    tokens = prompt_tokens(tokenizer, manifest["prompts"]["long"]["text"])
    if len(tokens) != 262:
        raise RuntimeError(f"expected 262 prompt tokens, got {len(tokens)}")

    active = {"phase": None, "rope_call": 0}
    tensors = {}
    op_meta = {}

    def save(name, value):
        mx.eval(value)
        if value.dtype == mx.bfloat16:
            data = np.array(value.view(mx.uint16), copy=True)
            encoding = "raw-bfloat16-bits"
        elif value.dtype == mx.float16:
            data = np.array(value.view(mx.uint16), copy=True)
            encoding = "raw-float16-bits"
        else:
            data = np.array(value, copy=True)
            encoding = f"numpy-{data.dtype}"
        path = out / f"{name.replace('.', '_')}.npy"
        np.save(path, data)
        tensors[name] = {
            "path": path.name,
            "shape": list(data.shape),
            "elements": int(data.size),
            "mlx_dtype": str(value.dtype),
            "encoding": encoding,
            "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        }

    class UnaryCapture(nn.Module):
        def __init__(self, inner, label):
            super().__init__()
            self.inner = inner
            self.label = label

        def __call__(self, value, *positional, **keywords):
            result = self.inner(value, *positional, **keywords)
            phase = active["phase"]
            if phase:
                save(f"{phase}.{self.label}.input", value)
                save(f"{phase}.{self.label}.output", result)
            return result

    class RopeCapture(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, value, *positional, **keywords):
            result = self.inner(value, *positional, **keywords)
            phase = active["phase"]
            if phase:
                label = "rope_q" if active["rope_call"] == 0 else "rope_k"
                active["rope_call"] += 1
                save(f"{phase}.{label}.input", value)
                save(f"{phase}.{label}.output", result)
                op_meta[f"{phase}.{label}"] = {
                    "offset": int(keywords.get("offset", 0))
                }
            return result

    layer = model.model.layers[0]
    layer.input_layernorm = UnaryCapture(layer.input_layernorm, "input_norm")
    layer.post_attention_layernorm = UnaryCapture(
        layer.post_attention_layernorm, "post_attention_norm"
    )
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(layer.self_attn, name, UnaryCapture(getattr(layer.self_attn, name), name))
    layer.self_attn.rope = RopeCapture(layer.self_attn.rope)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(layer.mlp, name, UnaryCapture(getattr(layer.mlp, name), name))
    model.model.norm = UnaryCapture(model.model.norm, "final_norm")

    original_sdpa = qwen2.scaled_dot_product_attention
    original_swiglu = qwen2.swiglu

    def capture_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        result = original_sdpa(
            queries, keys, values, cache, scale, mask, sinks=sinks
        )
        phase = active["phase"]
        if phase:
            save(f"{phase}.attention.query", queries)
            save(f"{phase}.attention.key", keys)
            save(f"{phase}.attention.value", values)
            mask_kind = "none"
            if isinstance(mask, str):
                mask_kind = mask
            elif mask is not None:
                save(f"{phase}.attention.mask", mask)
                mask_kind = "tensor"
            save(f"{phase}.attention.output", result)
            op_meta[f"{phase}.attention"] = {
                "scale": float(scale),
                "mask": mask_kind,
            }
        return result

    def capture_swiglu(gate, up):
        result = original_swiglu(gate, up)
        phase = active["phase"]
        if phase:
            sigmoid = mx.sigmoid(gate)
            silu = gate * sigmoid
            save(f"{phase}.sigmoid.input", gate)
            save(f"{phase}.sigmoid.output", sigmoid)
            save(f"{phase}.swiglu.gate", gate)
            save(f"{phase}.swiglu.up", up)
            save(f"{phase}.swiglu.silu", silu)
            save(f"{phase}.swiglu.output", result)
        return result

    qwen2.scaled_dot_product_attention = capture_sdpa
    qwen2.swiglu = capture_swiglu
    cache = make_prompt_cache(model)
    try:
        active["phase"] = "prefill"
        active["rope_call"] = 0
        prompt = mx.array(tokens, dtype=mx.int32)[None, :]
        logits = model(prompt, cache=cache)
        mx.eval(logits)
        save("prefill.lm_head.input", mx.array(np.load(out / "prefill_final_norm_output.npy")).view(mx.bfloat16))
        save("prefill.lm_head.output", logits)

        active["phase"] = "decode"
        active["rope_call"] = 0
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int32)
        decode_logits = model(next_token, cache=cache)
        mx.eval(decode_logits)
        save("decode.lm_head.input", mx.array(np.load(out / "decode_final_norm_output.npy")).view(mx.bfloat16))
        save("decode.lm_head.output", decode_logits)
    finally:
        qwen2.scaled_dot_product_attention = original_sdpa
        qwen2.swiglu = original_swiglu
        active["phase"] = None

    token_bytes = np.asarray(tokens, dtype=np.int32).tobytes()
    metadata = {
        "schema": "bf16-native-injected-ops/1",
        "platform": platform.platform(),
        "device": str(mx.default_device()),
        "device_info": mx.device_info(),
        "mlx_version": mx.__version__,
        "source_commit": args.source_commit,
        "model_revision": model_path.name,
        "model_config_sha256": hashlib.sha256(
            (model_path / "config.json").read_bytes()
        ).hexdigest(),
        "prompt_tokens": len(tokens),
        "prompt_token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "operations": op_meta,
        "tensors": tensors,
    }
    (out / "capture.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps(metadata, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

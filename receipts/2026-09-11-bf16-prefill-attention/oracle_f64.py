#!/usr/bin/env python3
"""Float64 oracle for the bf16 SDPA storage-precision move.

Three parts, one run on the M1:

1. Token streams (GPU, mlx): per-position argmax ids over the long (262)
   and ctx1024 (1053) prompts, with MLX_OMARCHY_SDPA_BF16_FAST unset
   (f32 composition) and =1 (bf16 composition). Upstream rounding is
   shared between the two runs, so stream differences isolate the
   attention composition.

2. f64 truth (CPU, numpy): the same model, same prompts, every weight
   lifted bf16->f64 exactly (bf16 bits are the top half of an f32), the
   whole forward in float64, per-position argmax. RoPE, RMS norm, softmax
   all f64.

3. Attention-block ULP (GPU + CPU): synthetic bf16 q/k/v at the real
   shapes (1, 14, m, 64), scale 1/8, m in {262, 1053}; GPU outputs under
   both compositions vs RNE(f64) attention truth. ULP distances use the
   root-cause receipt convention (rne via float32 bits, (b + 0x7FFF +
   ((b >> 16) & 1)) >> 16).

Writes NDJSON rows to --out and a summary to stdout.
"""

import argparse
import json
import os
import sys

import numpy as np

BF16_EXACT = np.uint32(0x7FFF)
ULP_FLOOR = 1e-6


def rne_bf16(a):
    b = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((b + BF16_EXACT + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_bits_to_f64(bits):
    bits = np.asarray(bits, dtype=np.uint32)
    return (bits << 16).view(np.float32).astype(np.float64)


def ordered_int16(bits):
    b = bits.astype(np.int32)
    return np.where(bits >= 0x8000, b - 0x10000, b)


def ulp_distance(bits, truth_bits, truth_f):
    keep = np.abs(truth_f) > ULP_FLOOR
    if not keep.any():
        return 0.0, 0.0, int(keep.sum())
    d = np.abs(
        ordered_int16(bits[keep]) - ordered_int16(truth_bits[keep])
    ).astype(np.float64)
    return float(d.mean()), float(d.max()), int(keep.sum())


def f64_attention_truth(q_f, k_f, v_f, scale):
    """(H, m, d) float64 -> (H, m, d) float64 attention block."""
    scores = np.einsum("hmd,hnd->hmn", q_f, k_f) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    return np.einsum("hmn,hnd->hmd", w, v_f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)
    os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

    from mlx_lm.utils import load

    import mlx.core as mx

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    model.eval()
    out = open(args.out, "w")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), file=sys.stderr)

    prompts = {
        "long": ("You are reviewing a measurement protocol for a small "
                 "language-model benchmark. The benchmark runs the same "
                 "workload matrix on two operating systems and compares "
                 "prefill and decode throughput. Each leg loads one pinned "
                 "model snapshot, applies the model chat template to the "
                 "prompt, runs greedy decoding with a fixed random seed, "
                 "and records the prompt phase, the per-token phase, the "
                 "generated token count, a hash of the generated token "
                 "ids, and peak memory. A leg is valid only when the model "
                 "snapshot revision matches the manifest pin, the power "
                 "state is recorded, and no other model-serving process is "
                 "holding the accelerator."),
    }
    long_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts["long"]}],
        add_generation_prompt=True)
    legs = {"long": long_ids, "ctx1024": (long_ids * 8)[:1053]}

    # ---- part 1: GPU token streams under both compositions ----
    streams = {}
    for name, ids in legs.items():
        x = mx.array(ids)[None]
        logits = model(x)
        mx.eval(logits)
        mx.synchronize()
        for gate, env in (("f32comp", None), ("bf16fast", "1")):
            if env is None:
                os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)
            else:
                os.environ["MLX_OMARCHY_SDPA_BF16_FAST"] = env
            lg = model(x)
            mx.eval(lg)
            mx.synchronize()
            ids_np = np.asarray(mx.argmax(lg, axis=-1), dtype=np.uint32)[0]
            streams[(name, gate)] = ids_np
            emit({"part": "stream", "leg": name, "gate": gate,
                  "positions": int(ids_np.size)})
    os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)

    # ---- part 3 (GPU half): synthetic attention under both gates ----
    gpu_attn = {}
    for m in (262, 1053):
        rng = mx.random.key(args.seed)
        kq, kk, kv = mx.random.split(rng, 3)
        q = mx.random.normal((1, 14, m, 64), key=kq).astype(mx.bfloat16)
        k = mx.random.normal((1, 14, m, 64), key=kk).astype(mx.bfloat16)
        v = mx.random.normal((1, 14, m, 64), key=kv).astype(mx.bfloat16)
        q_np = np.asarray(q.view(mx.uint16), dtype=np.uint16)
        k_np = np.asarray(k.view(mx.uint16), dtype=np.uint16)
        v_np = np.asarray(v.view(mx.uint16), dtype=np.uint16)
        np.save(os.path.join(os.path.dirname(args.out),
                             f"attn-inputs-m{m}.npy"),
                np.stack([q_np, k_np, v_np]))
        for gate, env in (("f32comp", None), ("bf16fast", "1")):
            if env is None:
                os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)
            else:
                os.environ["MLX_OMARCHY_SDPA_BF16_FAST"] = env
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
            mx.eval(o)
            mx.synchronize()
            gpu_attn[(m, gate)] = np.asarray(
                o.view(mx.uint16), dtype=np.uint16).reshape(-1)
        os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)
        emit({"part": "gpu_attn", "m": m})

    # ---- part 2: f64 truth streams (CPU) ----
    inner = model.model.layers[0]
    cfg = model.config
    theta = float(getattr(cfg, "rope_theta", 1000000.0))
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    n_layers = len(model.model.layers)

    def w64(w):
        return bf16_bits_to_f64(
            np.asarray(w.astype(mx.bfloat16).view(mx.uint16),
                       dtype=np.uint16))

    embed = w64(model.model.embed_tokens.weight)
    layers = []
    for layer in model.model.layers:
        layers.append({
            "ln1": w64(layer.input_layernorm.weight),
            "q": w64(layer.self_attn.q_proj.weight),
            "k": w64(layer.self_attn.k_proj.weight),
            "v": w64(layer.self_attn.v_proj.weight),
            "o": w64(layer.self_attn.o_proj.weight),
            "ln2": w64(layer.post_attention_layernorm.weight),
            "gate": w64(layer.mlp.gate_proj.weight),
            "up": w64(layer.mlp.up_proj.weight),
            "down": w64(layer.mlp.down_proj.weight),
        })
    final_norm = w64(model.model.norm.weight)
    lm_head = (w64(model.lm_head.weight)
               if getattr(model, "lm_head", None) is not None else embed)
    emit({"part": "weights_f64", "layers": n_layers})

    def rms64(x, w):
        return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)
                                  + eps)) * w

    def rope64(x, heads, m):
        # x: (B, H, m, D) f64, traditional=False, scale 1, offset 0.
        half = x.shape[-1] // 2
        inv = 1.0 / theta ** (2.0 * np.arange(half, dtype=np.float64)
                              / x.shape[-1])
        ang = np.arange(m, dtype=np.float64)[:, None] * inv[None, :]
        cos = np.cos(ang)[None, None]
        sin = np.sin(ang)[None, None]
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin,
                               x2 * cos + x1 * sin], axis=-1)

    def sdpa64(x, heads, kv_heads, m, scale):
        rep = heads // kv_heads
        b = x.shape[0]
        q = x[:, :heads]
        k = x[:, heads:heads + kv_heads]
        v = x[:, heads + kv_heads:]
        outs = np.empty((b, heads, m, v.shape[-1]), dtype=np.float64)
        for bi in range(b):
            for kv_i in range(kv_heads):
                for r in range(rep):
                    h = kv_i * rep + r
                    outs[bi, h] = f64_attention_truth(
                        q[bi, h], k[bi, kv_i], v[bi, kv_i], scale)
        return outs

    def forward(ids, capture_last_attn=None):
        tokens = np.asarray(ids, dtype=np.int64)
        m = tokens.size
        x = embed[tokens][None].astype(np.float64)
        heads, kv_heads, head_dim = (cfg.num_attention_heads,
                                     cfg.num_key_value_heads,
                                     getattr(cfg, "head_dim",
                                             cfg.hidden_size // cfg.num_attention_heads))
        scale = head_dim ** -0.5
        for i, lw in enumerate(layers):
            h = rms64(x, lw["ln1"])
            q = (h @ lw["q"].T).reshape(1, m, heads, head_dim).transpose(0, 2, 1, 3)
            k = (h @ lw["k"].T).reshape(1, m, kv_heads, head_dim).transpose(0, 2, 1, 3)
            v = (h @ lw["v"].T).reshape(1, m, kv_heads, head_dim).transpose(0, 2, 1, 3)
            q = rope64(q, heads, m)
            k = rope64(k, kv_heads, m)
            a = sdpa64(np.concatenate([q, k, v], axis=1),
                       heads, kv_heads, m, scale)
            a = a.transpose(0, 2, 1, 3).reshape(1, m, heads * head_dim)
            x = x + a @ lw["o"].T
            h2 = rms64(x, lw["ln2"])
            gate = h2 @ lw["gate"].T
            up = h2 @ lw["up"].T
            x = x + (gate / (1.0 + np.exp(-gate)) * up) @ lw["down"].T
        x = rms64(x, final_norm)
        return (x[0] @ lm_head.T, None)

    for name, ids in legs.items():
        logits64, _ = forward(ids)
        truth_ids = np.argmax(logits64, axis=-1).astype(np.uint32)
        emit({"part": "truth_stream", "leg": name,
              "positions": int(truth_ids.size)})
        for gate in ("f32comp", "bf16fast"):
            got = streams[(name, gate)]
            match = int((got == truth_ids).sum())
            emit({"part": "stream_match", "leg": name, "gate": gate,
                  "match": match, "of": int(truth_ids.size)})
        off = streams[(name, "f32comp")] == truth_ids
        on = streams[(name, "bf16fast")] == truth_ids
        emit({"part": "stream_cross", "leg": name,
              "both_right": int((off & on).sum()),
              "both_wrong": int((~off & ~on).sum()),
              "f32_only_right": int((off & ~on).sum()),
              "bf16_only_right": int((~off & on).sum())})

    # ---- part 3 (CPU half): f64 attention truth for the synthetic sets ----
    for m in (262, 1053):
        stacked = np.load(os.path.join(os.path.dirname(args.out),
                                       f"attn-inputs-m{m}.npy"))
        q_np, k_np, v_np = stacked[0], stacked[1], stacked[2]
        q_f = bf16_bits_to_f64(q_np).reshape(14, m, 64)
        k_f = bf16_bits_to_f64(k_np).reshape(14, m, 64)
        v_f = bf16_bits_to_f64(v_np).reshape(14, m, 64)
        truth_f = f64_attention_truth(q_f, k_f, v_f, 0.125).reshape(-1)
        truth_bits = rne_bf16(truth_f)
        exact_truth = np.mean(truth_bits
                              == rne_bf16(truth_f)).item()
        for gate in ("f32comp", "bf16fast"):
            got = gpu_attn[(m, gate)]
            exact = float(np.mean(got == truth_bits).item())
            mean_ulp, max_ulp, kept = ulp_distance(
                got, truth_bits, truth_f)
            emit({"part": "attn_ulp", "m": m, "gate": gate,
                  "exact_frac": round(exact, 6),
                  "mean_ulp": round(mean_ulp, 4),
                  "max_ulp": max_ulp, "kept": kept})
        off_bits = gpu_attn[(m, "f32comp")]
        on_bits = gpu_attn[(m, "bf16fast")]
        off_exact = off_bits == truth_bits
        on_exact = on_bits == truth_bits
        emit({"part": "attn_cross", "m": m,
              "both_exact": int((off_exact & on_exact).sum()),
              "both_off": int((~off_exact & ~on_exact).sum()),
              "f32_only_exact": int((off_exact & ~on_exact).sum()),
              "bf16_only_exact": int((~off_exact & on_exact).sum())})

    out.close()
    print("ORACLE-DONE")


if __name__ == "__main__":
    main()

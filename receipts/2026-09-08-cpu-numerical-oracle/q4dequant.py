#!/usr/bin/env python3
"""MLX affine 4-bit dequant, independently verified against the bf16
snapshot of the SAME model (both pinned mlx-community conversions of
Qwen2.5-0.5B-Instruct).

MLX affine quantization spec (group_size=64, bits=4):
  - packed uint32 weight, 8 nibbles per word, nibble j in
    bits [4j, 4j+4): q = (word >> (4*j)) & 0xF, q in [0, 15]
  - scales, biases: one pair per 64-element group
  - dequant: w ~= q * scale + bias  (per group)
  - qkv attention biases are kept UNquantized (separate .bias tensors)

The cross-check uses a per-tensor max-absolute-scale plus 1e-3 bound and
checks signed bias endpoints against the BF16 snapshot. It is not a
per-group precision proof.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import (OUT, Q4_DIR, REV_Q4, LegModel, EXPECTED_LEGS,
                    load_bf16, sha256_file, run_leg, compare_all)

GROUP = 64
BITS = 4
PER_WORD = 32 // BITS


def dequant_tensor(w, scales, biases):
    """w uint32 [out, in/8]; scales/biases [out, in/64]. Returns fp32 [out, in]."""
    out_dim, words = w.shape
    n_groups = scales.shape[1]
    assert words * PER_WORD == n_groups * GROUP, (w.shape, scales.shape)
    shifts = torch.arange(PER_WORD, dtype=torch.int64) * BITS
    q = (w.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << BITS) - 1)
    q = q.reshape(out_dim, n_groups, GROUP).to(torch.float32)
    s = scales.to(torch.float32).unsqueeze(-1)
    b = biases.to(torch.float32).unsqueeze(-1)
    return (q * s + b).reshape(out_dim, n_groups * GROUP)


def dequant_model():
    raw = load_file(str(Q4_DIR / "model.safetensors"))
    cfg = json.loads((Q4_DIR / "config.json").read_text())
    q = cfg["quantization"]
    assert q["group_size"] == GROUP and q["bits"] == BITS, q
    weights = {}
    for key in sorted(raw):
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        base = key[: -len(".weight")] if key.endswith(".weight") else None
        if base is not None and base + ".scales" in raw:
            weights[key] = dequant_tensor(
                raw[key], raw[base + ".scales"], raw[base + ".biases"])
        else:
            weights[key] = raw[key].to(torch.float32)
    return weights, raw, cfg


def validate_q4_weights(raw, cfg, tag):
    """Shape/dtype/key-mapping validation for the 4-bit snapshot."""
    L = cfg["num_hidden_layers"]
    hid, inter = cfg["hidden_size"], cfg["intermediate_size"]
    heads, kvh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = hid // heads
    voc, g = cfg["vocab_size"], GROUP
    lin = {"q_proj": (heads * hd, hid), "k_proj": (kvh * hd, hid),
           "v_proj": (kvh * hd, hid), "o_proj": (hid, heads * hd),
           "gate_proj": (inter, hid), "up_proj": (inter, hid),
           "down_proj": (hid, inter)}
    expected = {"model.embed_tokens.weight", "model.embed_tokens.scales",
                "model.embed_tokens.biases", "model.norm.weight"}
    for i in range(L):
        p = f"model.layers.{i}."
        expected |= {p + "input_layernorm.weight",
                     p + "post_attention_layernorm.weight"}
        for name, _ in lin.items():
            comp = "self_attn" if name.endswith("_proj") and name != "gate_proj" \
                and name != "up_proj" and name != "down_proj" else "mlp"
            expected |= {p + comp + "." + name + ".weight",
                         p + comp + "." + name + ".scales",
                         p + comp + "." + name + ".biases"}
            if comp == "self_attn" and name != "o_proj":
                expected |= {p + comp + "." + name + ".bias"}
    present = set(raw.keys())
    missing, extra = sorted(expected - present), sorted(present - expected)
    assert not missing, f"{tag}: MISSING keys: {missing[:8]}... ({len(missing)})"
    assert not extra, f"{tag}: UNEXPECTED keys: {extra[:8]}... ({len(extra)})"
    quant_keys = sorted(k for k in present if k.endswith(".weight")
                        and k.replace(".weight", ".scales") in present)
    assert len(quant_keys) == 7 * L + 1, f"{tag}: expected {7*L+1} quantized tensors, got {len(quant_keys)}"
    for wk in quant_keys:
        base = wk[:-len(".weight")]
        w, s, b = raw[wk], raw[base + ".scales"], raw[base + ".biases"]
        comp = base.split(".")[-2]
        o, inp = (lin[base.split(".")[-1]] if comp in ("self_attn", "mlp")
                  else (voc, hid))
        assert w.dtype == torch.uint32 and tuple(w.shape) == (o, inp // PER_WORD), \
            (wk, tuple(w.shape), str(w.dtype))
        assert tuple(s.shape) == (o, inp // g) and tuple(b.shape) == (o, inp // g), \
            (wk, tuple(s.shape))
        assert s.dtype in (torch.float16, torch.bfloat16), (wk, str(s.dtype))
    for k in present:
        if k.endswith(".bias") or k.endswith("layernorm.weight") or k == "model.norm.weight":
            assert raw[k].dtype in (torch.float16, torch.bfloat16), (k, str(raw[k].dtype))
    return {"tag": tag, "n_keys": len(present), "n_quantized": len(quant_keys),
            "lm_head_present": any("lm_head" in k for k in present),
            "group_size": g, "bits": BITS,
            "unquantized_dtypes": sorted({str(raw[k].dtype) for k in present
                                          if not k.endswith(".weight")
                                          or k.replace(".weight", ".scales") not in present})}


def check():
    weights, raw, cfg = dequant_model()
    val = validate_q4_weights(raw, cfg, "q4")
    bf16_weights, _, _ = load_bf16()
    checks, fails = [], []
    endpoint_violations = 0
    for wk in sorted(weights):
        if not wk.endswith(".weight") or wk not in bf16_weights \
                or wk.replace(".weight", ".scales") not in raw:
            continue
        base = wk[:-len(".weight")]
        s32 = raw[base + ".scales"].to(torch.float32)
        b32 = raw[base + ".biases"].to(torch.float32)
        ref = bf16_weights[wk]
        out_dim, G = s32.shape
        refG = ref.reshape(out_dim, G, 64)
        mins, maxs = refG.min(-1).values, refG.max(-1).values
        neg = s32 < 0
        b_err = torch.where(neg, (b32 - maxs).abs(), (b32 - mins).abs())
        endpoint_violations += int((b_err > 0).sum())
        diff = (weights[wk] - ref).abs()
        s_abs_max = s32.abs().max().item()
        bound = s_abs_max + 1e-3
        mx = diff.max().item()
        entry = {"key": wk, "max_abs_diff": round(mx, 6),
                 "bound_step": round(bound, 6)}
        if mx > bound:
            entry["FAIL"] = True
            fails.append(entry)
        checks.append(entry)
    report = {
        "schema": "q4-dequant-check/2",
        "spec": ("w = q*scale + bias, nibble j = (word >> 4j) & 0xF, group 64; "
                 "signed per-group scales; bias endpoint checked against BF16; "
                 "per-tensor max residual <= max(abs(scale)) + 1e-3"),
        "weight_validation": val,
        "n_checked_vs_bf16": len(checks),
        "max_abs_diff_overall": round(max(c["max_abs_diff"] for c in checks), 6),
        "endpoint_identity_violations": endpoint_violations,
        "fails": fails,
        "detail": checks,
    }
    (OUT / "q4-dequant-check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "detail"}, indent=1))
    if fails or endpoint_violations:
        print(f"DEQUANT CHECK FAILED: {len(fails)} tensors, "
              f"{endpoint_violations} endpoint violations", file=sys.stderr)
        raise SystemExit(1)
    print("dequant check: PASS")


def run_legs(topk):
    weights, raw, cfg = dequant_model()
    val = validate_q4_weights(raw, cfg, "q4")
    m = LegModel("q4", Q4_DIR, REV_Q4, weights, cfg,
                 sha256_file(Q4_DIR / "model.safetensors"), val)
    results = {pid: run_leg(m, pid, EXPECTED_LEGS[pid][1], topk)
               for pid in ["long", "ctx1024", "short"]}
    compare_all("q4", results)


if __name__ == "__main__":
    check()

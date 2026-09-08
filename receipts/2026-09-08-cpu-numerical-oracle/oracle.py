#!/usr/bin/env python3
"""Independent CPU fp32 numerical oracle for the pinned Qwen2.5-0.5B legs.

NOT a timing benchmark and NOT proof of native numerical identity: this is a
mathematical oracle in float32 on x86 CPU using stock PyTorch and the actual
pinned safetensors snapshots. Greedy argmax only (temp 0), EOS never
suppressed in logits (matches bench_decode: tokenizer.eos_token_ids=set()
only disables stopping; EOS ids can be and are emitted).

Engine parity notes (scripts/bench_decode.py):
  - prompt: tokenizer.apply_chat_template([{"role":"user", content}],
    add_generation_prompt=True) -- imported verbatim from
    scripts/bench_matrix.prompt_text.
  - the 4 warmup tokens run in a SEPARATE stream_generate() call with a fresh
    KV cache and are discarded; no state carries into the measured
    generation, so the oracle runs only the measured generation.
  - digest: sha256(",".join(ids))[:16] (bench_decode.ids_digest).

Subcommands:
  bf16                run the three BF16 legs (long128, ctx1024-32, short32)
  q4-dequant-check    verify MLX affine 4-bit dequant against the bf16
                      snapshot of the same model, per tensor
  q4                  run the three Q4 legs with verified dequant weights
  probe               forced-prefix logits at a divergence step:
                      prompt + --prefix-ids, dump fp32 logits of the step
                      that chooses generated index --index
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

WAVE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BF16_DIR = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
Q4_DIR = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
REV_BF16 = "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
REV_Q4 = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
EXPECT_COUNTS = {"short": 30, "long": 262, "ctx1024": 1053}
EXPECTED_LEGS = {
    "long": ("long-decode-128", 128),
    "ctx1024": ("longctx-1024-decode-32", 32),
    "short": ("short-decode-32", 32),
}

torch.set_grad_enabled(False)
torch.set_num_threads(min(16, torch.get_num_threads()))


def ids_digest(ids):
    h = hashlib.sha256()
    h.update(",".join(str(int(i)) for i in ids).encode("ascii"))
    return h.hexdigest()[:16]


def prompt_ids(prompt_id, snap_dir):
    """Exact prompt token ids, validated two tokenization paths + count."""
    sys.path.insert(0, str(WAVE / "scripts"))
    import bench_matrix
    from transformers import AutoTokenizer

    manifest = json.loads((WAVE / "scripts" / "bench_matrix.json").read_text())
    text = bench_matrix.prompt_text(manifest, prompt_id)
    tok = AutoTokenizer.from_pretrained(str(snap_dir), local_files_only=True)
    chat_ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True)
    chat_str = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True,
        tokenize=False)
    str_ids = tok(chat_str, add_special_tokens=False).input_ids
    assert chat_ids == str_ids, (
        f"{prompt_id}: tokenization path mismatch {len(chat_ids)} vs {len(str_ids)}")
    n = len(chat_ids)
    assert n == EXPECT_COUNTS[prompt_id], (
        f"{prompt_id}: token count {n} != pinned {EXPECT_COUNTS[prompt_id]}")
    info = {
        "prompt_id": prompt_id,
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "rendered_sha256": hashlib.sha256(chat_str.encode()).hexdigest(),
        "prompt_tokens": n,
        "first_ids": chat_ids[:6],
        "last_ids": chat_ids[-6:],
    }
    return chat_ids, info


class Qwen2Fp32:
    """Minimal eager Qwen2: fp32 math throughout, half-split RoPE, GQA."""

    def __init__(self, weights, cfg):
        self.w = weights
        self.layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["hidden_size"] // self.heads
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        assert cfg["tie_word_embeddings"], "oracle assumes tied embeddings"
        self.rope_len = 0

    def _rope(self, start, n):
        want = start + n
        if want > self.rope_len:
            self.rope_len = max(want, 4096)
            half = self.head_dim // 2
            inv = 1.0 / self.theta ** (np.arange(0, half, dtype=np.float64) * 2 / self.head_dim)
            pos = np.arange(self.rope_len, dtype=np.float64)
            ang = np.outer(pos, inv)
            self.cos = torch.tensor(np.cos(ang), dtype=torch.float32)
            self.sin = torch.tensor(np.sin(ang), dtype=torch.float32)
        half = self.head_dim // 2
        cos = torch.cat([self.cos[start:want]] * 2, dim=-1).view(1, 1, n, self.head_dim)
        sin = torch.cat([self.sin[start:want]] * 2, dim=-1).view(1, 1, n, self.head_dim)
        return cos, sin

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat((-x2, x1), dim=-1)

    def _attn(self, h, layer, kcache, vcache, start):
        n = h.shape[1]
        p = f"model.layers.{layer}.self_attn."
        q = F.linear(h, self.w[p + "q_proj.weight"], self.w.get(p + "q_proj.bias"))
        k = F.linear(h, self.w[p + "k_proj.weight"], self.w.get(p + "k_proj.bias"))
        v = F.linear(h, self.w[p + "v_proj.weight"], self.w.get(p + "v_proj.bias"))
        q = q.view(1, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(1, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, n, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self._rope(start, n)
        q = q * cos + self._rot(q) * sin
        k = k * cos + self._rot(k) * sin
        kcache.append(k)
        vcache.append(v)
        rep = self.heads // self.kv_heads
        K = kcache.cat().repeat_interleave(rep, dim=1)
        V = vcache.cat().repeat_interleave(rep, dim=1)
        t = K.shape[2]
        scores = q @ K.transpose(-1, -2) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.full((t, t), float("-inf")).triu_(1)[start:start + n, :]
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(1, n, -1)
        return F.linear(out, self.w[f"model.layers.{layer}.self_attn.o_proj.weight"])
    @staticmethod
    def _rms(x, w, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def forward_last(self, ids, caches, start):
        """ids: 1-D list/tensor. Returns fp32 logits at the last position."""
        h = F.embedding(torch.tensor([ids], dtype=torch.long),
                        self.w["model.embed_tokens.weight"])
        n = h.shape[1]
        for layer in range(self.layers):
            kcache, vcache = caches[layer]
            x = self._rms(h, self.w[f"model.layers.{layer}.input_layernorm.weight"], self.eps)
            h = h + self._attn(x, layer, kcache, vcache, start)
            x = self._rms(h, self.w[f"model.layers.{layer}.post_attention_layernorm.weight"], self.eps)
            gate = F.linear(x, self.w[f"model.layers.{layer}.mlp.gate_proj.weight"])
            up = F.linear(x, self.w[f"model.layers.{layer}.mlp.up_proj.weight"])
            down = F.linear(F.silu(gate) * up, self.w[f"model.layers.{layer}.mlp.down_proj.weight"])
            h = h + down
        h = self._rms(h, self.w["model.norm.weight"], self.eps)
        return h[:, -1:, :] @ self.w["model.embed_tokens.weight"].T

    def generate(self, prompt, n_gen, topk=16):
        caches = [(CatCache(), CatCache()) for _ in range(self.layers)]
        ids = list(prompt)
        logits = self.forward_last(ids, caches, 0)
        steps = []
        gen = []
        for i in range(n_gen):
            last = logits[0, -1]
            topv, topi = torch.topk(last, topk)
            chosen = int(torch.argmax(last).item())
            steps.append({
                "index": i,
                "token_id": chosen,
                "top": [[int(topi[j]), round(float(topv[j]), 6)] for j in range(topk)],
                "margin": round(float(topv[0] - topv[1]), 6),
            })
            gen.append(chosen)
            if i + 1 < n_gen:
                logits = self.forward_last([chosen], caches, len(ids))
                ids.append(chosen)
        return gen, steps


class CatCache:
    """Growable KV cache: keeps a preallocated tensor, appends in place."""

    def __init__(self, cap=4096):
        self.cap = cap
        self.buf = None
        self.n = 0

    def append(self, k):
        if self.buf is None:
            self.buf = torch.zeros(1, k.shape[1], self.cap, k.shape[3])
        d = k.shape[2]
        assert self.n + d <= self.cap
        self.buf[:, :, self.n:self.n + d, :].copy_(k)
        self.n += d

    def cat(self):
        return self.buf[:, :, :self.n, :]


def load_bf16(snap_dir=BF16_DIR):
    tensors = load_file(str(snap_dir / "model.safetensors"))
    weights = {k: v.to(torch.float32) for k, v in tensors.items()}
    cfg = json.loads((snap_dir / "config.json").read_text())
    return weights, cfg, tensors


def validate_weights(weights, cfg, raw, tag):
    """Key mapping / shape / tied-embed validation. Raises on any surprise."""
    L = cfg["num_hidden_layers"]
    hid, inter = cfg["hidden_size"], cfg["intermediate_size"]
    heads, kvh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = hid // heads
    voc = cfg["vocab_size"]
    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(L):
        p = f"model.layers.{i}."
        expected |= {
            p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
            p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
            p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight",
            p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
            p + "mlp.down_proj.weight"}
    present = set(raw.keys())
    extra = sorted(present - expected)
    missing = sorted(expected - present)
    assert not missing, f"{tag}: MISSING weight keys: {missing}"
    shapes = {
        "embed_tokens.weight": (voc, hid),
        "norm.weight": (hid,),
        "input_layernorm.weight": (hid,),
        "post_attention_layernorm.weight": (hid,),
        "q_proj.bias": (heads * hd,),
        "k_proj.bias": (kvh * hd,),
        "v_proj.bias": (kvh * hd,),
        "q_proj.weight": (heads * hd, hid),
        "k_proj.weight": (kvh * hd, hid),
        "v_proj.weight": (kvh * hd, hid),
        "o_proj.weight": (hid, heads * hd),
        "gate_proj.weight": (inter, hid),
        "up_proj.weight": (inter, hid),
        "down_proj.weight": (hid, inter),
    }
    for k, t in raw.items():
        if k in ("model.embed_tokens.weight", "model.norm.weight"):
            continue
        suffix = k.split(".")[-2] + "." + k.split(".")[-1]
        assert tuple(t.shape) == shapes[suffix], f"{tag}: {k} shape {t.shape} != {shapes[suffix]}"
        assert not k.endswith("lm_head.weight"), f"{tag}: unexpected lm_head"
    return {
        "tag": tag,
        "n_keys": len(present),
        "extra_keys_non_buffer": [k for k in extra if not k.endswith("inv_freq")],
        "buffer_keys": sorted(k for k in extra if k.endswith("inv_freq")),
        "lm_head_present": any("lm_head" in k for k in present),
        "weight_dtypes": sorted({str(v.dtype) for v in raw.values()}),
    }


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run_leg(model, prompt_id, n_gen, topk):
    ids, pinfo = prompt_ids(prompt_id, model.snap)
    t0 = time.time()
    gen, steps = model.net.generate(ids, n_gen, topk=topk)
    dt = time.time() - t0
    out = {
        "schema": "cpu-oracle/1",
        "leg": f"{model.tag}:{prompt_id}",
        "workload": EXPECTED_LEGS[prompt_id][0],
        "requested": n_gen,
        "prompt": pinfo,
        "ids": gen,
        "ids_sha256_16": ids_digest(gen),
        "topk": topk,
        "steps": steps,
        "engine": "pytorch-fp32-cpu",
        "torch_version": torch.__version__,
        "wall_s": round(dt, 1),
        "model_sha256": model.sha,
        "revision": model.rev,
        "weight_validation": model.val,
    }
    p = OUT / f"oracle-{model.tag}-{prompt_id}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p} digest={out['ids_sha256_16']} wall={dt:.0f}s")
    return out


class LegModel:
    def __init__(self, tag, snap, rev, weights, cfg, sha, val):
        self.tag, self.snap, self.rev = tag, snap, rev
        self.net = Qwen2Fp32(weights, cfg)
        self.sha, self.val = sha, val


def cmd_bf16(args):
    weights, cfg, raw = load_bf16()
    sha = sha256_file(BF16_DIR / "model.safetensors")
    val = validate_weights(weights, cfg, raw, "bf16")
    m = LegModel("bf16", BF16_DIR, REV_BF16, weights, cfg, sha, val)
    order = ["long", "ctx1024", "short"]
    results = {pid: run_leg(m, pid, EXPECTED_LEGS[pid][1], args.topk) for pid in order}
    compare_all("bf16", results)


def compare_all(tag, results):
    """Digest comparison vs Linux M1 baseline ids and native macOS digests."""
    linux = parse_rep_ids(WAVE / "receipts/parity-baseline-20260908/rep1.ids.jsonl")
    native = json.loads((WAVE / "receipts/native-baseline-2026-09-06/native-2026-09-06-summary.json").read_text())
    model_key = {"q4": "qwen25-0.5b-4bit"}.get(tag, f"qwen25-0.5b-{tag}")
    cmp_out = {}
    for pid, res in results.items():
        leg = EXPECTED_LEGS[pid][0]
        key = f"{model_key}:{leg}"
        mine = res["ids"]
        row = {"leg": key, "oracle_digest": res["ids_sha256_16"]}
        if leg in linux[model_key]:
            base = linux[model_key][leg]
            row["linux_digest"] = ids_digest(base)
            row["linux_first_diff_index"] = first_diff(mine, base)
            row["n_linux"] = len(base)
        if key in native:
            row["native_digest"] = native[key]["generated_ids_sha256_16"][0]
            row["native_matches_oracle"] = row["native_digest"] == res["ids_sha256_16"]
        if "linux_digest" in row:
            row["linux_matches_oracle"] = row["linux_digest"] == res["ids_sha256_16"]
        cmp_out[key] = row
        print(json.dumps(row))
    (OUT / f"compare-{tag}.json").write_text(json.dumps(cmp_out, indent=1))


def parse_rep_ids(path):
    """rep1.ids.jsonl order: 4bit short,long,ctx then bf16 short,long,ctx.
    Returns {model: {leg: ids}}."""
    legmap = {(262, 128): "long-decode-128",
              (1053, 32): "longctx-1024-decode-32",
              (30, 32): "short-decode-32"}
    order = ["short-decode-32", "long-decode-128", "longctx-1024-decode-32"]
    out = {"qwen25-0.5b-4bit": {}, "qwen25-0.5b-bf16": {}}
    models = iter(["qwen25-0.5b-4bit", "qwen25-0.5b-bf16"])
    model = next(models)
    seen = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        leg = legmap[(r["prompt_tokens"], r["requested"])]
        seen.append(leg)
        out[model][leg] = r["ids"]
        if len(seen) % 3 == 0:
            try:
                model = next(models)
            except StopIteration:
                pass
    assert all(out[m][l] for m in out for l in order), "rep1 leg mapping incomplete"


    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def cmd_probe(args):
    """Forced-prefix logits at divergence index d: prefix = prompt + baseline[0:d]."""
    if args.model == "bf16":
        weights, cfg, raw = load_bf16()
        snap, rev, tag = BF16_DIR, REV_BF16, "bf16"
        val = validate_weights(weights, cfg, raw, "bf16")
    else:
        import q4dequant
        weights, raw, cfg = q4dequant.dequant_model()
        snap, rev, tag = Q4_DIR, REV_Q4, "q4"
        val = q4dequant.validate_q4_weights(raw, cfg, "q4")
    m = LegModel(tag, snap, rev, weights, cfg, sha256_file(snap / "model.safetensors"), val)
    pids, pinfo = prompt_ids(args.prompt_id, snap)
    linux = parse_rep_ids(WAVE / "receipts/parity-baseline-20260908/rep1.ids.jsonl")
    base = linux[{"q4": "qwen25-0.5b-4bit"}.get(tag, f"qwen25-0.5b-{tag}")][EXPECTED_LEGS[args.prompt_id][0]]
    d = args.index
    prefix = pids + base[:d]
    caches = [(CatCache(), CatCache()) for _ in range(m.net.layers)]
    logits = m.net.forward_last(prefix, caches, 0)
    last = logits[0, -1]
    topv, topi = torch.topk(last, args.topk)
    ranked = int((last > last[base[d]]).sum().item())
    np.save(OUT / f"probe-{tag}-{args.prompt_id}-idx{d}-logits.npy", last.numpy())
    out = {
        "leg": f"{tag}:{args.prompt_id}",
        "forced_index": d,
        "prefix_len": len(prefix),
        "alignment": "logits computed after prompt + baseline generated ids [0, d); "
                     "the step chooses generated index d",
        "forced_prefix_token_ids_last6": base[max(0, d - 6):d],
        "baseline_token_at_index": base[d],
        "baseline_token_rank0": ranked,
        "baseline_token_logit": round(float(last[base[d]]), 6),
        "margin_top1_top2": round(float(topv[0] - topv[1]), 6),
        "top": [[int(topi[j]), round(float(topv[j]), 6)] for j in range(args.topk)],
        "top1_vs_baseline_margin": round(float(topv[0] - float(last[base[d]])), 6),
        "prompt": pinfo,
        "engine": "pytorch-fp32-cpu",
    }
    p = OUT / f"probe-{tag}-{args.prompt_id}-idx{d}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bf16")
    b.add_argument("--topk", type=int, default=16)
    sub.add_parser("q4-dequant-check")
    q = sub.add_parser("q4")
    q.add_argument("--topk", type=int, default=16)
    p = sub.add_parser("probe")
    p.add_argument("--prompt-id", required=True, choices=["long", "ctx1024", "short"])
    p.add_argument("--model", default="bf16", choices=["bf16", "q4"])
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--topk", type=int, default=32)
    args = ap.parse_args()
    if args.cmd == "bf16":
        cmd_bf16(args)
    elif args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd in ("q4", "q4-dequant-check"):
        import q4dequant
        if args.cmd == "q4-dequant-check":
            q4dequant.check()
        else:
            q4dequant.run_legs(args.topk)


if __name__ == "__main__":
    main()

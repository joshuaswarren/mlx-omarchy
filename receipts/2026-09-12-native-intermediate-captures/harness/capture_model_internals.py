#!/usr/bin/env python3
"""Capture native MLX internals on the canonical legs - macOS only.

Runs the exact bench_decode protocol (pinned snapshots, chat template,
greedy temp 0 seed 0, EOS suppressed, MLX_DISABLE_COMPILE=1, 4 warmup
tokens then the measured generation) with pure observers around
mx.fast.scaled_dot_product_attention and mx.quantized_matmul.

Per leg it banks:
  - decode sdpa q/k/v/out bits for all 28 layers at decode steps
    {0,1} (steps {0,1,2} on the short leg, covering KV 30/31/32);
  - first prefill sdpa call (layer 0, causal);
  - qmm layer-0 prefill: x, y, packed w, scales, biases for all seven
    linears, plus mx.dequantize operands for q_proj and down_proj;
  - shape/dtype/checksum metadata for every sdpa and qmm call;
  - single-pass dump-kernel intermediates on the real step-0 tensors of
    every layer (scores, running max/sum, exps, factors, lane partials,
    combine) validated against the captured native output at KV<1024;
  - the generated ids digest asserted against the committed native
    oracle digests.

Everything goes under --out/<model>/<leg>/ plus a per-run manifest.
"""

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "q4": {
        "repo": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "revision": "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
        "digests": {"short": "7fd25a869ff21678",
                    "long": "254d73fd93164b98",
                    "longctx": "7da83f06ec9f001d"},
    },
    "bf16": {
        "repo": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        "revision": "56d07e766edd7159fbe12ed12d9cf114bf38bf1e",
        "digests": {"short": "7fc0f968789b1882",
                    "long": "407b7624ed1b3b29",
                    "longctx": "ff502900d2a179a5"},
    },
}
LEGS = {"short": 32, "long": 128, "longctx": 32}
CURRENT_MODEL = None
CURRENT_LEG = None
OUT_DIR = None
N_LAYERS = 24
EXPECTED_PROMPT_TOKENS = {"short": 30, "long": 262, "ctx1024": 1053}
DECODE_STEPS = {"short": (0, 1, 2), "long": (0, 1), "longctx": (0, 1)}


import os as _os
def sha16(a):
    if _os.environ.get("CAP_SHA", "1") == "0":
        return "off"
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def f32(a):
    return np.array(a.astype(mx.float32))


def bits(a):
    """Bit-exact cheap bytes view for hashing (f16/bf16 -> uint16)."""
    if a.dtype in (mx.float16, mx.bfloat16):
        return np.array(a.view(mx.uint16))
    return np.array(a.view(mx.uint32))


def snapshot_path(repo, revision):
    base = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}"
        f"/snapshots/{revision}")
    if not os.path.isdir(base):
        sys.exit(f"REFUSING: snapshot for {repo}@{revision} missing at {base}")
    return base


class SdpaObserver:
    def __init__(self):
        self.real = mx.fast.scaled_dot_product_attention
        self.phase = "idle"
        self.decode_calls = 0
        self.prefill_seen = False
        self.meta = []

    def reset(self, phase):
        self.phase = phase
        self.decode_calls = 0
        self.prefill_seen = False
        self.meta = []

    def __call__(self, q, k, v, *args, **kwargs):
        out = self.real(q, k, v, *args, **kwargs)
        is_decode = q.shape[2] == 1
        layer = -1 if is_decode else 0
        if is_decode:
            step = self.decode_calls // N_LAYERS
            layer = self.decode_calls % N_LAYERS
            self.decode_calls += 1
            self.meta.append({"kind": "decode", "step": step,
                              "layer": layer, "kv": int(k.shape[2]),
                              "q_sha": sha16(bits(q)),
                              "out_sha": sha16(bits(out))})
        elif not self.prefill_seen:
            self.prefill_seen = True
            self.meta.append({"kind": "prefill_first", "layer": 0,
                              "q_sha": sha16(bits(q)),
                              "out_sha": sha16(bits(out))})
        else:
            self.meta.append({"kind": "prefill", "out_sha": sha16(bits(out))})
        want_dump = _os.environ.get("CAP_DUMP_SDPA", "1") == "1"
        if want_dump and self.phase == "capture" and (
                (is_decode and (self.decode_calls - 1) % N_LAYERS == 0
                 and (self.decode_calls - 1) // N_LAYERS
                 in DECODE_STEPS.get(CURRENT_LEG, ()))
                or (not is_decode and self.meta
                    and self.meta[-1]["kind"] == "prefill_first")):
            tag = (f"step{(self.decode_calls - 1) // N_LAYERS}"
                   if is_decode else "prefill0")
            rel = f"{CURRENT_MODEL}/{CURRENT_LEG}/sdpa/{tag}_L{layer if is_decode else 0}"
            for name, arr in (("q", q), ("k", k), ("v", v), ("out", out)):
                np.save(os.path.join(OUT_DIR, f"{rel}_{name}.npy"),
                        f32(arr))
        return out


class QmmObserver:
    def __init__(self):
        self.real = mx.quantized_matmul
        self.phase = "idle"
        self.calls = 0
        self.meta = []

    def reset(self, phase):
        self.phase = phase
        self.calls = 0
        self.meta = []

    def __call__(self, x, w, scales, biases, *args, **kwargs):
        out = self.real(x, w, scales, biases, *args, **kwargs)
        idx = self.calls
        self.calls += 1
        gsize = args[1] if len(args) > 1 else kwargs.get("group_size")
        gbits = args[2] if len(args) > 2 else kwargs.get("bits")
        entry = {"idx": idx, "x_shape": list(x.shape),
                 "y_shape": list(out.shape), "group": gsize, "bits": gbits,
                 "x_sha": sha16(bits(x)), "y_sha": sha16(bits(out)),
                 "w_sha": sha16(np.array(w.view(mx.uint32)))}
        self.meta.append(entry)
        if (_os.environ.get("CAP_DUMP_QMM", "1") == "1"
                and self.phase == "capture" and idx < 7):
            tag = ["q", "k", "v", "o", "gate", "up", "down"][idx]
            rel = f"{CURRENT_MODEL}/{CURRENT_LEG}/qmm/{tag}"
            np.save(os.path.join(OUT_DIR, f"{rel}_x.npy"), f32(x))
            np.save(os.path.join(OUT_DIR, f"{rel}_y.npy"), f32(out))
            np.save(os.path.join(OUT_DIR, f"{rel}_w_packed_u32.npy"),
                    np.array(w.view(mx.uint32)))
            np.save(os.path.join(OUT_DIR, f"{rel}_scales.npy"), f32(scales))
            np.save(os.path.join(OUT_DIR, f"{rel}_biases.npy"), f32(biases))
            if (_os.environ.get("CAP_DEQUANT", "1") == "1"
                    and (CURRENT_MODEL == "q4" or tag in ("q", "down"))):
                dq = mx.dequantize(w, scales, biases, gsize, gbits)
                np.save(os.path.join(OUT_DIR, f"{rel}_w_dequant_f32.npy"),
                        f32(dq))
        return out


def ids_digest(ids):
    h = hashlib.sha256()
    h.update(",".join(str(int(i)) for i in ids).encode("ascii"))
    return h.hexdigest()[:16]


def run_leg(model_id, leg, prompt_text, gen_tokens, sdpa_obs, qmm_obs,
            manifest):
    global CURRENT_MODEL, CURRENT_LEG
    CURRENT_MODEL, CURRENT_LEG = model_id, leg
    from mlx_lm.utils import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    info = MODELS[model_id]
    global N_LAYERS
    model, tokenizer = load(snapshot_path(info["repo"], info["revision"]))
    N_LAYERS = len(model.model.layers)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True)
    prompt_ids = mx.array(prompt)
    tokenizer.eos_token_ids = set()
    mx.random.seed(0)
    sampler = make_sampler(temp=0.0)
    assert len(prompt) == EXPECTED_PROMPT_TOKENS.get(
        {"short": "short", "long": "long", "longctx": "ctx1024"}[leg]), (
        f"{leg}: prompt tokens {len(prompt)}")

    def generate(n):
        return stream_generate(model, tokenizer, prompt_ids,
                               max_tokens=n, sampler=sampler)

    os.makedirs(os.path.join(OUT_DIR, model_id, leg, "sdpa"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, model_id, leg, "qmm"), exist_ok=True)

    sdpa_obs.reset("warmup")
    qmm_obs.reset("warmup")
    for _ in generate(4):
        pass
    sdpa_obs.reset("capture")
    qmm_obs.reset("capture")
    ids = []
    t0 = time.monotonic_ns()
    for item in generate(gen_tokens):
        ids.append(int(item.token))
    prefill_s = (time.monotonic_ns() - t0) / 1e9
    digest = ids_digest(ids)
    got = ids_digest(ids)
    want = info["digests"][leg]
    print(f"{model_id}/{leg}: digest {got} (oracle {want}) "
          f"prompt {len(prompt)}", flush=True)
    assert got == want, f"{model_id}/{leg}: digest {got} != oracle {want}"
    manifest[f"{model_id}/{leg}"] = {
        "digest": got, "prompt_tokens": len(prompt),
        "generated": ids, "wall_s": prefill_s,
        "sdpa_meta": sdpa_obs.meta, "qmm_meta": qmm_obs.meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/src/native-captures-20260912")
    ap.add_argument("--prompts",
                    default="~/src/mlx-bench-20260911/prompts.json")
    ap.add_argument("--models", default="q4,bf16")
    ap.add_argument("--legs", default="short,long,longctx")
    args = ap.parse_args()
    global OUT_DIR
    OUT_DIR = os.path.expanduser(args.out)
    prompts = json.load(open(os.path.expanduser(args.prompts)))
    prompt_for = {"short": prompts["short"], "long": prompts["long"],
                  "longctx": prompts["ctx1024"]}
    sdpa_obs = SdpaObserver()
    qmm_obs = QmmObserver()
    mx.fast.scaled_dot_product_attention = sdpa_obs
    mx.quantized_matmul = qmm_obs
    manifest = {"mlx_version": mx.__version__,
                "device": str(mx.metal.device_info())}
    for model_id in args.models.split(","):
        for leg in args.legs.split(","):
            t0 = time.time()
            run_leg(model_id, leg, prompt_for[leg], LEGS[leg],
                    sdpa_obs, qmm_obs, manifest)
            with open(os.path.join(OUT_DIR, "manifest_model.json"), "w") as f:
                json.dump(manifest, f, indent=1)
            print(f"  leg done in {time.time() - t0:.1f}s", flush=True)
    print("ALL LEGS DONE")


if __name__ == "__main__":
    main()

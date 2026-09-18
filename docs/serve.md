# Serve a local model on Omarchy

OpenAI-compatible HTTP on `127.0.0.1:8080`, using the mlx-omarchy GPU
backend (`mlx_lm.server`). Harnesses talk to that URL. Nothing here
replaces Mesa or edits Omarchy files.

Verified 2026-09-17: Hugging Face model cards below (mlx / mlx-lm,
Apache-2.0). The installer pins `mlx-lm==0.31.3`, which is known to load
Qwen2.5. Qwen3 cards exist on Hugging Face; upgrade `mlx-lm` only if a
Qwen3 load fails on 0.31.3.

## 1. Install

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
```

That puts `mlx-omarchy` on `PATH` (the venv Python with this wheel and
mlx-lm). Smoke test:

```bash
mlx-omarchy -c 'import mlx.core as mx; print(mx.device_info())'
```

The device name must be Apple (Honeykrisp), not `llvmpipe`.

## 2. Pick a 4-bit model by RAM

Unified memory is shared with the desktop. Leave about 6–8 GiB for Omarchy.
Sizes are 4-bit MLX weights plus working KV; long context needs more.

| Unified RAM | Use | Hugging Face id | Card checked |
|---|---|---|---|
| 8 GiB | Tiny chat / smoke | [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit) | demo default, mlx-omarchy verified |
| 16 GiB | Daily coding | [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) | 2026-09-17, mlx, Apache-2.0, 18k downloads |
| 24–32 GiB | Stronger local | [`mlx-community/Qwen3-14B-4bit`](https://huggingface.co/mlx-community/Qwen3-14B-4bit) | 2026-09-17, mlx-lm 0.24 convert, Apache-2.0 |
| 64 GiB | Large dense | [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 2026-09-17 HF listing (updated 3 days prior); VL-tagged — if `mlx_lm.server` refuses, use Qwen3-14B |
| 96 GiB | MoE / long ctx | [`mlx-community/Qwen3.5-122B-A10B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-4bit) | 2026-09-17 HF listing |

Current-generation text IDs on Hugging Face (same day search,
[mlx-community Qwen3 4bit](https://huggingface.co/models?search=mlx-community%20Qwen3%204bit)):
`mlx-community/Qwen3-0.6B-4bit`, `lmstudio-community/Qwen3-8B-MLX-4bit`,
`mlx-community/Qwen3-14B-4bit`, `lmstudio-community/Qwen3-32B-MLX-4bit`.
The 16 GiB default above is Qwen2.5-7B because that is what the
**installed** `mlx-lm` 0.31.3 is known to serve.

### What is actually SOTA in September 2026 (and what loads here)

The table above lists IDs **verified to load** on the installed
`mlx-lm` 0.31.3. It is a compatibility list, not a freshness list.
As of 2026-09-18 the open-weight leaders in these size tiers are
Qwen3.6-27B, Gemma 4 31B, and the 8B/14B Ministral 3 / Qwen3 dense
models. MLX quants exist for the two big ones —
`mlx-community/Qwen3.6-27B-mxfp4` (arch `qwen3_5`) and
`mlx-community/gemma-4-31b-it-4bit` (arch `gemma4`) — but both need a
newer `mlx-lm` than 0.31.3 (that pin lacks the `qwen3_5`/`gemma4`
model code). Neither has been run on this stack; treat them as
experimental until verified.

**Ternary Bonsai 2 27B** (PrismML, Apache-2.0): there is **no Q4 of
it, and a Q4 would defeat the point** — Bonsai 2 *is* the quant, a
ternary {-1, 0, +1} packing of Qwen3.8-27B at a true ~1.72 bits per
weight (5.9 GB language model) reporting 98.2% of the FP16 benchmark
average. The published packings are:

- `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` — MLX 2-bit container,
  8.60 GB including the FP16 vision tower. It declares
  `model_type: prism_hadamard_qwen35` and **requires the loader
  bundled in the repo's `runtime/`** (Hadamard activation transform +
  inverse embedding lookup; ordinary MLX loaders produce wrong output
  silently rather than erroring), plus `mlx_vlm`. Full-speed ternary
  kernels live in PrismML's MLX fork.
- `prism-ml/Ternary-Bonsai-2-27B-gguf` — PTQ1_0 (5.95 GB) / PQ2_0
  (7.21 GB), **requires the PrismML llama.cpp fork**; stock llama.cpp
  rejects or garbles them.

None of these load paths exist in this stack's pinned runtime, and
the model is untested on MLX-over-Vulkan. If 27B-class on a 16 GiB
machine is the goal, Bonsai 2 is the interesting artifact — as an
upstream-mlx-lm upgrade + loader-port task, not a drop-in.

First download goes to `~/.cache/huggingface`.

## 3. Start the server

```bash
mlx-omarchy -m mlx_lm.server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8080
```

Smoke:

```bash
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

Listen on localhost only. The mlx-lm server is not a production
front door.

Optional memory knobs (mlx-lm): `--kv-bits 4` for long context (disables
batching). `--draft-model` for speculative decode if you have a smaller
draft of the same family.

## 4. Point harnesses at it

All of these speak OpenAI `/v1`. Use a dummy key. Keep the server running.

**Codex**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=mlx
codex
```

**omp**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=mlx
# then pick the openai-compatible model id your omp build uses
omp
```

**pi**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=mlx
pi
```

**Hermes**

Set the OpenAI-compatible endpoint to `http://127.0.0.1:8080/v1` and API
key `mlx` in Hermes provider settings (same `/v1/chat/completions` contract).

**OpenClaw**

Same URL: `http://127.0.0.1:8080/v1`, key `mlx`. oMLX’s integrations
screen uses this contract on macOS; mlx-omarchy is the Linux GPU backend
behind the same HTTP shape.

**Claude Code**

Claude Code speaks the Anthropic Messages API, not OpenAI. Put LiteLLM
in front of mlx-lm (separate process):

```bash
pip install 'litellm[proxy]'
OPENAI_API_KEY=mlx litellm --model openai/mlx-community/Qwen2.5-7B-Instruct-4bit \
  --api_base http://127.0.0.1:8080/v1
```

Then:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_API_KEY=sk-litellm
claude
```

If LiteLLM’s Anthropic port differs, use whatever `litellm` printed.

**Claude Code / Codex model name**

The server’s `/v1/models` `id` is the Hugging Face repo id. Use that
string as the model name in the harness when it asks.

## 5. Chat without a harness

```bash
mlx-omarchy-demo
mlx-omarchy-demo --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

Or `mlx-omarchy -m mlx_lm.generate --model … --prompt '…'`.

## 6. Stop

Ctrl-C the server. Uninstall: `bash install.sh --uninstall` (models stay
in `~/.cache/huggingface`).

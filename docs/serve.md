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
`mlx-community/gemma-4-31b-it-4bit` (arch `gemma4`). The installed
0.31.3 does carry the `qwen3_5`/`gemma4`/`ministral3` loader code
(verified 2026-09-19 in the installed venv), but loader presence is not
serve support: Ministral-3-8B crashes `mlx_lm.server` 0.31.3 in its
mistral3/tekken serve path while oMLX serves it (see §4 and the
[bench receipt](../receipts/2026-09-19-serve-options-bench-t6001-test-host.md)).
One verified data point
on this stack (2026-09-18 recert matrix, greedy 96, wheel `2def345c…`):
gemma-4-31b-it-4bit generated coherently at 2.37 tok/s compiled ON and
2.45 tok/s eager, digest `9f1fe40101db3a4b` identical under both modes
([receipt](../receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md)).
`Qwen3.6-27B-mxfp4` remains untested on MLX-over-Vulkan; treat it as
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

**Verified on this stack 2026-09-18** (t6001-test-host, recert wheel from main
`b283a16f`, wheel sha256 `2def345c…`): the `2bit` pack loads through its
bundled `runtime/` schema-2 loader with stock `mlx_vlm` 0.7.1 and generates
coherently at **1.44 tok/s** with compile ON (96 greedy steps, digest
`9252095e0de70235`, `nan_at` null — the F7 NaN class is fixed at `da43969e`).
The runtime refusal-ablation arm (129 writers, α=1.5) is coherent at
**1.25 tok/s** with the projection proven live by probe (refusal-adjacent
argmax 40 → 47; benign-prompt digest identical to stock). This is not an
`mlx_lm.server` drop-in: serve-path models in the ladder above stay on the
standard loader; Bonsai 2 runs through the pack's loader as the receipts do.
Receipts: [`receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md`](../receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md),
[`receipts/2026-09-18-ablit-bonsai2-t6001-test-host-recert-wheel.md`](../receipts/2026-09-18-ablit-bonsai2-t6001-test-host-recert-wheel.md).

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

## 4. Which server? mlx_lm.server / oMLX / mlx-serve

Three servers can put an OpenAI-compatible endpoint on local MLX
weights. Measured head-to-head on t6001-test-host (M1 Max, Asahi) in one
exclusive GPU window with the published v0.7.1 wheel (`libmlx` sha
`df3d4e74c597956c` pinned inside the bench), Qwen2.5-7B-Instruct-4bit
(37 prompt tokens on every leg), single-stream greedy, 128 max_tokens,
median of 8 interleaved rounds
([four-leg receipt](../receipts/2026-09-19-mlxserve-linux-port-t6001-test-host.md),
earlier two-leg run:
[receipt](../receipts/2026-09-19-serve-options-bench-t6001-test-host.md)):

| | `mlx_lm.server` | oMLX (`omlx`) | mlx-serve (Linux port) |
|---|---|---|---|
| Runs on omarchy/Linux | yes (default) | yes | yes — **from source**, via the fork's `linux-vulkan-port` (upstream: [ddalcu/mlx-serve#473](https://github.com/ddalcu/mlx-serve/pull/473)); release binaries are macOS-only |
| Install | bundled by `install.sh` (mlx-lm 0.31.3) | from source (`github.com/jundot/omlx`, not on PyPI at 0.6.4) | Mac: `brew install mlx-serve`; Linux: build (below) |
| Run | `mlx-omarchy -m mlx_lm.server --model <hf-id> --host 127.0.0.1 --port 8080` | `python -m omlx.server --model-dir <dir> --host 127.0.0.1 --port 8082` | `mlx-serve --model <dir> --serve` → `:11234` |
| Decode tok/s, Qwen2.5-7B-4bit | 10.36 | **33.5** (≈3.2×) | 5.65 — PLD speculative decode (on by default) made no difference on this prompt (5.70 with `--no-pld`) |
| Model breadth on this stack | standard mlx-lm loaders; mistral3/tekken serve path crashes on Ministral-3-8B (0.31.3 bug) | broad, incl. VLM models and Ministral-3-8B (3.3–3.4 tok/s) | Mac: every MLX arch + GGUF; Linux port: MLX safetensors only (llama.cpp/ds4/ANE engines stubbed) |
| APIs | OpenAI `/v1` (chat + completions) | OpenAI `/v1`, auto-discovers the HF cache | OpenAI + Anthropic + Ollama + Responses |

Choose `mlx_lm.server` (§3) as the default — it ships with the
installer and every harness above works against it. Choose oMLX when
decode speed matters or you need VLM models over HTTP; it is a source
install and discovers models from `~/.cache/huggingface` by itself.
Choose mlx-serve when you want the Anthropic and Ollama APIs on one
port — on a Mac it is a `brew install` away; on Omarchy it builds from
the fork's Linux port but is currently the slowest of the three
(~2× slower than `mlx_lm.server`, ~6× slower than oMLX on 7B 4-bit),
so pick it for the API surface, not the speed.

### mlx-serve on Omarchy: the Linux port

Upstream states the server is "macOS / Apple Silicon only" and the
v26.9.4 release ships only macOS assets. A Linux build exists on the
fork ([`linux-vulkan-port`](https://github.com/joshuaswarren/mlx-serve/tree/linux-vulkan-port),
opened upstream as
[ddalcu/mlx-serve#473](https://github.com/ddalcu/mlx-serve/pull/473)):
a target-gated Zig build graph that links the same `mlx-c` binding
against the mlx-omarchy Vulkan fork of MLX (the same `libmlx` the
wheel uses — the server and the Python stack agree token-for-token in
greedy mode). Build:

```bash
git clone --recurse-submodules -b linux-vulkan-port \
  https://github.com/joshuaswarren/mlx-serve && cd mlx-serve
./scripts/fetch-zig.sh && export PATH="$PWD/.zig-toolchain:$PATH"
# stage the Linux mlx tree once (mlx-omarchy checkout):
#   cd <mlx-omarchy> && scripts/prepare-mlx.sh   # prints .work/mlx
MLX_SOURCE=<mlx-omarchy>/.work/mlx ./scripts/build-mlx-linux.sh
zig build -Doptimize=ReleaseFast
./zig-out/bin/mlx-serve --version
```

System deps: cmake, Vulkan headers (a Vulkan 1.2 driver at runtime),
`libwebp`, `libdns_sd` (Arch: avahi; Debian:
`libavahi-compat-libdnssd-dev`).

Serve a model (note: `--model`, not `--model-dir` — the latter is a
discovery root and boots an empty server):

```bash
mlx-serve --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/<hash> \
  --serve --host 127.0.0.1 --port 11234
```

Requests work with `"model": "default"`; `/v1/models` reports an
internal hash id, not a name.

Rough edges of the Linux path (all observed on 2026-09-19, see the
[four-leg receipt](../receipts/2026-09-19-mlxserve-linux-port-t6001-test-host.md)
for the full list): MLX safetensors only (no GGUF — the embedded
llama.cpp is stubbed); context pinned to the model card maximum
(4096 for Qwen2.5-7B); decode ~5.7 tok/s on 7B 4-bit because the
standard MLX op set over Vulkan carries none of the Metal fast paths
the macOS build uses; both required submodules must be checked out or
the build fails (loudly, since the port landed configure-time
checks).

Bench model verification (2026-09-19): both HF repos exist, are not
gated, Apache-2.0 (`mlx-community/Qwen2.5-7B-Instruct-4bit`, last
modified 2024-11-06; `mlx-community/Ministral-3-8B-Instruct-2512-4bit`,
2025-12-06). Qwen2.5-7B is pinned as the verified-loadable bench
default, not as current SOTA — see the September-2026 SOTA paragraph in
§2 for the current generation.

## 5. Point harnesses at it

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

## 6. Chat without a harness

```bash
mlx-omarchy-demo
mlx-omarchy-demo --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

Or `mlx-omarchy -m mlx_lm.generate --model … --prompt '…'`.

## 7. Stop

Ctrl-C the server. Uninstall: `bash install.sh --uninstall` (models stay
in `~/.cache/huggingface`).

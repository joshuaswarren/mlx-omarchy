# Local generation and HTTP serving on Omarchy

mlx-omarchy provides the Linux GPU backend. A successful CLI generation run
is not proof that an HTTP server supports the same model.

## Current model and qualification

The current example is
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit).
Verified online **2026-09-20**: Apache-2.0, ungated, last modified
2026-09-14; [Hugging Face metadata](https://huggingface.co/api/models/mlx-community/Qwen3.8-27B-4bit).
The tested revision is `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.
This is a multimodal checkpoint; text-only generation passed a fresh
Python 3.14 environment on t6021-test-host with candidate
`0.32.3.dev202609201346+a1251aaa`, `mlx-vlm==0.7.1` and mlx-lm 0.31.3.
The candidate is not yet a published release.

Use the [README text-generation example](../README.md#quick-start) and its
[pinned install receipt](../receipts/2026-09-20-qwen38-text-install/receipt.json).
Compilation was enabled; the older diagnostic wheel
`b744f4dd` required `MLX_DISABLE_COMPILE=1`, which is not a current-main limitation.
Image inference and Qwen3.8 HTTP serving on this Linux stack are not yet
qualified. Model-card server snippets alone do not establish compatibility.

The checkpoint contains approximately 15 GB of weights. Disk size is not
peak runtime memory. A 16 GiB M1 has not passed this model's memory and
generation gates; no smaller current-generation Q4 replacement is qualified
here yet. The MTP checkpoint is an auxiliary prediction component, not a
standalone smaller language model.

## Install and check the backend

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
mlx-omarchy -c 'import mlx.core as mx; print(mx.device_info())'
```

Confirm the Apple GPU / Honeykrisp backend, not a software Vulkan device.
Install additional loaders in the same environment as mlx-omarchy; do not
replace its wheel with the upstream macOS package.

## HTTP server contract

The installer includes `mlx-lm==0.31.3`. The following is a server template,
**not a qualified Qwen3.8 invocation**. Set `MODEL_ID` only after validating
that exact checkpoint with the selected server and wheel.

```bash
: "${MODEL_ID:?Set a model already qualified with this server}"
mlx-omarchy -m mlx_lm.server \
  --model "$MODEL_ID" --host 127.0.0.1 --port 8080
```

Keep the server on loopback. These development servers are not an
authenticated production front door; do not expose them directly to a LAN
or the internet. A dummy API key is appropriate only for a loopback endpoint
that does not require authentication.

Inspect `/v1/models`, then use its actual model ID in requests. A successful
model listing is not a generation smoke test: `/v1/chat/completions` must
return a completion ID and nonempty assistant content.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/models
```

For OpenAI-compatible clients, configure:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=mlx
```

Select the reported model ID in the client's provider settings. Codex,
omp, pi, Hermes and OpenClaw have client-specific configuration; the common
contract is the endpoint and model ID, not necessarily these environment
variables. Claude Code uses Anthropic Messages rather than OpenAI chat;
it needs a separately configured compatible proxy. Do not assume an
OpenAI URL alone makes that protocol work. Stop a foreground server with
Ctrl-C.

## Historical server measurements — not current model recommendations

The 2026-09-19 comparison used **Qwen2.5-7B-Instruct-4bit**, v0.7.1 on
t6001-test-host (M1 Max), 37 prompt tokens, greedy decoding, 128 maximum output tokens,
and eight interleaved rounds. These are archival results, not predictions
for Qwen3.8 or evidence that those servers load it.

| Server | Historical decode tok/s | Linux scope at measurement |
|---|---:|---|
| mlx_lm.server | 10.36 | Bundled Python server |
| oMLX | 33.5 | Source installation |
| mlx-serve | 5.65 | `linux-vulkan-port` source build; MLX safetensors only |

See the [four-leg receipt](../receipts/2026-09-19-mlxserve-linux-port-t6001-test-host.md)
and [earlier comparison](../receipts/2026-09-19-serve-options-bench-t6001-test-host.md)
for reproduction, versions and limitations. Linux mlx-serve's GGUF and ANE
engines were not operational in that comparison. Its `/v1/models` ID may
be an internal hash rather than a Hugging Face repository name.

Other historical loader experiments, including the specialized Bonsai
runtime, remain in the [v0.7.0 recertification receipt](../receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md).
They are not drop-in HTTP-serving recommendations.

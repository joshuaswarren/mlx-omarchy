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
The published prerelease [v0.7.2-rc.1](https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.7.2-rc.1)
(`5b18306`) re-ran the same smoke on the same model revision with the same
result ([receipt](../receipts/2026-09-20-release-v0.7.2-rc.1/m2-receipt.json)).
Image inference and Qwen3.8 HTTP serving on this Linux stack are not yet
qualified. Model-card server snippets alone do not establish compatibility.

The checkpoint contains approximately 15 GB of weights. Disk size is not
peak runtime memory. A 16 GiB M1 has not passed this model's memory and
generation gates; no smaller current-generation Q4 replacement is qualified
here yet. The MTP checkpoint is an auxiliary prediction component, not a
standalone smaller language model.

## Model status for serving (2026-09-20)

"Recommended" here requires a qualification pass on real hardware — a
generation gate alone does not make a model recommended. As of the
2026-09-20 integration, three catalog entries have passed both
generation and HTTP on device: the Qwen3.8-27B-4bit chat checkpoint, the
Bonsai-2-27B module backend, and the Laya typed-decision endpoint. The
catalog still flags no recommendation pending the integration decision.
Per-entry status:

| Model | Verified online (HF API) | Serving status here |
|---|---|---|
| [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 2026-09-20, Apache-2.0, ungated | **Qualified: text CLI and functional HTTP on device (t6001-test-host)** — revision `10c35caa`. Text CLI: [install receipt](../receipts/2026-09-20-qwen38-text-install/receipt.json). HTTP: against **raw `mlx_lm.server` 0.31.3** — 4 measured + 3 streamed requests, all HTTP 200 with identical text ([raw qualification receipt](../receipts/2026-09-20-qwen38-http-mlxlm-t6001-test-host-raw-qualification.md)); observed rates in that receipt are explicitly not a performance claim. Limits: the managed CLI/shim launch smoke is still queued (this receipt does not validate the shim), HTTP-vs-direct numerical equivalence is under investigation, image input is not qualified, and the recommendation flag stays off pending the integration decision. |
| [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | 2026-09-20, Apache-2.0, ungated | **Qualified: generation and HTTP on device (t6001-test-host)** via the dedicated `mlx_omarchy_bonsai2` module backend ([receipt](https://github.com/joshuaswarren/ane-linux-experiments/commit/2e4b78f)). The pack's bundled runtime, which is remote code, is never executed; the server enforces a hard context cap and reports the pack's LICENSE/NOTICE with the required attribution. Historical coherent decode on this pack: ~1.44 tok/s ([v0.7.0 recert receipt](../receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md)). |
| [`empero-ai/Qwen3.8-35B-A3B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) | 2026-09-20, Apache-2.0, ungated | Not qualified. No load or serve test recorded on this stack; device qualification is planned. Scripting trap for this GDN/hybrid family: `mlx_lm.generate_step` takes a 1-D `[S]` prompt tensor while direct `model()` calls take `[B,S]` — use the CLI or handle shapes explicitly. |
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) — served through the in-repo `mlx_omarchy_laya` conversion @ `1c5edc17` | 2026-09-20, Apache-2.0, ungated | **Typed decisions endpoint qualified on device (t6001-test-host)**: generation and HTTP both pass, 6/6 frozen numerical gates, with concurrent real co-serving against an external resident chat service ([receipt](../receipts/2026-09-20-laya-gpu-qual-t6001-test-host.md)). Managed-reservation co-serving: not exercised. Not in a release, and the catalog still recommends nothing pending the integration decision — see below. |

## Laya typed-decision serving

Laya is not a chat LLM: it is a ~421M-parameter non-autoregressive typed
decision model (choice / score / yes-no questions) with calibrated
probabilities and an explicit escalate/abstain probability per answer,
served as a second model with its own model id and its own typed
decision endpoint — a shape `mlx_lm.server` does not offer. Serving it
is two steps: the in-repo `mlx_omarchy_laya` converter first converts
convaiinnovations/laya @ `1c5edc17` into a local checkpoint, then
`mlx_omarchy_laya.server` serves that checkpoint — conversion and
serving are separate commands.

On 2026-09-20 the decisions endpoint passed device qualification on an
M1 Max: 6/6 frozen numerical gates (fp16 GPU against an fp32 CPU
fixture), managed memory admission before load, and concurrent real
co-serving — the standing chat service stayed resident and both
endpoints answered real requests in the same second
([receipt](../receipts/2026-09-20-laya-gpu-qual-t6001-test-host.md)). Scope limits
that receipt records: co-serving evidence is the external-resident chat
(observed through `MemAvailable`); co-serving between two managed
reservations was not exercised. The code ships with the release that
contains it, and the catalog's recommendation flags stay off pending the
integration decision.

## Serving catalog CLI

The serve front door is `mlx-omarchy-serve`, exposed as `omarchy mlx
serve …` when the command-center launcher is installed. It is in this
source tree (serving packages integrated 2026-09-20) but **not yet in a
published release or the installer**: a release install gains it only
when a release carries it, and old installs never poll anything. From a
source checkout, run it from the repository root as
`PYTHONPATH=serve python -m mlx_omarchy_serve` — the PYTHONPATH is
required so the server's child processes can import the top-level
`mlx_omarchy_*` packages; a bare `python -m mlx_omarchy_serve` fails,
and the `serve.mlx_omarchy_serve` spelling resolves `plan`/`--help` but
is not the validated child-launch environment.

Downloads are approve-first: interactive runs require typed approval
before anything downloads, and a memory-aware admission gate must pass
first (details below). The catalog currently flags **every entry
`recommended: false`**, so the automatic pick names no model — name a
target explicitly. Manual targets, exactly as `--help` defines them:

```bash
PYTHONPATH=serve python -m mlx_omarchy_serve plan qwen3.8-27b-4bit --offline
PYTHONPATH=serve python -m mlx_omarchy_serve serve mlx-community/Qwen3.8-27B-4bit
PYTHONPATH=serve python -m mlx_omarchy_serve serve /path/to/local/model --context 4096
PYTHONPATH=serve python -m mlx_omarchy_serve catalog list|status|refresh
PYTHONPATH=serve python -m mlx_omarchy_serve reserve NAME GIB [--note TEXT]   # memory another local service owns
PYTHONPATH=serve python -m mlx_omarchy_serve unreserve NAME
```

`target` is a catalog id, a Hugging Face `org/name`, or a local model
directory. `plan` runs the admission and disk checks only — it downloads
nothing; `serve` plans, requires approval, downloads, and launches in
the foreground. Unqualified manual targets proceed only with a loud
warning.

The catalog it reads seeds ten pinned entries — six Qwen3.8-27B
quantizations (4-bit through bf16/mxfp4/nvfp4/mxfp8), the
Qwen3.8-35B-A3B-Distill pair, Ternary-Bonsai-2-27B, and the laya-mlx
decision model. Three entries carry qualification records, each
generation and HTTP: `qwen3.8-27b-4bit`, `bonsai-2-27b-mlx-2bit`, and
`laya-mlx`. Every other entry is an untested placeholder until it passes
the same bars — generation and HTTP are gated separately, and the
catalog flags no recommendation pending the integration decision.

Before anything downloads, a memory-aware gate budgets `MemAvailable` as
weights (the full total for MoE; a 35B-A3B counts its full 35B) plus KV at
the model's explicit context limit, plus a labeled workspace-margin
estimate and a safety reserve, minus reservations declared with
`reserve` (a pending reservation subtracts from `MemAvailable`; a
resident one does not, because the memory is already in use). A no-fit
is rejected: the gate never trades swap or OOM for
a download, and a pass is a conservative pre-flight check, not a live
admission coordinator between concurrent servers. Disk free is checked
before download. Interactive runs
require typed approval before any download; noninteractive runs fail
closed without `--yes`, and `--yes` requires an explicit target — a
recommendation is never downloaded silently.

Selection: `recommend` orders the catalog by curated priority within a
kind among tested/compatible fits, not "largest model that fits". Manual
use accepts a catalog id, an explicit Hugging Face repo, or a local path.
The automatic pick requires an entry that is recommended, generation- and
HTTP-qualified, with a working backend and a memory fit — every catalog
entry currently reads `recommended: false`, so auto-pick names no model
today; manual targets with unqualified status proceed only with a loud
warning.

Context is enforced on the mlx-lm path by a project shim
(`_mlxlm_server.py`): every request is capped so prompt and output
together stay within the admitted context — upstream `--max-tokens` alone
does not do this (it is an output-only per-request default). The shim
pins mlx-lm to 0.31.3 and refuses other versions loudly, rejects
non-integer token arguments, and launches with decode/prompt concurrency
1 so exactly one request's tokens are resident; a nonzero
`--prompt-cache-size` is sized into the memory admission before launch.
(The shim's own HTTP acceptance suite validates this cap independently;
the Qwen HTTP qualification above ran against raw `mlx_lm.server`.)
The omlx backend has no verified server-side cap and launches with an
honest warning saying so. `--server module` routes only through an
audited in-repo allowlist (`mlx_omarchy_laya.server`,
`mlx_omarchy_bonsai2.server`); catalog-named modules are never executed
directly.

Catalog refresh fetches from GitHub raw only: conditional ETag, 5 s
timeout, atomic rename over a last-known-good copy, with a bundled
fallback before the first successful fetch. `MLX_OMARCHY_OFFLINE=1` (or
per-command `--offline`) disables refresh entirely,
`MLX_OMARCHY_CATALOG_TTL_HOURS` (default 24) gates staleness,
`MLX_OMARCHY_CATALOG_URL` can repoint the source, and
`MLX_OMARCHY_HOME` relocates state. The TTL check runs only inside
serve/catalog invocations — no daemon, no telemetry.

Safety boundaries as designed: loopback bind by default, `trust_remote_code`
is never enabled, unsupported models error honestly, nothing downloads in
the background. The unit suite is green (71 tests, 2026-09-20) and the
command shapes above are real, but green tests and help output are not
behavioral proof: the open review findings above gate acceptance, HTTP
serving is unqualified, and no performance claim is made.

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

The measured rate is end-to-end HTTP wall-clock divided by
`usage.completion_tokens`: it includes prefill, detokenization and HTTP
handling, and is not a decode-only figure. All legs were single-stream —
no concurrency was measured — and all four servers were co-resident on
one GPU in one shared window, which is fair across legs but understates
each server's solo absolute rate. mlx-serve's prompt-lookup decoding made
no measurable difference on this prompt (5.65 tok/s on vs 5.70 off). The
earlier two-leg run that day measured the mlx_lm.server leg at 9.33
tok/s versus 10.36 in the four-leg run; the four-leg numbers are the
canonical comparison because every leg shared that window.

See the [four-leg receipt](../receipts/2026-09-19-mlxserve-linux-port-t6001-test-host.md)
and [earlier comparison](../receipts/2026-09-19-serve-options-bench-t6001-test-host.md)
for reproduction, versions and limitations. These measurements establish
nothing about current-generation models: they predate Qwen3.8
qualification and must not be read as a "fastest server for current
models" claim. Linux mlx-serve's GGUF and ANE engines were not
operational in that comparison. Its `/v1/models` ID may be an internal
hash rather than a Hugging Face repository name.

Other historical loader experiments, including the specialized Bonsai
runtime, remain in the [v0.7.0 recertification receipt](../receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md).
They are not drop-in HTTP-serving recommendations.

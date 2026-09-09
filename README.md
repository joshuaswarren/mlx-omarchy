# mlx-omarchy

MLX on Apple Silicon Linux.

![Primitive coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning. Upstream it runs on Metal, which means macOS. Mesa's Honeykrisp driver now gives Apple GPUs a conformant Vulkan 1.4 stack on Linux, and mlx-omarchy is the MLX GPU backend built on it. Your code still says `import mlx.core as mx` and `mx.gpu`; it runs on the Apple GPU under Linux, with no Metal and no CPU fallback.

This is early, actively developed software. Check the [compatibility table](docs/compatibility.md) and the [defect ledger](docs/known-defects.md) before depending on it.

## Demo

https://github.com/user-attachments/assets/7b2326f0-4679-4784-9622-e403b99be853

One-command install on an M1 running Omarchy, the first model download, the streamed answer with its measured tokens per second, and the launcher entry; 2:47, unedited, no narration. Also at [joshuaswarren.github.io/mlx-omarchy](https://joshuaswarren.github.io/mlx-omarchy/).

## Hardware

Supported today: Apple M1 running [Omarchy](https://github.com/omarchy-mac/omarchy-mac) (Asahi-based) with Mesa Honeykrisp. Later Apple Silicon generations follow once the M1 path is complete.

## Install (v0.4.0)

On an M1 running Omarchy, one command installs the release wheel into a private
venv under `~/.local/share/mlx-omarchy`, adds `mlx-omarchy` and
`mlx-omarchy-demo` to `~/.local/bin`, and registers **MLX Chat (Apple GPU)** in
the Omarchy launcher. It never replaces Mesa or edits Omarchy files.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
mlx-omarchy-demo
```

[demo/README.md](demo/README.md) walks through what to expect. Remove it all
with `bash install.sh --uninstall`.

Manual install, or any other Linux box:

```bash
# Apple Silicon (M1, Honeykrisp) - Python 3.14; any Linux box (x86_64, software Vulkan, development only) - Python 3.11
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install <wheel URL for your platform from https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.4.0>
```

Wheel filenames carry the build commit, so the exact URLs and SHA256 sums are in the release notes and the `SHA256SUMS` asset, not here.

Building from source is covered in [docs/install-omarchy.md](docs/install-omarchy.md). Build dependencies: Python 3.10+, CMake 3.25+, Vulkan headers, a C++ compiler, and LAPACK/BLAS development packages; the wheel needs `liblapack.so.3` and `libblas.so.3` at runtime.

Do not install the upstream `mlx` package beside this wheel; both provide the `mlx` module. `mlx-lm` depends on upstream `mlx`, so install it with `pip install --no-deps mlx-lm` and add its own dependencies (`transformers[sentencepiece] numpy protobuf pyyaml jinja2 huggingface_hub`) as `install.sh` does.

## Quick start

```python
import mlx.core as mx

x = mx.array([[1.0, 2.0], [3.0, 4.0]])
w = mx.array([[0.5], [0.25]])

def loss(w):
    return mx.exp(x @ w).sum()

value, grad = mx.value_and_grad(loss)(w)
print(value, grad)
```

Run a language model with [mlx-lm](https://github.com/ml-explore/mlx-lm):

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 32
```

## Performance

Published v0.4.0 aarch64 wheel, installed by `install.sh`, on an Apple M1 with the stock Omarchy Mesa 26.1.7 Honeykrisp driver, AC power, the pinned Qwen2.5-0.5B-Instruct-4bit snapshot, greedy decoding, fixed output lengths, one run each ([receipt](receipts/2026-09-09-v0.4.0-release.md)):

| Prompt / generated tokens | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| 30 / 32 | 78.60 | 100.3 |
| 262 / 128 | 64.79 | 255.9 |
| 1053 / 32 | 44.95 | 272.9 |

The BF16 model decodes at about 11 tok/s on the stock driver and about 40 on the optional [Honeykrisp fork build](docs/install-omarchy.md#honeykrisp-driver-with-the-fork-fixes).

Since that wheel, main carries four paired-verified changes with every generated token unchanged on every measured leg: the exact SwiGLU chain fused into one dispatch by default ([receipt](receipts/2026-09-09-fused-chain-default/verdict.json)), the Q4 GEMV rewritten to native Metal's arithmetic order ([receipt](receipts/2026-09-09-q4-gemv-order/verdict.json)), and the prefill wave: four-wide 16-bit binary kernels, a register-blocked f16 attention matmul, causal softmax without a materialized mask, and a straight-line SwiGLU kernel ([receipt](receipts/2026-09-09-prefill-speed/verdict.json), Q4 prefill +21% at 262 tokens and +81% at 1053). Main on the M1 now ([smoke](receipts/2026-09-09-main-prefill-smoke/verdict.json)):

| Prompt / generated tokens | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| 30 / 32 | 87.93 | 112.4 |
| 262 / 128 | 71.32 | 544.7 |
| 1053 / 32 | 48.48 | 854.7 |

`MLX_OMARCHY_FUSED_CHAIN=0`, `MLX_OMARCHY_QMM_VEC_Q4_WORD=0`, and `MLX_OMARCHY_QMM_TILE_RB=0` disable the fused chain, the decode kernel, and the prefill kernel for comparison.

Performance parity is still open. Historical macOS MLX 0.32.2 medians on this M1 were 150.8 / 146.6 / 140.3 decode tok/s and 294 / 1213 / 1838 prefill tok/s for the same three workloads. The historical short and 1024-context token-ID hashes match Linux, but the long-prompt hashes differ: native `254d73fd93164b98`, Linux `4cc08910089477fd`. These cross-OS timings do not establish numerical parity. [macOS receipts](receipts/native-baseline-2026-09-06).

To reproduce a leg, use the fixed-length runner, which suppresses EOS:

```bash
MLX_DISABLE_COMPILE=1 python3 scripts/bench_decode.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the capital of France? Answer in one word." \
  --tokens 64
```

The full matrix runs through `scripts/bench_matrix.py --mode run` on either operating system.

## Feature parity

Primitive coverage and upstream test coverage measure different things:

| Measure | Result | Date |
|---|---|---|
| MLX primitives with a working GPU kernel (the badge above) | 128 / 130 | 2026-09-05 |
| Upstream MLX C++ test cases passing on the GPU device | 251 / 251 | 2026-09-06 |
| Upstream MLX Python test cases passing on the GPU device | 10,767 / 11,437 | 2026-09-06 |

The first counts operations: a primitive counts once it computes on the GPU and a test verifies its values against a host reference. The other two run upstream's own test suites, pinned at the commit the backend is built from (MLX 0.32.2, `1f8e74e3`); one test case exercises many primitives across many dtypes and layouts, so they are the stricter measure. The table is a dated full-suite snapshot, not a qualification of the current development branch.

The [Python failure classification](receipts/upstream-suite-2026-09-06-py4/case-classification.csv) at `ef188d58` records 471 named refusals, 149 assertion failures, and 50 other errors, including 40 watchdog timeouts. All 670 remain failures in that snapshot; a named refusal does not count as support. GPU operations never fall back silently to CPU execution. Per-primitive status is generated from source in [docs/compatibility-matrix.md](docs/compatibility-matrix.md).

At `f5ba1c82`, fresh native M1 checks pass all 40 runtime cases and the packed-word prefill case (11,140 assertions). The full native upstream C++ run passes 250 of 251 cases. Its remaining exact-equality failure is `log(3)`, one float32 ULP below the host result; an installed-wheel comparison reproduces the same value in v0.3.7. It remains a failure, not qualified parity.

Earlier development checks ran on x86_64 software Vulkan. The [fresh parent build](receipts/2026-09-07-complex-matmul-parent.json) at `1947f2e5` passes all 33 complex-number cases, 98 primitive cases, and 13 convolution cases. It includes the complex unary fixes, singleton-batch matmul broadcasting, and separate convolution flip, stride, and dilation handling. The runtime suite passes all 40 cases on three runs after replacing timing-dependent and exact-wording assertions with observable contract checks.

Earlier installed-wheel [operation checks](receipts/2026-09-07-operation-parity.json) and [expanded BLAS checks](receipts/2026-09-07-dense-coverage.json) retain their failures and source identities. Those wheels predate these fixes. The [fresh 17-file core Python qualification](receipts/2026-09-08-core-python-requal/core-python-qualification.json) at `e3ee3c51` records 584 passed, 25 skipped, and three failed on software Vulkan, running NumPy 2.2.6 oracles so the prior `unstack` oracle gap is closed. The FFT dispatch, complex broadcast extraction, and repeated reflect-padding defects no longer reproduce; large sine, int64 sorting, and real-FFT output at 2^24 remain unsupported, the last as an explicit named refusal rather than wrong values. The quantized/fast partition remains in progress. These separate checks do not establish full upstream or native parity.

### What works

- Arrays, elementwise math with general broadcasting, reductions (including bool sums), softmax, logsumexp, cumulative sum, searchsorted
- Comparisons over float, integer, bool, int64 (where the device supports it), and complex64
- Real and complex dense matmul, including transposed and high-rank batched inputs; grouped-query attention; convolution in 1-D, 2-D, and 3-D, forward and transposed, with groups and dilation
- Quantized matmul and dequantize: affine, 2/3/4/5/6/8-bit, group sizes 32/64/128, plus gathered expert matmul
- Autograd: `value_and_grad`, `vjp`, `jvp` on device; `vmap`
- `mx.compile` for every operation class upstream fuses
- Sort, argsort, argpartition, top-k, argmax, argmin; bit-exact threefry random generation at 8/16/32-bit widths
- FFT at arbitrary lengths, including rfft/irfft on any axis
- complex64 arithmetic and transport
- Linear algebra in float32: Cholesky, inverse, LU, QR, eigh, SVD, `linalg.solve`
- Scatter with atomic sum/product, bool scatter, masked scatter
- Safetensors load and save
- Explicit CPU streams run upstream's CPU implementations through the same API; GPU streams never fall back silently
- Two-rank distributed ring over loopback: AllReduce, AllGather, Send, Recv

### Known gaps

- Compiled bfloat16 graphs are refused (`MLX_DISABLE_COMPILE=1` runs them eagerly); see the [ledger entry](docs/known-defects.md).
- `ReduceScatter` remains unavailable with the Linux ring transport.

The full list of open defects, with the platform each was observed on, is in [docs/known-defects.md](docs/known-defects.md).

## Neural Engine

The Apple Neural Engine is a planned internal accelerator for static graph regions, not a user-facing device. The open-source [MIL-to-HWX compiler](https://github.com/joshuaswarren/mil-hwx-compiler) builds on Linux and emits HWX programs without Apple's toolchain. Its M1 backend now has partial native operation and program-chain receipts; connected MLX graph integration remains unqualified. Design and bundle contract: [docs/architecture.md](docs/architecture.md), [docs/ane-bundles.md](docs/ane-bundles.md).

## Contributing

### Send us your hardware results

The most useful thing an M-series owner can do is run the collector and submit the report. It records chip, kernel, Mesa and Vulkan versions, correctness probes, and a benchmark sweep; it redacts user names, host names, paths, and addresses before anything is written, shows you the exact payload, and sends nothing without your explicit consent. Reports feed the public [community dataset](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results), which decides what gets fixed next.

```bash
python3 -m venv ~/.venvs/mlx-collect
~/.venvs/mlx-collect/bin/pip install <mlx-omarchy wheel for your machine>
~/.venvs/mlx-collect/bin/python scripts/collect_deep.py \
  --out mlx-omarchy-deep.tar.gz \
  --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
```

`scripts/collect_quick.py` is the no-install version: hardware and driver identity only, a few seconds, no network. Details of what is collected and how it is redacted are in [CONTRIBUTING.md](CONTRIBUTING.md). Query the dataset with `python3 scripts/query_community_data.py list`.

### Code

Start with the [contributor guide](docs/CONTRIBUTOR-GUIDE.md); it lists the open work and the verification each change needs. Development on any Linux machine works with a software Vulkan driver by setting `MLX_OMARCHY_ALLOW_NON_APPLE=1`; GPU kernel changes are verified on Apple hardware before release.

- [docs/roadmap.md](docs/roadmap.md): release plan
- [docs/compatibility.md](docs/compatibility.md): feature status by area
- [docs/known-defects.md](docs/known-defects.md): open and fixed defects

## License

MIT. The prepared MLX source keeps Apple's MIT license and copyright notices. This project is not affiliated with Apple.

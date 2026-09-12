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

## Install (v0.4.2)

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
~/.venvs/mlx/bin/pip install <wheel URL for your platform from https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.4.2>
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

Historic release measurement, kept for reference — the published v0.4.0 aarch64 wheel, installed by `install.sh`, on an Apple M1 with the stock Omarchy Mesa 26.1.7 Honeykrisp driver, AC power, the pinned Qwen2.5-0.5B-Instruct-4bit snapshot, greedy decoding, fixed output lengths, one run each ([receipt](receipts/2026-09-09-v0.4.0-release.md)):

| Prompt / generated tokens | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| 30 / 32 | 78.60 | 100.3 |
| 262 / 128 | 64.79 | 255.9 |
| 1053 / 32 | 44.95 | 272.9 |

### Current main: measured, not qualified

The composed tree at `a2e38c3e` was measured as a single unit on 2026-09-12 (later commits carry receipts and documentation only at this writing). All 36 canonical digest cells hold on both drivers. The standing M1 battery passes 29 of 30 targets; the one failure is the [ledgered, still-open quantized-matmul affine-offset wrong value](docs/known-defects.md), so the baseline gate stays open and neither the tree nor these numbers are a qualification ([receipt](receipts/2026-09-12-parity-status/README.md)).

Measured on one wheel built from this tree, fork driver, median of three fresh-process repetitions per driver after a discarded warmup, AC power. The denominator is the committed macOS MLX 0.32.2 baseline on an Apple M1, five repetitions with stable digests ([receipt](receipts/native-baseline-2026-09-06/native-2026-09-06-summary.json)): Q4 decode 150.57 / 146.77 / 140.38 tok/s and prefill 294.1 / 1213.0 / 1840.9; BF16 decode 56.43 / 55.72 / 54.55 and prefill 232.6 / 1007.7 / 1655.7. The fork build reaches these fractions of native:

| Model | Prompt / generated tokens | Decode vs native | Prefill vs native |
|---|---|---:|---:|
| Q4 | 30 / 32 | 0.74 | 1.13 |
| Q4 | 262 / 128 | 0.74 | 0.80 |
| Q4 | 1053 / 32 | 0.69 | 0.60 |
| BF16 | 30 / 32 | 0.56 | 0.57 |
| BF16 | 262 / 128 | 0.55 | 0.58 |
| BF16 | 1053 / 32 | 0.46 | 0.41 |

Stock driver (same wheel, same protocol; the FMA prefill route engages
only on drivers without cooperative matrix, so these are the numbers stock
users see):

| Model | Prompt / generated tokens | Decode vs native | Prefill vs native |
|---|---|---:|---:|
| Q4 | 30 / 32 | 0.67 | 0.58 |
| Q4 | 262 / 128 | 0.55 | 0.52 |
| Q4 | 1053 / 32 | 0.38 | 0.36 |
| BF16 | 30 / 32 | 0.57 | 0.57 |
| BF16 | 262 / 128 | 0.55 | 0.48 |
| BF16 | 1053 / 32 | 0.46 | 0.30 |

`MLX_OMARCHY_FUSED_CHAIN=0`, `MLX_OMARCHY_QMM_VEC_Q4_WORD=0`, and `MLX_OMARCHY_QMM_TILE_RB=0` disable the fused chain, the decode kernel, and the prefill kernel for comparison.

The step-by-step path, with its predecessor tables and per-change receipts, is history: the 2026-09-10 canonical 12-leg matrix ([receipt](receipts/2026-09-10-main-parity-12-matrix/verdict.md)); the paired-verified kernel work — SwiGLU chain fusion, native-order Q4 GEMV, the prefill wave, the retiled prefill `qmm_t`, grouped decode GEMVs, native-order decode attention, and paired KV-cache updates ([fused chain](receipts/2026-09-09-fused-chain-default/verdict.json), [Q4 GEMV](receipts/2026-09-09-q4-gemv-order/verdict.json), [prefill wave](receipts/2026-09-09-prefill-speed/verdict.json), [prefill qmm](receipts/2026-09-09-prefill-qmm/verdict.json), [decode fusion](receipts/2026-09-09-decode-fusion/verdict.json)); the dense BF16 decode GEMV with digest re-pins under the [2026-09-10 policy amendment](docs/parity-id-policy.md) ([receipt](receipts/2026-09-10-bf16-decode-gemv-land/verdict.json)); the BF16 prefill staging form ([receipt](receipts/2026-09-11-bf16-prefill/verdict.json)); and the composed-tree qualification with its two corrections — prefill-span parsing and the scalar-fill ordering fix ([qualification](receipts/2026-09-12-composed-main-qualification/README.md), [corrections](receipts/2026-09-12-composed-regressions/README.md), [prefill gap](receipts/2026-09-11-bf16-prefill-gap/README.md)).

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
| Upstream MLX C++ test cases passing on the GPU device | 251 / 251 | 2026-09-11 |
| Upstream MLX Python test cases passing on the GPU device | 11,483 / 11,847 | 2026-09-11 |

The first counts operations: a primitive counts once it computes on the GPU and a test verifies its values against a host reference. The other two run upstream's own test suites, pinned at the commit the backend is built from (MLX 0.32.2, `1f8e74e3`); one test case exercises many primitives across many dtypes and layouts, so they are the stricter measure. The table is a dated full-suite snapshot, not a qualification of the current development branch.

The 2026-09-11 snapshot ran both suites at `2a9add42` on an Apple M1 (Honeykrisp Vulkan): every C++ case passes, and the [Python failure classification](receipts/2026-09-11-upstream-suite/case-classification.csv) records 342 named refusals, 13 assertion failures, and 9 other errors — 364 failures, down from 670 (471 / 149 / 50) at the [2026-09-06 snapshot](receipts/upstream-suite-2026-09-06-py4/case-classification.csv), with no watchdog timeouts in either suite phase. A named refusal does not count as support, and GPU operations never fall back silently to CPU execution. The two standing refusal clusters are the quantized-matmul weight layout (258 cases) and bfloat16-input quantize (83 cases); the 13 remaining wrong-value cases, ranked by shared root cause with primitive and shape/dtype signatures, are in the [snapshot notes](receipts/2026-09-11-upstream-suite/notes.md). The executed-case denominator grew by 410 since 2026-09-06 because upstream's device-conditional sweeps run more subtests on this host and the earlier environment skipped the quantization sweep; the per-category deltas and case-level analysis account for it. Per-primitive status is generated from source in [docs/compatibility-matrix.md](docs/compatibility-matrix.md).

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

A Core ML / Parakeet compatibility lane is [planned](docs/plans/2026-09-12-coreml-parakeet-ane-plan.md) on this stack: a pinned public Parakeet Core ML model, its encoder compiled by the pinned MIL-to-HWX compiler and executed on the ANE, decoder and joint on the GPU, no inference server and no CPU tensor fallback. It is planned, not shipped — no Core ML or Parakeet capability exists in any release today.

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

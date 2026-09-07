# mlx-omarchy

MLX on Apple Silicon Linux.

![Primitive coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning. Upstream it runs on Metal, which means macOS. Mesa's Honeykrisp driver now gives Apple GPUs a conformant Vulkan 1.4 stack on Linux, and mlx-omarchy is the MLX GPU backend built on it. Your code still says `import mlx.core as mx` and `mx.gpu`; it runs on the Apple GPU under Linux, with no Metal and no CPU fallback.

This is early, actively developed software. Check the [compatibility table](docs/compatibility.md) and the [defect ledger](docs/known-defects.md) before depending on it.

## Hardware

Supported today: Apple M1 running [Omarchy](https://github.com/omarchy-mac/omarchy-mac) (Asahi-based) with Mesa Honeykrisp. Later Apple Silicon generations follow once the M1 path is complete.

## Install (v0.3.8)

```bash
# Apple Silicon (M1, Honeykrisp) - Python 3.14
python3.14 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install \
  https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.3.8/mlx_omarchy-0.32.2.dev202609071529%2Bf5ba1c8-cp314-cp314-linux_aarch64.whl

# Any Linux box (x86_64, software Vulkan, development only) - Python 3.11
python3.11 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install \
  https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.3.8/mlx_omarchy-0.32.2.dev202609071536%2Bf5ba1c82-cp311-cp311-linux_x86_64.whl
```

Building from source is covered in [docs/install-omarchy.md](docs/install-omarchy.md). Build dependencies: Python 3.10+, CMake 3.25+, Vulkan headers, a C++ compiler, and LAPACK/BLAS development packages; the wheel needs `liblapack.so.3` and `libblas.so.3` at runtime.

Do not install the upstream `mlx` package beside this wheel; both provide the `mlx` module.

The published aarch64 v0.3.8 wheel completed all three pinned Qwen workloads on M1 with the normal compilation settings and default kernels. Generated-token counts and hashes match the paired eager runs. [Public-wheel smoke results](receipts/2026-09-07-q4-second-wave/public-v038-default.json) and [output comparison](receipts/2026-09-07-q4-second-wave/public-default-comparison.json).

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

The default kernels at `f5ba1c82` improve Linux decode by 3.5% to 6.5% and prefill by 12.9% to 21.1% over v0.3.7 (`417c06e6`). Five alternating baseline/candidate pairs ran on the same Apple M1 with Honeykrisp, AC power, the pinned Qwen2.5-0.5B-Instruct-4bit snapshot, greedy decoding, and fixed output lengths. Every generated token ID matched in every pair. These measurements use eager execution, `MLX_DISABLE_COMPILE=1`.

| Prompt / generated tokens | v0.3.7 decode tok/s | Current decode tok/s | v0.3.7 prefill tok/s | Current prefill tok/s |
|---|---:|---:|---:|---:|
| 30 / 32 | 66.02 | 70.28 | 87.464 | 99.010 |
| 262 / 128 | 56.25 | 59.12 | 210.104 | 254.122 |
| 1053 / 32 | 40.77 | 42.27 | 227.725 | 273.294 |

Values are medians of five runs; percentage gains are medians of the paired ratios. Prefill now unpacks each Q4 word once for eight weights. Increasing the bounded graph batch from 100 to 256 nodes reduces submission overhead without changing synchronization or buffer ownership. The temporary comparison kernel and its switch were removed.

[Raw comparisons, token arrays, numerical checks, and runtime receipts](receipts/2026-09-07-q4-second-wave) cover the new changes. [Earlier Q4 results](receipts/2026-09-07-q4-kernel-gains) retain the v0.3.7 measurements. The existing `MLX_OMARCHY_QMM_VEC_Q4_WORD=0` and `MLX_OMARCHY_QMM_TILE_RB=0` switches disable the decode and prefill optimizations for comparison.

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

Earlier installed-wheel [operation checks](receipts/2026-09-07-operation-parity.json) and [expanded BLAS checks](receipts/2026-09-07-dense-coverage.json) retain their failures and source identities. Those wheels predate these fixes. The [fresh 17-file core Python qualification](receipts/2026-09-07-q4-second-wave/core-python-qualification.json) at `f5ba1c82` records 582 passed, 24 skipped, and six failed on software Vulkan. Three failures expose FFT dispatch, complex broadcast extraction, and repeated reflect-padding defects; large sine and int64 sorting remain unsupported. The sixth fails because NumPy 1.26 lacks `unstack`; that unchanged test [passes separately with NumPy 2.2.6](receipts/2026-09-07-q4-second-wave/numpy2-unstack.json). The quantized/fast partition remains in progress. These separate checks do not establish full upstream or native parity.

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
- A single `mx.eval` over a very long full-sequence forward (about 2,048 tokens in one operation) can wedge the GPU queue; chunked prefill, which mlx-lm uses, is unaffected.
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

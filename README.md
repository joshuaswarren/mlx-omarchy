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

The development branch raises Linux decode by 75% to 156% and prefill by 60% to 165% over `f5ba1c82` on the same Apple M1 with Honeykrisp (Mesa 26.1.7), AC power, the pinned Qwen2.5-0.5B-Instruct-4bit snapshot, greedy decoding, fixed output lengths, and eager execution (`MLX_DISABLE_COMPILE=1`). Each step was measured as alternating same-wheel or cross-wheel pairs on an otherwise idle machine; values are medians of the latest pairs.

| Prompt / generated tokens | f5ba1c82 decode tok/s | Current decode tok/s | f5ba1c82 prefill tok/s | Current prefill tok/s | macOS MLX 0.32.2 decode / prefill |
|---|---:|---:|---:|---:|---:|
| 30 / 32 | 68.2 | 119.6 | 85.2 | 135.8 | 150.8 / 294 |
| 262 / 128 | 55.8 | 111.8 | 207.0 | 462.9 | 146.6 / 1213 |
| 1053 / 32 | 41.3 | 105.8 | 225.8 | 598.6 | 140.3 / 1838 |

What changed, in order of effect. The 4-bit decode matrix-vector kernel now streams 8 bytes per lane with two rows per subgroup in 128-thread workgroups; the 68 MB output projection moved from 34 to 64 GB/s. Elementwise, RoPE, copy, and RMSNorm kernels compile one pipeline per operation and layout through specialization constants, which cuts a small dispatch from 8.7 to about 3.4 microseconds of GPU time; Honeykrisp flushes after every dispatch, so this floor is the driver's, not the barrier's. Decode attention runs as one flash-decoding dispatch per layer (two above 256 keys) with float32 scores. The prefill matmuls use 64x64 and 64x128 register-blocked tiles (0.28 to 0.60 TFLOP/s at 1,053 tokens; the Honeykrisp compiler serializes shared-memory loads, which caps exact fp32 GEMM near 0.6 TFLOP/s). KV-cache growth no longer joins the GPU queue for each of the 48 zero-filled buffers per prefill, and cache updates donate their buffer instead of copying it. Kill switches: `MLX_OMARCHY_QMM_VEC_Q4_V2=0`, `MLX_OMARCHY_SDPA_FUSED=0`, `MLX_OMARCHY_QMM_TILE_V2=0`, `MLX_OMARCHY_MATMUL_GEMM=0`, `MLX_OMARCHY_ROPE_GRID=0`, `MLX_OMARCHY_COPY_GRID=0`, `MLX_OMARCHY_RMS_NORM_SUBGROUP=0`, `MLX_OMARCHY_COPY_DONATE=0`, `MLX_OMARCHY_NO_PUSH_DESCRIPTORS=1`.

Short and 1,024-context token IDs are byte-identical to `f5ba1c82`; the 262-token workload diverges at generated token 48, where the float32 attention scores flip a near tie (the fused path has lower error than the old path against a double-precision reference on every tested shape; [evidence](receipts/2026-09-07-perf-dispatch/sdpa-fusion-report.md)). Neither the old nor the new long-prompt output matches the macOS hash `254d73fd93164b98`. Native C++ suites: 863 of 869 pass; the six failures (float32 `log(3)` one ULP off, three complex-number edge cases, and their two aggregates) fail identically on `f5ba1c82` on this GPU. [Step-by-step paired runs](receipts/2026-09-07-perf-dispatch), [prior-art survey](receipts/2026-09-07-perf-dispatch/prior-art-survey.md), [native suite log](receipts/2026-09-07-perf-dispatch/perfsnap8-native-ctest.log).

Performance parity is still open: decode is at 75% to 79% of macOS and prefill at 33% to 46%. Every remaining dispatch pays Honeykrisp's per-dispatch cache flush (Mesa `hk_cmd_dispatch.c`), so the next steps are fewer dispatches per token and a higher-throughput GEMM, both documented in the survey. [macOS receipts](receipts/native-baseline-2026-09-06). Earlier Q4 results: [second wave](receipts/2026-09-07-q4-second-wave), [first wave](receipts/2026-09-07-q4-kernel-gains).

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

Earlier installed-wheel [operation checks](receipts/2026-09-07-operation-parity.json) and [expanded BLAS checks](receipts/2026-09-07-dense-coverage.json) retain their failures and source identities. Those wheels predate these fixes. The [fresh 17-file core Python qualification](receipts/2026-09-07-q4-second-wave/core-python-qualification.json) at `f5ba1c82` records 582 passed, 24 skipped, and six failed on software Vulkan. Two failures expose FFT dispatch and complex broadcast extraction defects; large sine and int64 sorting remain unsupported. Both defects are fixed on the development branch: the [fixed build](receipts/2026-09-07-core-parity-fixes/corefix-build-source.json) passes the native M1 FFT, complex-layout, and shape suites (4,751 assertions) and 15 of the 16 runnable upstream FFT cases, refusing by name the 2^24-point irfft whose real-extraction pass exceeds the single-dispatch group limit (65,535 groups of 256) ([log](receipts/2026-09-07-core-parity-fixes/fft-python-corefix.xml)). The other two come from NumPy 1.26: it lacks `unstack` and has a [known repeated-reflection bug](receipts/2026-09-07-core-parity-fixes/reflect-oracle-analysis.json). Both unchanged tests pass separately with NumPy 2.2.6; the original six-failure aggregate remains intact. The recovered quantized/fast run completed quantized tests, reached its 1,800-second SDPA deadline, and completed the remaining 14 files separately. SDPA qualification continues in isolated per-case processes. These separate checks do not establish full upstream or native parity.

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

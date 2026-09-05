# mlx-omarchy

MLX on Apple Silicon Linux.

![MLX coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning. It is tied to Metal, which means macOS. Mesa now ships Honeykrisp, a conformant Vulkan 1.4 driver for Apple GPUs on Linux. mlx-omarchy is the Vulkan backend that runs MLX on that path. Your model code still reads `import mlx.core as mx`. It now runs on the Apple GPU under Linux.

This is early work in the open. Read the compatibility table and the defect ledger before you plan anything around it.

## Hardware

The supported target is an Apple M1 running [Omarchy](https://github.com/omarchy-mac/omarchy-mac) with Mesa Honeykrisp. Later Apple Silicon generations come after the M1 path is complete. Progress by area is in [docs/compatibility.md](docs/compatibility.md); the design is in [docs/architecture.md](docs/architecture.md).

Honeykrisp is the Apple GPU Vulkan driver inside [Mesa](https://gitlab.freedesktop.org/mesa/mesa) under `src/asahi/vulkan/` (the `hk_` prefixed files), sharing the AGX shader compiler with the OpenGL driver. Our fork is [`joshuaswarren/mesa`](https://github.com/joshuaswarren/mesa). All five shader miscompiles this project isolated live in that compiler; one crash bug lived in Mesa's shared submit-thread runtime.

## Install (v0.3.5)

v0.3.5 ships wheels for both platforms. Pick the one that matches your machine.

```bash
# Apple Silicon (M1, Honeykrisp) — Python 3.14
python3.14 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install \
  https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.3.5/mlx_omarchy-0.32.2.dev202609040917%2B0535e62-cp314-cp314-linux_aarch64.whl

# Any Linux box (x86_64, llvmpipe, development only) — Python 3.11
python3.11 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install \
  https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.3.5/mlx_omarchy-0.32.2.dev202609041010%2B0535e62-cp311-cp311-linux_x86_64.whl
```

A clean install from source is in [docs/install-omarchy.md](docs/install-omarchy.md). Build dependencies: Python 3.10+, `cmake` 3.25+, Vulkan development headers, a C++ compiler, and `liblapack-dev libblas-dev liblapacke-dev` on Debian-family distributions; the wheel needs `liblapack.so.3` and `libblas.so.3` at runtime.

Do not install the upstream `mlx` package beside this wheel; the module name is the same.

## Performance

Native Metal parity is not yet established. The archived native decode runs
stopped at EOS after 2–10 tokens and used mlx-lm 0.30.2; current Linux
measurements use pinned output lengths and mlx-lm 0.31.3. The former README
ratios of 9.4× and 7.0× slower mixed those protocols and are withdrawn.
A matched same-chip comparison must use the same model revisions, prompts,
output lengths, and timing definitions on both operating systems.
See the [native measurement](receipts/2026-09-01-m1-same-chip-parity.md) and
[pinned-length correction](receipts/2026-09-03-decode-metric-fix.md).

Validated M1 Linux prefill results, five alternating pairs on one development wheel:

| Prompt / generated tokens | Untiled prefill tok/s | Tiled prefill tok/s | Throughput ratio | Decode tok/s OFF → ON |
|---|---|---|---|---|
| 30 / 32 | 36.058 | 79.365 | 2.20× | 30.96 → 31.07 |
| 262 / 128 | 36.587 | 187.545 | 5.13× | 28.41 → 28.32 |
| 1053 / 32 | 35.963 | 197.524 | 5.49× | 23.53 → 23.53 |

Tiled quantized prefill is enabled by default in development builds. Every
comparison above preserved generated token IDs; no decode gain is claimed.
Both bf16 RoPE/SDPA candidates changed the 128-token greedy output and remain
disabled. These are Linux OFF/ON measurements, not Metal comparisons or
published-wheel results. [Conditions and gate decisions](receipts/2026-09-04-m1-performance-gates.md).

Development code also keeps KV state slices as views. Native checks recorded
no slice-evaluation copy and no replacement copy in the tested matmul and
attention consumers ([receipt](receipts/2026-09-04-kv-state-views.md)).
Fusion passes its numeric tests but stays off by default. In model runs, it
did not cut GPU calls or show a clear speed gain
([receipt](receipts/2026-09-04-swiglu-fused-chain.md)).

> **A green run on a software Vulkan driver (llvmpipe, lavapipe) proves nothing about the Apple GPU.** Four of the v0.3.0 defects never appeared on a development box: bool scatter, 33-element `LogicalAnd`, broadcast `select`, and the `mx.sin`/`mx.cos` range-reduction collapse. llvmpipe passed the full battery the whole night those shipped. Numbers in this README were measured on Honeykrisp; verify them on Honeykrisp before quoting them.

## What works

- Arrays, elementwise math with general broadcast, reductions, softmax, logsumexp, cumulative sum, sorted-row search
- Dense, transposed, and broadcast-batch matmul up to rank 5; grouped-query attention with scores computed in float32; grouped, depthwise, 1-D, and dilated forward convolution
- Quantized matmul and dequantize: affine, 4-bit and 8-bit, group sizes 32 and 64, plus gathered expert matmul
- Autograd: `value_and_grad`, `vjp`, `jvp` on device; `vmap` over elementwise closures
- `mx.compile` over 51 op classes (elementwise, comparison, logical, select, broadcast); the fused RoPE pair (per-batch vector offsets, the inverse / VJP path) is fenced to the composed path because the fused variants have not passed equivalence
- Sort, argsort, argpartition, and top-k over float and integer rows up to 1024 elements; argmax and argmin; threefry-exact random sampling
- FFT at arbitrary lengths: composites decompose into radix-2 passes, primes ride a Bluestein chirp-z, and non-trailing-axis rfft and irfft compute
- complex64 end to end: arithmetic, Conjugate, Real, Imag, and transport through reshape, slice, pad, concatenate
- Linear algebra in float32: Cholesky, inverse, LU with pivots, QR, Eigh, SVD; upstream's composed `linalg.lu()` and `linalg.solve()` run end to end
- Scatter with float32 atomic sums and products (hardware atomics where the driver advertises them, compare-exchange kernels where it does not), bool scatter, masked scatter
- Safetensors load and save
- Explicit CPU streams run upstream's CPU implementations — sort, eigh, topk, LAPACK-backed linear algebra — through the same API. GPU streams never fall back silently.
- A two-rank distributed ring over loopback: AllReduce, AllGather, Send, Recv are value-proven at group size 2; ReduceScatter refuses by name
- An installable wheel: distribution `mlx-omarchy`, module `mlx`

## Known gaps and defects

The honest list, with the platform each one was observed on. The full ledger is [docs/known-defects.md](docs/known-defects.md).


Anything not on the defect ledger fails loudly with a named `[omarchy] ... is not implemented` error. This is the contract.

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

The forward pass and the gradient both run on the Apple GPU. No Metal, no macOS.

Run a language model:

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 32
```

For steady-state decode, use the pinned-length harness instead of `--max-tokens` (which is a cap; generation usually stops at EOS first):

```bash
python3 scripts/bench_decode.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the capital of France? Answer in one word." \
  --tokens 64
```

It suppresses EOS, asserts the produced token count, and prints "decode X tok/s over Y tokens" with prefill reported separately.

For the C++ test battery and the dispatch-count profiler, see [docs/install-omarchy.md](docs/install-omarchy.md). Development machines without an Apple GPU can run everything on any Vulkan 1.3 driver, llvmpipe included, with `MLX_OMARCHY_ALLOW_NON_APPLE=1`. That flag proves nothing about the Apple GPU — read the callout above.

## The Neural Engine plan

The ANE has no public compiler, so this project splits the work. A macOS machine compiles supported graph regions into versioned bundles. Each bundle holds the compiled program, its weights, and a manifest that pins graph identity, tensor contracts, compiler and firmware identity, and payload hashes. Linux validates every field before it maps a single byte, then executes the bundle through the open [eiln/ane](https://github.com/eiln/ane) driver. A region without a valid bundle stays on Vulkan.

Today the exporter (`tools/ane-export`) compiles fp16 add and multiply regions on macOS, and the Linux validation gate (`mlx-omarchy-info --check-bundle`) accepts or rejects them. Execution on Linux is still blocked on the ANE device node. See [docs/ane-bundles.md](docs/ane-bundles.md) and [docs/ane-hwx-format-notes.md](docs/ane-hwx-format-notes.md).

## Contributing

Start at the [contributor guide](docs/CONTRIBUTOR-GUIDE.md). It splits the open work by whether it needs Apple hardware, names the strategies that were tried and stood down (with the numbers), and pins the verification bar. The project owns one Apple Silicon machine (`jwm1`); `MLX_OMARCHY_ALLOW_NON_APPLE=1` development is the rest.

- [CONTRIBUTING.md](CONTRIBUTING.md): community data collection, source rules, proof required
- [docs/roadmap.md](docs/roadmap.md): proof-gated release plan
- [docs/CONTRIBUTOR-GUIDE.md](docs/CONTRIBUTOR-GUIDE.md): open work, killed strategies, entry-point commands
- [docs/known-defects.md](docs/known-defects.md): the live defect ledger
- [docs/compatibility.md](docs/compatibility.md) and [docs/compatibility-matrix.md](docs/compatibility-matrix.md): primitive-by-primitive status, generated from source

Hardware receipts from M-series machines always help.

## License

MIT. The prepared MLX source keeps Apple's MIT license and copyright notices. Apple is not involved with this project.
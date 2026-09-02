# mlx-omarchy

MLX on Apple Silicon Linux.

![MLX coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning. It is fast and well designed. It is also tied to Metal, which means macOS. Mesa now ships Honeykrisp, a conformant Vulkan 1.4 driver for Apple GPUs on Linux. mlx-omarchy connects the two. Your model code still reads `import mlx.core as mx`. It now runs on the Apple GPU under Linux.

This is early work in the open. Read the two tables below before you plan anything around it.

## Why this exists

An M1 MacBook running [Omarchy](https://omarchy.org) is a great Linux machine with a GPU that Linux can finally drive well. [omarchy-mac](https://github.com/omarchy-mac/omarchy-mac) puts Omarchy on Apple Silicon in an afternoon. But no ML framework treats the result as a first-class target. PyTorch sees a CPU. MLX sees nothing at all.

So the goal here is bigger than a port. Run the full MLX stack, gradients included, on the Apple GPU under Linux. Then bring up the Apple Neural Engine, the most locked-down accelerator Apple ships, as a second backend behind the same API. An Apple Silicon laptop running Omarchy should give up nothing for local ML.

## Where it actually stands

**Coverage: 87.2% of MLX primitives (116 of 133), up from 30.1% on 2026-09-01.** A primitive counts only on two conditions. The backend must have a real computing path for it. A test must verify its values against a host reference. The generator enforces both, so neither is a claim. An eval body that only ever refuses is not an implementation. A test that pins error messages, or carries a skip marker, is not an anchor. Tightening those rules on 2026-09-02 first dropped the number from 80.5% to 76.7%. Real work earned it back, and the same pass found two undercounts. One caveat: a `partial` primitive counts. It computes for the dtypes, layouts, and modes it implements, and refuses the rest by name. So 87.2% means "does something correct and proven." It never means "is complete." The full breakdown, with every named-error constraint per primitive, is generated from source into [docs/compatibility-matrix.md](docs/compatibility-matrix.md). Regenerate it with:

```bash
python3 tools/gen-compat-matrix.py --json-out docs/coverage.json > docs/compatibility-matrix.md
```

**Upstream MLX's own test suite, run against this backend.** These numbers come from the 2026-09-01 sweep. They predate the six coverage waves of 2026-09-02, so they understate the current state. They have not been re-run, and no newer figure is claimed here. The C++ suite executed 251 cases: 62 passed, 189 failed. The python suite executed 10,500: 1,730 passed, 8,770 failed, 56 skipped. Almost every failure was a named `[omarchy] ... is not implemented` error. Integer-dtype Sum, ErfInv, Abs, and Scatter led that list, and all four compute now. The sweep also found five real defects, all fixed: `log2` and `log10` returned the natural log, sums over expanded axes wrote out of bounds, a backend error crossing numpy's buffer protocol killed the interpreter, cross-thread stream lookup aborted, and saving a zero-size array segfaulted.

Of that sweep's two caveats, one is now resolved. The suite's own `array_equal` helper could not run, because of a bool And/Equal gap. Many of its passing cases were therefore unverified rather than proven correct. `array_equal` now computes correctly here, so a re-run would carry real signal for the first time. Still open: 72 python tests were excluded, because they crashed the interpreter before the buffer-protocol fix landed. Re-run the sweep with `tools/run-upstream-suite.sh`; raw logs from the original are in [receipts/](receipts/2026-09-01-upstream-suite-coverage.md).

**Performance against native MLX, same chip.** Both columns are the same Apple M1 (T8103, 8 GPU cores, 16 GB), same model revisions, same prompts, `--max-tokens 32 --temp 0 --seed 0`, warm run, Qwen2.5-0.5B-Instruct:

| Measurement | macOS 13.7.8, MLX on Metal | Omarchy Linux, mlx-omarchy on Vulkan | Ratio |
|---|---|---|---|
| bf16 prefill | 377.9 tok/s | 18.3 tok/s | 20.6x slower |
| bf16 decode | 61.5 tok/s | 2.5 tok/s | 24.9x slower |
| bf16 peak memory | 1.025 GB | 0.993 GB | about equal |
| 4-bit prefill | 705.6 tok/s | 19.2 tok/s | 36.8x slower |
| 4-bit decode | 290.3 tok/s | 4.2 tok/s | 68.7x slower |
| 4-bit peak memory | 0.320 GB | 0.292 GB | about equal |

Generated text is identical on both platforms: `Hello! How can I assist you today?` for bf16, `Paris` for 4-bit, matching token counts and stop positions. Numerical correctness is there. Speed is not.

The gap is expected and unfixed. Every kernel is written for correctness first. There is no fusion, no tuning, and no fused attention kernel. The v0.2.0 microbenchmarks reached [more than 80% of a pinned llama.cpp Vulkan build](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.2.0-m1-kernel.json) on matmul and attention. The hardware path is not the problem. Full decode runs are where the work remains.

One caveat on the macOS column. That slice runs macOS 13.7.8. MLX dropped macOS 13 wheels after 0.29.3, so it measured mlx 0.29.3 and mlx-lm 0.30.2 against mlx-lm 0.31.3 on Linux.

## What makes it different

**A patch set, not a fork.** This repository contains no MLX source and no MLX history. `mlx.lock` pins one upstream release by SHA-256. `overlay/` adds the Vulkan backend beside Metal and CUDA, `patches/` carries a few small diffs, and `scripts/prepare-mlx.sh` assembles the tree. You can read every line this project adds in one sitting. Tracking upstream MLX is a version bump, not a rebase.

**No CPU fallback.** When a program hits an operation the backend does not support, it fails with the operation name, dtype, and shape. It never falls back to CPU silently. A number you measure on this backend is a number the GPU earned.

**Receipts, not claims.** Every number above comes from a recorded run on real hardware, stored in [receipts/](receipts/). The [M1 development gate](receipts/2026-08-31-m1-development-gates.md), the [MLX-LM generation attempts](receipts/2026-08-31-m1-mlxlm-fp16-smoke.md), and the [same-chip parity run](receipts/2026-09-01-m1-same-chip-parity.md) all record the exact commands and their output.

## What works today

- Language model inference from prompt to tokens: Qwen2.5-0.5B in bf16, fp16, and 4-bit affine quantization, greedy and temperature sampling
- Arrays, elementwise math with general broadcast, reductions, softmax, logsumexp, cumulative sum, sorted-row search
- Dense, transposed, and broadcast-batch matmul up to rank 5; native float32 attention scores; 2D convolution
- Quantized matmul and dequantize, affine, 4-bit and 8-bit, group sizes 32 and 64
- Autograd: `value_and_grad`, `vjp`, and `jvp` run on device; `vmap` over elementwise closures
- `mx.compile` interprets fused elementwise tapes
- Sort, argsort, argmax, argmin, threefry-exact random sampling
- Safetensors load and save without a CPU backend
- An installable wheel: distribution `mlx-omarchy`, module `mlx`

## Known gaps and defects

Honest list. Each one fails loudly with a named error rather than returning wrong numbers, except where noted.

- **v0.3.0-alpha.1 ships known silent wrong-value defects.** Ten operations return confidently wrong numbers and raise nothing, so watching for errors cannot catch them. The two worst hit mainstream use: attention is wrong for grouped-query models, which is what current Llama, Qwen, and Mistral all are, and the `mx.fast.layer_norm` weight gradient is wrong above 512 columns. Item-by-item repros and affected-use groups sit in [docs/known-defects-v0.3.0-alpha.1.md](docs/known-defects-v0.3.0-alpha.1.md). Fixes are in progress.
- Top-k sampling needs argpartition over vocabulary-width rows; the bitonic sort caps at 1024 elements. Temperature sampling works.
- No training. Optimizers, LoRA, and full backward coverage are untested.
- Linear algebra and FFT now compute: Cholesky, Inverse, QR, Eigh, and SVD run on real Jacobi and Householder kernels, and FFT covers power-of-two lengths 2 to 2048. Their remaining holes refuse by name: LU is gated pending numeric verification, QR refuses batches, non-power-of-two FFT lengths need Bluestein or mixed-radix, and Eigh and SVD refuse rather than return unconverged factors when the Jacobi sweep limit trips. Complex dtypes, float64, and distributed still raise named errors throughout.
- Three silent wrong-value defects were found and fixed on 2026-09-02, all in paths that had just become reachable: `eigvalsh` returned `[1, 1]` for a matrix whose spectrum is (7 ± √5)/2, `pinv` returned all zeros, and SVD returned an all-zero `Vt`. None of them raised. They are the reason the linalg suite now checks values against analytic references instead of properties like positivity and sortedness, which the identity matrix happens to satisfy.
- Performance is 20-69x behind Metal, as measured above.
- ANE export works: it exports and validates bundles but does not execute them yet; see below.

## Quick start

On an M1 running [Omarchy](https://github.com/omarchy-mac/omarchy-mac) with Mesa Honeykrisp:

```bash
git clone https://github.com/joshuaswarren/mlx-omarchy.git
cd mlx-omarchy
./scripts/build-wheel.sh
python3 -m venv ~/.venvs/mlx && ~/.venvs/mlx/bin/pip install dist/mlx_omarchy-*.whl
```

Then write MLX like you would anywhere:

```python
import mlx.core as mx

x = mx.array([[1.0, 2.0], [3.0, 4.0]])
w = mx.array([[0.5], [0.25]])

def loss(w):
    return mx.exp(x @ w).sum()

value, grad = mx.value_and_grad(loss)(w)
print(value, grad)
```

Both the forward pass and the gradient run on the Apple GPU. No Metal, no macOS.

Running a language model needs `mlx-lm` and one flag while the bf16 tape defect is open:

```bash
pip install mlx-lm
MLX_DISABLE_COMPILE=1 python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 32
```

For the C++ tests and tools, see [docs/install-omarchy.md](docs/install-omarchy.md). Development machines without an Apple GPU can run everything on any Vulkan 1.3 driver, llvmpipe included, with `MLX_OMARCHY_ALLOW_NON_APPLE=1`.

## The Neural Engine plan

The ANE has no public compiler, so this project splits the work. A macOS machine compiles supported graph regions into versioned bundles. Each bundle holds the compiled program, its weights, and a manifest. The manifest pins graph identity, tensor contracts, compiler and firmware identity, and payload hashes. Linux validates every field before it maps a single byte, then executes the bundle through the open [eiln/ane](https://github.com/eiln/ane) driver. A region without a valid bundle stays on Vulkan.

Today the exporter and the Linux validation gate both work. `tools/ane-export` compiles fp16 add and multiply regions on macOS. `mlx-omarchy-info --check-bundle` then accepts or rejects them on Linux. Execution on Linux is still blocked on the ANE device node. See [docs/ane-bundles.md](docs/ane-bundles.md) and the [HWX format notes](docs/ane-hwx-format-notes.md).

## Hardware

The supported target is an M1 running Omarchy with Mesa Honeykrisp; [omarchy-mac](https://github.com/omarchy-mac/omarchy-mac) is the installer. Later Apple Silicon generations come after the M1 path is complete. Progress by area lives in [docs/compatibility.md](docs/compatibility.md), and the design in [docs/architecture.md](docs/architecture.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](docs/roadmap.md). The most useful work right now is the missing primitives in the matrix. Kernel performance against the 20-69x gap comes next. Hardware receipts from M-series machines always help.

## License

MIT. The prepared MLX source keeps Apple's MIT license and copyright notices. Apple is not involved with this project.

# mlx-omarchy

MLX on Apple Silicon Linux.

![MLX coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning. It is fast and well designed. It is also tied to Metal, which means macOS. Mesa now ships Honeykrisp, a conformant Vulkan 1.4 driver for Apple GPUs on Linux. mlx-omarchy connects the two. Your model code still reads `import mlx.core as mx`. It now runs on the Apple GPU under Linux.

This is early work in the open. Read the two tables below before you plan anything around it.

## Why this exists

An M1 MacBook running [Omarchy](https://omarchy.org) is a great Linux machine with a GPU that Linux can finally drive well. [omarchy-mac](https://github.com/omarchy-mac/omarchy-mac) puts Omarchy on Apple Silicon in an afternoon. But no ML framework treats the result as a first-class target. PyTorch sees a CPU. MLX sees nothing at all.

So the goal here is bigger than a port. Run the full MLX stack, gradients included, on the Apple GPU under Linux. Then bring up the Apple Neural Engine, the most locked-down accelerator Apple ships, as a second backend behind the same API. An Apple Silicon laptop running Omarchy should give up nothing for local ML.

## Where it actually stands

**Coverage: 98.5% of upstream MLX's full primitive list (131 of 133), and 98.5% Mac parity (128 of 130 Mac-usable primitives), up from 30.1% on 2026-09-01.** A primitive counts only on two conditions. The backend must have a real computing path for it. A test must verify its values against a host reference. The generator enforces both, so neither is a claim. An eval body that only ever refuses is not an implementation. A test that pins error messages, or carries a skip marker, is not an anchor. Distributed primitives carry one further bar: they count only through a two-rank harness whose every case proves the multi-rank group in its own body, because a single-process run never constructs them. One caveat: a `partial` primitive counts. It computes for the dtypes, layouts, and modes it implements, and refuses the rest by name. So these numbers mean "this share of primitives does something correct and proven". They never mean "this share is complete".

Mac parity is the headline. It asks the question that matters on an Apple Silicon Linux machine: of the primitives MLX implements and a Mac user can actually reach, what share does this backend implement and value-test? The denominator comes mechanically from upstream's own source, never from a list curated by hand. A primitive is Mac-usable when upstream's Metal backend gives it a real `eval_gpu` body, or when upstream's CPU backend gives it a real `eval_cpu` body. Three primitives fail both tests: `fast::CrossEntropy`, `fast::CrossEntropyVJP`, and `fast::ScaledDotProductAttentionVJP`. All three are operations this backend implements and Metal does not. The redefinition therefore removed covered primitives from the numerator and denominator alike, and lowered this backend's own number. Perfect achievable parity is 129 of 130, or 99.2%. The one unclosable gap is `fast::CustomKernel`: it compiles user-supplied Metal shading-language source, and no Metal-to-SPIR-V translator exists in this stack.

The full breakdown, with every named-error constraint per primitive and every exclusion citation, is generated from source into [docs/compatibility-matrix.md](docs/compatibility-matrix.md). Regenerate it with:

```bash
python3 tools/gen-compat-matrix.py --json-out docs/coverage.json > docs/compatibility-matrix.md
```

**Upstream MLX's own test suites, run against this backend.** The 2026-09-02 sweep executed 251 C++ cases: 133 passed, 118 failed. It executed 10,932 python cases: 2,827 passed, 8,105 failed, 68 skipped, and one crash-excluded. The C++ pass count more than doubled since the 2026-09-01 sweep, and python passing grew by 1,097 cases. Most of that gain is measurement honesty: the bool Equal and buffer-protocol holes that masked or killed cases are closed, so a pass now carries real signal. Almost every failure is still a named `[omarchy] ... is not implemented` refusal: 7,909 of the 8,105 python failures. Five C++ cases and six python test functions exposed silent wrong-value defects, twelve distinct root causes in all. Most are fixed on this branch now; the still-open ones are listed below. The sweep ran at pinned commit `5f8ba16`, before the day's final coverage commits, so it understates this tree. Re-run it with `tools/run-upstream-suite.sh`. The defect ledger and raw logs sit in [receipts/](receipts/2026-09-02-upstream-suite-coverage.md); the 2026-09-01 sweep is in [receipts/](receipts/2026-09-01-upstream-suite-coverage.md).

**Performance against native MLX, same chip.** Both columns are the same Apple M1 (T8103, 8 GPU cores, 16 GB), same model revisions, same prompts, `--max-tokens 32 --temp 0 --seed 0`, warm run, Qwen2.5-0.5B-Instruct:

| Measurement | macOS 13.7.8, MLX on Metal | Omarchy Linux, mlx-omarchy on Vulkan | Ratio |
|---|---|---|---|
| bf16 prefill | 377.9 tok/s | 23.9 tok/s | 15.8x slower |
| bf16 decode | 61.5 tok/s | 3.56 tok/s | 17.3x slower |
| bf16 peak memory | 1.025 GB | 0.993 GB | about equal |
| 4-bit prefill | 705.6 tok/s | 25.3 tok/s | 27.9x slower |
| 4-bit decode | 290.3 tok/s | 6.46 tok/s | 44.9x slower |
| 4-bit peak memory | 0.320 GB | 0.292 GB | about equal |

Generated text is identical on both platforms: `Hello! How can I assist you today?` for bf16, `Paris` for 4-bit, matching token counts and stop positions. Numerical correctness is there. Speed is not.

The Vulkan tok/s column was measured on commit `ceae628`. It moved a long way from the previous revision, on the same machine in the same session: 4-bit prefill up 35.5%, 4-bit decode up 58.6%, bf16 prefill up 45.8%, bf16 decode up 71.2%. The memory rows carry over from the earlier revision and were not re-measured.

Two things about the Vulkan column are not like-for-like, and both are the product's fault rather than the benchmark's. The Vulkan legs run `MLX_DISABLE_COMPILE=1`. Compiled bf16 tapes corrupt nondeterministically on this driver, and the gate that catches them is deliberately kept. So a bf16 leg with compile at its default cannot be measured here at all. The macOS column ran compile at its default, where it works. 4-bit with compile enabled is unmeasured on Vulkan so far.

**Where the remaining time goes**, measured on the M1 over 7,853 dispatches of a profiled 4-bit run. A ranked list beats an apology. Recording a dispatch is 0.3% of host wall. A per-pipeline descriptor cache took that from 160-260 microseconds to under one, so per-dispatch cost is not a driver-side wall. Submitting is 14.3%, joining 0.4%. Waiting for a free submission slot is 45.4%. That is the GPU-backlog signature: the host blocks because the GPU is the slower side in those windows, so more slots would not help. The remaining 39.6% sits outside the encoder, in MLX's evaluator throttle and Python.

So the encoder's controllable overhead is now about 15% of wall, and that lever is close to spent. The next wins are kernel time and upstream's evaluation model. Elementwise and copies alone are 64% of GPU busy. Submission batching was tried and deleted: it fights the evaluator's own throttle and measured 4.5x slower. The v0.2.0 microbenchmarks reached [more than 80% of a pinned llama.cpp Vulkan build](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.2.0-m1-kernel.json) on matmul and attention, so the hardware path was never the problem.

One caveat on the macOS column. That slice runs macOS 13.7.8. MLX dropped macOS 13 wheels after 0.29.3, so it measured mlx 0.29.3 and mlx-lm 0.30.2 against mlx-lm 0.31.3 on Linux.

## What makes it different

**A patch set, not a fork.** This repository contains no MLX source and no MLX history. `mlx.lock` pins one upstream release by SHA-256. `overlay/` adds the Vulkan backend beside Metal and CUDA, `patches/` carries a few small diffs, and `scripts/prepare-mlx.sh` assembles the tree. You can read every line this project adds in one sitting. Tracking upstream MLX is a version bump, not a rebase.

**No silent CPU fallback.** When a program hits an operation the backend does not support, it fails with the operation name, dtype, and shape. Nothing reroutes GPU work to CPU silently; run the operation on an explicit CPU stream to use the CPU implementation. A number you measure on a GPU stream is a number the GPU earned.

**Receipts, not claims.** Every number above comes from a recorded run on real hardware, stored in [receipts/](receipts/). The [M1 development gate](receipts/2026-08-31-m1-development-gates.md), the [MLX-LM generation attempts](receipts/2026-08-31-m1-mlxlm-fp16-smoke.md), the [same-chip parity run](receipts/2026-09-01-m1-same-chip-parity.md), and the [M1 qualification of the coverage waves](receipts/2026-09-02-m1-qualification.md) all record the exact commands and their output.

## What works today

- Language model inference from prompt to tokens: Qwen2.5-0.5B in bf16, fp16, and 4-bit affine quantization, greedy and temperature sampling
- Arrays, elementwise math with general broadcast, reductions, softmax, logsumexp, cumulative sum, sorted-row search
- Dense, transposed, and broadcast-batch matmul up to rank 5; grouped-query attention scored in float32; grouped, depthwise, 1-D, and dilated forward convolution
- Quantized matmul and dequantize, affine, 4-bit and 8-bit, group sizes 32 and 64, plus gathered expert matmul
- Autograd: `value_and_grad`, `vjp`, and `jvp` run on device; `vmap` over elementwise closures
- `mx.compile` interprets tapes of 51 op classes: elementwise, comparison, logical, select, and broadcast
- Sort, argsort, argpartition, and top-k over float and integer rows up to 1024 elements; argmax and argmin; threefry-exact random sampling
- FFT at arbitrary lengths: composites decompose into radix-2 passes, primes ride a Bluestein chirp-z, and non-trailing-axis rfft and irfft compute
- complex64 end to end: complex arithmetic, Conjugate, Real, and Imag, and transport through reshape, slice, pad, and concatenate
- Linear algebra in float32: Cholesky, inverse, LU with pivots, QR, Eigh, and SVD; upstream's composed `linalg.lu()` and `linalg.solve()` run end to end
- Scatter with float32 atomic sums and products - hardware atomics where the driver advertises them, compare-exchange kernels where it does not - bool scatter, and masked scatter
- Safetensors load and save
- Explicit CPU streams run upstream's CPU implementations - sort, eigh, topk, and LAPACK-backed linear algebra - through the same API. GPU streams still never fall back silently.
- A two-rank distributed ring over loopback: AllReduce, AllGather, Send, and Recv are value-proven at group size 2; ReduceScatter refuses by name.
- An installable wheel: distribution `mlx-omarchy`, module `mlx`

## Known gaps and defects

Honest list. Each one fails loudly with a named error rather than returning wrong numbers, except where noted.

- **v0.3.0 shipped four defects that v0.3.1 fixes.** A Vulkan semaphore lifetime bug in the shared completion path could segfault any primitive - measured at 6 of 50 runs of one test binary in the v0.3.0 build configuration on Mesa lavapipe, and invisible to the validation layers. Real-M1-only wrong values in bool scatter, 33-element `LogicalAnd`, and `select` with broadcast or strided conditions: four suites were red on the M1 while the dev-box battery stayed green, because the miscompile never executes on llvmpipe. Twelve float scatter operations that refused by name on the M1 while running fine on the dev box. And `mx.sin`/`mx.cos` degradation on the M1 from arguments of about 1e4 upward - now refused by name above 1e5, eagerly and inside `mx.compile` alike: this backend's compiled tape dispatches every node through its `eval_gpu`, where the gate lives. An earlier commit message claimed a compiled-tape bypass; [docs/known-defects.md](docs/known-defects.md) retracts that claim with the evidence.
- **Still open, disclosed with the platform each one was observed on.** Eight of the twelve silent wrong-value root causes v0.3.0-alpha.1 shipped were fixed in v0.3.0 with value tests at the exact upstream failing shapes: grouped-query attention, `layer_norm` weight gradients above 512 columns, NaN-dropping first-axis reductions, `array_equal(equal_nan=True)`, int multi-axis sums, negative-axis `take`, dtype-converting `full_like`, and `log10`. The ninth, large-argument `sin`, became the v0.3.1 eager gate above. Three remain open in v0.3.1: `gelu_approx` under pytest process context, the fast SDPA vector path disagreeing with its own decomposition, and a one-ulp vjp. `cummax`/`cummin` NaN carry - a thirteenth defect, outside the twelve - was fixed in v0.3.0. The full ledger, with affected versions, symptoms, and platforms: [docs/known-defects.md](docs/known-defects.md).
- Top-k and argpartition over rows wider than 1024 elements refuse by name: the bitonic sort caps there. A wide-row ArgPartition kernel was built and then reverted, so the cap is pinned rather than open-ended. Rows up to 1024 partition exactly. Temperature sampling works.
- No training story yet. Optimizers, LoRA, and full backward coverage are unproven here, and upstream's optimizer tests still hit named gaps.
- Linear algebra and FFT compute but not everywhere. QR refuses batches pending numeric verification, and complex and float64 linalg refuse. Eigh and SVD refuse rather than return unconverged factors when the Jacobi sweep limit trips. Prime FFT lengths above 32768 refuse, because the chirp needs k squared exact in u32. Float64 raises named errors throughout.
- Three silent wrong-value defects were found and fixed on 2026-09-02, all in paths that had just become reachable: `eigvalsh` returned `[1, 1]` for a matrix whose spectrum is (7 ± √5)/2, `pinv` returned all zeros, and SVD returned an all-zero `Vt`. None of them raised. They are the reason the linalg suite now checks values against analytic references instead of properties like positivity and sortedness, which the identity matrix happens to satisfy.
- Performance is 16-45x behind Metal, as measured above, down from 20-69x.
- ANE export works: it exports and validates bundles but does not execute them yet; see below.

Five Honeykrisp driver miscompiles are isolated with minimal reproducing shaders, and all five are fixed or worked around in this repo. Bool-word loads inside divergent lane loops read zero except at 16-byte-aligned words ([receipt](receipts/2026-08-31-m1-mlxlm-fp16-smoke.md)). A data-dependent shift-and-mask feeding a shared-memory scan stops propagating mid-scan ([receipt](receipts/2026-09-02-masked-scatter-m1-fix.md)). A wide op selector over a per-byte path miscompiles bool comparisons, so they live in their own shader (commit `3c7d257`). Shift-then-mask byte extraction with a data-dependent shift amount drops bool scatter writes and corrupts `LogicalAnd` and `select` on the M1, and the workaround is per-site: neither byte-extraction form is safe by default on this hardware, so an eight-variant device probe pins every site (commit `959c7a0`, [receipt](receipts/2026-09-02-m1-red-suites-root-cause.md)). The same dynamic shift-then-mask inside `reduce_general.comp`'s `load_truthy` misreduced boolean results past the first 32-bit word - `mx.all` and `mx.any` both - and the fix replaced it with the constant-shift chain, device-probed at every boundary rather than trusted by analogy (commit `cf68e7d`). Anyone building compute on this driver should read those five first. A sixth defect stays open and gated: bf16 compiled tapes corrupt inside the real mlx-lm forward. On real M1 hardware, 15 identical-seed runs produced 15 different garbage outputs, prefill matched eager bit for bit through all 24 layers, and divergence began at decode step 2 ([receipt](receipts/2026-09-02-m1-bf16-compiled-tape.md)). No minimal shader repro exists, so it is suspected but not confirmed as a miscompile. `MLX_DISABLE_COMPILE=1` avoids it.

## Quick start

On an M1 running [Omarchy](https://github.com/omarchy-mac/omarchy-mac) with Mesa Honeykrisp:

```bash
git clone https://github.com/joshuaswarren/mlx-omarchy.git
cd mlx-omarchy
./scripts/build-wheel.sh
python3 -m venv ~/.venvs/mlx && ~/.venvs/mlx/bin/pip install dist/mlx_omarchy-*.whl
```

Build dependencies: Python 3.10+, `cmake` 3.25+, Vulkan development headers, a C++ compiler, and the BLAS/LAPACK development packages the CPU backend links (`liblapack-dev libblas-dev liblapacke-dev` on Debian-family distributions; the wheel needs `liblapack.so.3` and `libblas.so.3` at runtime). See [docs/install-omarchy.md](docs/install-omarchy.md).

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

Honeykrisp is not a separate project. It is the Apple GPU Vulkan driver inside [Mesa](https://gitlab.freedesktop.org/mesa/mesa), under `src/asahi/vulkan/` (the `hk_` prefixed files), and it shares the AGX shader compiler in `src/asahi/compiler/` with the OpenGL driver. There is a [GitHub mirror](https://github.com/Mesa3D/mesa). Anyone reading our driver-defect receipts will want those two directories: all five shader miscompiles this project isolated live in the compiler, and the one crash bug lived in Mesa's shared submit-thread runtime rather than the compiler.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](docs/roadmap.md). The most useful work right now is kernel time against the 16-45x gap - elementwise and copies are 64% of GPU busy, and the host encoder is already down to about 15% of wall, so kernels are where the remaining wins are. Hardware receipts from M-series machines always help.

### A gap that needs a project of its own

`fast::CustomKernel` is the one primitive this backend cannot implement, and the reason is worth stating plainly for anyone looking for a substantial problem to solve.

It compiles user-supplied Metal Shading Language source at runtime. Running that same source here needs an MSL front-end that targets SPIR-V, and no such compiler exists. The traffic all goes the other way: SPIRV-Cross translates SPIR-V into MSL, clspv handles OpenCL C, DXC handles HLSL. Apple's own Metal compiler does not run on Linux.

An MSL-to-SPIR-V compiler would unblock this, and it would unblock a great deal more than this project. Every Metal shader in every application becomes portable to Vulkan the day it exists. That work belongs in its own repository, not here. Anyone who takes it on should know at least one backend is waiting to use it.

A narrower option stays open to us, and it is not the same thing. Accepting GLSL or SPIR-V source through the same entry point would give Linux users custom kernels. It would not run a Mac user's Metal source, so it closes a usability gap rather than a parity gap, and the compatibility matrix would still count the primitive as unimplemented.

## License

MIT. The prepared MLX source keeps Apple's MIT license and copyright notices. Apple is not involved with this project.

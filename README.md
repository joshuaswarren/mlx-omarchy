# mlx-omarchy

MLX on Apple Silicon Linux.

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning. It is fast, well designed, and tied to Metal, which means macOS. Meanwhile Mesa now ships Honeykrisp, a conformant Vulkan 1.4 driver for Apple GPUs on Linux. mlx-omarchy connects the two. Your model code still reads `import mlx.core as mx`. It now runs on the Apple GPU under Linux.

## Why this exists

An M1 MacBook running [Omarchy](https://omarchy.org) is a great Linux machine with a GPU that Linux can finally drive well. [omarchy-mac](https://github.com/omarchy-mac/omarchy-mac) puts Omarchy on Apple Silicon in an afternoon. But no ML framework treats the result as a first-class target. PyTorch sees a CPU. MLX sees nothing at all.

So the goal here is bigger than a port. Run the full MLX stack, gradients included, on the Apple GPU under Linux. Then bring up the Apple Neural Engine, the most locked-down accelerator Apple ships, as a second backend behind the same API. An Apple Silicon laptop running Omarchy should give up nothing for local ML.

## What makes it different

**A patch set, not a fork.** This repository contains no MLX source and no MLX history. `mlx.lock` pins one upstream release by SHA-256. `overlay/` adds the Vulkan backend beside Metal and CUDA, `patches/` carries a few small diffs, and `scripts/prepare-mlx.sh` assembles the tree. You can read every line this project adds in one sitting. Tracking upstream MLX is a version bump, not a rebase.

**No CPU fallback.** When a program hits an operation the backend does not support, it fails with the operation name, dtype, and shape. It never falls back to CPU silently. A number you measure on this backend is a number the GPU earned.

**Receipts, not claims.** Every capability listed below links to a recorded run on real hardware. The [v0.2.0 kernel receipt](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.2.0-m1-kernel.json) holds 300 samples per case. The [current development receipt](receipts/2026-08-31-m1-development-gates.md) covers the primitive, autograd, and install runs on an M1.

## What works today

- Arrays, elementwise math, suffix reductions, dense and transposed matmul, and addmm on the GPU
- FP32, FP16, and emulated BF16 storage with rounding-correct conversion kernels
- Autograd: `value_and_grad`, `vjp`, and `jvp` run end to end on device
- `vmap` over elementwise closures, and `mx.compile` in `no_fuse` mode
- An installable wheel: distribution `mlx-omarchy`, module `mlx`
- Named errors for everything else: unsupported linear algebra, `float64`, complex dtypes

Kernel performance on M1, measured against a pinned `llama.cpp` Vulkan baseline: prefill above 92%, decode above 83%, attention above 80%.

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

For the C++ tests and tools, see [docs/install-omarchy.md](docs/install-omarchy.md). Development machines without an Apple GPU can run everything on any Vulkan 1.3 driver, llvmpipe included, with `MLX_OMARCHY_ALLOW_NON_APPLE=1`.

## The Neural Engine plan

The ANE has no public compiler, so this project splits the work. A macOS machine compiles supported graph regions into versioned bundles: the compiled program, its weights, and a manifest that pins graph identity, tensor contracts, compiler and firmware identity, and payload hashes. Linux validates every field before it maps a single byte, then executes the bundle through the open [eiln/ane](https://github.com/eiln/ane) driver. A region without a valid bundle stays on Vulkan.

The Linux validation gate is in the tree today; see [docs/ane-bundles.md](docs/ane-bundles.md). The compiler and execution stages are in progress, with groundwork recorded in [ane-linux-experiments](https://github.com/joshuaswarren/ane-linux-experiments).

## Hardware

The supported target is an M1 running Omarchy with Mesa Honeykrisp; [omarchy-mac](https://github.com/omarchy-mac/omarchy-mac) is the installer. Later Apple Silicon generations come after the M1 path is complete. Progress by area lives in [docs/compatibility.md](docs/compatibility.md), and the design in [docs/architecture.md](docs/architecture.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](docs/roadmap.md). The most useful contributions right now are missing primitives, kernel performance work, and hardware receipts from M-series machines.

## License

MIT. The prepared MLX source keeps Apple's MIT license and copyright notices. Apple is not involved with this project.

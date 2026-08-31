# mlx-omarchy

Use `mlx-omarchy` to run MLX on Apple Silicon Linux.
It uses Honeykrisp Vulkan.
Model code still uses `import mlx.core as mx` and `mx.gpu`.
The project does not copy Metal.

Release `v0.2.0` supports M1 systems with Omarchy.
The build handles Vulkan memory, copies, streams, events, and short waits.
In matched tests, prefill reached more than 92% of the pinned `llama.cpp` Vulkan result.
Decode reached more than 83%.
Attention reached more than 80%.
The [M1 receipt](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.2.0-m1-kernel.json) has 300 samples for each case.
Work on primitives, transforms, packages, and ANE support is not complete.

## Repository layout

This repository has the Linux backend, tests, build patch, and hardware receipts.
It does not have the MLX source or MLX commit history.

- `mlx.lock` pins the MLX archive and its SHA-256.
- `overlay/` has new files for the prepared MLX source.
- `patches/` has small changes to current MLX files.
- `scripts/prepare-mlx.sh` gets, checks, and prepares the source.

## Prepare and build

Install CMake, Ninja, and a C++ compiler.
Install Vulkan headers, a shader compiler, `curl`, `tar`, and `patch`.

```bash
./scripts/prepare-mlx.sh

cmake -S .work/mlx -B .work/build -G Ninja \
  -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=OFF \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF

cmake --build .work/build --target \
  omarchy_runtime_tests \
  omarchy_copy_offset_tests \
  omarchy_primitive_tests \
  mlx-omarchy-info
```

The prepare script gets MLX `v0.32.2` and puts it in the ignored `.work/` folder.
It checks the archive hash before it adds the project files.
Run gate scripts from the prepared source under `.work/mlx`.

## Product rules

- Vulkan runs all tensor work when ANE is off.
- A failed ANE gate sends that graph part back to Vulkan.
- The CPU never runs a tensor primitive.
- A missing primitive fails with its name, dtype, and shape.
- ANE runs as an internal graph-region accelerator.
- The first shared path uses host staging buffers.

## Roadmap

1. Vulkan runtime core
2. Vulkan kernel performance gate
3. Vulkan primitives and dtypes
4. Transforms, compilation, and automatic differentiation
5. Installable Linux package
6. ANE graph integration

Each release needs an M1 receipt.
The first ANE release must also pass the full hybrid gate.

## Project docs

- [System design](docs/architecture.md)
- [Proof gates](docs/roadmap.md)
- [Current status](docs/compatibility.md)
- [Full plan](docs/plans/2026-08-29-mlx-omarchy-ane-compatibility-plan.md)
- [How to help](CONTRIBUTING.md)

## Upstream projects

[MLX](https://github.com/ml-explore/mlx) supplies the public API and core code.
`mlx-omarchy` keeps a pinned Linux patch set for MLX.
`ane-linux-experiments` holds early ANE hardware tests.
[`eiln/ane`](https://github.com/eiln/ane) owns the Linux ANE driver and `libane` ABI.

## License

The project uses the MIT license.
The prepared MLX source keeps Apple's MIT license and copyright notices.
Apple does not sponsor this project.

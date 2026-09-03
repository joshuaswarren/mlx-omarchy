# Install mlx-omarchy

mlx-omarchy is MLX with the Omarchy Vulkan backend. The distribution name is
`mlx-omarchy`. The Python module is `mlx`.

## Supported target

Apple-silicon Honeykrisp GPUs are the supported target. The wheel builds on
Linux only.

## Development override

`MLX_OMARCHY_ALLOW_NON_APPLE=1` allows a desktop or software Vulkan driver,
for example llvmpipe, on a development machine. Do not set it on a supported
machine. Receipts from a development run must record that the device is a
development device, not Honeykrisp.

`MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1` lets compiled tapes run on a real
Apple GPU. The default runs them eager: at device discovery the runtime
disables compilation and prints one warning, because the tape
interpreter has produced silently wrong values there and the defect is
unpinned ([docs/known-defects.md](known-defects.md)). Eager output is
identical, only slower. Set the override only to investigate that defect
deliberately; it permits wrong values. The differential harness
(`scripts/differential_compile.py`, `scripts/probe_tape_eager.py`) sets
it for itself; `scripts/probe_compile_ordering.py` measures the default
path and runs without it.

## Build the wheel

1. Install the build tools: Python 3.10 or newer with `venv`, `cmake` 3.25 or
   newer, Vulkan development headers, a C++ compiler, and the BLAS/LAPACK
   development packages the CPU backend links (`liblapack-dev libblas-dev
   liblapacke-dev` on Debian-family distributions).
2. Run `./scripts/build-wheel.sh`
3. Read the wheel path, size, and sha256 from the receipt lines.

The script prepares the pinned upstream tree, builds with
`MLX_BUILD_OMARCHY=ON`, the CPU backend on, and the Metal and CUDA backends
off, and writes one wheel into `dist/`. The built wheel needs
`liblapack.so.3` and `libblas.so.3` at runtime.

## Install and smoke-test

1. Run `./tools/ci/run-clean-omarchy-install.sh`
2. Expect `clean install verified` as the last line.

The script creates a fresh venv, installs the newest wheel from `dist/`, and
runs an import check, an add, a matmul, and a gradient check under
`MLX_OMARCHY_ALLOW_NON_APPLE=1`.

## Install by hand

1. `python3 -m venv ~/.venvs/mlx-omarchy`
2. `~/.venvs/mlx-omarchy/bin/pip install dist/mlx_omarchy-*.whl`
3. `MLX_OMARCHY_ALLOW_NON_APPLE=1 ~/.venvs/mlx-omarchy/bin/python -c 'import mlx.core as mx; print(mx.default_device())'`

Do not install the upstream `mlx` package beside this wheel. The module name
is the same, so the two distributions conflict. Remove upstream `mlx` before
you install `mlx-omarchy`.

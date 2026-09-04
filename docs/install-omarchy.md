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

## Compiled-tape debug switches

Three switches exist to bisect the compiled-tape corruption on Honeykrisp
([docs/known-defects.md](known-defects.md)). They are diagnostics for the
defect hunt, not product configuration. All three are off by default;
with none set, nothing changes for any user. Each is read once per tape
evaluation and prints one line to stderr the first time a tape runs with
any of them active. All of them slow execution down, some severely. They
combine freely with `MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1`, which the
hardware hunt needs first.

- `MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1` submits each tape node as its own
  command buffer, one submission per node, matching the eager execution
  shape while running the tape's own code path. If the corruption
  disappears under this switch, the defect lives in the
  many-dispatches-per-command-buffer recording. If it persists, the
  command buffer is innocent and the tape's resource handling stays
  suspect.
- `MLX_OMARCHY_TAPE_FULL_BARRIERS=1` adds a full memory dependency
  (all commands, all memory access, both directions) before and after
  every dispatch, on top of the regular per-dispatch barriers. If
  per-node submission fixes the corruption but this switch does not, the
  driver is not honouring an in-buffer dependency it should, and that is
  a Honeykrisp finding worth its own minimal shader reproduction.
- `MLX_OMARCHY_TAPE_NO_REUSE=1` gives every dispatch fresh resources:
  the buffer cache is bypassed for the duration of a tape recording, so
  every intermediate lands in new device memory, and every dispatch gets
  its own descriptor pool with exactly one set. If per-node submission
  does not fix the corruption but this switch does, the defect is
  aliasing or lifetime, not ordering.
- `MLX_OMARCHY_TAPE_SYNC_EVERY=1` drains the stream after every tape
  evaluation, so nothing queued behind the tape executes while the host
  runs ahead. Tests whether host run-ahead is load-bearing for the
  corruption.
- `MLX_OMARCHY_NO_BUFFER_CACHE=1` turns the buffer cache off for the
  whole process, not just the tape window: every freed buffer is
  destroyed instead of recycled. Tests whether cross-window buffer
  recycling carries the corruption. Unlike the tape-scoped switches it
  is read once at runtime init, so it must be set before the process
  starts, not just before the tape runs.
- `MLX_OMARCHY_POISON_FREED=1` fills every buffer with the float32 word
  123456789.0 when it is recycled into the allocator cache. Any stale
  read of recycled storage then announces itself: a read that reaches
  the Cos gate aborts with exactly that magnitude in the message, and
  no legitimate f16 tensor can contain the word (f16 max finite is
  65504). Run the model prompt with this armed as a regression check:
  a correct answer proves no recycled-storage read served the run. Read
  once at first use, so set it before the process starts.

The first three switches ran the M1 decision tree on 2026-09-03: none
of them removed the corruption, and the refused magnitude tracked the
switch class (receipts/2026-09-03-tape-layer-isolation-switches.md,
MEASURED OUTCOME). These two switches bisect the remaining space.

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

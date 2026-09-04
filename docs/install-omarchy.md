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

Compiled tapes run by default on Apple GPUs: the stale-shape
corruption that closed them is root-caused and fixed
([docs/known-defects.md](known-defects.md)). The former
`MLX_OMARCHY_ALLOW_UNSAFE_COMPILE` override was retired with the fix;
setting it now does nothing.
To enable compilation, unset `MLX_DISABLE_COMPILE`; setting it to `0`
still disables compilation because upstream checks its presence.

Development builds use tiled quantized prefill by default. Set
`MLX_OMARCHY_QMM_TILE=0` to compare with the untiled path; single-row
decode still uses GEMV. This change is not in the v0.3.5 wheels.
The experimental `MLX_OMARCHY_ROPE_BF16_DIRECT` and
`MLX_OMARCHY_SDPA_BF16_FAST` flags remain off: both changed generated
token IDs on M1. See the [hardware gate receipt](../receipts/2026-09-04-m1-performance-gates.md).

Compiled-tape elementwise chains can fuse into one dispatch behind
`MLX_OMARCHY_FUSED_CHAIN`. It defaults off pending the native paired
gate (`receipts/2026-09-04-swiglu-fused-chain.md`).

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

## Benchmark matrix

`scripts/bench_matrix.py` runs the declared workload matrix (models x
prompts x pinned-length decode) through `scripts/bench_decode.py`. It
never downloads models: a snapshot missing from the local Hugging Face
cache is reported `skipped`, never passing, and revisions are read from
the cache, never guessed.

On Linux, inside the venv that holds the mlx-omarchy wheel, add `mlx-lm`
to the same venv, then:

```sh
python3 scripts/bench_matrix.py --mode plan
python3 scripts/bench_matrix.py --mode run \
  --python ~/.venvs/mlx-omarchy/bin/python --wheel dist/mlx_omarchy-*.whl
```

`--wheel` hands the file to `bench_decode`'s provenance gate, which
refuses to emit numbers from a mismatched binary. On a development
machine without an Apple GPU, add `--allow-non-apple`; llvmpipe results
are correctness checks, never performance claims.

On macOS (16-inch M1 Max baseline), keep the benchmark in its own venv
and never install into system Python, Homebrew, or an existing venv:

```sh
/opt/homebrew/bin/python3.12 -m venv ~/src/mlx-bench-$(date +%Y%m%d)
~/src/mlx-bench-<date>/bin/pip install "mlx" "mlx-lm==0.31.3"
python3 scripts/bench_matrix.py --mode metadata \
  --python ~/src/mlx-bench-<date>/bin/python
python3 scripts/bench_matrix.py --mode run \
  --python ~/src/mlx-bench-<date>/bin/python --host-label <label>
```

`metadata` records chip, core count, memory, OS version and build, MLX
and mlx-lm versions, Metal identity, source commit and dirty state, power
state, and any running model-serving processes. Hostnames, user names,
and serial numbers are excluded. A run while `llama-server`, `ollama`, or
similar processes are serving is labeled contended; contended timings are
never compared against clean numbers.

The matrix covers ~262, ~1024, and ~4096 prompt-token prefill plus
32/128-token pinned decode; exact prompt token counts are recorded per
leg from bench_decode's own measured generation response, never assumed
or probed separately. The ~4096 workload is explicit selection only, to
bound normal runs:

```sh
python3 scripts/bench_matrix.py --mode run --select longctx-4096-decode-32
```

Every run records a pins map: each ready model with its exact revision,
labeled `pinned` (manifest SHA) or `resolved-from-cache` (optional
models). To compare two machines, pass machine A's pins map to machine
B with `--expect-pins MODEL_ID=REVISION`; a different resolved revision
refuses the run with exit 4 before anything executes, because the same
model id with different weights is not a comparison.

# Compatibility

This file records observed status.
A row changes only when its receipt is public and repeatable.

## Status terms

Not started means this repository has no qualifying run.
In progress means code or hardware work exists, but one required gate remains open.
Supported means a public release passed its numerical, dispatch, stability, and install gates.
Blocked means an external or hardware condition prevents the qualifying run.

## Hardware

M1 Omarchy is in progress.
Vulkan runtime and matched kernel gates passed through `v0.2.0`.
The receipt used Vulkan API `1.4.354`, `MESA_HONEYKRISP`, Mesa `26.1.7`, and an Apple M1 device.
The device reported vendor `0x10005`.
ANE research continues in [`ane-linux-experiments`](https://github.com/joshuaswarren/ane-linux-experiments).
The package gate has not started.

M2, M3, and M4 Omarchy Linux work is deferred.
Vulkan, ANE, and install gates are not qualified on those systems.

## MLX core

Arrays and memory are runtime verified.
The tests cover allocation, copies, views, aliases, and lifetime.
See the [v0.1.0 M1 runtime receipt](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.1.0-m1-runtime.txt).

Streams and events are runtime verified.
The tests prove correct order without a global device wait.
See the [v0.1.0 M1 runtime receipt](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.1.0-m1-runtime.txt).

Matched kernel speed is verified through `v0.2.0`.
The gate covers matched prefill, decode, and attention operations against pinned `llama.cpp` Vulkan operations.
See the [v0.2.0 M1 kernel receipt](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.2.0-m1-kernel.json).

Primitive operations are in progress.
The development gate covers FP32 and FP16 elementwise work, suffix Sum and Max, offsets, and grid-stride dispatch.
It also covers dense Matmul and AddMM with tiled kernels, transposed inputs, and trailing-dimension bias broadcast.
Transposed-input Matmul now passes the gate for 2D views of either operand.
Batched Matmul and AddMM pass the gate for rank-3 and rank-4 operands with equal batch dims.
Each operand may be row-major or per-matrix transposed, and batch strides must equal the matrix size.
The batch count rides in the push-constant dims field, and workgroup z selects the batch.
Batched AddMM keeps the scalar, per-row, and full per-batch bias broadcasts.
Rank beyond 4 fails with the named `matrix rank` error, broadcast batch strides with the named `matrix layout` error, and batches beyond 65535 with the named `batch count` error.
`mx.fast.scaled_dot_product_attention` evaluates through the composed fallback and matches a host reference within `1e-3`.
Subtract, Negative, non-zero scalar fill, and same-dtype general strided copy pass the gate.
Elementwise binary ops broadcast operands on any axis up to a collapsed rank of 4.
Trailing broadcasts keep the modulo fast path, and a higher collapsed rank fails with the named `broadcast rank` error.
Suffix Softmax passes the gate for FP32, FP16, and BF16.
The Softmax kernel subtracts the row max and accumulates in float32.
Large logits stay finite, and both precise modes produce the same values.
A strided slice view materializes through the general strided-copy engine at eval, so elementwise ops read it as a normal array.
An offset-only slice keeps sharing the parent buffer.
Gapless strided views such as transposes run through the elementwise stride path.
Other layouts fail with the named layout error.
`mx.take` passes the gate for an axis-0 lookup in a 2D row-contiguous table with 1D int32 indices.
Out-of-range and negative indices write zero rows.
Upstream negative-index wrapping is not provided.
Other ranks, layouts, and index dtypes fail with named errors.
The softmax gradient passes the gate.
`value_and_grad` of sum(softmax(x) * x) matches a host reference within 1e-4 through the keepdims-sum broadcast views.
Dtype-converting strided copy, rank greater than 4, and negative strides stay unsupported with named errors.
The [M1 development gate receipt](../receipts/2026-08-31-m1-development-gates.md) records 20/20 primitive cases on Honeykrisp.
The pinned upstream matrix remains open.
`mx.concatenate` and `mx.slice_update` pass the development gate through the shared strided-copy engine.
Concatenate copies each input into an output window.
Row-contiguous axis-0 inputs use a plain buffer copy; other layouts use the general strided-copy kernel.
`mx.slice_update` supports the None reduce mode.
It copies the source first, then pastes the update into the strided window.
Other reduce modes fail with the named `SliceUpdate reduce` error.
The `omarchy_kv_ops_tests` binary covers exact-value 2D and 3D concatenates, fp16, and KV-cache growth.
`mx.argmax` and `mx.argmin` pass the development gate for a last-axis reduce over row-contiguous FP32, FP16, and BF16 inputs.
The kernel keeps one (value, index) pair per thread in shared memory and writes uint32 indices.
Ties keep the first occurrence, and NaN never wins a comparison, which matches the upstream CPU and Metal comparators.
Non-suffix axes, non-contiguous inputs, and non-float inputs fail with named errors.
`mx.sort` and `mx.argsort` pass the development gate for a last-axis sort of row-contiguous FP32 and FP16 rows up to 1024 elements.
The bitonic kernel sorts one row per workgroup in shared memory and pads the row to a power of two with NaN keys.
The comparator orders NaN after every number and breaks value ties on the smaller source index, which mirrors the upstream CPU `stable_sort` rule.
ArgSort writes uint32 source indices, and the tie rule makes the index order unique.
`mx.partition` and `mx.argpartition` route to the same full sort, the redirect the upstream Metal backend makes, so every kth position holds the sorted value.
Rows beyond 1024, non-suffix axes, non-contiguous inputs, and non-float inputs fail with named errors.
`mx.topk` returns the k largest values in ascending order through the partition path, and the strided tail slice now passes for 2-D inputs.
The BF16 sort variants build, but they have no gate receipt yet.
`mx.cos` and `mx.sin` pass the development gate for FP32 against host references at `1e-5`, including negative inputs.
`mx.arange` passes the gate for FP32 and FP16 fills of the form start plus step times index.
Upstream derives the arange length from `ceil((stop - start) / step)`, so a negative step over a descending range is valid.
The kernel applies the step as a signed multiplier, so it covers the negative-step case.
Non-float dtypes fail with the named `Arange dtype` error.
The BF16 arange kernel variant builds, but it has no gate receipt yet.
The gradient of `sum(sin(x))` matches `cos(x)` at `1e-5`.
The Sin vjp lowers to Cos and Multiply only, so the gradient stays inside supported operations.
`mx.fast.rope` evaluates through the composed fallback on Vulkan.
The int32 scalar offset cast to float32 runs as a one-element device kernel.
The half-split slice views `x[..., 0:dims/2]` and `x[..., dims/2:dims]` materialize at eval, so the trig multiply and subtract run over contiguous data.
The `{2,1,4,12}` case with `dims=8` and `base=10000` matches a host-computed rotation within `1e-4`.

Dtype work is in progress.
FP16 and FP32 casts pass the development gate.
Emulated BF16 passes the development gate.
BF16 arrays store as 16-bit bit patterns.
BF16 compute expands to float32 inside the shader.
int32 casts to and from float32, float16, and bfloat16 pass the development gate.
The float-to-int side truncates toward zero, which matches the upstream CPU `static_cast` semantics; upstream pins `-1.7` to `-1` in `mlx/random.cpp`.
int32 to float16 and int32 to bfloat16 keep the 16-bit storage capability gates; int32 to float32 needs none.
Scalar data of size one converts through the same kernel, so the RoPE offset cast runs.
Other int widths, bool, and uint64 casts remain unsupported with the named `dtype converting copy` error.
Low-bit formats remain open.

Transform work is in progress.
`grad` and `vjp` pass the development gate for supported operations.
`jvp` passes the gate for `sum(exp(x))` and matmul tangents with value checks at `1e-4`.
`vmap` passes the gate for batched `exp` and `add` with value checks.
Batched matmul under `vmap` passes the gate with value checks.
`mx.compile` fuses `exp` then `multiply` into one `Compiled` primitive and fails with the named `Compiled` error.
`CompileMode::no_fuse` keeps the tape unfused and matches the uncompiled values.

Compilation work is in progress.
The proof covers the fused path, the `no_fuse` fallback, values, and named errors.
Pre-fusion ANE partitioning and cache tests remain open.

The runtime has no CPU tensor fallback.
The release build and backend trace prove this state.
See the [v0.1.0 M1 runtime receipt](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.1.0-m1-runtime.txt).

Explicit exclusions are in progress.
Named errors now cover unsupported linear algebra, `float64`, and complex dtypes in the development gate.
The M1 development gate receipt covers these named errors on Honeykrisp.

Package work is in progress.
`scripts/build-wheel.sh` builds a `mlx-omarchy` wheel that provides the `mlx` module.
`tools/ci/run-clean-omarchy-install.sh` verifies a fresh-venv install with add, matmul, and gradient receipts.
The M1 clean-install receipt is recorded: aarch64 wheel installs in a fresh venv and passes add, matmul, and gradient checks on `Apple M1 (G13G B1)`.

## ANE

Linux descriptor submission is in progress.
Production task graphs run through the experimental KMD.
Full graph tensor parity remains open.

Qwen graph export is in progress.
macOS exports the 13-layer and 11-layer HWX graphs.
The corrected Linux 13-layer run remains open.

The complete token path is in progress.
Buffer geometry and the workspace role are mapped.
The 13-layer output and state must connect to the 11-layer tail.

MLX-to-MIL lowering has not started.
Known fixtures pass existing compiler stages.
A hand-authored one-operation MIL proof remains open.

Bundle validation is in progress.
The Linux host gate parses `manifest_version: 1` bundles.
It verifies graph identity, tensor geometry, tile-aligned strides, compiler and firmware identity, and payload sha256 before any mapping.
A missing bundle directory is the keep-on-Vulkan outcome.
See `docs/ane-bundles.md`.
The bundle validation gate also passes on the M1 (12/12 aarch64).
The macOS export proof and M1 execution of a validated bundle remain open.

MLX graph partitioning has not started.
The architecture is defined.
The Vulkan baseline and Linux bundle validation now exist; it still needs a stable worker ABI.

GPU and ANE shared memory is blocked.
Honeykrisp supports Linux external memory.
The ANE driver still needs the PRIME and dma-buf capability gate.

The detailed receipts live in [`ane-linux-experiments`](https://github.com/joshuaswarren/ane-linux-experiments).
Do not mark a research result as Supported.

## Reference model

The exact 32-token contract uses `Qwen3.8-2B-Q4_K_M.gguf`.
Its SHA-256 is `4aa0fb13c431514262f259d420ecc95a8714df58ac2a2384514e20b93983f0ff`.
Other models use their pinned numerical tolerance and fixture contract.

## Applications

MLX-LM must pass text generation and HTTP server workflows.
MLX Whisper must pass one public speech-to-text example.
MLX-VLM must pass one public image-and-text inference example.
MLX-Audio must pass one public speech workflow.
`mlx-openai-server` must pass one OpenAI-compatible generation request.
`mlx-serve` must pass one native server generation request.
`mlxcel` must pass one native Rust generation request.
No application gate has started.

## Release evidence

A Supported row must link every applicable record.

(1) Link the source commit and wheel hash.
(2) Record the kernel, Mesa, firmware, and Vulkan device identity.
(3) Record the ANE driver and compiler identity when ANE runs.
(4) Record the model and quantization hash.
(5) Link the numerical comparison and backend dispatch trace.
(6) Record prefill, decode, first-token, memory, and thermal results.
(7) Record the repeated-request stability result.
(8) Link the clean-install command output.

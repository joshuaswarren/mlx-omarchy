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
Subtract, Negative, non-zero scalar fill, and same-dtype general strided copy pass the gate.
Dtype-converting strided copy, rank greater than 4, and negative strides stay unsupported with named errors.
The pinned upstream matrix and M1 receipt remain open.

Dtype work is in progress.
FP16 and FP32 casts pass the development gate.
Emulated BF16 passes the development gate.
BF16 arrays store as 16-bit bit patterns.
BF16 compute expands to float32 inside the shader.
Low-bit formats remain open.

Transform work is in progress.
`grad` and `vjp` pass the development gate for supported operations.
`jvp` and `vmap` remain open.

Compilation work has not started.
The proof covers `mx.compile`, pre-fusion ANE partitioning, and cache tests.

The runtime has no CPU tensor fallback.
The release build and backend trace prove this state.
See the [v0.1.0 M1 runtime receipt](https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.2.0/mlx-omarchy-v0.1.0-m1-runtime.txt).

Explicit exclusions are in progress.
Named errors now cover unsupported linear algebra, `float64`, and complex dtypes in the development gate.
The M1 receipt remains open.

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

MLX graph partitioning has not started.
The architecture is defined.
It still needs a Vulkan baseline, versioned ANE bundles, and a stable worker ABI.

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

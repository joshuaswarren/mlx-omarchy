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
Batched Matmul and AddMM pass the gate for rank-3, rank-4, and rank-5 operands.
Each operand may be row-major or per-matrix transposed, and batch axes may
broadcast: a size-1 axis against a wider axis carries a zero stride through
the shared-buffer broadcast view and repeats one matrix per batch step.
An operand whose batch strides are uniform but not contiguous materializes
to a standard row-major batch through the general strided-copy engine, and
the dispatch runs normally on the copy. A KV-cache state prefix slice from
a just-written cache composes this way: the cold-cache GQA decode scores
matmul of shape `[1, 2, 7, 1, 1]` reads the slice view and matches a host
reference. The other operand keeps the zero-copy path when its layout
already conforms.
The push constants carry the batch axis count, the batch extents, and the
per-operand batch strides in elements, and workgroup z unravels over the
batch shape. Batched AddMM keeps the scalar, per-row, and full per-batch
bias broadcasts.
Rank beyond 5 fails with the named `matrix rank` error, mismatched batch
dims with the named `batch dimensions` error, and collapsed batches beyond
65535 with the named `batch count` error.
`mx.fast.scaled_dot_product_attention` evaluates through the composed
fallback and matches a host reference within `1e-3` for float32, including
grouped-query attention (`n_q_heads != n_kv_heads`, which emits rank-5
matmuls over stride-0 broadcast batch views) and causal masks.
`mx.quantized_matmul` passes the gate for the mlx-lm Linear shape:
affine mode, 4-bit and 8-bit codes, group sizes 32 and 64, transposed
packed weights `[N, K * bits / 32]`, and f32, f16, and bf16 activations.
The kernel dequantizes in registers with per-group scale and bias,
accumulates in float32, and matches a double-precision host reference
and a dequantized dense matmul on the same device within `2e-4` for
float32. Leading x dims flatten into M when x is row-contiguous.
Scales and biases pack as two halves of one buffer binding, built by
two device copies. Other modes, other bits and group sizes,
non-transposed weights, rank-3 weights, and non-row-contiguous operands
fail with the named `QuantizedMatmul mode`, `bits`, `group size`,
`transpose`, `weight layout`, and `non-contiguous input` errors.
`mx.dequantize` passes the gate for the affine mode that QuantizedEmbedding
feeds: 4-bit and 8-bit codes, group sizes 32 and 64, packed uint32 words,
and f16 or f32 scales and biases. One kernel thread owns one packed word,
reads the word at its own linear address, and writes the `32 / bits`
dequantized values `q * scale + bias` consecutively, so the shape the
Honeykrisp driver reads correctly is preserved. Output dtype follows the
promoted scales dtype, and a `[1, 40, 112]` word block dequantizes to the
`[1, 40, 896]` embedding shape. Device values match a host unpack of the
same packed words bit for bit with dyadic group parameters, and the
quantizer's own parameters round-trip within float32 rounding noise.
The quantize direction and non-affine modes such as mxfp4 fail with the
named `Quantize direction` and `Quantize mode` errors; other bits, other
group sizes, non-row-contiguous operands, and bfloat16 parameters fail
with the named `Quantize bits`, `Quantize group size`,
`non-contiguous input`, and `Quantize scales dtype` errors.
Subtract, Negative, non-zero scalar fill, and same-dtype general strided copy pass the gate.
Elementwise binary ops broadcast operands on any axis up to a collapsed rank of 4.
Trailing broadcasts keep the modulo fast path, and a higher collapsed rank fails with the named `broadcast rank` error.
Suffix Softmax passes the gate for FP32, FP16, and BF16.
The Softmax kernel subtracts the row max and accumulates in float32.
Large logits stay finite, and both precise modes produce the same values.
`mx.logsumexp` passes the gate for a last-axis reduce over row-contiguous
FP32, FP16, and BF16 inputs.
The kernel keeps the row max, accumulates `exp(x - max)` in float32, and
writes one value per row, which is the upstream keepdims contract.
The `[1, V]` logits epilogue in mlx-lm reduces to `[1, 1]`, large logits
stay finite, and an infinite row max stays the answer as on the upstream
CPU.
Non-contiguous inputs fail with the named layout error.
`mx.log` passes the gate for FP32, FP16, and BF16 through the elementwise
kernel.
A strided slice view materializes through the general strided-copy engine at eval, so elementwise ops read it as a normal array.
An offset-only slice keeps sharing the parent buffer.
Gapless strided views such as transposes run through the elementwise stride path.
Other layouts fail with the named layout error.
`mx.take` passes the gate for an axis-0 lookup in a 2D row-contiguous
table with row-contiguous int32, uint32, or int64 indices of any rank.
The dispatch passes an index mode to the kernel, and int64 indices read
as two little-endian words where a nonzero high word writes the zero row.
Out-of-range and negative indices write zero rows.
Upstream negative-index wrapping is not provided.
Other ranks, layouts, and index dtypes fail with named errors.
The uint32 path is proven with a real `mx.argmax` output feeding the
gather at the mlx-lm decode shapes `[1, 1]` and `[2, 3]`, and the int64
path covers values above 2^31 and negatives.
A uint32 table gathers through a raw word-copy kernel with no float
conversion, so the packed QuantizedEmbedding weight words keep values
above 2^31 bit-exact.
An int32 table shares that kernel unchanged because the copy is bitwise
and signedness never participates.
The gather keeps one straight-line per-thread load at a linear address,
the shape the Honeykrisp driver reads correctly.
Other table dtypes, such as float64, keep the named dtype error.
`RandomBits` passes the gate for the width-4 uint32 case that mlx-lm
sampling uses: `mx.random.bits`, the `mx.random.split` key shape, and the
bits behind `mx.random.uniform` and `mx.random.categorical`.
The kernel reproduces the upstream threefry2x32 rotation constants and
5-round key schedule, and the words match a host reference built from
the upstream constants bit for bit, including the odd word count with
its middle counter.
Other widths fail with the named `RandomBits width` error.
`mx.random.uniform` with a pinned key is deterministic across runs and
`mx.random.categorical` over uniform four-class logits hits every class
across 1000 draws.
The uint32-to-float32 cast and the `Minimum` elementwise op that the
uniform path composes pass the same gate.
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
`mx.arange` passes the gate for FP32 and FP16 fills of the form start plus
step times index.
Upstream derives the arange length from `ceil((stop - start) / step)`, so a
negative step over a descending range is valid.
The kernel applies the step as a signed multiplier, so it covers the
negative-step case.
int32 aranges run through an exact integer kernel: the host keeps `|start|`,
`|step|`, and `|start + step * count|` below `2^24`, so the float transport
is exact, and larger ranges fail with the named `Arange range` error.
Other non-float dtypes fail with the named `Arange dtype` error.
`mx.greater_equal` passes the gate for two int32 index arrays with
broadcast views and a bool output; the mask bytes move through 32-bit word
packing, so no 8-bit storage feature is required.
`mx.equal` shares that comparison machinery through one compare kernel
family and passes the gate for float32, float16, bfloat16, and int32
operands of one dtype, over scalar, suffix-broadcast, and stride-view
broadcast shapes, including the scalar-only `shape=[]` case.
The output stays word-packed bool, and equality with NaN stays false.
`NotEqual`, `Less`, `LessEqual`, and `Greater` stay named rejections;
no mlx-lm sampling path needs them yet.
`mx.logical_or` passes the gate for word-packed bool inputs and outputs
and serves the `isinf` composition `or(isposinf, isneginf)`.
`LogicalAnd` and `LogicalNot` stay named rejections.
`mx.where` serves the composed causal mask: a strided bool condition view
picks between a row-contiguous value and a scalar floor, for float32,
float16, and bfloat16. The false operand now also accepts the same
suffix-aligned shapes as the true operand, which the sampler's
`where(isinf(m), eq, exp)` composition needs.
`astype` of bool to float32 passes the gate and yields exact `0.0` and
`1.0`; other bool casts stay named rejections.
Other comparisons, dtypes, and layouts stay named rejections.

Sampling work is in progress.
`mx.cumsum` passes the gate for float32, float16, and bfloat16 suffix
scans over row-contiguous rows, in both the inclusive and the exclusive
form the sampler chain uses, with one invocation per row and a float32
accumulator. Reverse scans, non-suffix axes, and other reduce types fail
with named errors.
`mx.searchsorted` passes the gate for one sorted 1-D row against any
row-contiguous value array, on both the `left` and `right` sides, in
float32, float16, bfloat16, int32, and uint32, and writes uint32 indices.
`Subtract` extends to int32 and uint32 through an integer elementwise
kernel, so the `searchsorted` minus one epilogue runs on device.
`mx.random.categorical` over size-equal logits takes the inverse-CDF
path end to end on Vulkan: over `[1, 32]` float32 class logits with a
pinned key it draws in-range varied samples deterministically, and every
draw matches a host searchsorted finished on the same device
intermediates bit for bit.
The full mlx-lm temp sampling shape also passes: one `[1, 151936]`
bfloat16 logprob row scaled by `1/temp` and sampled in range.
Temp-only sampling needs no `ArgPartition`: with the sampler defaults
`make_sampler` chains nothing but `categorical_sampling`, so `--temp`
works while the row-length limit holds.
Wide-row `ArgPartition` and `top-k` stay unsupported: rows beyond 1024
fail with the named `sort row length` error, which is the one remaining
mlx-lm sampler limitation.
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

Model file io is in progress.
`mx.save_safetensors` and `mx.load` of a safetensors file pass the development gate for FP32 and BF16 arrays with exact-value round trips.
The io stream selection uses the default stream when the CPU backend is absent, and the Load primitive reads file bytes straight into host-visible output buffers.
The `.npy` loader follows the same stream rule but has no gate receipt yet.
GGUF load has no stream selection to fix and no receipt.

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

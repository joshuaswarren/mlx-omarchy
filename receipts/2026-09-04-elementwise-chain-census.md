# Elementwise-chain census: decode dispatch attribution

Date: 2026-09-04. Worktree off `origin/main` (`b7bde25`), branch
`ew-fused-elementwise`. This receipt is MEASUREMENT ONLY: no fused
kernel shipped. The step-1 stop rule fired; see Verdict.

## Question

The fusion assignment assumed ~1,530 cast/copy/elementwise dispatches
per decode token and set the target at fusing chains inside compiled
tapes. Two prior findings disagreed with that premise. This census
attributes every GPU dispatch in a real decode step to its path (compiled
tape vs eager) and enumerates the elementwise chains that actually occur.

## Environment and provenance

- llvmpipe (Mesa, LLVM 15.0.6) via `MLX_OMARCHY_ALLOW_NON_APPLE=1`;
  software Vulkan, x86_64 Linux 6.17.2.
- Wheel built in this worktree with `MLX_OMARCHY_GPU_PROFILING=ON`
  (`mlx_omarchy-0.32.2.dev202609040054+b7bde25`, libmlx sha256 recorded
  in the commit's receipt addendum); installed into a fresh venv with
  `mlx-lm==0.31.3` and numpy.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, snapshot
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` (repo-pinned).
- Command: `scripts/profile_generate.py --model <snapshot> --prompt
  "What is the capital of France?" --max-tokens 32 --temp 0 --seed 0`
  with `MLX_OMARCHY_GPU_PROFILE`, `MLX_OMARCHY_TAPE_CENSUS` set.
- Analysis: `scripts/chain_census.py` (new, committed with the
  instrument).

## Instruments added (committed)

1. `trace::compiled_tape_dispatches` and
   `trace::compiled_tape_node_evaluations` counters, incremented in
   `eval_compiled_tape`.
2. GPU-profiler dispatch events carry `"tp":0|1` (tape vs eager path)
   via `CommandEncoder::in_tape_recording`.
3. `MLX_OMARCHY_TAPE_CENSUS=<path>` dumps every tape evaluation: node
   list with op name, dtype kind, tape-wide use count, input wiring,
   output flag.

## Result 1: tape path share

Per median decode token: **1,953 dispatches, of which 72 (3.7%) are
tape-path**. Eager: 1,881 (96.3%). The only tape fragment in the entire
run is the mlx-lm `swiglu` activation
(`mlx_lm/models/activations.py`, `@partial(mx.compile, shapeless=True)`)
- 24 layers, one 7-node tape per layer per token, tracing to 3 real
dispatches per eval (four Broadcast nodes collapse to zero-dispatch
views):

```
n0=Sigmoid  in=[gate]        uses=2
n1=Broadcast in=[gate,n0]    uses=1
n2=Broadcast in=[n0,gate]    uses=1
n3=Multiply in=[n1,n2]       uses=2   (silu(gate))
n4=Broadcast in=[n3,up]      uses=1
n5=Broadcast in=[up,n3]      uses=1
n6=Multiply in=[n4,n5]       OUT      (silu(gate)*up)
```

mlx_lm 0.31.3 compiles NOTHING else in the generate loop
(`generate.py` has no `mx.compile`; `sample_utils.py` compiles only the
temp>0 sampling methods, unused at temp 0).

## Result 2: eager composition per median token

| family | dispatches/token | note |
|---|---|---|
| ElementwiseF32 | 456 | rope trig+scale, mul/add mass |
| ElementwiseF16 | 409 | 72 tape + 337 eager (mul/add around qkv, kv) |
| CopyGeneralF16 | 288 | kv-cache writes (sizes 64x96, 448x96, 128x48, 5248x48) |
| QmmF16 | 169 | quantized matmuls (compute) |
| CastF32F16/F16F32/I32F32 | 240 | f32-compute/f16-storage boundaries |
| ArangeF32 | 96 | rope inv_freq/position rebuild per call |
| ReduceF32 | 96 | rope norm scans |
| FastRmsNormF16 | 49 | 2/layer + 1 |
| MatmulF32 | 48 | composed-attention pair |
| SoftmaxF32 | 24 | 1/layer |
| dispatches with count==1 | 290 | scalar-tensor ops (rope freq math) |

Eager elementwise-class mass (elementwise+cast+copy+arange): **1,489
dispatches/token = 76.2%**.

## Result 3: chains that actually occur

Buffer-flow-linked elementwise runs (writer->reader, same count) find
exactly TWO shapes, each 24x per token (once per layer):

- len 3: `ElementwiseF16/sigmoid + mul + mul` - the swiglu tape itself
  (tape path).
- len 2: `CastF16F32 + ElementwiseF32/mul` - eager.

840 of 1,953 dispatches sit inside those runs; everything else is
chain-length 1 in the dispatch stream. The only use-count-clean chains
are degenerate: every interior node of the swiglu tape has uses=2
(both multiply operands pass through individual Broadcast views), so a
sole-consumer tape fuser would fuse nothing there without carrying the
swiglu shape as a special case.

## Discrepancy flagged for the M1 leg

README documents ~95 GPU dispatches per decode token on the M1; this
census measures 1,953 on llvmpipe with the same mlx-lm pin and model.
Candidates: capability-gated fast paths differing per device, an older
measurement mode, or a superseded baseline. The consolidated protocol
below re-measures dispatches/token on the M1 with the same instrument;
treat the two numbers as unresolved until then.

## Verdict (stop rule fired)

The assignment's own stop rule: "If the measurement shows the chains
are mostly length one or two, say so and stop." Measured: the tape path
owns 3.7% of decode dispatches; its only chain is length 3 with uses=2
interiors; the eager path owns 96.3% and has no tape to inspect.
Tape-level fusion cannot pay on the decode path mlx-lm 0.31.3 actually
runs. Not built, by design.

## What the numbers say would pay instead

1. **fast::RoPE composition** (~576-700 dispatches/token: arange, abs,
   cos, sin, exp, reduce, scalar mul/add): a dedicated fused RoPE
   kernel - already FusedRopeKernel's assignment.
2. **kv-cache CopyGeneralF16** (288/token): cache-layout or fused
   write kernel work.
3. **f32<->f16 boundary casts** (240/token): fuse into producer
   kernels (rmsnorm/rope writing f16 directly).
4. **290 count==1 scalar dispatches/token**: host-side constant
   caching of rope frequency tables across steps.
5. **Compile the model forward** (upstream/mlx-lm change): makes tapes
   exist for the whole step, at which point tape-level fusion has a
   real ceiling. This is what upstream Metal users get; it does not
   belong in this backend alone.

## Reproduce

```bash
bash scripts/prepare-mlx.sh
CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_OMARCHY_GPU_PROFILING=ON" \
  pip wheel --no-build-isolation --no-deps -w dist .work/mlx
MLX_OMARCHY_ALLOW_NON_APPLE=1 \
MLX_OMARCHY_GPU_PROFILE=/tmp/prof.jsonl \
MLX_OMARCHY_TAPE_CENSUS=/tmp/census.txt \
  venv/bin/python scripts/profile_generate.py \
  --model <4bit snapshot> --prompt "What is the capital of France?" \
  --max-tokens 32 --temp 0 --seed 0 --markers /tmp/m.jsonl
python3 scripts/chain_census.py /tmp/prof.jsonl \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h \
  --markers /tmp/m.jsonl
```

## Addendum: compile-the-forward probe (pivot step 1, same night)

`scripts/probe_compile_forward.py` re-expresses the Qwen2 decode layer
functionally (explicit cache arrays in/out, offset as a scalar int32
array, full-slot cache select + additive -inf mask instead of in-place
KV writes) and wraps each whole decoder layer in `mx.compile`. Same
model, venv, and box; both legs in one process, split by markers.

| leg | dispatches/token | tape-path/token | median inter-token |
|---|---|---|---|
| eager | 1,951 | 72 | 1,412 ms (profiled run) |
| compiled layers | ~2,167 median, 3,950 first step | 1,272 | 1,757 ms |

**Compile alone does NOT collapse dispatches on this backend.** The
tape share went 3.7% -> ~59%, but every tape node still records its own
dispatch: our Compiled path is a tape interpreter, Metal's generates a
fused kernel. Compiling the model forward moves primitives from the
eager path into tapes one-for-one; the dispatch count is structurally
unchanged, and the probe's functional full-slot attention made it
slower on top.

Findings recorded for whoever continues this road:

1. The chain compile -> tape -> dispatch collapse needs BOTH halves:
   compiled graphs (mlx_lm/upstream change) AND tape-level kernel
   generation or fused variants (backend change). Either alone moves
   nothing; the probe measured the "compiled graph" half.
2. mlx-lm's KVCache mutates arrays in place; mx.compile tolerates only
   the functional re-expression. The re-expression costs full-slot
   attention and diverges from eager greedy after one token (masked-
   softmax numerics; output stays coherent, token argmax flips).
3. The backend refuses int32 Add ("no GPU kernel exists"), so a
   decode-loop offset increment cannot ride the GPU graph; the probe
   increments offsets host-side.
4. No shapeless retrace storm during decode: shapes stay constant, the
   prefill->decode transition retraces once (the "[1,7,4864] serving
   [1,1,4864]" notice).

Standing conclusion unchanged: on the decode loop mlx-lm 0.31.3
actually runs, the only dispatch reducers that work today are dedicated
named kernels (rope ~576-700, kv-copy 288, casts 240, scalar folding
290 dispatches/token) - plus, behind the compile-the-forward change,
the tape-fusion engine this census ruled out for tonight.

## Addendum 2: kv-cache copy and boundary-cast attribution (same night)

`scripts/kvcopy_decompose.py` isolates one layer's decode path in three
stages (update only; update+sdpa; full step with rope) under the
profiler, Qwen2.5-0.5B-4bit, 7-token prefill, 8 decode steps.

- update_and_fetch alone: **0** strided copies. The cache append is
  innocent.
- update+sdpa: 4 copies/step - the 128-element pieces are the new k/v
  (transposed post-projection views) written into the cache, and the
  growing pieces are the cache.state slices.
- full step: 10 of the 12 copies/layer reproduce in isolation.

Root cause found in our own SDPA path (primitives.cpp,
ScaledDotProductAttention::eval_gpu): `to_f32()` runs
`contiguous_copy_gpu` on any input that is not row-contiguous, then a
separate cast to float32, for q, k, AND v; scores run as MatmulF32; the
output is cast back. The cache slice `keys[..., :offset, :]` is never
row-contiguous, so every layer every token pays:

- 2x CopyGeneralF16 growing with offset (k, v dense materialization)
- casts in both directions around the f32 scores matmul (the census's
  CastF32F16 120 + CastF16F32 72 + CastI32F32 48 per token; the int32
  pair is rope's offset-to-position arithmetic)
- GQA repeat expansion materializing 448-class pieces (7 repeats x 64
  head_dim per kv head)

Verdict per Main's question: the copies and most casts should be
DELETED, not accelerated. The boundary crossing is our implementation's
choice, not the graph's:

1. The scores matmul can take f16 inputs with f32 accumulation (the
   qmm kernel already does), deleting to_f32 for q/k/v and the output
   downcast - most of the 240 casts/token plus the 2 growing
   CopyGeneralF16 per layer.
2. A stride-aware cast (or strided-input matmul) removes the
   contiguous materialization of the cache slices outright.
3. The GQA expansion should ride broadcast views (the pattern
   broadcast_view already uses for the scale), not materialized
   repeats.

These are changes to ScaledDotProductAttention::eval_gpu in
primitives.cpp - shared file; coordinate before implementing.
Equivalence guard: greedy decode of the pinned prompt must stay
token-identical to the current path on the same build (the composed
attention path is the current reference; upstream's fast SDPA
vector-path divergence is a known separate defect and not a
justification).

## Addendum 3: SDPA rework - function-level implementation plan

Coordination settled: FusedRopeKernel's primitives.cpp window is closed
(final commit ca58b89 touches RoPE only, not yet on origin/main; main
at 238a977 carries no diff in the SDPA region). The plan below is
mechanical; the implementing session should rebase and re-read the
function first.

Target function: `ScaledDotProductAttention::eval_gpu` (primitives.cpp,
~6543-6729 on this base).

1. Delete `to_f32(q/k/v)`; keep the existing reshape/transpose views on
   the f16 arrays (views are free). The scale multiply moves to a
   f16-elementwise multiply (same broadcast_view trick for the scalar)
   or folds into the matmul alpha parameter - `dispatch_matmul` already
   carries alpha; prefer alpha.
2. scores: `dispatch_matmul(tag, {qs_f16, keys_t_f16}, scores_f16, ...)`
   with `ComputeKernel::MatmulF16`. matmul.comp line 111 accumulates in
   `float` under USE_FP16 (verified this session), so each dot product
   keeps f32 accumulation; the only new rounding is f16 storage of the
   scores intermediate. That is the stated tolerance: bounded by f16
   rounding of scores (~5e-4 relative), no accumulation-order change;
   softmax still runs from a materialized tensor as today.
3. probs: `dispatch_softmax` on f16 scores - verify softmax.comp's f16
   leg upcasts internally before trusting it; if it does not, keep the
   softmax leg on the f32 kernel with one scores-cast (still deletes
   three upcasts and two dense copies per call).
4. result: MatmulF16 probs @ v32T; the final `copy_gpu(result, out)`
   downcast disappears (result is already f16) - the
   `result.dtype() == out.dtype()` buffer-share branch then takes over,
   GQA 5-D stride case included.
5. GQA expansion: repeats>1 currently reshapes to 5-D and relies on
   matmul batching; audit whether the 448-class materializations die
   with the f32 views or need the broadcast-view treatment; measure
   before changing further.
6. Mask legs: causal mask is built in f32 host-side then added; either
   build it in f16 or keep one cast - measure.

llvmpipe storage-defect caveats from FusedRopeKernel apply: do not
alias a converted buffer across readonly/writeonly slots; prefer f32
blocks for accumulate buffers; see receipts/2026-09-03-llvmpipe-
storage-defects/.

Equivalence suite required before merge (Main's bar, above the greedy
guard): value-level SDPA versus the current composed path on the same
build, across (a) decode shapes [1, heads, 1, 64], (b) non-contiguous
cache slices at offsets 1, 7, 41, 256, (c) GQA repeat factors 1, 2, 7,
(d) head_dim 48 (not a power-of-two multiple), (e) q_len 1 and q_len
>1; tolerance: scores/probs agree within 2e-3 absolute after f16
storage rounding, outputs within 1e-3, plus the pinned-prompt greedy
token-identity end-to-end check. Measured acceptance: CopyGeneralF16
per layer per token drops from 12 toward the update-path minimum (0
appends + write-back only), casts per token drop from 240; report the
gap against the 4-6-copy prediction either way.

Not implemented this session: the implementing leg needs a full
build-equivalence-measure cycle; this session ended at the design
handoff.

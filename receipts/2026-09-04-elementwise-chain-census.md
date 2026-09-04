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

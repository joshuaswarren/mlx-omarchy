# Decode fusion plan: current census, mass attribution, ranked hand-fusion designs

Date: 2026-09-04. Branch `decode-fusion-plan` off main `94d967c`. This
receipt is MEASUREMENT AND DESIGN ONLY: no fused kernel ships in it. It
re-censuses the decode dispatch stream on current main (two merges landed
after `2026-09-04-elementwise-chain-census.md`: the SDPA f16-scores rework
and the quantized GEMV decode path), attributes every dispatch to a named
mass, and specifies the fusion candidates Main can weigh against the M1
per-kernel profile.

## Environment and provenance

All numbers in this receipt are **llvmpipe dispatch COUNTS and structure
only** — no speed claim about the Apple GPU may be derived from them.

- llvmpipe via `MLX_OMARCHY_ALLOW_NON_APPLE=1`; software Vulkan, Mesa
  22.3.6-1+deb12u2 (LLVM 15.0.6), x86_64 Linux 6.17.2-1-pve, 16-core EPYC.
- Wheel built in this worktree from main `94d967c` with
  `MLX_OMARCHY_GPU_PROFILING=ON`:
  `mlx_omarchy-0.32.2.dev202609041204+94d967c-cp311-cp311-linux_x86_64.whl`,
  wheel sha256 `b4c1ab2f3a12cfb115407cfb22007e71a907afcce61d049f8fb2da2ea02557a3`,
  `libmlx.so` sha256 `f2115ea5119da45aaca6a6c45c9f3a2f0239cad4fda8b1af62899579f3a2ea63`
  (installed == wheel member, provenance gate green).
- `mlx-lm==0.31.3`, numpy, fresh venv, no freeze pins (the 2026-09-03
  provisioning rule).
- Models: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` snapshot
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` (the repo-pinned snapshot);
  `mlx-community/Qwen2.5-0.5B-Instruct-bf16` snapshot
  `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`.
- Prompt: chat-template "What is the capital of France?", 32 max tokens,
  temp 0, seed 0, `scripts/profile_generate.py`, greedy text
  "The capital of France is Paris." on every leg that ran.

## Result 1: current per-token dispatch stream

`scripts/chain_census.py`, median decode token:

| leg | dispatches/token | tape-path | note |
|---|---|---|---|
| 4-bit, `MLX_DISABLE_COMPILE=1` | **633** | 0 | matches the integrated-tree number in the SDPA rework receipt |
| 4-bit, default (compile on) | **633** | 72 | identical stream; the swiglu tape's 3 real dispatches x 24 layers just re-attribute to the tape path |
| bf16, `MLX_DISABLE_COMPILE=1` | **774** | 0 | different composition; see Result 3 |
| bf16, default (compile on) | refused | — | `RuntimeError: [omarchy] Compiled tape bfloat16 is refused` at `mx.eval` in `generate_step` — the documented bf16 tape gate, loud by design |

Kernel composition, 4-bit median token (633 total, all paths):

| kernel | /token | share |
|---|---|---|
| ElementwiseF16 | 193 | 30.5% |
| QmmVecF16 | 169 | 26.7% |
| CopyGeneralF16 | 96 | 15.2% |
| FastRmsNormF16 | 49 | 7.7% |
| FastRopeF16 | 48 | 7.6% |
| MatmulF16 | 48 | 7.6% |
| SoftmaxF16 | 24 | 3.8% |
| GatherF16 | 2 | 0.3% |
| GatherU32 | 1 | 0.2% |
| DequantF16 | 1 | 0.2% |
| LogSumExpF16 | 1 | 0.2% |
| ArgReduceF16 | 1 | 0.2% |

Kernel composition, bf16 median token (774 total):

| kernel | /token | share |
|---|---|---|
| MatmulBF16 | 169 | 21.8% |
| ElementwiseBF16 | 121 | 15.6% |
| CastBF16F32 | 120 | 15.5% |
| CopyGeneralBF16 | 96 | 12.4% |
| CastF32BF16 | 72 | 9.3% |
| FastRmsNormBF16 | 49 | 6.3% |
| FastRopeF32 | 48 | 6.2% |
| MatmulF32 | 48 | 6.2% |
| ElementwiseF32 | 24 | 3.1% |
| SoftmaxF32 | 24 | 3.1% |
| GatherBF16 | 1 | 0.1% |
| LogSumExpBF16 | 1 | 0.1% |
| ArgReduceBF16 | 1 | 0.1% |

Reproduce commands at the end of this receipt.

## Result 2: mass attribution

Note: the two snapshots have different Qwen2 graph shapes. The 4-bit quantization keeps the q/k/v bias tensors (HF Qwen2 config defaults attention_bias=true and the converter preserved them); the bf16 conversion dropped them (q_proj.bias absent from the bf16 safetensors, so mlx-lm's nn.Linear initializes bias=False). Bias adds therefore appear in the 4-bit stream only. Per layer per decode token the eager stream is 26 dispatches (4-bit, verified against the flow-linked token dump, token 4):

```
ln1                      FastRmsNormF16 896
k_proj                   QmmVecF16 128
v_proj                   QmmVecF16 128
k_bias                   ElementwiseF16 Add 128
v_bias                   ElementwiseF16 Add 128
q_proj                   QmmVecF16 896
q_bias                   ElementwiseF16 Add 896
k rope                   FastRope 64          (2 kv heads x half_dims 32)
k cache write            CopyGeneral 128      slice-update of new k
v cache write            CopyGeneral 128      slice-update of new v
k state materialize      CopyGeneral 4480+    growing, offset x 128
v state materialize      CopyGeneral 4480+    growing, offset x 128
q rope                   FastRope 448         (14 q heads x half_dims 32)
scores matmul            Matmul 14 x offset   532 at offset 38
softmax                  Softmax 14 x offset
probs @ v                Matmul 896
o_proj                   QmmVecF16 896
residual 1               ElementwiseF16 Add 896
ln2                      FastRmsNormF16 896
gate_proj                QmmVecF16 4864
sigmoid                  ElementwiseF16 Sigmoid 4864
up_proj                  QmmVecF16 4864
gate * sig               ElementwiseF16 Mul 4864
* up                     ElementwiseF16 Mul 4864
down_proj                QmmVecF16 896
residual 2               ElementwiseF16 Add 896
```

Plus per token, outside the layers: 4 embedding dispatches (GatherF16 x2 =
the 14 per-group scales/biases, GatherU32 112 = the packed row,
DequantF16), final norm, lm_head QmmVecF16 n=151936, and sampling
(LogSumExp, Subtract n=151936 = `logits - logsumexp`, ArgReduce = argmax).

Named masses, per median token:

| mass | 4-bit | bf16 | owner |
|---|---|---|---|
| qmm/matmul compute | 217 | 217 | QmmVecF16/MatmulBF16 169 (7 projections + lm_head) + SDPA matmuls 48 (MatmulF16/MatmulF32) |
| KV-cache copies | 96 | 96 | 2 new-k/v slice-update writes (128 each) + 2 growing state-slice materializations (offset x 128 each), per layer |

| q/k/v bias adds | 72 | 0 | 3 graph-level Add primitives per layer (4-bit only; bf16 snapshot has no bias weights) |
| swiglu (sigmoid + 2 muls) | 72 | 72 | eager primitives (MLX_DISABLE_COMPILE=1) or the swiglu tape's 3 real dispatches (compiled 4-bit); bf16 compiled is refused, so bf16 eager always |
| SDPA f32 composition (bf16 only) | 0 | 144 | 3 CastBF16F32 (q/k/v up) + result CastF32BF16 + ElementwiseF32 (scale) + SoftmaxF32 = 6/layer = 144; MatmulF32 2/layer = 48 belongs to compute above |
| bf16 rope cast wrap | 0 | 96 | 2 casts x 2 rope calls per layer around FastRopeF32 |
| residual adds | 48 | 48 | 2 graph-level Add primitives per layer (the 24 ElementwiseBF16 includes 48 residual + 72 swiglu + 1 sampling subtract = 121) |
| RMSNorm | 49 | 49 | fused kernel, 2/layer + final |
| RoPE (already fused) | 48 | 48 core | FastRopeF16 2/layer; bf16 core is FastRopeF32, wraps with 4 casts/layer (see row above) |
| softmax (4-bit, SDPA-internal) | 24 | (in f32 row) | fused SoftmaxF16, 1/layer |
| sampling | 3 | 3 | LogSumExp + Subtract(151936) + ArgReduce |
| embedding | 4 | 1 | 4-bit: Gather x2 + GatherU32 + Dequant (4 graph primitives); bf16: GatherBF16 (one) |
| casts (4-bit) | 0 | — | the landed SDPA rework deleted all of them |
| **total** | **633** | **774** | |

Kernel sums are ground truth: 4-bit 633 and bf16 774 from the table of
kernel counts (Result 1). The mass table is the per-primitive
attribution; the kernel numbers are what decide the fusion plan.

## Result 3: attribution findings the census settles

### The growing KV copies are the state-slice read, not SDPA and not the write

`2026-09-04-sdpa-f16-scores-rework.md` left open who owns the two growing
per-layer copies ("cache realloc vs slice-assign vs a consumer densifying").
`scripts/kvcopy_owner.py` (committed with this receipt) isolates the three
stages, each its own `mx.eval`, cache shape `[1, 2, 256, 64]` f16, offset
41:

| stage | dispatches |
|---|---|
| s1: `cache[..., 41:42, :] = new_kv` (slice-update only) | 2 x CopyGeneralF16 n=128, nothing else |
| s2: s1 + state slice `cache[..., :42, :]` consumed by a plain matmul | + 2 x CopyGeneralF16 n=5376 |
| s3: s1 + the same slices through `mx.fast.scaled_dot_product_attention` | + 2 x CopyGeneralF16 n=5376 |

n=5376 = 2 heads x 42 slots x 64 dim: the full state extent. The slice
UPDATE writes only the new 128 elements; the growing copies appear exactly
when the state slice is READ. The owner is `Slice::eval_gpu` -> `slice_gpu`
(shared `mlx/backend/gpu/slicing.cpp`: it materializes the slice into a
fresh dense buffer) feeding the primitive that consumes the state. SDPA is
innocent (consistent with that receipt's dense-k/v probe), the write path
is innocent, and this is NOT a defect - it is how every MLX backend
executes a slice: a slice evaluates to its own dense buffer. Metal pays the
same class of copy. It is nonetheless deletable on our backend; see Design
3.

### The bf16 rope cast wrap wraps a dead shader leg

`RoPE::eval_gpu` (primitives.cpp ~6820-6869) routes bf16 through
`copy_gpu` up-cast -> the F32 rope kernel -> `copy_gpu` down-cast,
documented as avoiding bf16 word stores. But `fast_rope.comp`'s USE_BF16
variant already exists and already reads packed-bf16 uint words directly
(the proven reduce_general load form, commit cf68e7d's select-chain
discipline; no uint16_t block IO - that is the llvmpipe recycled-garbage
hazard from 2026-09-03) and writes f32. The host never selects it: the
up-cast produces an f32 input, so `select_float_kernel(rope_output.dtype())`
picks the F32 kernel. The BF16 leg is dead code on the current host path.
The down-cast is one dispatch; the up-cast is pure redundancy against a
shader that already speaks packed bf16.

### f16 vs bf16 asymmetry is the whole bf16 story

The 4-bit (f16) model pays ZERO casts: rope is direct (FastRopeF16) and
SDPA runs the f16-scores fast path. The bf16 model pays 192 cast
dispatches per token because neither fast path accepts bf16. Both are the
same two primitives; the deltas below are backend-internal routing, not new
model semantics.

## Designs

RoPE-shaped means: one primitive's `eval_gpu` stops composing several
kernels and dispatches one (or none), exact upstream semantics preserved,
fenced behind the proven path until equivalence passes. All three designs
below follow it. Ranked by (dispatches removed per token) / risk.

### Design 1 (rank 2): SDPA fast path for bf16 — -120/token bf16

- **Mass**: the bf16-only SDPA f32 composition. Per call currently:
  3 CastBF16F32 (q/k/v up) + ElementwiseF32 (scale) + MatmulF32 +
  SoftmaxF32 + MatmulF32 + CastF32BF16 = 8 dispatches; the bf16 fast
  path dispatches MatmulBF16 + SoftmaxBF16 + MatmulBF16 = 3. The
  non-matmul cuts (-5/call = 120/token) are the design's payoff; the
  matmuls stay (the 48 MatmulF32 become 48 MatmulBF16 — count unchanged,
  GPU work smaller).
- **Hook**: `ScaledDotProductAttention::eval_gpu`
  (overlay/mlx/backend/omarchy/primitives.cpp, `q.dtype() == float16`
  branch). Widen to `float16 || bfloat16` once equivalence passes; the
  conjunct is the fence.
- **Mechanism**: every kernel the f16 path dispatches already has a bf16
  twin used elsewhere in the same model today (MatmulBF16 runs the
  projections; SoftmaxBF16, ElementwiseBF16 proven). Verified in shader:
  matmul.comp USE_BF16 loads bf16 via `uintBitsToFloat(uint(x) << 16)` and
  accumulates in `float` (line 117); softmax_suffix.comp USE_BF16 computes
  float math over bf16 storage with a rounded `bf16_store`. No new shader.
  Scores materialize as bf16, probs as bf16, output bf16 end to end.
- **Shader sketch**: none new — the exact matmul.comp/softmax_suffix.comp
  USE_BF16 legs.
- **Semantics delta**: bf16 score STORAGE. Rounding per stored score is
  2^-8 relative (8-bit mantissa), 8x coarser than f16 storage. Softmax is
  scale-invariant so the additive-constant cancellation still holds; the
  causal addend becomes bf16's finite maximum (-3.39e38), overflow-immune
  (bf16 max 3.39e38 vs f16's documented 65504 cap — strictly safer there),
  and fully-masked rows stay defined exactly as in the f16 rework.
- **Equivalence test**: extend `scripts/sdpa_equivalence.py` (17/17 f16
  today) with the bf16 dtype leg over the same coverage: cache slices at
  offsets 1/7/41/256, GQA repeats 1/2/7, head_dim 48, q_len 1 and >1,
  causal + additive masks, fully-masked row, batch 2. Bar: primitive vs
  f32-reference outputs within a DERIVED bf16-storage bound (the f16 leg's
  2e-3 storage-emulation analog scales to ~1.6e-2 on scores; outputs bar
  derived from the suite, not fitted) + greedy token identity on the pinned
- **Dispatch delta**: 8 -> 3 per call = **-5 x 24 = -120/token bf16**
  (774 -> 654). 4-bit unaffected.

### Design 2 (rank 1): select the existing bf16 rope leg — -48/token bf16 phase A, -96/token total at phase B

- **Mass**: the bf16 rope cast wrap, 96/token (4 cast dispatches per
  layer), plus the dead-shader finding above.
- **Hook**: `RoPE::eval_gpu` (primitives.cpp ~6829-6869). Phase A deletes
  the input up-cast: bind the bf16 input directly, select
  `FastRopeBF16`/`FastRopeFreqsBF16` (existing kernels, existing enum
  entries), keep the output f32 temporary + the single proven
  CastF32BF16. The `out.dtype() == bfloat16` cast-wrap block is the fence;
  remove the input-cast conjunct only after equivalence.
- **Shader sketch**: none new for phase A — the existing USE_BF16 leg:
  packed uint word loads (low half `word & 0xFFFF`, high half
  `word >> 16`, constant shifts only), f32 rotation, f32 stores. Phase B
  (optional, later): in-kernel bf16 output via PAIRED 32-bit stores — each
  thread owns two ADJACENT output elements (2t, 2t+1), packs them into one
  uint32 word, no per-element RMW of a shared word (the documented hazard),
  no uint16_t block IO (the documented llvmpipe hazard). Both hazards
  respected by construction.
- **Semantics delta**: none. The composition rounds once (f32 rope ->
  f32-to-bf16 cast); phase A keeps that exact rounding; phase B's
  `bf16_store` (round-to-nearest-even, the matmul.comp form) rounds once
  identically. Phase A is expected bit-exact against the current path;
  the RoPE value tests pin it.
- **Equivalence test**: the existing RoPE suite (f64 host reference,
  offset sweep across every tolerance band, the scalar-offset fence case)
  with the bf16 leg enabled; plus one decode-shape A/B of the full fused
  path vs the cast-wrap path, bit-exact expected.
- **Dispatch delta**: phase A: -1 per rope call x 2 calls x 24 layers =
  **-48/token bf16** (774 -> 726 with Design 1 already applied). Phase B:
  **-96/token bf16** total (-48 phase A + -48 phase B). 4-bit unaffected
  (f16 rope is already direct).

### Design 3 (rank 3): view-valued slices for the cache state — -48/token on BOTH models

- **Mass**: the growing KV state-slice materializations, 48/token
  (2 per layer), on 4-bit and bf16 alike. This is the non-compute mass the
  SDPA receipt could not attribute; Result 3 pins it to the slice read.
- **Hook**: `slice_gpu` (shared `mlx/backend/gpu/slicing.cpp`, wired to
  omarchy via the shared GPU build; reachable through a `patches/` diff —
  the mechanism `mlx-linalg-gpu.patch` already uses — or an omarchy-local
  override wired in the omarchy CMake). `Slice::eval_gpu` ->
  `slice_gpu` currently materializes every slice into a fresh dense
  buffer.
- **Mechanism**: a slice of constant strides is a pure re-indexing: the
  output can `copy_shared_buffer` the input's storage with the derived
  strides and data offset (the regroup-view idiom SDPA already uses at
  ~6948), flags contiguous=false, and dispatch NOTHING. The consumer then
  either handles strides (the landed stride-aware matmul: the cache-state
  consumer) or refuses by name. No kernel at all — the dispatches are
  deleted, not accelerated.
- **Shader sketch**: none. Zero dispatch.
- **Semantics delta**: none for values. A behavior change for CONSUMERS:
  code that today receives a silently-densified slice receives a strided
  array; primitives that require row_contiguous (RMSNorm/LayerNorm's
  `require_norm_input`, CrossEntropy, ...) REFUSE BY NAME where they used
  to work. The project contract (refuse by name, never a wrong number) is
  preserved; the blast radius is what demands the risk rank.
- **Equivalence test**: (a) full llvmpipe battery + upstream suite
  (non-negotiable for a shared-path change), (b) a value-identity probe on
  the decode path: greedy token identity with the view path on vs the
  materializing path on, (c) the cache-state consumer map: SDPA (strided,
  proven), generate_step's `mx.eval([c.state ...])` (eval of a view with a
  materialized producer = no-op), and any strided-input refusals surfaced
  by the battery fixed or fenced per-primitive before the default flips.
- **Dispatch delta**: **-2 x 24 = -48/token both dtypes** (4-bit
  633 -> 585 with nothing else), plus the same class of deletion anywhere
  else slices feed stride-tolerant consumers (prefill included;
  unquantified here).
## The masses with no backend-only design (the wall)

Four of the largest masses are chains of SEPARATE MLX graph primitives.
The backend sees one primitive per eval and cannot see neighbors; that is
precisely why fused RoPE was possible (fast::rope is ONE primitive whose
backend implementation composed many kernels) and these are not:

- **q/k/v bias adds, 72/token**: `nn.Linear`'s `mx.add(qmm_out, bias)`.
  Absorbing the bias into the qmm kernel requires seeing the Add; the Add
  knows nothing of its producer. No `fast` primitive carries a bias.
- **swiglu, 72/token**: Sigmoid + Mul + Mul as three eager primitives (or
  three tape-node dispatches through the interpreter on the compiled path
  — the tape interpreter is a per-node dispatcher, `compiled.cpp:146`, so
  compilation changes attribution, not count).
- **residual adds, 48/token**: two graph-level Adds per layer.
- **sampling, 3/token; embedding, 4/token**: each a distinct primitive.

What would unlock them, and why it is out of scope here: (a) new `fast`
primitives adopted by mlx-lm (upstream model-code change, not backend);
(b) fusing chains at the graph or tape level — generic tape fusion is
killed at a 7.3% ceiling (CONTRIBUTOR-GUIDE, killed strategies) and a
hardcoded swiglu tape-pattern matcher inherits that kill (narrow coverage
was the stated reason; a single-pattern matcher is the narrowest possible
coverage). They are listed so Main can price the upstream route against
the M1 profile; they are not proposed.

## Priority (dispatches removed per token / risk)

| rank | design | /token | models | risk | note |
|---|---|---|---|---|---|
| 1 | bf16 rope leg selection (phase A) | -48 | bf16 | LOW | routing only; kernel + load form already shipped and device-probed; bit-exact expected |
| 2 | SDPA bf16 fast path | -120 | bf16 | MODERATE | reuses 3 proven kernels; tolerance re-derivation and M1 greedy identity gate the unfence |
| 3 | view-valued cache-state slices | -48 | both | HIGH | shared-path semantics change; loud refusals possible until the consumer audit lands |
| 4 | SDPA scores+softmax fusion (f16), flash later | -24 (flash: -48) | 4-bit | MODERATE-HIGH | new fused kernel class; decode q_len=1 keeps the row bounded; listed for completeness — it is the largest 4-bit mass after the KV copies that is backend-reachable |

Combined potential: bf16 774 -> 774 -120 (d1) -48 (d2 phase A)
-48 (d3) = **558** at phase A, **510** at d2 phase B; 4-bit 633 -> 585
(d3) -> 561/537 with rank 4 scores+softmax / flash. Dispatch counts
are the currency the M1 profile prices: Main merges this with the
per-kernel GPU-time profile before picking.

## Reproduce

```bash
# wheel: current main with the profiling instrument
bash scripts/prepare-mlx.sh
CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_OMARCHY_GPU_PROFILING=ON" \
  pip wheel --no-build-isolation --no-deps -w dist .work/mlx

# fresh venv: wheel first, then mlx-lm --no-deps (2026-09-03 provisioning rule)
python3 -m venv runenv
runenv/bin/pip install dist/mlx_omarchy-0.32.2.dev202609041204+94d967c-cp311-cp311-linux_x86_64.whl
runenv/bin/pip install --no-deps "mlx-lm==0.31.3" numpy
runenv/bin/pip install huggingface_hub transformers sentencepiece jinja2 protobuf
# provenance gate: installed libmlx.so sha256 must equal the wheel member's
unzip -p dist/mlx_omarchy-*.whl mlx/lib/libmlx.so | sha256sum
sha256sum runenv/lib/python3.11/site-packages/mlx/lib/libmlx.so

# four census legs (bf16 default refuses by design - that IS the result)
M4=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
MB=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e
for leg in "4b-eager:$M4:1" "4b-comp:$M4:" "bf16-eager:$MB:1" "bf16-comp:$MB:"; do
  name=${leg%%:*}; rest=${leg#*:}; model=${rest%:*}; dis=${rest#*:}
  [ -n "$dis" ] && d=1 || d=
  MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_DISABLE_COMPILE=$d \
  MLX_OMARCHY_GPU_PROFILE=/tmp/prof-$name.jsonl \
    runenv/bin/python scripts/profile_generate.py --model "$model" \
    --prompt "What is the capital of France?" --max-tokens 32 --temp 0 \
    --seed 0 --markers /tmp/m-$name.jsonl
done

python3 scripts/chain_census.py /tmp/prof-4b-eager.jsonl \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h \
  --markers /tmp/m-4b-eager.jsonl
# (and the other three profiles the same way)

# KV copy ownership probe (Result 3)
runenv/bin/python scripts/kvcopy_owner.py
```

Provenance line for every number above: measured on llvmpipe (Mesa 22.3.6,
LLVM 15.0.6) via `MLX_OMARCHY_ALLOW_NON_APPLE=1`, wheel
`mlx_omarchy-0.32.2.dev202609041204+94d967c` (libmlx sha256 `f2115ea5...`),
mlx-lm 0.31.3, Qwen2.5-0.5B-Instruct snapshots `a5339a41...` (4-bit) /
`56d07e76...` (bf16), pinned prompt, temp 0, seed 0 — **dispatch counts
and structure only; no Apple-GPU speed claim**.

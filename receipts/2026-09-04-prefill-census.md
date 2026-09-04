# Prefill census: dispatch structure of the first token (Qwen2.5-0.5B, 36/256/1024, 4-bit and bf16)

Date: 2026-09-04. Worktree off `origin/main` (`94d967c`), branch
`prefill-census`. MEASUREMENT ONLY: no backend change ships with this
receipt. Prefill on this backend had never been profiled; this census
establishes the baseline the M1 per-kernel profile (running on jwm1) will
be merged against.

Provenance for every number in every table below: llvmpipe (Mesa, LLVM
15.0.6) via `MLX_OMARCHY_ALLOW_NON_APPLE=1`, software Vulkan, x86_64
Linux 6.17.2, 16-core EPYC host; wheel
`mlx_omarchy-0.32.2.dev202609041201+diag.94d967c`
(`scripts/build-wheel.sh --diagnostics` at `94d967c`,
`scripts/mlx_provenance.py` verified=match against the installed venv
wheel); mlx-lm 0.31.3; `HF_HUB_OFFLINE=1`, temp 0, seed 0,
max-tokens 1. Per the fleet rule, llvmpipe numbers are dispatch COUNTS
and element counts only - they are never speed evidence about the Apple
GPU.

## Method

Six legs: {4-bit, bf16} x prompt {36, 256, 1024} chat-templated tokens,
constructed to the exact token count and fed as ids
(`/tmp/prefill_gen.py`, a marker-compatible clone of
`scripts/profile_generate.py`), `MLX_OMARCHY_GPU_PROFILE` +
`MLX_OMARCHY_TAPE_CENSUS`, analyzed with `scripts/profile_analyze.py`,
`scripts/chain_census.py`, and a section splitter over the dispatch
stream. The splitter classifies each submission structurally: a forward
STARTS at the embedding submission (GatherF16/GatherBF16 + Dequant or
Matmul), ENDS at the submission containing LogSumExp + ArgReduce, and
its m is the max FastRmsNorm element count divided by 896 (hidden size).
Each leg splits into exactly three sections:

- **chunk** - the prompt[:-1] forward at m = N-1 (35 / 255 / 1023);
- **step-1** - the m=1 forward over the last prompt token plus the
  sample that produces token 1 (this is what the markers call prefill);
- **step-2** - a second full m=1 forward. mlx-lm 0.31.3 pipelines one
  token ahead: `generate_step` enqueues `next_y = _step(y)` whenever
  `n != max_tokens`, so even a max-tokens=1 run enqueues and executes
  the next token before the generator exits. For real serving this is
  the next token's compute, not waste; for this census it is excluded
  from every per-prefill number and reported separately.

Dispatch events are counted per section by submission membership, which
is exact; join-time phase windows (profile_analyze.py) smear boundary
submissions and undercount (they reported 1,882 for the m4b36 prefill
window vs 704+633+633=1,970 by section).

Calibration: two France-prompt decode legs (compile on and off) gave
identical dispatch counts per token; the swiglu tape collapses to 3
dispatches where eager silu+mul is 3, so compile mode does not move
counts on this wheel. The bf16 legs ran with `MLX_DISABLE_COMPILE=1`
(the bf16 tape gate refuses by design). The m=1023 chunk on llvmpipe
exceeds the watchdog's default 10 s no-progress window (one huge
submission; the timeline advances only at completion), so the long legs
raised `MLX_OMARCHY_HANG_NO_PROGRESS_NS=1800000000000`. No wedge
occurred; every leg completed.

Models: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` snapshot
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`,
`mlx-community/Qwen2.5-0.5B-Instruct-bf16` snapshot
`56d07e766edd7159fbe12ed12d9cf114bf38bf1e` - the same pinned snapshots
every prior receipt used.

## Table 1: dispatches per prefill and per input token

Provenance: same line as above; llvmpipe dispatch counts, max-tokens 1,
one run per leg.

| leg | chunk (m=N-1) | step-1 | step-2 (artifact) | useful/leg (chunk+step-1) | per prompt token (useful/N) | chunk per prompt token |
|---|---|---|---|---|---|---|
| 4-bit, N=36 | 704 | 633 | 633 | 1,337 | 37.1 | 19.6 |
| 4-bit, N=256 | 704 | 585 | 729 | 1,289 | 5.0 | 2.8 |
| 4-bit, N=1024 | 704 | 585 | 729 | 1,289 | 1.3 | 0.69 |
| bf16, N=36 | 886 | 774 | 774 | 1,660 | 46.1 | 24.6 |
| bf16, N=256 | 886 | 726 | 870 | 1,612 | 6.3 | 3.5 |
| bf16, N=1024 | 886 | 726 | 870 | 1,612 | 1.6 | 0.87 |

**The chunk's dispatch count is FLAT in prompt length**: 704 (4-bit) and
886 (bf16) for m = 35, 255, and 1023 - a 29x range of m with zero
dispatch growth. Per input token, prefill dispatches FALL as ~1/N:
0.69/token at m=1023 vs 633/token for decode. Dispatch count is not
where prefill cost lives; per-dispatch WORK is (Table 3).

Submissions per chunk: 72 (both dtypes, all lengths). Each submission
waits on the stream's previous submission, so a prefill chunk is 72
serialized host round trips regardless of length; decode's ~49/token is
the same family of cost.

## Table 2: per-kernel composition of the chunk

Provenance: same line as above; counts are dispatches inside the chunk
section; percentages of the chunk total; identical at m=35/255/1023
(counts; the element counts differ per Table 3).

4-bit (total 704):

| kernel | dispatches | share |
|---|---|---|
| ElementwiseF16 | 209 | 29.7% |
| CopyGeneralF16 | 165 | 23.4% |
| QmmF16 (per-element kernel, m>1) | 163 | 23.2% |
| FastRmsNormF16 | 47 | 6.7% |
| FastRopeF16 | 47 | 6.7% |
| MatmulF16 (scores, probs@v) | 46 | 6.5% |
| SoftmaxF16 | 23 | 3.3% |
| embedding (GatherF16/GatherU32/DequantF16) | 4 | 0.6% |

bf16 (total 886):

| kernel | dispatches | share |
|---|---|---|
| CopyGeneralBF16 | 212 | 23.9% |
| MatmulBF16 | 163 | 18.4% |
| CastBF16F32 | 116 | 13.1% |
| ElementwiseBF16 | 115 | 13.0% |
| CastF32BF16 | 70 | 7.9% |
| FastRmsNormBF16 | 47 | 5.3% |
| FastRopeF32 | 47 | 5.3% |
| ElementwiseF32 | 46 | 5.2% |
| MatmulF32 | 46 | 5.2% |
| SoftmaxF32 | 23 | 2.6% |
| embedding (GatherBF16) | 1 | 0.1% |

Tape share: the swiglu tape runs 3 dispatches/layer in the 4-bit chunk
(69 tape-marked); bf16 has none (gate). bf16 pays +182 dispatches vs
4-bit for the same chunk, all of it the attention f32 round trip
(casts + f32 scores/softmax) - see Lever 3.

## Table 3: what scales with prompt length (element counts, not counts)

Provenance: same line as above; element counts are the profiler's per-
dispatch `n` fields, summed per class inside the chunk section.

| class (4-bit chunk) | count | m=35 | m=255 | m=1023 | growth |
|---|---|---|---|---|---|
| scores matmul / mask add / softmax (each) | 23 | 17,150 | 910,350 | 14,651,406 | O(m^2) |
| QmmF16 + ElementwiseF16 at m x 896 | 69 each | 31,360 | 228,480 | 916,608 | O(m) |
| QmmF16 + ElementwiseF16 at m x 4864 | 69 each | 170,240 | 1,240,320 | 4,975,872 | O(m) |
| CopyGeneralF16 at m x 128 (kv/cache) | 142 | 4,480 | 32,640 | 130,944 | O(m) |
| CopyGeneralF16 at m x 896 | 46 | 31,360 | 228,480 | 916,608 | O(m) |

The only kernels whose DISPATCH count moves anywhere in the census:

- step-1's growing cache-state copies disappear at N>=256 (633 -> 585
  dispatches: the two per-layer `[1,2,offset,64]` state materializations
  of the small-cache path stop being issued once the cache allocation
  crosses a growth boundary). A count decrease, recorded for the record.
- step-2's growing state copies SPLIT at >=32,768 elements: 96 copies
  of n=4,608 at N=36 become 96 of n=32,768 PLUS 48 of n=32,896 at
  N=256/1024 (729 vs 633 dispatches). CopyGeneralF16 therefore has a
  per-copy element ceiling that splits one logical copy into several
  dispatches on long caches - the only count-scaling kernel found.

Attention at q_len>1 is 4 dispatches per layer (scores matmul, mask
add, softmax, probs@v) over a MATERIALIZED scores tensor [14, m, m]
(14.65M f16 elements = 29 MB at m=1023, written and re-read by the add,
softmax, and probs matmul). The causal mask itself is built host-side
(`ScaledDotProductAttention::eval_gpu`, the `values[row * k_len + col]`
loop) and costs no dispatch.

## Table 4: elementwise-chain census, chunk vs decode

Provenance: same line as above; chain = maximal run of consecutive
elementwise-class dispatches linked by buffer flow (writer->reader, same
element count), `scripts/chain_census.py` logic applied to the chunk
section's dispatch sequence.

| section | chains>=2 | dispatches in chains | chains |
|---|---|---|---|
| 4-bit chunk (any m) | 23 | 69 of 704 | len-3 x23: swiglu tape (EW/op5+op1+op1) |
| 4-bit step-1 (= decode) | 24 | 72 of 633 | len-3 x24: swiglu tape |
| bf16 chunk (any m) | 116 | 278 of 886 | len-2 x70 (CopyGeneralBF16+CastBF16F32 kv path), len-3 x46 (CastF32BF16+CastBF16F32+EW round trip x23, eager swiglu x23) |

Same verdict as the decode census: between matmuls there are no fusable
elementwise chains beyond the swiglu tape; 90% of chunk dispatches are
chain-length 1. Fusable-chain work has no prefill payoff on this path.

## Comparison against decode's known structure

step-1 at N=36 measures **633 dispatches** - exactly the
`receipts/2026-09-04-sdpa-f16-scores-rework.md` integrated-tree number
(casts 0, MatmulF16 48, SoftmaxF16 24, CopyGeneralF16 96, QmmVec 169)
- which validates the instrument chain end to end. At N>=256 step-1 is
585 for the cache-growth reason above. A full-prompt prefill chunk is
704 dispatches - 1.11x ONE decode token's count for the whole prompt.
Prefill's disadvantage is not dispatch count; it is what each dispatch
does: the m^2 attention chain, the m x weight dequant inside
`qmm.comp`, and the copy mass.

## Ranked levers (at most three)

Provenance for the expected-delta numbers: llvmpipe dispatch counts and
element counts from Tables 1-3, never speed claims.

### 1. `qmm.comp` has no m-blocking: weight dequant repeats m times per prefill

`overlay/mlx/backend/omarchy/shaders/qmm.comp` computes one
`out[m, n]` element per thread (`for index...; row = index / n`), and
the inner k-loop unpacks the quantized weight word and applies
scale/bias INLINE PER MAC. Every weight element is therefore unpacked
once per output row: unpack+MAC pairs = m x n x k per matmul. At
m=1023 the whole model runs ~462G pairs per prefill chunk (per layer:
q/o ~0.82G each, k/v ~0.12G each, gate/up/down ~4.46G each = 19.3G x
24), each carrying ~5 extra ALU ops of unpack (~2.3T ALU per prefill);
a 16-row m-tiled variant amortizes the unpack across the tile, dropping
unpack work ~16x (to ~29G) while MAC count is unchanged, and cutting
weight re-reads by the same factor. This is the 4-bit prefill analogue
of the qmm_vec win (a 1-row GEMV replaced the untiled kernel at
m=1; prefill needs an m-BLOCKED GEMM, which the current kernel is not -
`matmul.comp`'s 16x16 tile does block, but QuantizedMatmul does not use
it). Dispatch count: unchanged (still one per matmul).

- Files: `overlay/mlx/backend/omarchy/shaders/qmm.comp` (new blocked
  variant or tile loop), dispatch in
  `overlay/mlx/backend/omarchy/primitives.cpp`
  `QuantizedMatmul::eval_gpu` (the `matrix_m == 1` gate at line ~4536
- Expected delta: per-prefill unpack element-ops ~462G -> ~29G at
  m=1023 (16x), weight global re-reads cut by the same factor; the
  chunk dispatch count is unchanged at 704.
- M1 confirmation: the per-kernel GPU-time profile now running on jwm1
  should show QmmF16's total GPU time per prefill collapsing
  disproportionately between prompt 256 and 1024 vs 36; after the
  change, an M1 A/B prefill wall ratio (prompt 1024 : prompt 36)
  shrinks. Equivalence reuses the staged qmm tree-vs-subgroup protocol
  (`receipts/2026-09-04-qmm-gemv-subgroup-m1.md`, 144-shape sweep).

### 2. Prefill copy mass: kv-cache densification and the attention-output materialization

Two CopyGeneral families dominate the chunk (Table 3): 142 copies of
`m x 128` and 46 copies of `m x 896` per chunk. At m=1023 that is
60.8M element-copies (~122 MB of f16 moved) per chunk vs the useful
kv produced (6.3M elements). The `m x 896` family is the attention
output `[B,H,L,D]` being materialized by the graph-level
transpose+reshape before the o-proj QuantizedMatmul (which refuses
non-row-contiguous operands,
`primitives.cpp` `QuantizedMatmul::eval_gpu` line ~4446); the
`m x 128` family is the cache-update path densifying new k/v and
materializing cache state slices (the same 4/layer class
`receipts/2026-09-04-sdpa-f16-scores-rework.md` left open at decode,
here multiplied by m and by 24 layers).

- Files: `overlay/mlx/backend/omarchy/primitives.cpp` -
  `QuantizedMatmul::eval_gpu` operand check (accept the strided operand
  the way fix-2's stride-aware matmul does) and/or
  `ScaledDotProductAttention::eval_gpu` result layout
  (`commit_result`, line ~6916); the cache-state slice attribution is
  open work named in the SDPA receipt (instruments:
  `MLX_OMARCHY_TRACE_MATERIALIZE`, `scripts/kvcopy_decompose.py`).
- Expected delta: chunk dispatches 704 -> 516 (-188, -27%); element-
  copies per chunk at m=1023 drop by 60.8M (142 x 130,944 + 46 x
  916,608).
- M1 confirmation: per-kernel GPU time of CopyGeneralF16/BF16 per
  prefill at each prompt length in the running jwm1 profile (today it
  should scale ~linearly with m and be a visible share at 1024);
  after, the share collapses and prefill GPU-busy drops by that
  amount. Greedy token-identity on the pinned prompt is the equivalence
  guard (same as the SDPA rework).

### 3. bf16 attention still pays the f32 round trip the f16 path deleted

The 4-bit/f16 SDPA runs f16 scores end to end (fix 1-3, on main). The
bf16 path still upcasts q/k/v to f32 (CastBF16F32 116/chunk), runs
scores + mask + softmax + probs@v in f32 (MatmulF32 46, SoftmaxF32 23,
ElementwiseF32 46), and downcasts (CastF32BF16 70): +182 dispatches
per chunk (886 vs 704) and every m^2-class tensor moves at 4 bytes
instead of 2. bf16 storage has no 65504 cap, so the overflow caveat
that motivated keeping bf16 on the f32 composition does not apply to a
bf16-storage/f32-accumulate rework - it would DELETE the caveat.

- Files: `overlay/mlx/backend/omarchy/primitives.cpp`,
  `ScaledDotProductAttention::eval_gpu` - a bf16 branch mirroring the
  f16 branch (matmul.comp already accumulates in float under
  USE_BF16-styled storage; verify the bf16 store path in
  `softmax_suffix.comp` before trusting it).
- Expected delta: chunk 886 -> ~704 (-182 dispatches); the scores/
  softmax/probs chain bytes at m=1023 drop from ~5 passes x 14.65M x 4B
  to 5 x 14.65M x 2B (~-88 MB moved per chunk); step-1 726 -> 633-
  equivalent.
- M1 confirmation: `scripts/sdpa_equivalence.py` extended with bf16-
  storage legs (scores/probs <= 2e-3, outputs <= 1e-3 bars as in the
  f16 rework) plus greedy token-identity; per-kernel cast share in the
  running jwm1 profile at prompt 256/1024.

Not ranked, answered for the record: the dense `matmul.comp` 16x16 tile
(IS the tile for m>1; at m=1023, n=896 the grid is 3,584 workgroups -
no parallelism shortage on 8 cores; its intensity is 0.5 MAC/byte so a
32x32 tile is a plausible secondary knob once Lever 1 lands, but dense
matmul is only 46/704 of chunk dispatches and the scores chain it
serves is Lever 3's/flash-fusion's territory). Per-token-scaling
dispatches that should be per-sequence: none found - chunk dispatch
counts are flat in m; per-token structure exists only in decode steps.

## Reproduce

```bash
bash scripts/prepare-mlx.sh
./scripts/build-wheel.sh --diagnostics
python3 -m venv /tmp/prefill-venv
/tmp/prefill-venv/bin/pip -q install "mlx-lm==0.31.3" numpy dist/<diag wheel>
export MLX_OMARCHY_ALLOW_NON_APPLE=1 HF_HUB_OFFLINE=1
# long legs: MLX_OMARCHY_HANG_NO_PROGRESS_NS=1800000000000
# bf16 legs: MLX_DISABLE_COMPILE=1
MLX_OMARCHY_GPU_PROFILE=/tmp/p/<tag>/profile.jsonl \
MLX_OMARCHY_TAPE_CENSUS=/tmp/p/<tag>/tape.txt \
  /tmp/prefill-venv/bin/python /tmp/prefill_gen.py \
  --model <snapshot> --tokens {36,256,1024} --max-tokens 1 \
  --temp 0 --seed 0 --markers /tmp/p/<tag>/markers.jsonl
/tmp/prefill-venv/bin/python scripts/profile_analyze.py ... --markers ... \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h
/tmp/prefill-venv/bin/python scripts/chain_census.py <profile> \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h --markers <markers>
# section split rules (this receipt's tables): forward starts at the
# embedding submission (GatherF*/GatherBF* + Dequant*/Matmul*), ends at
# LogSumExp*+ArgReduce*; m = max FastRmsNorm n / 896; per-kernel counts
# and chains per section follow chain_census.py's definitions.
```

Driver and splitter live at `/tmp/prefill_gen.py`,
`/tmp/prefill_sections.py` on the dev box (throwaway instruments, not
committed); the profile JSONL + markers + tape census for all six legs
are preserved under `/tmp/prefill/<tag>/`.

## Open items

1. The second `m x 896` copy per layer (46, not 23, in the chunk) is
   not attributed to a named site yet; `MLX_OMARCHY_TRACE_MATERIALIZE`
   on a prefill run would pin it.
2. The cache-growth threshold that removes step-1's two growing copies
   at N>=256 (633 -> 585) and the CopyGeneralF16 ~32k-element split
   ceiling are unpinned in code; both are count-visible knobs a cache-
   path owner should name.
3. The M1 leg: merge this baseline with the per-kernel GPU-time profile
   (jwm1, running) to rank the levers by GPU time, not by count.

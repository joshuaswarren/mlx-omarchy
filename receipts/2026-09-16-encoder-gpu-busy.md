# Encoder GPU busy: fp16-in coopmat linear with fp16 partials — identical
# bytes, coopmat bucket 1448.8 to 0 ms, GPU busy −782 ms, encoder wall
# 7978.3 to 7135.6 ms median (2026-09-16)

Attack on the two largest GPU-busy buckets priced by `receipts/
2026-09-16-encoder-attribution-2.md` (MatmulF32Coopmat 1453.6 ms and the
"SliceUpdatePairF16" 320.8 ms bucket on the pre-residual baseline).
Verdict: **LAND.** One rewrite of the leftover `linear` op collapses both
buckets with byte-identical output: `encoder_hidden` pin `38c73261…` and
transcript `db501a8c…` on every run, 104/104 six of six in both arms.

## The two assignment targets, relabeled precisely

- **MatmulF32Coopmat (194 dispatches, 1448.8 ms in the fresh same-session
  record):** every dispatch is one of the 194 `linear` statements — the
  batched-blocked partials matmul `xb @ wb` over the 16-wide K blocks
  ([K/16] batches, M=375, K=16, N∈{640,1024,4096}; 145 N=1024, 48 N=4096,
  1 N=640). Its dominant cost is neither MMA nor staging bandwidth but the
  **f32 partials buffer** (`[K/16, 375, N]`, 98–393 MB per linear):
  written by the matmul, then streamed back by the chain kernel, ≈85 GB
  per encoder pass at the measured buckets.
- **The "SliceUpdatePairF16 (605 dispatches, 320.8 ms)" bucket is actually
  `CopyGeneralF16`** — the attribution receipt's label is off by one enum
  entry (kernel enum 55 = CopyGeneralF16, 56 = SliceUpdatePairF16; the
  pair kernel fires zero times in the encoder — there are no adjacent
  same-state slice_update pairs in this graph). The 605 dispatches are
  full-tensor staging copies; 388 of them (two per linear) are the
  reshape-materializations of `x` and `weight` feeding the transpose views
  in the old linear path. The KV-window-style pair-copy machinery is
  decoder-side and not reachable from the encoder graph.

Both buckets share one root: the linear path materializes four things it
does not need — an f16 copy of `x`, an f16 copy of `weight`, f32 `xb`/`wb`
staging tensors, and f32 partials.

## The cut

`_linear_f16_coopmat_kernel` in `overlay/tools/coreml/vulkan_encoder.py`
plus a rewired `linear` op: one custom coopmat dispatch per linear
computes every 16-wide block's partial directly from the fp16 operands.

- **fp16-in, f32 math:** A and B are read from the fp16 `x` and `weight`
  buffers (no materializations at all) and widened exactly when staged
  into the same 2 KiB shared layout (`a_s[256]`/`b_s[256]`), then the
  identical 8x8x8 f32 `coopMatMulAdd` sequence runs (same tile split
  32×32, same two-k-step loop, same 16-MMA order, same drain through
  shared). f32 f16→f32 widening is exact, so every staged value equals the
  value the f32 batched matmul staged.
- **fp16 partials:** the drain rounds each f32 partial once to fp16
  (hardware RNE) — the same single round-to-nearest-even the old path
  applied when the chain kernel read the f32 partials — and writes
  `[K/16, 375, N]` fp16. The partials write + chain-read traffic halves;
  the chain kernel (`_leftover_chain_kernel`) is unchanged except its
  input dtype, and its fp16 ascending accumulation sees identical bits.
- **Cooperative matrix in a custom kernel:** the runtime GLSL→SPIR-V path
  (glslangValidator) carries `GL_KHR_cooperative_matrix` via the header
  parameter; parameter names avoid `.x/.y/.z` swizzle collisions with the
  generated guard's `#define` macros (lhs/rhs/dst).
- **Guards:** the coopmat path requires fp16 operands, K%16==0 (already
  enforced), blocks ≤ 65535, and sane tile counts; anything else falls
  back to the unchanged f32 batched-matmul route. All 194 program linears
  take the coopmat path.

The prior attack's lesson is respected: the fp16 chain stays a separate
dispatch (the block-rounded single-kernel attempt, `531c48a7`, was slower
twice); only the operand feed and the partials dtype changed, not the
chain structure.

## Bit-exactness evidence, in order

1. Unit harness (`lincheck.py` in this receipt dir, release wheel): every
   path vs the old dispatch chain on the four exact program shapes
   (375,512,640), (375,1024,1024), (375,1024,4096), (375,4096,1024) —
   partials compared as uint16 views after the old path's own fp16
   rounding, chain outputs, and full linear outputs with bias: **ALL-EXACT,
   0 mismatches** (7.68M–98.3M elements per shape).
2. Standalone encoder legs, both wheels: `encoder_hidden` `38c73261…` on
   every leg (cut warm r1/r2 release wheel, cut record and base record on
   the diag wheel).
3. Full E2E, six runs per arm (below): pin + transcript on all 12.

## A/B: full E2E battery, clean runners, release wheel, 6 runs per arm

Both batteries this session, same host, same release wheel, same lock;
base = `ecd3e81f` runner, cut = this branch's runner.

| quantity | base (`ecd3e81f`) | cut (this branch) |
| --- | ---: | ---: |
| encoder_ane median, r1–r6 | **7978.318 ms** | **7135.603 ms (−842.7, −10.6%)** |
| encoder_ane individual | 7950.0–8039.3 | 7023.0–7182.8 |
| total_pipeline median, r1–r6 | 9354.036 ms | 8460.852 ms (−893.2) |
| vk compute dispatches (encoder leg) | 4485 | **3709 (−776)** |
| GPU busy (device ticks, diag record) | 3715.5 ms | **2933.2 ms (−782.3, −21.1%)** |
| `encoder_hidden` sha256 | `38c73261…` | **`38c73261…`, all 6 runs** |
| transcript sha256 | `db501a8c…` | **`db501a8c…`, all 6 runs** |

Both arms, every run r1–r6: status `match`, emissions 104/104 with
matching_prefix 104, mel bit-exact, encoder bounds PASS, decode `gpu-loop`
with `tdt_fallback_reason: null`, island batch engaged (1 batch
submission, 1 worker start, 0 timeouts), `cpu_tensor_events` 0.

### Per-bucket GPU busy (record captures, diag wheel)

| kernel | before: disp / ms | after: disp / ms | delta ms |
| --- | ---: | ---: | ---: |
| MatmulF32Coopmat | 194 / 1448.8 | 0 / 0 | −1448.8 |
| custom kernels (Count; chain + new coopmat) | 937 / 835.2 | 1131 / 1748.7 | +913.5 |
| CastF16F32 | 1138 / 128.7 | 750 / 56.4 | −72.3 |
| CopyGeneralF16 (the mislabeled "SliceUpdatePairF16" bucket) | 605 / 319.8 | 217 / 85.8 | −234.0 |
| everything else (MatmulF32, ElementwiseF32, Reduce*, Conv, CopyGeneralF32, …) | 1611 / 983.0 | 1611 / 1042.3 | +59.3 |
| **total GPU busy** | **4485 / 3715.5** | **3709 / 2933.2** | **−782.3** |

The coopmat bucket moves into the custom-kernel bucket (194 new dispatches)
and comes out net ≈ −535 ms there; the two eliminated staging families
account for the rest. Dispatches −776, dispatch count is a wall lever too
(the host residual scales with statement/dispatch count).

## Identity

- Host `jwm1-linux`, aarch64, kernel `7.1.6-1-1-ARCH`, `/dev/accel/accel0`,
  Vulkan Mesa Honeykrisp (26.3.0-devel git-6f6afc8968), device Apple M1
  (G13G B1), `timestamp_period_ns: 1`, glslangValidator runtime compile.
- Lock `/tmp/m1-gpu.lock`, inode 29 throughout, `flock -w 900`, never
  stolen, never unlinked; `flock -n` free after every battery.
- Release wheel `mlx_omarchy-0.32.2.dev202609160525+7d82ec94`
  (site `/var/tmp/ParakeetE2EBaseline7d82/site`) for every timing and
  identity claim — the change is runner-only (Python + runtime-compiled
  SPIR-V), no libmlx byte changes, so the baseline wheel serves both arms.
  Diag wheel `mlx_omarchy-0.32.2.dev202609160555+diag.1f830a0d`
  (`/var/tmp/enc-gpubusy/site-diag`, sha256 `56ebf422…`) used ONLY for
  the profiled record captures (lowerings identical: `1f830a0d` and
  `ecd3e81f` differ only in the runner file + receipt).
- Cut runner sha256 `eedb7c4802ca678183deb3a55c134f2f9b11571e487d82f8d21c
  8b38316d851f` = committed `overlay/tools/coreml/vulkan_encoder.py`;
  base runner `4e60c5a49038e046750516e153b3a1cce700b3c9ade75ab9ebcdda10d
  0dea955` = `ecd3e81f`'s file.
- Strict worker + libane-strict (`/var/tmp/mlx-main-strict/.work/mlx/
  build-ane-device/tools/mlx-omarchy-ane-worker/…`,
  `/var/tmp/island-reexport/libane-strict.so`), bundles
  `/var/tmp/island-reexport/bundles`, `ANE_ISLAND_MODE=resident-batch`:
  1 batch submission, 1 worker start, 72 rounds, 0 timeouts on every run.
- Artifacts on jwm1: `/var/tmp/enc-gpubusy/` (runners, `lincheck.py`,
  `cut-e2e.sh`/`base-e2e.sh` batteries, `cut-enc.sh` legs,
  `base-record.jsonl`/`cut-record.jsonl` profiles, `buckets-base.txt`/
  `buckets-cut.txt`, `ver-out-r1..r6` and `base-e2e-out-r1..r6` E2E
  reports, `e2e-run.log`/`base-e2e-run.log`).
- Scripts in this receipt dir: `lincheck.py` (bit harness),
  `collect_ab.py` (battery collector), `analyze_buckets.py` (profile
  bucketizer), `cut-enc.sh`, `cut-e2e.sh`, `base-e2e.sh`.

## Not claimed

- No claim on the remaining custom-kernel bucket (now the largest line,
  ≈1.75 s: the coopmat+chain pair itself) — a fused single-kernel chain
  was measured slower twice and stays rejected.
- No claim on the attention `MatmulF32` bucket (50 disp, ≈353 ms), the
  ElementwiseF32 residue, the host residual, macOS, other fixtures, or
  the TDT leg.
- The remaining 217 CopyGeneralF16 dispatches (85.8 ms) are weight
  uploads and graph-mandated transpose materializations outside the
  linear path; not attacked here.
- `63c1d3cf` not merged, not touched.

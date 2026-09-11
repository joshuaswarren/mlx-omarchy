# QMM coopmat data path vs native Metal: structural analysis

Assignment: raise the cooperative-matrix quantized matmul from ~1.03
TFLOP/s toward the native ~1.7 TFLOP/s on the real 1K prefill shapes,
working at the MMA-pipeline level. This file records the upstream
Metal structure, our kernel's structure, and the per-tile cost model
that motivated the bench arms. Numbers below are derived from source
reading; measured numbers land in `verdict.json`.

## Upstream Metal (`mlx/backend/metal/kernels/quantized.h`)

Native MLX 0.32.2 dispatches f16 affine 4-bit/group-64 transposed
prefill (`M >= qmv_limit`) to `affine_qmm_t` -> `qmm_t_impl`
(`quantized.cpp`: `bm=32, bn=32`, grid `(ceil(N/32), ceil(M/32), B)`,
threadgroup `(32, wn=2, wm=2)` = 4 simdgroups = 128 lanes).

- **Tile geometry**: 32x32 output tile per threadgroup; each simdgroup
  owns 16x16 (`TM = BM/(8*WM) = 2`, `TN = BN/(8*WN) = 2`).
- **Accumulator fragments per lane**: 4 fragments of 8x8 f32
  (`MMATile<AccumType, 2, 2>`), 2 thread elements each = 8 f32
  registers of accumulator per lane.
- **MMA issue rate**: `BlockMMA::mma` walks `BK=32` in four 8-wide
  steps; per step `tile_matmad` issues `TM*TN*TM = 2*2*2 = 8`
  `simdgroup_multiply_accumulate` per lane, i.e. **32 MMA issues per
  lane per 32-k slice** (16 per 16-k equivalent - the same as our
  kernel). Operand fragments reloaded from threadgroup each step: 2 A
  frags + 2 B frags = 4+4 f16 elements per lane per 8-k step.
- **Operand staging**: both operands go through threadgroup memory as
  **f16** (`threadgroup T Xs[BM * BK_padded]`), leading dimension
  padded `BK_padded = BK + 16/sizeof(T) = 40` f16 = 80 B row pitch,
  explicitly to break shared-memory bank aliasing. 2 x 32x40 f16 =
  5 KiB shared per threadgroup.
- **Dequantisation**: `QuantizedBlockLoader` reads packed u8 weight
  words from device memory and dequantizes **straight into the f16
  threadgroup tile** (`dequantize<T=f16>`): `scale * nibble + bias`
  in f16 arithmetic. There is no f32 intermediate anywhere.
- **Reuse**: per k slice each A row-block is read by 2 simdgroups and
  each B col-block by 2 simdgroups (WM=WN=2), so shared reads are
  ~2x(A tile) + 2x(B tile) = 8 KiB f16 per slice against 4 KiB of
  writes -> **12 KiB shared traffic per 65536 FLOP = 0.19 B/FLOP**.
- **Double buffering**: none. Single buffer; barrier, load, barrier,
  MMA, next slice - loads serialize with MMAs, **2 barriers per
  65536 FLOP**.
- **Weight re-reads across M**: the same per-threadgroup structure as
  ours: weights are re-read and re-dequantized for every 32-row tile
  band (33 bands at M=1053). Upstream does not amortize this either;
  it is not the differentiator.

## Our kernel (`shaders/qmm_coopmat.comp`)

64 lanes = 2 subgroups; 32x32 output tile; each subgroup owns a
16-row half via `acc[2][4]` = **8 f32 8x8 fragments per subgroup =
16 f32 accumulator per lane** (2x upstream's accumulator density).

- **MMA issue rate**: per 8-k `ks` step each subgroup issues 8
  `coopMatMulAdd` (2 rb x 4 cb) = 16 per lane per 16-k step.
- **Staging**: `x_s[32x16]` and `w_s[16x32]` **f32** (4 KiB total).
  x: 4 packed f16x2 words per lane unpacked to f32 (exact); w: one
  packed word per lane dequantized to f32 (`scale*float(nibble)+bias`,
  same expression as upstream but in f32).
- **Shared traffic per 16-k step**: writes 768 f32 = 3 KiB; reads per
  ks per subgroup 128 f32 (A) + 256 f32 (B), doubled across 2 subgroups
  x 2 ks = 6 KiB -> **9 KiB per 16384 FLOP = 0.56 B/FLOP**, ~2.9x
  upstream per FLOP, at twice the bytes per value.
- **Barriers**: 2 per 16-k step = 2 per 16384 FLOP = **4x upstream's
  barrier rate per FLOP**.
- **Bank aliasing**: `w_s` row pitch 32 f32 = 128 B = the full 32-bank
  cycle, so every k row maps to an identical bank pattern; `x_s` pitch
  16 f32 = 64 B aliases every second row. Upstream pads instead (40
  f16 pitch). Conflict degree depends on the driver's coopMatLoad
  lowering; arm 2 (padded strides 66/34) measures the aggregate cost.
- **Dequant work per reused weight word**: 8 nibble extracts + 8 f32
  fma + 8 f32 shared stores per word per tile-band pass, repeated for
  every one of the 33 M-bands at M=1053 (same as upstream's 33 f16
  re-dequant passes).
- **Redundant operand loads**: none inside a step (each word lane maps
  1:1 to distinct data); the redundancy is the per-band weight
  re-dequant above, which upstream shares.

## Structural conclusion

Upstream's advantage is not exotic: f16 staging width (2x less shared
bytes), 4x fewer barrier events per FLOP, 2.9x less shared traffic per
FLOP, padded strides, and a native `simdgroup_multiply_accumulate`
lowering. A Vulkan cooperative matrix cannot dequantize "directly into
fragments": SPIR-V cooperative matrix objects have no per-lane element
access (GL_KHR_cooperative_matrix exposes only coopMatLoad / Store /
MulAdd), so some memory round trip is unavoidable on this API. The
achievable digest-preserving restructures are (a) chunk-level staging
to cut barrier count 4x and amortize staging over 4x more MMA work per
barrier, (b) padded strides against bank aliasing, and (c) a
device-memory pre-dequant pass so the GEMM loop is loads + MulAdd only
(largest diff, needs host scratch plumbing). The driver's own 8x8x8
fp32 MulAdd rate - measured by bench arms 3 and 4 - bounds every
shader-side restructure: if the pure-MMA ceiling sits near 1.03
TFLOP/s, the shipped kernel is already at the platform limit and no
data-path change can close the gap to 1.7.

## Bench arms (`shaders/qmm_coopmat_bench.comp`, MLX_OMARCHY_QMM_COOP_BENCH)

| arm | data path | digest-preserving |
|-----|-----------|-------------------|
| 0   | shipped kernel (step staging) | reference |
| 1   | chunk staging, one barrier pair per 64-k chunk | yes, bitwise |
| 2   | arm 1 + padded strides (66/34) | yes, bitwise |
| 3   | shared-load + MulAdd ceiling (synthetic operands, no staging in loop) | measurement only |
| 4   | pure MulAdd ceiling (fragments loaded once) | measurement only |

The k32 precedent (`receipts/2026-09-10-qmm-prefill-tile`): an
order-preserving source schedule already shifted f16 outputs once
(Honeykrisp cooperative-matrix anomaly). Arms 1/2 are exactly such a
schedule change, so the arm host-reference gates and the probe digest
gate decide bit-identity on hardware; a mismatch is reported as that
anomaly, not as a landing candidate.

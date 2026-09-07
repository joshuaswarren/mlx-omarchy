# PrefillGemm: blocked GEMM kernels for prefill (M1, 2026-09-07)

Tree: `/tmp/mlx-release-f5-final` at `81e9f53a` plus the uncommitted
perf-dispatch work. Machine: jwm1 (Apple M1 G13G, 8 GPU cores, Linux
7.1.6, Mesa 26.1.7 Honeykrisp, glslc 2026.3). Every timed run held
`~/benchq/gpu.lock`; builds ran at `nice -n 19`.

## What changed

* `overlay/mlx/backend/omarchy/shaders/qmm_gemm.comp` (new): blocked
  GEMM for the m > 1 affine QuantizedMatmul, transposed word-packed
  weights (bits 2/4/8, any group size). A 256-thread workgroup owns a
  64x64 (4x4 per thread) or 64x128 (4x8 per thread) output tile; per
  32-wide k step it stages x as `float16_t` (exact for float16 storage)
  and the dequantized weights as f32 (`scale * q + bias`, once per weight
  element per m-tile) in shared vec4 tiles; every thread then runs 32
  outer products from two or three shared vec4 loads per 16 or 32 FMAs.
  Stage loads are branch-free (clamped index plus select). A third
  geometry, 32x32 with four in-workgroup k-partitions (`KS=4`), is
  compiled for tiny grids.
* `overlay/mlx/backend/omarchy/shaders/matmul_gemm.comp` (new): the same
  64x64 / 4x4 blocking with the full `matmul.comp` contract (alpha, beta
  and C, transposed operand views through the gaps, four collapsed batch
  axes), f16 shared tiles for float16 storage.
* `primitives.cpp`: `QuantizedMatmul::eval_gpu` affine m > 1 branch
  dispatches `QmmGemm64F16` / `QmmGemm64x128F16` for float16 transposed
  word-packed weights (default; `MLX_OMARCHY_QMM_TILE_V2=0` restores
  QmmTile / QmmTileRb; `MLX_OMARCHY_QMM_TILE=0` still selects the
  per-element kernel). The 64x128 tile is taken when the 128-wide grid
  still has at least 64 workgroups. `MLX_OMARCHY_QMM_GEMM_SPLITK=1`
  enables the 32x32 split-K tile for grids under eight 64x64 tiles.
  `dispatch_matmul` dispatches `MatmulGemm{F32,F16,BF16}` for m >= 64
  (`MLX_OMARCHY_MATMUL_GEMM=0` restores the 16x16 tile). Grids come from
  `gemm_group_count`.
* `compute.h` / `compute.cpp` / backend `CMakeLists.txt`: six kernels
  appended (`QmmGemm64F16`, `QmmGemm32K4F16`, `MatmulGemmF32/F16/BF16`,
  `QmmGemm64x128F16`).
* `overlay/tests/omarchy/bench_prefill_gemm.cpp` (new,
  `omarchy_prefill_gemm_bench`, not a ctest): N back-to-back dispatches
  of the four Qwen2.5-0.5B projection shapes and the two batched SDPA
  matmuls at L = 41 / 262 / 1053, host double reference on a row sample,
  best of three, one JSON row per (shape, L, variant) with us/call,
  TFLOP/s, GB/s and `max_diff_vs_v1` (largest |v2 - v1| over the full
  output).
* `docs/install-omarchy.md`: the three switches.

## Accumulation order

Every output element is one f32 accumulator summing `x[r][k] *
(scale * q + bias)` over ascending k, exactly the QmmTile / matmul.comp
order, and the driver compiles both to the same fused `ffma` chain: the
bench reports `max_diff_vs_v1 = 0` on every shape and L (the outputs are
bit-identical). The only order-changing path is the opt-in split-K tile
(partials combined as `((p0 + p1) + p2) + p3`), which measured 2.9x on
kv_proj at L = 41 (257 -> 120 us) with a max difference of 0.03 against
v1; it stays off until the parent's paired token check accepts it.

## Before / after (us per call, best of 3, `prefill-gemm-ab-m1.jsonl`)

| op | shape | L | v1 us | v2 us | speedup | v2 TFLOP/s |
|---|---|---|---|---|---|---|
| qmm | qo_proj 896x896 | 41 | 393.0 | 268.0 | 1.47x | 0.246 |
| qmm | qo_proj | 262 | 1767.7 | 988.5 | 1.79x | 0.426 |
| qmm | qo_proj | 1053 | 6103.1 | 2835.3 | 2.15x | 0.596 |
| qmm | kv_proj 128x896 | 41 | 345.3 | 256.7 | 1.35x | 0.037 |
| qmm | kv_proj | 262 | 324.7 | 272.5 | 1.19x | 0.221 |
| qmm | kv_proj | 1053 | 1058.3 | 650.5 | 1.63x | 0.371 |
| qmm | gate_up 4864x896 | 41 | 2098.3 | 1062.1 | 1.98x | 0.337 |
| qmm | gate_up | 262 | 9068.9 | 4487.3 | 2.02x | 0.509 |
| qmm | gate_up | 1053 | 32445.6 | 14980.6 | 2.17x | 0.613 |
| qmm | down 896x4864 | 41 | 2063.3 | 1354.9 | 1.52x | 0.264 |
| qmm | down | 262 | 9627.4 | 5229.2 | 1.84x | 0.437 |
| qmm | down | 1053 | 33297.2 | 15202.4 | 2.19x | 0.604 |
| matmul | sdpa_scores 14x[L,64]x[64,L] | 41 | 97.5 | 91.2 | 1.07x | 0.033 |
| matmul | sdpa_scores | 262 | 962.5 | 582.2 | 1.65x | 0.211 |
| matmul | sdpa_scores | 1053 | 13639.8 | 6176.3 | 2.21x | 0.322 |
| matmul | sdpa_probs 14x[L,L]x[L,64] | 41 | 63.9 | 62.3 | 1.03x | 0.048 |
| matmul | sdpa_probs | 262 | 738.5 | 458.5 | 1.61x | 0.268 |
| matmul | sdpa_probs | 1053 | 10087.9 | 5085.3 | 1.98x | 0.391 |

Baseline-only run before any change: `prefill-gemm-baseline-m1.jsonl`.
The L = 41 dense rows are the 16x16 tile on both sides (gate m >= 64).

Per layer at L = 1053 the seven projections drop from 126 ms to 61 ms
and the two attention matmuls from 24 ms to 11 ms.

Targets: >= 1.2 TFLOP/s at L = 1053 is not met (0.60-0.61); >= 2x at
L = 41 is met on gate_up (1.98x) only, 1.35-1.52x elsewhere.

## Why 0.6 TFLOP/s is the ceiling reached

Measured on the same shapes (all bit-identical unless noted):

| variant | qo_proj L=1053 | notes |
|---|---|---|
| 4x4, 64x64, 256 threads, f32 x tile | 3049 us / 0.554 TF | first cut |
| 8x4, 64x64, 128 threads | 3692 / 0.458 | |
| 8x4, 128x64, 256 threads | 3588 / 0.471 | |
| 8x8, 64x64, 64 threads | 5117 / 0.330 | |
| 8x8, 128x128, 256 threads | 2976 / 0.568 | 199 half-registers, 512 threads/core |
| 4x4 + register prefetch of step + 1 | 3758 / 0.450 | |
| 4x4, BK = 16 | 3900 / 0.433 | |
| 4x4 unrolled 2 / 4 / 8 k per block, loads first | 3092-3121 / 0.54 | driver re-sinks the loads |
| 4x4 unrolled + `memoryBarrierShared` / subgroup fence | 3119-3122 / 0.54 | fence dropped by the driver |
| 4x4 `fma()` instead of `a * w + acc` | 3037 / 0.557 | identical code: the driver already fuses |
| 4x4 rolled k loop (`[[dont_unroll]]`) | 3370 / 0.502 | |
| 4x4 branch-free stage loads | 3079 / 0.549 | kept: no regression, fewer branches |
| 4x8, 64x128, f16 x tile | 2827-2850 / 0.60 | kept for large grids |
| 4x4 with the FMA loop removed (stage only) | 120 us | staging is not the cost |
| 4x4 with shared loads replaced by ALU constants | 1252-2403 us | ALU-only ceiling 0.7-1.35 TF |

`AGX_MESA_DEBUG=shaders` shows the compiled k loop as `lload x2, ffma
x16, iadd x4` repeated: the driver sinks every shared load to just
before its first use regardless of source order, fences, or unrolling,
so each k pays the full threadgroup-memory latency and the only cover is
occupancy (1024 threads per core at <= ~96 half-registers). Bigger
micro-tiles halve the loads per FMA but also halve the resident
simdgroups, which is why 8x8 lands on the same number. The Asahi
community profile reached the same conclusion (prior-art survey, ranked
item 4). Getting past this needs the driver to schedule loads early or
a fp16 path that the exactness policy rules out.

## Tests

M1 (`prefill-gemm-tests-m1-*.log`): omarchy_primitive_tests
`-tc='*uantiz*,*atmul*'` 16/16 cases, 1978949 assertions;
omarchy_matmul_family_tests 13/13, 46227; omarchy_quantized_batch_layout_tests
4/4, 1488; omarchy_fast_ops_tests 30/30, 45211. All with the new kernels
default-on.

llvmpipe (`MLX_OMARCHY_ALLOW_NON_APPLE=1`, `prefill-gemm-tests-llvmpipe-*.log`,
same four suites on the final tree): 16/16 cases, 1978949 assertions;
13/13, 46011; 4/4, 1488; 30/30, 57158. All passed.

End-to-end (parent, same wheel perfsnap5, three alternating pairs, token
IDs equal): prefill 272 -> 422 tok/s at 262 tokens, 300 -> 595 at 1053,
105 -> 112 at 41; decode unchanged
(`prefill-gemm-ab-m1-perfsnap5`).

## Reproduce

    # jwm1, tree at ~/benchq/PrefillGemm (QmmBandwidth clone + this overlay)
    nice -n 19 ninja -C build omarchy_prefill_gemm_bench
    MLX_OMARCHY_PREFILL_BENCH_QMM_VARIANTS=v1,v2,v2splitk \
      flock ~/benchq/gpu.lock ./build/tests/omarchy/omarchy_prefill_gemm_bench

## Short prefill: column-batched GEMV (`shaders/qmm_vec_cols.comp`)

Parent's follow-up slice. Decode-shaped kernel (eight 32-lane slots per
workgroup, two output rows per slot, one weight uvec4 per row per lane
per half group) that applies each dequantized weight to eight activation
rows held as float16_t registers, with the per-(column, half-group)
activation sums tabulated once per workgroup in shared memory. Every
column is bit-identical to the m == 1 decode GEMV (same products, same
order, same lane fold); it is a different order from the tile kernels.
Kernels `QmmVecColsF16` / `QmmVecColsSubgroupF16`, grid x = ceil(n/16),
y = ceil(m/8); gate: transposed affine q4/g64 float16, k <= 4864, aligned
x and w, 2 <= m <= 16 or (m <= 64 and fewer than eight 64x64 GEMM tiles).

us per call, best of 3 (`prefill-gemm-cols-m1.jsonl`, v2 = blocked GEMM):

| shape | m=8 v2 / cols | m=16 | m=41 | m=64 |
|---|---|---|---|---|
| qo_proj | 262 / 70 | 267 / 122 | 271 / 319 | 271 / 422 |
| kv_proj | 213 / 30 | 217 / 38 | 226 / 65 | 234 / 83 |
| gate_up | 1053 / 299 | 1055 / 572 | 1063 / 1629 | 1066 / 2190 |
| down | 1334 / 235 | 1339 / 446 | 1357 / 1263 | 1367 / 1707 |

The kernel is ALU-bound at ~0.2-0.3 TFLOP/s (shaderdb: 231 half-registers,
448 threads per core), so it only wins where the GEMM grid is starved.
Gated as above the L = 41 layer saves ~320 us of ~4.5 ms of projections
(the two k/v projections); the projection sum at L = 41 is ~4.5 ms per
layer against ~15 ms per layer end to end, so the short-prefill gap is
mostly outside the GEMMs.

Default OFF (`MLX_OMARCHY_QMM_VEC_COLS=1` enables): with it default-on
the tile-vs-untiled bound tests fail (omarchy_primitive_tests 168
assertions, omarchy_matmul_family_tests 29) because those tests pin the
tile summation order; all four suites pass with it opt-in (M1 logs
`prefill-gemm-tests-m1-*.log`, rerun after the change).

# DecodeQ4Vec v3: qmv_fast-shaped affine q4 decode GEMV (M1, 2026-09-07)

Agent QmmBandwidth3. Source: overlay at HEAD 81e9f53a plus the uncommitted
sibling work of 2026-09-07 plus this change. Device: Apple M1 (G13G B1),
Honeykrisp Mesa 26.1.7, jwm1. Bench: `overlay/tests/omarchy/bench_qmm_bandwidth.cpp`
(`omarchy_qmm_bandwidth_bench`), N = 100 back-to-back `quantized_matmul`
dispatches per eval, best of 7 trials after one warm-up, configurations
interleaved trial by trial in one process, every run
`flock gpu.lock taskset -c 4-7` on an idle box unless noted. Files:

* `qmm-bandwidth-v3-final-m1.jsonl`: repo-state build (the tree as shipped),
  configs `old` (DecodeQ4Word, `MLX_OMARCHY_QMM_VEC_Q4_V2=0`), `r2t128`
  (the shipped default), `r8t64`, in both orders (runs repoF, repoG).
* `qmm-bandwidth-v3-sweeps-m1.jsonl`: the scratch-build sweeps (runs A-E,
  gate1/2) with the previous uvec4 kernel (`v2`) built alongside the new
  one under scratch enums, so v2 and v3 are timed in the same process.
* `qmm-bandwidth-v3-shaderdb-m1.txt`: `AGX_MESA_DEBUG=shaderdb` lines.

## What changed

`overlay/mlx/backend/omarchy/shaders/qmm_vec_q4.comp` is rewritten in the
shape both MLX `qmv_fast` and ggml-Metal `mul_mv_q4_0` use on AGX
(prior-art-survey.md ranked item 2): small workgroups of 32-lane slots,
ROWS consecutive output rows per slot, one uvec2 (8 B = 16 elements) per
lane per row per K-step so a slot covers 512 contiguous elements per
step, all ROWS weight loads of a step issued before the first FMA, x
loaded once per step as 16 pre-scaled f32 and reused across the rows, row
bases (weights and scale/bias) computed once before the K loop so no
multiply remains in it, nibbles masked in place (no shifts), subgroupAdd
epilogue (shared-memory tree on drivers without a 32-wide arithmetic
subgroup). ROWS and the workgroup size are specialization constants
(constant 0 = rows from `params.shape[3]`, constant 1 = threads from
`params.in_strides[0]`, `compute.cpp` `specialization()`), so the row
loop unrolls and registers are allocated for exactly the rows in use and
one SPIR-V serves every policy. Rows past n read the clamped last row and
are never written, so the K loop has no row guard.

Dispatch (`QuantizedMatmul::eval_gpu`, m == 1 affine branch): default
2 rows per slot, 128 threads (4 slots) per workgroup, grid =
ceil(n / 8); `MLX_OMARCHY_QMM_Q4_ROWS` (1..8) and `MLX_OMARCHY_QMM_Q4_WG`
(32..256) pin them for measurement; if the grid would exceed the 65535
one-axis limit the rows then the threads double until it fits (the
previous code clamped the grid silently). `MLX_OMARCHY_QMM_VEC_Q4_V2=0`
still falls back to DecodeQ4Word. Same six kernel enums and CMake targets
as before; the shared-memory flavor's scratch is sized for 8 rows x 256
threads (8 KB).

Accumulation order (documented in the shader header): per lane per
step an f32 dot of 16 terms (one 8-term chain per word, chains summed),
then `s * dot + b * xsum` accumulated over the lane's steps, then one
subgroupAdd across 32 lanes. This is a rounding-order change from the
previous 32-term half-group chains; every product x_j * q_j is still
exact. Max abs error against the dense reference is unchanged on all
five shapes (f16 output rounding dominates; see the jsonl `max_abs_err`).

The row policy is flat: 2 rows wins or ties on every decode shape from
n = 128 to n = 151936 (below). The survey's 4-rows/8-rows consensus does
not transfer to Honeykrisp: shaderdb shows 73 GPRs at 2 rows (1024
threads/core) but 115 at 4 rows (832 threads/core) and 179 at 8 rows
(576 threads/core); the compiler keeps every row's 16 masked products
live, so occupancy loss outweighs the extra x reuse.

## Results, repo-state build (us per dispatch / GB/s), `qmm-bandwidth-v3-final-m1.jsonl`

| shape | bytes | old, repoF / repoG | r2t128 (default), repoF / repoG | r8t64, repoF / repoG |
|---|---|---|---|---|
| floor 8x64 | 288 | 17.0 / 17.7 | 14.8 / 15.2 | 23.2 / 23.8 |
| k_proj 128x896 (57 KB) | 64,512 | 17.9 / 28.6 | 15.0 / 21.9 | 20.8 / 36.6 |
| q_proj 896x896 (401 KB) | 451,584 | 24.4 / 54.6 | 16.3 (28) / 27.9 (16) | 19.6 / 37.2 |
| gate 4864x896 (2.18 MB) | 2,451,456 | 84.3 (29) / 104.9 (23) | 48.7 (50) / 69.4 (35) | 62.7 / 83.3 |
| down 896x4864 (2.18 MB) | 2,451,456 | 71.4 (34) / 71.3 (34) | 40.3 (61) / 42.1 (58) | 50.5 / 52.5 |
| lm_head 151936x896 (68 MB) | 76,575,744 | 2233 (34) / 2226 (34) | 1204 (64) / 1221 (63) | 1336 (57) / 1343 (57) |

repoF ran with the box at load 1.69 (the idle wait timed out), repoG at
1.09; the small shapes move with host clock state, lm_head is stable to
2%. The `env` column of repoG inherited r8t64's variables and is not a
measurement (bench comment updated).

## Results, previous uvec4 kernel vs v3 in one process (scratch build, runs D and E, us / GB/s)

| shape | old | v2 (uvec4, 2 rows, 256 thr) | r2t128 | r2t64 | r2t256 | r1t128 | r4t128 | r8t64 | r8t128 |
|---|---|---|---|---|---|---|---|---|---|
| floor 8x64 (D/E) | 17.6 / 18.6 | 16.8 / 19.5 | 15.4 / 16.7 | 15.3 / 16.5 | 15.5 / 16.1 | 15.8 / 16.7 | 15.5 / 19.3 | 24.0 / 24.3 | 24.3 / 24.6 |
| k_proj 57 KB | 30.6 / 30.8 | 25.9 / 26.1 | 24.6 / 24.7 | 24.4 / 24.6 | 24.5 / 24.8 | 26.6 / 26.6 | 29.9 / 29.9 | 39.2 / 39.2 | 39.3 / 39.5 |
| q_proj 401 KB | 58.9 / 61.6 | 44.1 / 38.1 | 32.8 / 34.7 | 39.6 / 34.9 | 33.5 / 33.7 | 54.2 / 47.7 | 41.6 / 40.0 | 37.6 / 37.5 | 38.4 / 38.0 |
| gate 2.18 MB | 114.8 / 121.7 | 86.2 / 90.0 | 73.7 / 76.3 | 78.1 / 73.9 | 79.6 / 73.0 | 121.5 / 103.1 | 78.0 / 79.4 | 86.3 / 85.2 | 87.5 / 85.8 |
| down 2.18 MB | 74.5 / 74.6 | 53.1 / 53.2 | 45.7 / 46.1 | 46.9 / 47.5 | 45.7 / 45.9 | 50.7 / 51.2 | 58.5 / 58.9 | 56.0 / 56.2 | 56.6 / 57.0 |
| lm_head 68 MB | 2275 / 2236 | 1453 / 1361 | 1252 / 1205 | 1268 / 1215 | 1256 / 1217 | 1991 / 1924 | 1340 / 1297 | 1361 / 1340 | 1365 / 1323 |

lm_head: 56-58 GB/s (v2) -> 63-64 GB/s (r2t128); the 68 MB stream is at
92% of the 68.25 GB/s LPDDR4X ceiling. gate/down in this bench stay
resident in the 8 MB system cache across the 100 repetitions (the 2.18 MB
shapes reach 50-61 GB/s wall with the 15 us floor in series, above what
DRAM allows), so per-token decode sees the lm_head figure for them, plus
the 2-3 us lower per-dispatch floor of the new kernel (45 vs 54 preamble
instructions, 88 vs 108 uniforms) on all 169 GEMV dispatches per token.

Gate-only runs (gate1/gate2, 15 trials, both orders) rank the same:
v2 55.5 / 59.0, r2t128 45.4 / 48.5, r2t64 48.4 / 45.4, r2t256 45.8 / 46.9,
w4r2t256 51.3 / 51.4, r4t128 54.3 / 56.2, r8t64 62.4 / 62.3.

## Variants measured and not shipped (runs A-C, scratch build)

* 16 B per lane per row (uvec4, `w4r*`): never better than 8 B at the
  same rows/threads (lm_head 1281-1312 at 2 rows vs 1215-1228; k_proj
  and q_proj equal within noise). The x slice doubles to 4 uvec4 loads
  per step and the second K step of k = 896 idles 12.5% of lanes either
  way.
* 4 and 8 rows per slot (`r4*`, `r8*`): 115 / 179 GPRs, 832 / 576
  threads per core; slower on every shape (table above). The predecessor's
  sequential 8-row chunk experiment (qmm-bandwidth-v2-experiment-outer4.txt)
  reached lm_head 1389-1394 us and gate 86-93 us; r2t128 beats both.
* 1 row per slot: 51 GPRs but half the x reuse; lm_head 1924-1991 us.
* K-loop unroll 2 and 4 (all weight loads of two or four steps issued
  first): 103 GPRs at unroll 2; slower everywhere except gate at unroll 4
  (run C: 72.7 vs 73.8), lm_head 1576 at unroll 2. Not kept.
* 32-thread workgroups (`r2t32`): equal to 64/128/256 on the four shapes
  it fits; n = 151936 needs the grid bump (75968 groups).
* 128-thread workgroups for the small shapes: k_proj is the same at 64,
  128, 256 threads (24.1-25.5 us, floor 15-16); q_proj prefers 128/256
  by 1-2 us. 128 is the default because it wins or ties everywhere.

## Bench harness changes

* Configuration grammar `old` | `env` | `rN[tT]` (rows, threads).
* Each configuration is checked on its own scaled copy of x (distinct
  sign and magnitude) against a fresh dense reference, and `ok` is
  reported per configuration. The previous single-x check passed a
  kernel that left 20,866 rows of lm_head unwritten (scratch run B,
  `r2t32` before the grid bump) because the recycled output buffer still
  held the previous configuration's values; `-x` alone was not enough
  either (run C). In isolation that configuration reads
  `max_abs_err 136`.

## Tests

* M1 (repo-state build `benchq/QmmBandwidth3/repo`): omarchy_quantized_batch_layout_tests
  1488/1488 and omarchy_primitive_tests `*quantized*` 8110/8110 with the
  default, with `MLX_OMARCHY_QMM_VEC_Q4_V2=0`, and with
  `MLX_OMARCHY_QMM_Q4_ROWS=8 MLX_OMARCHY_QMM_Q4_WG=64`.
* llvmpipe (`/tmp/qb3-work`, `MLX_OMARCHY_ALLOW_NON_APPLE=1`, shared-memory
  flavor): the same two suites 1488/1488 and 8110/8110 with the default,
  the kill switch, rows 8 / 64 threads, and rows 1 / 256 threads; bench
  correctness gate ok on the five small shapes for `old,r2t128,r1t64,r8t256,r4t32`.
* Greedy-token A/B against the previous kernel: parent's paired run
  (not part of this receipt).

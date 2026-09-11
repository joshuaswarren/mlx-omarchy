# Q4 decode GEMV native thread-mapping port — analysis and window plan

- schema: mlx-omarchy/q4-gemv-native-map/1 (CLOSED 2026-09-11: receipt-only negative - see verdict.json)
- branch: wave/Q4GemvNativeMapping (off origin/main ccc25c0f)
- llvmpipe bit-identity: DONE (see below); M1 window: pending (window.sh)

## Native Metal qmv mapping (mlx/backend/metal/kernels/quantized.h, upstream 0.32.2)

`qmv_impl` (plain tile, K % 512 != 0) and `qmv_fast_impl` (K % 512 == 0 and
N % 8 == 0), bits=4, group_size=64, T=half, U=float:

- **Rows per tile:** one threadgroup = `num_simdgroups`=2 simdgroups (64
  threads) covers 8 output rows: `out_row = tid.y*8 + simd_gid*4`; each
  simdgroup owns `results_per_simdgroup`=4 consecutive rows.
- **Words per lane:** Q4 packs 8 nibbles per uint32 (`pack_factor`=8,
  `bytes_per_pack`=4). `qmv_fast`: `packs_per_thread`=2 → each lane owns
  16 K-values (2 words, 8 bytes) per 512-wide K block;
  `block_size = values_per_thread * SIMD_SIZE` = 16*32 = 512.
  `qmv_impl`: `packs_per_thread`=1 → 8 values (1 word) per 256 block.
- **Addressing:** lane l reads words `simd_lid*packs*4` bytes into the row,
  advancing one full `block_size*4/8` bytes per K iteration — lanes are
  fully coalesced (256 contiguous bytes per row per iteration), and the
  `row` loop strides by whole rows (`in_vec_size_w`) to reach rows +1..+3.
- **x reuse:** `load_vector` loads the lane's 16 (or 8) x values ONCE per
  K block into `x_thread[16]` registers; the `row` loop reuses those
  registers across all 4 rows. x traffic per column is therefore 1/4 of a
  one-column-per-warp design.
- **Scale/bias:** one scale+bias pair per lane per row per K block
  (`simd_lid / scale_step_per_thread`, `scale_step_per_thread` = 4 for
  the fast tile), advanced by `block_size/group_size` = 8 groups per
  iteration.
- **Reduction:** per-lane `result[4]` accumulate across the whole K loop;
  ONE `simd_sum(result[row])` per row at the end (xor strides 1,2,4,8,16);
  lane 0 stores all 4 rows. 2 simdgroups = 8 rows per threadgroup.

## Our pre-change kernel (qmm_vec_q4_multi_subgroup_*)

- workgroup 256 threads = 8 subgroups; workgroup covers 8 columns; each
  32-lane subgroup owns ONE column; grid = sum(ceil(n_i/8)) (grids 144,
  112, 1216, 112 for the four Qwen2.5-0.5B decode dispatches).
- Per column the lane word order, quad half-chains, fma ladder, block
  finish `fma(scale, dot, sum*bias)`, and the 32-lane stride-1-first
  subgroupAdd are bit-pinned to native qmv
  (receipts/2026-09-09-q4-gemv-order). That per-column arithmetic was and
  remains exactly native.
- The DIFFERENCE vs native was only the mapping: x is re-loaded per
  column from L1 instead of being reused in registers across 4 columns
  (4x the x load instructions per column), and each lane carries a single
  accumulator chain (4x less per-lane memory-level parallelism).

## The change (this branch)

`overlay/mlx/backend/omarchy/shaders/qmm_vec.comp`, QMM_VEC_MULTI main:

- one 32-lane slot now owns `COLUMNS_PER_SLOT`=4 columns (native's
  results_per_simdgroup), workgroup = 64 threads = 2 slots = 8 columns,
  grid UNCHANGED (no host/dispatch change; `-DQ4_MULTI_32=1` optionally
  packs four slots into a 256-thread workgroup covering 32 columns for
  the occupancy axis).
- x values load ONCE per lane per K block (uvec4 for f16; 8/16 scalar
  loads hoisted for f32/bf16, the same count native issues) and feed all
  four per-column chains; per-column word order, chains, ladder, finishes
  are byte-for-byte the old Q4_ROW order, and each column is still
  reduced across the same 32 lanes with the same pairing.

## Verification so far

- glslang: 18/18 define-combination variants compile.
- llvmpipe (tree flavor), tools/q4-bw-bench: base (frozen ccc25c0f
  production) vs nativemap vs nm32 on the four production shapes PLUS
  fasttile (K%512==0) and oddn (N%8!=0) coverage shapes:
  **18/18 compares, 0 bit mismatches**.
- llvmpipe regression test
  (overlay/tests/omarchy/test_matmul_family.cpp "q4 decode gemv fused
  multi dispatch matches per-node bits"): fused multi dispatch vs
  per-node single-weight kernels, 3 dtypes x 6 shapes, Add epilogues,
  host f64-reference sanity under the documented bound family —
  **224/224 assertions pass**.
- M1 window RESULT (2026-09-11, verdict.json): bit-identity HOLDS - M1
  bench eq 18/18 (0 mismatches), 36/36 canonical digest legs across fork
  and stock under the corrected per-driver pins (commit 328218c; run1
  gate failure was the stale pre-re-pin BF16 262/128 fork value in the
  packaged driver, not the candidate). Speed does NOT improve: isolated
  per-dispatch ratios 0.98-1.05x vs base on all four production shapes,
  occupancy sweep flat x1->x16 at a ~30 us dispatch floor. Candidate
  fork medians 107.43 / 103.79 / 93.14 tok/s = 0.713 / 0.707 / 0.663 of
  the committed base baseline (shortfall is end-to-end environment, not
  the mapping: clean and loaded runs agree within 1.5%). RECEIPT-ONLY
  NEGATIVE: do not land; the kernel is latency-bound, not mapping-bound.

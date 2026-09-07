# Decode small-kernel cuts on the M1: rope, KV-row copy, rms_norm, cache donation (2026-09-07, DecodeKernels)

Repo `/tmp/mlx-release-f5-final` @ `81e9f53a` plus the uncommitted sibling work (per-op specialized
elementwise, q4 GEMV v2, fused SDPA, PrefillGemm's GEMM kernels as of each snapshot). Hardware: jwm1
Apple M1 (G13G B1), Mesa 26.1.7 Honeykrisp; second driver llvmpipe (LLVM 22) on the x86 build host
(`MLX_OMARCHY_ALLOW_NON_APPLE=1`). Every M1 command ran under `/home/joshuawarren/benchq/DecodeKernels/`
with `flock benchq/gpu.lock`, builds `nice -n 19 ninja -j 6`. No mlx-lm generation was run for
measurement; the parent's end-to-end A/B is `decode-kernels-ab-m1-perfsnap5/` (same wheel, three env
switches off vs on, 3 pairs: decode +4.7% short-32, +3.5% long-128, -1.3% 1024-ctx with the box busy,
token ids equal in every pair).

Ranking input: `decode-profile-after-v2-sdpa.md` (per-token device time of the tree before these changes).

## Result (raw production dispatch, GPU us per dispatch, `omarchy_dispatch_floor_bench raw 1000`, best of 3 runs, `taskset -c 4-7`)

| kernel, decode shape | run | 1-D / old | grid / new | shaderdb old (instrs / uniforms / preamble) | shaderdb new |
|---|---|---|---|---|---|
| FastRopeF16 query rope, 14 heads x 32 pairs, T=1 | snap1 (quiet box) | 6.00 | **3.51** | 244 / 182 / 203 | **66 / 62 / 57** |
| | snap4 (final tree, box busy) | 6.74 | **3.75** | | |
| | snap4 count0 (launch + preamble only) | 5.21 | 3.28 | | |
| CopyGeneralF16 KV-cache row paste, [2 heads, 64] into (2, 256, 64) | snap1 | 4.59 | **3.12** | 97 / 54 / 41 | **20 / 44 / 42** |
| | snap4 | 5.06 | **3.65** | | |
| | snap4 count0 | 3.73 | 3.67 | | |
| FastRmsNormF16 one 896-row (FastRmsNormSubgroupF16 new) | snap2 | 8.12 | **7.34** | 117 / 64 / 46 | 180 / 62 / 46 |
| | snap4 | 9.05 | **7.71** | | |
| | snap4 count0 | 4.32 | 3.76 | | |
| reference: ElementwiseF16 add (specialized) / FillF16 | snap1 | 3.06 / 2.53 | | 20 / 32 / 26, 14 / 24 / 22 | |
| reference: `alternate-production` (Add, RmsNormSg, RopeGrid, CopyGrid, Fill cycling) | snap1 / snap4 | 3.92 / 4.11 | | | |

Cross-run noise on the busy M1 is +-1 us (the parent's A/Bs and builds ran alongside snap2-4; snap1 ran
on a quiet box, load 0.1); deltas within a run are the numbers to trust: rope -2.5 to -3.0 us, copy
-1.4 to -1.5 us, rms_norm -0.8 to -1.3 us per dispatch. Per token that is 48 x ~2.7 + 48 x ~1.4 + 49 x
~1.0 = ~0.25 ms of the ~10 ms token, consistent with the parent's +3.5-4.7%. `alternate-production`
(five different pipelines, bindings and grids per five dispatches) costs the same as repeating one
kernel: pipeline switching is free on this driver, so the ~20 us mean per small dispatch that the decode
profile shows is neither kernel body nor pipeline-switch cost. The KV-cache copy elimination (item 4)
removes 48 transfer nodes per token (each a Honeykrisp compute launch, `copy-prepost` row 1.5-2.3 us,
plus the cache bytes: 64 KB per cache at <=256 ctx, 256 KB at 1024) and is not in the perfsnap5 A/B.

Shaderdb lines: `shaderdb-snap4.txt` (each `CS shader:` line follows its `PIPELINE <name>` marker; the
bench prints the marker before creating the pipeline).

## Changes (overlay, uncommitted; all default on, each with a kill switch)

1. `shaders/fast_rope.comp`: specialization constants `HALF_DIMS` (id 0) and `GRID_FLAGS` (id 1). Nonzero
   `HALF_DIMS` compiles `rope_grid()`: x = the T*half_dims (time, frequency) pairs of one head matrix
   (i = x % HALF_DIMS, t = x / HALF_DIMS, constant divisors), y = head, z = batch; the same expressions in
   the same order as the 1-D unravel, so bit-identical. The 1-D path (constant 0 = 0) is unchanged.
   `primitives.cpp` `RoPE::eval_gpu`: forward, no-passthrough dispatches set `kRopeGridFlag` (flags bit
   16) and dispatch (ceil(T*half_dims/256), N, B); `MLX_OMARCHY_ROPE_GRID=0` keeps the 1-D dispatch.
2. `shaders/copy_general.comp`: specialization constant `GRID` (id 1): y = outer row, x = inner element
   over `shape[0]`/`shape[1]` and the two stride pairs, no division. `copy.cpp` `copy_gpu_inplace`: a
   same-dtype window (not the packed-byte dtypes) whose collapsed rank is <= 2 with at most 64 rows sets
   `kCopyGridFlag` (flags bit 4) and dispatches (ceil(inner/256), rows); a rank-0/1 window rides as one
   row. `MLX_OMARCHY_COPY_GRID=0` keeps the runtime unravel.
3. `shaders/fast_norm.comp`: `block_reduce()`; with `-DUSE_SUBGROUP=1` the last five stages of the
   256-lane tree run inside subgroup 0 through `subgroupShuffleDown` with the same pairing (lane t adds
   lane t + stride), so the sum is bit-identical and the row needs 5 barriers instead of 11. Also for
   every variant: the barrier after reading the reduced value is gone (the next write to shared memory
   is the next row's store behind the row-loop barrier) and the trailing barrier is skipped on the last
   row (uniform condition). New blobs `fast_rms_norm_sg_{f32,f16,bf16}` (`CMakeLists.txt`), enums
   `FastRmsNormSubgroupF32/F16/BF16` appended before `Count` (`compute.h`, `compute.cpp`);
   `RMSNorm::eval_gpu` selects them on `subgroup_size == 32` with `VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT`;
   `MLX_OMARCHY_RMS_NORM_SUBGROUP=0` keeps the shared-memory tree. LayerNorm keeps the tree (no
   subgroup blob built; not on the decode path).
4. `copy.cpp` `copy_gpu`: a same-dtype Vector copy of a donatable input reuses the input's buffer and
   records nothing (upstream `set_copy_output_data` semantics). `SliceUpdate::eval_gpu` used to copy the
   whole KV cache through `copy_buffer` before every row paste; with donation the paste is in place.
   Verified by `dk_grid_check`: 40 chained `slice_update`s -> `vk_buffer_copies` 0 (was 40), result
   equals the CPU, a still-referenced input stays intact. `MLX_OMARCHY_COPY_DONATE=0` restores the copy.
5. `compute.h`: `kRopeGridFlag`, `kCopyGridFlag`; `compute.cpp` `specialization()`: FastRope* (key =
   {dims/2, traditional|transpose bits} when the flag is set) and CopyGeneral* except bool/u8 (key =
   {0, 1} when the flag is set); the unspecialized pipeline stays the runtime path.
6. `tests/omarchy/bench_dispatch_floor.cpp`: `Kernel` rows (Fill, Rope, RopeGrid, CopyRows, CopyGrid,
   RmsNorm, RmsNormSg) with their decode shapes, `count0` keeps the grid (the earlier count0 rows
   dispatched zero workgroups), `alternate-*` rows, `PIPELINE` stderr markers for shaderdb pairing, the
   shaderdb name table regenerated from `compute.h`.
7. `docs/install-omarchy.md`: the four switches.

Tried and reverted: keeping each thread's 4 values and weights in registers between the rms_norm
reduction and normalize passes (`bench-snap3.jsonl`, `run-snap3.out`): no measurable gain and the
preamble grew 46 -> 77 (uniform count 64 -> 141), so the shader is back to the snap2 form.

Not changed, with the measurement that says why: CastF16F32 / CastF32F16 are 17 instrs / 28 uniforms /
24 preamble, at the FillF16 floor, and there are 0 casts per decode token; MatmulF16 and SoftmaxF16 no
longer appear in the decode token (fused SDPA), only in prefill.

## Bit-exactness and suites

`dk_grid_check.cpp` (throwaway, in this directory, not in the tree): for every case the old path
(three switches off) and the new path (on) are evaluated and compared byte for byte. 270 cases on the M1
(`grid-check-snap4.log`) and on llvmpipe (`grid-check-llvmpipe.log`): rope in f16/f32/bf16 over 6 shapes
x traditional/half-split x base/freqs/partial-rotation/scale, head-seq transposed views, 3-D input;
slice_update KV-row, 41-row and strided-row pastes, 1-D windows, transposes and column updates in
f16/f32/bf16/int32/uint16; rms_norm in f16/f32/bf16 over 7 row shapes with and without a weight
(subgroup-32 path exercised on the M1); the donation chain and refcount cases above. All identical.

M1 (`suite-snap4-*.log`, final tree): omarchy_fast_ops_tests 30/30, omarchy_fast_regression_tests 2/2,
omarchy_primitive_tests 99/99, omarchy_runtime_tests 40/40, omarchy_copy_offset_tests 25/25,
omarchy_kv_ops_tests 14/14, all SUCCESS; primitive also 99/99 in 3 x 2 runs with `MLX_OMARCHY_COPY_DONATE`
0/1 (`run-snap3.out`).

llvmpipe (`suite-llvmpipe-*.log`, final tree): fast_ops 30/30, fast_regression 2/2, runtime 40/40,
copy_offset 25/25, kv_ops 14/14 SUCCESS; primitive 99/99 or 98/99 depending on the run:
`test_primitives.cpp:4898` (categorical determinism, `repeat` vs `token`) flakes on llvmpipe under
load, independent of these changes: 12 alternating full runs with `MLX_OMARCHY_COPY_DONATE` 0/1 failed
3/6 and 4/6 (`llvmpipe-primitive-categorical-flake-ab.txt`), the case passes in isolation, and the
DispatchFloor2 receipt saw the same case flake after unrelated barrier changes. Pre-existing; not
touched here (elementwise/random path, not this slice).

## Files in this directory

`bench-snap{1,2,3,4}.jsonl` (three `raw 1000` runs each), `shaderdb-snap{1,2,4}.txt`,
`grid-check-snap{1,2,4}.log`, `grid-check-llvmpipe.log`, `run-snap{1,2,3,4}.out` (the M1 driver
scripts' output incl. suite summaries), `suite-snap{2,4}-*.log`, `suite-llvmpipe-*.log`,
`dk_grid_check.cpp`. snap1 = rope grid + copy grid; snap2 = + rms subgroup; snap3 = + donation + the
reverted register cache; snap4 = the final tree.

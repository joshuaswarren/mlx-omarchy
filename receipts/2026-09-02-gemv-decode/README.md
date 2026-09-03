# 2026-09-02 — DecodeGemvPath: GEMV decode path built, gated, and stood down on benchmarking

**Measurement condition (added 2026-09-03):** all jwm1-linux timings in this document were taken with only 1 of 8 CPU cores online (GRUB entry `Omarchy ANE test`, which supplies a static device tree and left cores 1-7 offline). The machine now boots all 8 cores. GPU-bound figures were materially unaffected (matmul TFLOP/s median 0.1556 -> 0.1576); these host-bound figures (wall-clock shares, ms/token joins) may improve on re-measurement.

## The structural question, answered

**Yes — before this change, decode ran every matrix multiply through the general
tiled path.** Evidence from the tree as of main `f449ed2`:

- `dispatch_matmul` (`overlay/mlx/backend/omarchy/primitives.cpp`) had no
  single-row branch. Every call, including `m == 1`, dispatched
  `matmul.comp`: a 16x16 tiled kernel with `local_size 16x16`, shared-memory
  tile staging, and **two workgroup barriers per 16-element k step**. At
  `m == 1`, 15 of every 16 tile rows compute padding that is discarded.
- `QuantizedMatmul::eval_gpu` dispatched `qmm.comp`: **one thread per output
  element**, each thread walking the full K row serially with per-element
  shift-and-mask unpack. Coalescing analysis: with lanes assigned to
  consecutive output columns, the per-lane weight read stride is
  `words_per_row * 4` bytes (448 bytes at K=896, 4-bit). A 32-lane fetch
  touches 32 separate cache lines — the opposite of coalesced. This is the
  68.7x decode gap's kernel-side shape.

## What is now in the tree (uncommitted, tests green)

Two new dedicated kernels, gated on `matrix_m == 1` (the decode shape; every
decode GEMM is `m == 1` including batched attention `q @ k.T` and `p @ v`):

- `shaders/matmul_vec.comp` (`MatmulVecF32/F16/BF16`): one 32-lane subgroup
  slot per output column, lanes striding K. Consecutive lanes read
  consecutive addresses of the transposed weight row (the `x @ w.T` decode
  layout), which is the coalesced streaming read a memory-bound contraction
  wants. Runtime-bounded k loop: no unroll, low register pressure (the
  llama.cpp AGX lesson). Reduction: one workgroup-shared log2(32) tree at the
  end — five barrier rounds, no GLSL float subgroup arithmetic (Honeykrisp
  float reduce is software-lowered; hardware scan/reduce are integer-only,
  per Main's corrected brief). The synthetic slot split is plain workgroup
  arithmetic, correct on any device subgroup size.
- `shaders/qmm_vec.comp` (`QmmVecF32/F16/BF16`): the fused 4/8-bit kernel.
  Dequantization happens in registers inside the dot product — no
  dequantized intermediate pass, no extra dispatch. Lanes stride single k
  steps, so consecutive lanes read consecutive x elements and consecutive
  weight-row words; one weight row streams linearly per subgroup slot.
  Scale/bias per k-step group computed per lane (`k / group_size`), which
  keeps group boundary crossings correct mid-slot.
- `primitives.cpp`: `gemv_group_count` helper plus the two `matrix_m == 1`
  dispatch branches (Matmul and QuantizedMatmul). Identical push-constant
  transport; `use_c`, alpha/beta, batched z-unravel, and both b orientations
  behave exactly as in the tiled kernels. m > 1 falls through to the
  original general paths unchanged — nothing existing got slower.
- `compute.h` / `compute.cpp` / backend `CMakeLists.txt`: six new kernel
  registrations (append-only, coordinated with FuseDecodeChains).

## Verification (dev box, llvmpipe — no jwm1 numbers, see below)

- `glslangValidator -V --target-env vulkan1.3`: all 8 shader variants rc=0
  (matmul_vec and qmm_vec, each in f32/f16/bf16), plus the A/B variant below.
- `omarchy_matmul_family_tests`: **8 cases, 20,055 assertions, 0 failures**,
  including the two new value tests:
  - "matmul vector path matches general path across shapes and dtypes":
    shapes {1x1x1 (single-element vector), 33x17, 64x8, 130x7, 256x72}
    (k and n missing both the 8-column slot and 32-lane k step), batched
    2x(1,k) stacks, row-major b orientation, AddMM alpha/beta/c, f32/f16/bf16;
    every result checked against a host double reference AND against the
    general tiled path's row 0 on the same device.
  - "quantized matmul vector path matches general and host references":
    bits 4 and 8 x group sizes 32 and 64, k = 3 groups (lane crossings land
    mid-slot on purpose), n in {1,7,8,9,37}, all three dtypes, host double
    dequant reference with 16-bit-rounded parameters, plus the
    vector-vs-general same-device comparison.
  - 16-bit tolerances are one-output-ulp aware (bf16 rtol 1e-2, f16 4e-3);
    a wrong lane, index, or unpack misses by O(1) and still fails.
- Full battery: **not run** — the jwm1 window was cancelled (below); the dev
  box is llvmpipe and siblings hold parts of the shared tree. Whoever lands
  this runs the jwm1 battery per the handoff script.

## Why there are no tokens-per-sec numbers

Main stopped the jwm1 window on PerfProfileBaseline's measured profile: GPU
busy is 5.6-6.9 percent of wall, kernels are floor-bound (median 17.4 us,
which is the Vulkan per-dispatch floor), matmul tiling ranks last at ~0.3
percent of wall time. The decode gap is host-side: ~372 ms/token of join
wait and ~372 ms of host record cost across ~1390 joins, with descriptor
pool creation per dispatch and 1.14 dispatches per submission. Faster
matmul kernels cannot move a number that is 93 percent host wait. This is a
clean negative on tonight's lever ranking, not a failed implementation.

## Handoff for whoever benchmarks after the host work lands

1. The dev box has my slice with green value tests; jwm1 has an isolated
   worktree `~/src/mlx-omarchy-gemv` at `f449ed2` plus venv
   `.work/venv-gemv` (mlx-lm 0.31.3, setuptools 84, cmake) and the pinned
   mlx archive staged in `.work/`.
2. Copy the slice files to the worktree (tracked:
   `overlay/mlx/backend/omarchy/{primitives.cpp,compute.cpp,compute.h,CMakeLists.txt}`,
   `overlay/tests/omarchy/test_matmul_family.cpp`; new:
   `shaders/{matmul_vec,qmm_vec}.comp`), run
   `receipts/2026-09-02-gemv-decode/jwm1-build-battery.sh` (corrected:
   cmake source is the staged `.work/mlx`, NOT the repo root — that cost
   the first attempt a configure failure), then `bench-ab.sh` — same
   mlx-lm invocations as `receipts/2026-09-01-m1-same-chip-parity.md`
   (greedy, temp 0, seed 0, models under `~/models/*-mlx`, warm second run).
3. A/B legs must run alone on the box (PerfProfileBaseline's noise warning:
   the wall is host-dominated, so even ssh bursts show up in tok/s).
4. `MLX_OMARCHY_GPU_PROFILE=<path>` + `scripts/profile_analyze.py`
   (PerfProfileBaseline) will show `MatmulVec*`/`QmmVec*` rows per dispatch
   once the overlay is on the device — use it to confirm the path is taken
   and to quantify kernel-side gains after the host fix.

## The open A/B the assignment asked to measure, not assume

Shared-memory tree (shipped default) vs `subgroupAdd` float reduction:
the variant is written, validated (rc=0), and staged at
`receipts/2026-09-02-gemv-decode/matmul_vec_subgroup.comp`. Expectation from
the hardware facts: software-lowered float subgroup reduce likely loses to
the five-round shared tree; that is why the tree ships. The variant is
correct ONLY at real subgroup size 32 (one slot must equal one hardware
subgroup), so a winning variant needs a device subgroup-size gate in
dispatch before it can ship. One jwm1 build + bench leg answers it.

## Files

- `overlay/mlx/backend/omarchy/shaders/matmul_vec.comp` (new)
- `overlay/mlx/backend/omarchy/shaders/qmm_vec.comp` (new)
- `overlay/mlx/backend/omarchy/primitives.cpp` (+47 lines)
- `overlay/mlx/backend/omarchy/compute.h`, `compute.cpp`, `CMakeLists.txt`
  (kernel registrations)
- `overlay/tests/omarchy/test_matmul_family.cpp` (+269 lines, 2 test cases)
- `receipts/2026-09-02-gemv-decode/` (this receipt, both scripts, the A/B
  variant shader)

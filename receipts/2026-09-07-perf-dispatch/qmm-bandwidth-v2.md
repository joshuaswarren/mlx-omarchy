# DecodeQ4Vec: affine q4 decode GEMV bandwidth (M1, 2026-09-07)

Agent QmmBandwidth2. Source: overlay at HEAD 81e9f53a plus this work.
Device: Apple M1 (G13G B1), Honeykrisp Mesa 26.1.7, glslc 2026.3, jwm1.
Bench: `overlay/tests/omarchy/bench_qmm_bandwidth.cpp` (target
`omarchy_qmm_bandwidth_bench`, not registered with ctest). It quantizes
the five Qwen2.5-0.5B-4bit decode shapes, bit-checks every configuration
against the dequantized dense matmul, then times N = 100 back-to-back
`quantized_matmul` dispatches per eval, best of `MLX_OMARCHY_QMM_BENCH_TRIALS`
trials after one untimed warm-up pass, with the configurations from
`MLX_OMARCHY_QMM_BENCH_CONFIGS` interleaved trial by trial. Every run was
`flock gpu.lock taskset -c 4-7`. An 8x64 GEMV row reports the per-dispatch
floor (host graph + record + launch) on the same harness.

## What changed

* New shader `overlay/mlx/backend/omarchy/shaders/qmm_vec_q4.comp`
  (ComputeKernel `QmmVecQ4V2{F32,F16,BF16,SubgroupF32,SubgroupF16,SubgroupBF16}`,
  CMake targets `qmm_vec_q4_v2_*`), selected by `QuantizedMatmul::eval_gpu`
  for m == 1, transposed, affine, bits 4, group 64, k <= 4864, 16-byte
  aligned x and w offsets. `MLX_OMARCHY_QMM_VEC_Q4_V2=0` is the kill switch
  back to the previous DecodeQ4Word kernel, which stays built and gated as
  before. `MLX_OMARCHY_QMM_Q4_ROWS` pins rows per slot (1..2) for
  measurement; `params.shape[3]` carries it to the shader.
* Kernel shape: 256 threads = 8 slots of 32 lanes; each slot owns two
  consecutive output rows; each lane loads one uvec4 (16 B = 32 nibbles)
  per row per step so a slot streams 512 contiguous bytes per load and
  fetches one scale/bias pair per 16 weight bytes. The 32 activations a
  lane needs are read straight from the cached x buffer as uvec4 words
  into registers and shared across the slot's rows; there is no shared
  memory staging and no barrier (a transposed shared staging of x cost
  more than the weight stream on AGX: 32 two-byte shared loads per lane
  per step). shaderdb (f16 subgroup flavor): 382 instrs, 111 gprs, 108
  uniforms, 54 preamble instrs, 896 threads/core (old DecodeQ4Word: 114
  instrs, 27 gprs, 66 uniforms, 45 preamble).
* Accumulation order (documented in the shader header): per 32-element
  half group, f32 dot of x_j * q_j (products exact: nibbles are masked in
  place, `float(word & (0xF << 4j))` = q_j * 2^4j, and x_j is pre-scaled by
  2^-4j, both exact power-of-two rescalings), then
  `s * dot + b * sum(x)` (the Metal qmv `qdot` identity), accumulated per
  lane across its half groups, then one `subgroupAdd` (or the shared-memory
  tree on drivers without a 32-wide arithmetic subgroup, e.g. llvmpipe).
  Max abs error against the dense reference is identical to the old
  kernel on all five shapes (f16 output rounding dominates).

## Results (us per dispatch, same process, interleaved, best of 7)

`qmm-bandwidth-v2-final-m1.jsonl`, box idle (load 0.5):

| shape | bytes | old (DecodeQ4Word) | V2 (default) | V2 GB/s |
|---|---|---|---|---|
| floor 8x64 | 288 | 17.6 | 18.5 | - |
| k_proj 128x896 (57 KB) | 64,512 | 27.0 | 22.0 | 2.9 |
| q_proj 896x896 (401 KB) | 451,584 | 46.1 | 35.8 | 12.6 |
| gate 4864x896 (2.18 MB) | 2,451,456 | 111.4 | 81.3 | 30.2 |
| down 896x4864 (2.18 MB) | 2,451,456 | 75.9 | 54.6 | 44.9 |
| lm_head 151936x896 (68 MB) | 76,575,744 | 2248.4 | 1418.2 | 54.0 |

Earlier idle-box pairs (`taskset`, 2 alternating runs, not interleaved)
put V2 at 20.1-20.5 / 27.3-29.2 / 67-68 / 51.3 / 1591-1604 vs old
25.5 / 35.3 / 94.5 / 73.4 / 2293-2341; the gate/q_proj wall numbers move
+-15% run to run with GPU clock state and host load, lm_head and down are
stable to ~5%. The parent's end-to-end A/B (perfsnap2/3, IDs equal) shows
decode +25% / +21% / +15% (short / long / 1024 ctx) from this kernel alone.

Targets: 68 MB shape reaches 54-57 GB/s (>= 55 met in most runs). The
2.18 MB shapes reach 45 GB/s (down) and 30-36 GB/s (gate) wall-clock;
with the 18 us per-dispatch floor in series a 2.18 MB dispatch cannot
show 55 GB/s wall (that would need 27 us of GPU time = 92 GB/s), so the
gate/up shapes remain floor + latency bound. k_proj sits 3.5 us above the
18 us floor; <= 15 us is below the harness floor itself.

## Experiments that did not ship (raw logs alongside this file)

* Shared-memory transposed x staging (first V2): slower than old on
  four of five shapes; 16-bit shared loads dominate. Removed.
* MAX_ROWS = 4 rows per slot in registers: 135 gprs -> 768 threads/core;
  lm_head 1442 but q_proj/k_proj/gate worse (maxrows4 matrix).
* Sequential row chunks per slot (`experiment-outer-chunk.comp`, rows up
  to 8 in chunks of 2, 97 gprs): lm_head 1332-1370 us (56-57 GB/s), gate
  ~77 us with rows = 8, but each extra chunk adds a ~2.5 us dependent round
  trip so n = 128/896 shapes lose 3-10 us; needs an n-based rows policy
  and an end-to-end A/B before it can replace the shipped kernel.
* Software prefetch of the next (chunk, step) pair: compiler keeps both
  sets live (133-159 gprs, 640-768 threads/core); slower everywhere.
* 128- and 64-thread workgroups: one run showed 60/21 us for gate/q_proj,
  the confirmation run did not reproduce it (84/31); treated as noise.
* AGX_MESA_DEBUG=shaderdb is the only reliable GPU-side signal found; the
  compile-time GPU timestamp profiler reports 42 us for the 68 MB dispatch
  on Honeykrisp and is unusable for kernel timing.

## Tests

* M1 (jwm1 tree, V2 default and `MLX_OMARCHY_QMM_VEC_Q4_V2=0`):
  omarchy_quantized_batch_layout_tests 1488/1488;
  omarchy_primitive_tests `quantized matmul*` + `quantized decode gemv*`
  8110/8110 assertions.
* llvmpipe (this host, `MLX_OMARCHY_ALLOW_NON_APPLE=1`, tree-reduction
  flavor): the same two suites pass (1488/1488, 8110/8110) with the default
  and with the kill switch; bench correctness gate ok on the four small
  shapes (68 MB skipped, minutes on llvmpipe).
* New test `quantized decode gemv covers multi-step rows, batched weights,
  and unaligned x` (test_primitives.cpp): batched weights [2, 37, 1088/8],
  k = 1088 (two lane steps), n = 37 (partial slot), plus a sliced x at a
  4-byte offset that must fall back to DecodeQ4Word; both against the
  host double-precision reference.
* Pre-existing, not mine: `quantize and dequantize pin named errors outside
  the affine gate` fails on any tree prepared from HEAD 81e9f53a because
  the quantize-errors patch now accepts bf16 while the test still expects
  the error (reported to Main).

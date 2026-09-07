# Prior-art survey: q4 decode GEMV, prefill GEMM, dispatch overhead on AGX/Honeykrisp

Status: COMPLETE (sections 1-6, ranked list, directly-actionable map). Author: PriorArtSurvey subagent, 2026-09-07. Read-only research; no source edits, no builds, no jwm1 runs.

Our baseline (from parent): Qwen2.5-0.5B-4bit on M1 via Honeykrisp. Decode 97.5/88.9/80.9 tok/s vs macOS MLX 150.8/146.6/140.3; prefill 103/252/289 vs 294/1213/1838. Measured: q4 GEMV ~48 GB/s on lm_head, 32-36 GB/s on the 2.18 MB gate/up/down shapes, ~28 us for the 401 KB shapes against an ~18 us dispatch floor; ~537 dispatches per decode token; per-dispatch GPU floor 3-5 us; prefill GEMM ~12% of fp32 peak.

All sources below were fetched on 2026-09-07 from `main`/`master` unless a commit is named. Line numbers refer to those fetches (copies under `/tmp/prior-art/` on this host).

## 1. Apple MLX (ml-explore/mlx, Metal backend) - what the reference implementation does

### 1.1 Affine q4 GEMV: `qmv_fast_impl` (mlx/backend/metal/kernels/quantized.h)

Source: https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/quantized.h, `qmv_fast_impl` (lines ~653-715 of the fetched file), `qdot` (lines ~197-300), `load_vector` (lines ~29-120).

Facts (read from source):
- Threadgroup = 2 simdgroups x 32 lanes = 64 threads (`num_simdgroups = 2`, dispatch `group_dims(bk=32, 2, 1)` in quantized.cpp `qmv()` line ~497).
- Each simdgroup owns `results_per_simdgroup = 4` output rows; grid = `(M, ceil(N/8), B)` threadgroups, so 8 rows per threadgroup, N/8 threadgroups. N=896 -> 112 TGs; N=4864 (gate/up) -> 608 TGs; N=151936 (lm_head) -> 18992 TGs.
- 4-bit: `packs_per_thread = 2`, `pack_factor = 8` -> `values_per_thread = 16` elements = 8 weight bytes per lane per row per K-step; `block_size = 16*32 = 512` elements per simdgroup K-step. The 4 rows' 8-byte loads are issued in the row loop before any reduction (4 loads x 8 B = 32 B in flight per lane).
- x is loaded once per K-step (`load_vector`), pre-scaled by 1, 1/16, 1/256, 1/4096, and reused for 4 rows; the nibble dot is computed with masks, not shifts (`qdot`, `bits == 4` branch uses `uint16_t` words and masks 0x000f/0x00f0/0x0f00/0xf000).
- Affine identity is exactly ours: `return scale * accum + sum * bias;` (qdot last line) with `sum` = plain sum of the 16 x values.
- One scale+bias read per lane per row per K-step (`sl[0]`, `bl[0]`; scale pointer advances by `block_size/group_size` per step).
- Final reduce: `simd_sum` per row, lane 0 writes (no shared memory, no barrier).
- `qmv_quad_impl` exists for K == 64/128 only (quantized.cpp line ~1755): 4-lane quads own a row, 8 rows per quad-set.
- `qmv_wide` (multiple x vectors per weight stream) is gated to gen-15+ for affine (`use_qmv_wide`, quantized.cpp line ~538), so on M1 (gen 13) MLX uses plain `qmv_fast` for every M below the batch limit.
- Routing (quantized.cpp `get_qmv_batch_limit`, lines 85-140; `QuantizedMatmul::eval_gpu` lines 1799-1875): on M1 (arch_gen 13, `default` size branch) the limit is 14 for D,O <= 2048 (10 for <= 4096, 6 above), so `M < 14` goes to `qmv` with grid.x = M (the GEMV kernel is re-dispatched per token row, not a GEMM), and only `M >= 14` goes to `qmm_splitk`.

Comparison with our `qmm_vec_q4.comp` (overlay/mlx/backend/omarchy/shaders/qmm_vec_q4.comp): we use 256-thread workgroups = 8 slots x 32 lanes, each slot owns 1-2 rows (`MAX_ROWS = 2`), each lane streams one uvec4 (16 B) per row per step, so 32 B in flight per lane at rows=2 (same as MLX's 4 x 8 B) but x is reused across only 2 rows instead of 4, and only 1 row at rows=1. MLX's 8 rows/TG with 64 threads gives 8x more threadgroups than our 256-thread/2-row layout for the same N.

### 1.2 How MLX Metal avoids paying a barrier per dispatch (mlx/backend/metal/device.cpp)

Source: https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/device.cpp lines ~341-426 and ~511-626.
- The compute encoder is created with `MTL::DispatchTypeConcurrent` (line ~580).
- A `memoryBarrier(MTL::BarrierScopeBuffers)` is inserted only when the next kernel's inputs intersect the previous epoch's outputs or vice versa (`set_input_array` / `set_output_array` track `prev_outputs_` / `next_outputs_`, `maybeInsertBarrier` line ~393). Independent dispatches (q/k/v projections, gate and up) run back to back with no barrier.
- Command buffers are committed every 40 ops on base/pro M-series ('g', line ~608-610) or 40 MB; so a decode token of ~500 ops spans ~12 command buffers, but the GPU never sees a per-dispatch flush.

Why it matters on Honeykrisp: `hk_dispatch_with_usc_launch` (Mesa src/asahi/vulkan/hk_cmd_dispatch.c lines 56-69) emits `agx_cdm_launch` then unconditionally `hk_cdm_cache_flush` -> `agx_cdm_barrier` after EVERY dispatch. There is no dependency tracking in the driver; the flush is the driver's only coherence mechanism. `hk_CmdPipelineBarrier2` (hk_cmd_buffer.c lines 327-345, comment "The big hammer ... XXX: perf") just ends the current compute control stream; adjacent CDM streams are re-merged with a jump at `vkEndCommandBuffer` (`merge_control_streams`, hk_cmd_buffer.c lines 271-295). So on Honeykrisp the cost model is: every vkCmdDispatch = launch + CDM barrier; vkCmdPipelineBarrier adds only CPU-side cost. The only app-side lever for the per-dispatch floor is fewer dispatches (fusion), not smarter barriers.

### 1.3 Prefill GEMM in MLX on M1: `qmm_splitk` + `steel` tiles

Source: quantized.cpp `qmm_splitk` lines 1120-1217, `qmm` lines 1024-1118.
- Non-NAX (M1) quantized GEMM uses `bm = 32, bn = 32`, 128 threads (`group_dims(32, 2, 2)`, i.e. 2x2 simdgroups each owning a 16x16 sub-tile through `simdgroup_matrix`), and `split_k = max(1, 512 / (n_tiles*m_tiles))` chosen "to target ~512 threadgroups", limited so each K partition is a whole number of 32-wide K tiles and whole quant groups. Partials go to an fp intermediate `[split_k, M, N]` and are reduced with a separate strided-sum reduction (lines 1210-1216). No float atomics needed.
- For Qwen2.5-0.5B prefill of 128 tokens, N=896: n_tiles=28, m_tiles=4 -> 112 TGs -> split_k = 4 (K=896 / 64 = 14 groups, 4 | 14? no -> loop decrements until K % (split_k*64) == 0 -> split_k = 2). So MLX still splits K even at 128 tokens to get >= 224 threadgroups on an 8-core GPU.
- Metal's dense `steel` GEMM uses `simdgroup_multiply_accumulate` (hardware `simd_matrix_fmadd` on G13; see section 5). Vulkan on Honeykrisp has no cooperative-matrix extension, so this is the one MLX technique we cannot copy; the scalar-tile findings in section 3 and 5 are the substitute.

### 1.4 Decode attention: `sdpa_vector` (mlx/backend/metal/kernels/sdpa_vector.h, scaled_dot_product_attention.cpp)

Source: sdpa_vector.h lines 15-177; scaled_dot_product_attention.cpp lines 364-452 and 878-884.
- On M1 (arch char not 'd'/'s') single-pass `sdpa_vector` is used unless `k_len >= 4096` with GQA; the 2-pass split-K variant is for Max/Ultra or long contexts.
- One threadgroup of 1024 threads = 32 simdgroups per (batch*head, query). `BN = 32` simdgroups stride over keys (`for (i = simd_gid; i < N; i += BN)`), `BD = 32` lanes split the head dim (`qk_per_thread = D/32`, so D=64 -> 2 elements per lane). Each key's score is a `simd_sum` over 32 lanes; online softmax state (max, sum, o[v_per_thread]) lives in registers per simdgroup; the 32 simdgroups combine once at the end through one barrier for max/sum plus two per output element per lane (lines 149-169: 5 barriers for D=64, 9 for D=128).
- No shared-memory score arrays, no per-key barriers; the only barriers are in the final combine.

Comparison with our `sdpa_decode.comp`: one workgroup per (batch, head, query, 256-key block), 256 lanes each own one key, block max / exp-sum via shared-memory trees ("five barriers per workgroup" in the subgroup variant, eighteen otherwise), then regroup to accumulate probs @ v. MLX's layout has each simdgroup consume keys serially with register-resident online softmax and only the final-combine barriers (5 for D=64).

### 1.5 RoPE / RMSNorm in MLX

Source: https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/rms_norm.metal and rope.metal (fetched, 427 and 229 lines).
- `rms_single_row` (rms_norm.metal lines 12-80): one threadgroup per row, `N_READS` contiguous elements per thread held in `thread_x[]`, one `simd_sum` + threadgroup reduce, `metal::precise::rsqrt(acc/axis_size + eps)` once (line 61), then the output pass from the cached registers (lines 66-79, no re-read); `rms_looped` (line 84) handles rows longer than threadgroup*N_READS.
- `rope_single` / `rope`: each thread handles a pair (or 4 with `N_READS`) computing `cos/sin` from `metal::fast::` functions using `theta = pos * base^(-2i/D)` via `exp2`/`log2`; no precomputed cos/sin table is read from memory in the default path (the `freqs` variant reads a per-dim freq array). So RoPE is one elementwise dispatch with no extra buffers.
- mlx-lm's Qwen2 (mlx_lm/models/qwen2.py) calls `mx.fast.rope`, `mx.fast.rms_norm`, `mx.fast.scaled_dot_product_attention` (fused primitives), so per layer the decode graph is: rms_norm, q/k/v qmv (3), rope (2), sdpa (1), o_proj qmv, add, rms_norm, gate/up qmv (2), silu*mul (1 fused via compile or 2), down qmv, add. That is ~14-16 dispatches per layer -> 24 layers ~350-400 + lm_head + sampling. Our 537/token is in the same order; the gap is the per-dispatch cost, not the count.

## 2. mlx-lm (ml-explore/mlx-lm) - what changes the dispatch count per token

Source: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/generate.py (2137 lines), mlx_lm/models/cache.py (1775 lines), mlx_lm/models/qwen2.py (221 lines). Facts:
- `generate_step` pre-allocates the KV cache and updates it in place (`KVCache.update_and_fetch`, cache.py: grows by `step = 256` and slices views; no per-token concat copies).
- The per-token step is not `mx.compile`d; generation relies on the lazy graph plus one token of lookahead: `generate_step` calls `mx.async_eval(next_y, next_logprobs)` for token t+1 before `mx.eval(y)` of token t (generate.py lines 454-463), so CPU encoding of token t+1 overlaps GPU execution of token t and encode cost never sits on the GPU critical path. Prompt prefill is chunked by `prefill_step_size` (2048 default, line 312) with `mx.eval` of the cache state per chunk (lines 427-439).
- Quantized KV cache (`QuantizedKVCache`, cache.py) is opt-in (`kv_bits`), and increases dispatches (quantize per token); not used in the baseline numbers.
- `prompt_cache` avoids re-prefill across turns; not relevant to single-shot decode tok/s.

Takeaway: mlx-lm does not reduce dispatch count below ~14/layer; parity comes from MLX's per-dispatch cost on Metal being far below Honeykrisp's 3-5 us floor (concurrent dispatch, no forced flush; exact Metal figure not measured here [INFERENCE]).

## 3. llama.cpp Vulkan backend (ggml/src/ggml-vulkan) - what a tuned Vulkan backend does on AGX

### 3.1 Apple / Honeykrisp specific device tuning (ggml-vulkan.cpp)

Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp (fetched 2026-09-07, 20859 lines).
- Lines 7259-7266 and 7286-7297: for `VK_VENDOR_ID_APPLE` and for `vk::DriverId::eMesaHoneykrisp`, only the MEDIUM matmul tile is enabled (`mul_mat_l = false, mul_mat_m = true, mul_mat_s = false`). PR https://github.com/ggml-org/llama.cpp/pull/24306 (merged 2026-06): llama-2-7B Q4_0 on M1 Ultra pp512 207.6 -> 259.0 t/s (+25%) from the tile change alone; tg128 unchanged.
- Lines 2971-3048 + 3149-3163: on Honeykrisp the `mul_mm.comp` BK loop's SPIR-V `Unroll` hint is rewritten to `DontUnroll` at pipeline creation (`ggml_vk_roll_bk_loop`). PR https://github.com/ggml-org/llama.cpp/pull/24663: "possibly because the unrolled loop body is otherwise too large to fit in Apple's instruction cache (12 KB)"; M1 Ultra pp512 306.8 -> 375.6 (+22.4%), tg128 -2%.
- Medium scalar (no coopmat) warptile, lines 4474-4480: `m_warptile = { 128 threads, BM=64, BN=64, BK=16, WM=mm_warp_8(=32 on AGX), WN=32, WMITER=2, TM=4, TN=2, TK=1, warp=32 }` for f16/f32 A and `m_warptile_mmq = { 128, 64, 64, BK=32, ... same }` for quantized A; `m_wg_denoms = {64, 64, 1}`. Per thread: 8 x 4 = 32 fp32 accumulators (`ACC_TYPEV2 sums[WMITER*TM*WNITER*TN/2]`, mul_mm.comp line 279), A strip of 8 and B strip of 4 in registers per k-step; 128 threads = 4 subgroups per 64x64 tile; shared tiles for A and B. Honeykrisp's `maxComputeSharedMemorySize` = 32 KB (hk_physical_device.c line 749; `HK_MAX_SHARED_SIZE = 32*1024` in hk_private.h line 35) is the reason the LARGE 128x128 tile is disabled (its shared footprint exceeds it, ggml-vulkan.cpp 4530-4590 checks).
- Honeykrisp reports `integerDotProduct4x8BitPackedSignedAccelerated = false` (hk_physical_device.c lines 911-913) so llama.cpp's int8 `mul_mat_vecq`/`mul_mmq` (q8_1 activations, `dotPacked4x8`) paths are NOT used on AGX ("int dot: 0" in the llama-bench header on Asahi, issue #10982). fp16 storage and arithmetic are on (`shaderFloat16 = true`, `storageBuffer16BitAccess = true`), subgroup size is exactly 32 (`minSubgroupSize = maxSubgroupSize = 32`, lines 901-902), subgroup arithmetic/shuffle/ballot/quad/vote supported (lines 825-830).

### 3.2 Decode GEMV: `mul_mat_vec.comp` + `mul_mat_vec_base.glsl`

Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp and mul_mat_vec_base.glsl; pipeline creation ggml-vulkan.cpp lines 5416-5530; workgroup-size heuristic lines 8107-8128.
- Default (non-AMD/Intel) rows per workgroup for q4_0: `rm_stdq = 1` -> `{2*rm_stdq, 1, 1}` = 2 rows per workgroup (line 5508: `mul_mat_vec_q4_0_f16_f32 ... {wg_size_subgroup, 2*rm_stdq, i+1}`); k-quants `rm_kq = 2`. On NVIDIA/Intel a `DMMV_WG_SIZE_LARGE` variant with 4x subgroup workgroup is picked when `m < 4096 && k >= 1024`; for everyone else (including AGX) the workgroup is exactly one subgroup (32 threads) with `SHADER_REDUCTION_MODE_SUBGROUP` (subgroupAdd, no shared memory; `USE_SUBGROUP_ADD_NO_SHMEM` branch of `reduce_result`).
- `K_PER_ITER = 8` for quantized A (mul_mat_vec.comp line 12): each lane handles 8 elements of a row per iteration = 4 bytes of q4_0 per row per iteration, with `[[unroll]] 4` then 2 then 1 manual unrolling of the K loop (lines 165-236) so 4 iterations x 2 rows x 4 B = 32 B of weight loads in flight per lane.
- `NUM_COLS` up to `mul_mat_vec_max_cols = 8` (line 389): when 2..8 activation columns are present (small-batch prefill, speculative decode), the same weight stream serves up to 8 columns (`temp[NUM_COLS][NUM_ROWS]`), which is the Vulkan analog of MLX `qmv_wide`.
- B (activation) is read as `vec4` (`data_b_v4`) with a fixed 4-float stride pattern per quant block (lines 56-64).
- Fusion flags (`MAT_VEC_FUSION_FLAGS_BIAS0/1`, `SCALE0/1`) fold a following bias add or scale into the GEMV epilogue (mul_mat_vec_base.glsl lines 100-130), removing one dispatch per fused op.

Measured decode data on Asahi from issue #10982 (M1 Ultra, llama-2-7B Q4_0, 2026-06-06): Vulkan/Honeykrisp tg128 32.6 t/s vs MoltenVK 33.2 vs Metal 103.3. On a 7-core M1 Air, Llama-3.2-3B q4_0 tg128 "~20 t/s ... memory-bound (~68 GB/s)" (2026-06-18 comment). So the best public Vulkan-on-AGX decode is ~3x slower than Metal on the same weights; the same author states llama.cpp's own Metal kernel hits ~100 GB/s-class on M1 Ultra decode. This means nobody has published a Vulkan q4 GEMV that reaches Metal bandwidth on AGX; our 48 GB/s on lm_head is already above the 32-36 GB/s regime llama.cpp Vulkan reports (68 GB/s "memory-bound" claim in that thread is stated, not measured per kernel).

### 3.3 Prefill GEMM without cooperative matrices: `mul_mm.comp`

Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp (475 lines) + mul_mm_funcs.glsl (688 lines).
- Register-blocked outer product: each thread owns `TM*WMITER` rows x `TN*WNITER` cols of fp32 accumulators (`WNITER = WM*WN/(WARP*TM*TN*WMITER)`, line 173), loads a column strip of A (TM per WMITER) and a row strip of B (TN per WNITER) from shared memory, then FMAs. Shared tiles are stored as `FLOAT_TYPEV2` (f16 pairs, lines 132-133) and the inner loop steps `BK_STEP` k values at a time: 4 (f16vec4 register cache) for f16/f32 A, 2 (f16vec2) for quantized A (lines 115-121, 281-286, 324-351). With the Apple medium tile (BM=BN=64, 128 threads, TM=4, TN=2, WMITER=2, WNITER=2): 32 accumulators per thread; per k-step 8 A + 4 B shared loads feed 32 `dot_product` calls = 64 (quant) or 128 (f16) FMAs -> ~5-10 FMAs per shared load. The K-tile loop loads a 64xBK A tile and 64xBK B tile (quantized A is dequantized into shared as f16 at load time, `mul_mm_funcs.glsl` `load_a_to_shmem`) with one barrier per BK step (16 for f16, 32 for quant).
- Loads from global into shared use `vec4`/`f16vec4` loads and `[[unroll]]`; the BK inner loop is the one rolled on Honeykrisp (section 3.1).

### 3.4 Flash attention shaders (`flash_attn.comp`, no-coopmat variant)

Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp (760 lines), flash_attn_base.glsl (223 lines).
- The scalar FA path (`flash_attn.comp`) is selected when no coopmat; it uses `Br` query rows x `Bc` key columns per workgroup with `split_k` across the KV length for decode (`flash_attn_split_k_reduce.comp` merges partials, same structure as our PARTIAL + combine path). Not AGX-specific; included for completeness. llama.cpp's Asahi benchmarks in #10982 report "flash-attn ... dead ends" (no gain) for prefill on M1, so no evidence that the FA shader shape matters for our decode numbers.

## 4. llama.cpp Metal (ggml/src/ggml-metal/kernels/mul_mv.metal) - what AGX likes for q4 GEMV

Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/kernels/mul_mv.metal (3225 lines), ggml-metal-impl.h lines 30-31.
- `N_R0_Q4_0 = 4` rows per simdgroup, `N_SG_Q4_0 = 2` simdgroups per threadgroup -> 8 rows per 64-thread threadgroup, identical to MLX `qmv_fast`. `mul_vec_q_n_f32_impl` (lines 217-303): `NQ = 16` blocks per simdgroup K-step; lane `ix = tiisg/2` picks the block, `il = (tiisg%2)*8` picks the half block, so each lane reads 8 bytes (`uint16_t qs[4]`) of one q4_0 block per row per step and 16 floats of y (pre-scaled by 1, 1/256, 1/16, 1/4096 in `yl[]`, lines 274-282). Per step per lane: 4 rows x 8 B = 32 B of weight loads in flight, then `block_q_n_dot_y` (lines 85-100) does the masked dot with `d * (sumy * -8 + acc...)`: same masked-nibble, scale-outside trick as MLX and as our kernel.
- Epilogue: `simd_sum` per row, lane 0 writes (lines 296-302). No shared memory.
- `ggml-metal-ops.cpp` line 2420: for the `mul_mv_ext` (small-batch) kernels `nsg = 2` is hard-coded with the comment that higher nsg values showed tail effects ("the work grid is not evenly divisible for different nsg values").
- Dispatch: `ggml_metal_encoder_dispatch_threadgroups(enc, ceil(ne11/nr1), ceil(ne01/nr0), ne12*ne13, 32, nsg, 1)` (line 2528) i.e. grid.y = N/8 threadgroups of 64 threads.

Ground-truth conclusion for AGX q4 GEMV (both Apple's own MLX and ggml's tuned Metal kernel agree): 64-thread threadgroups, 2 simdgroups, 4 rows per simdgroup, 8 contiguous weight bytes per lane per row per K-step (256 B per row per simdgroup step = 4 cache lines... 512 elements per step), x pre-scaled once and reused across 4 rows, masked nibble dot, subgroup reduce only. Our kernel differs in: 256-thread workgroups, 1-2 rows per slot, 16 B per lane per row per step, and shared-memory reduce in the non-subgroup flavor.


## 5. Mesa Honeykrisp / AGX compiler and driver behaviour relevant to compute

Sources: src/asahi/vulkan/hk_cmd_dispatch.c, hk_cmd_buffer.c, hk_cmd_buffer.h, hk_queue.c, hk_device.c, hk_device.h, hk_physical_device.c, hk_private.h (all fetched from https://gitlab.freedesktop.org/mesa/mesa/-/raw/main/...), src/asahi/genxml/cmdbuf.xml; plus dougallj/applegpu docs (https://dougallj.github.io/applegpu/docs.html) and philipturner/metal-benchmarks README.

5.1 Dispatch cost model (driver, verified in source)
- `hk_dispatch_with_usc_launch` (hk_cmd_dispatch.c 56-69): `agx_cdm_launch(...)` then `hk_cdm_cache_flush(dev, cs)` -> `agx_cdm_barrier` on every dispatch, unconditionally; `cs->stats.flushes++`. The CDM Barrier packet (cmdbuf.xml lines 1056-1082) is a 4-byte word of mostly undocumented bits plus `USC cache inval`. This is the 3-5 us floor the parent measured; nothing in the app can remove it per dispatch.
- `hk_CmdPipelineBarrier2` (hk_cmd_buffer.c 327-345): "The big hammer ... XXX: perf": it ends the current compute control stream (`hk_cmd_buffer_end_compute`) so the next dispatch allocates a fresh 64 KB control-stream chunk (`hk_cmd_buffer_get_cs_general`, hk_cmd_buffer.h 590-637); at `vkEndCommandBuffer` adjacent CDM streams are merged with a jump (`merge_control_streams` 271-295, `hk_cs_merge_cdm` hk_cmd_buffer.h 424-445). So in-command-buffer barriers cost CPU time and a control-stream jump, not extra GPU flushes. `HK_PERFTEST=nobarrier` (hk_device.c 41-48, 327) skips them entirely (test knob only; unsafe in general).
- Submission: `max_commands_per_submit` is 1 unless `HK_PERFTEST=batch` (hk_queue.c 248-257, with the comment that batching caused rare CTS flakes). Each `vkQueueSubmit` of a command buffer becomes one `drm_asahi_cmd_compute` (hk_queue.c 829-840). No hardware preemption on Asahi (issue #10982 2026-06-06 comment), so long compute batches starve the display; llama.cpp lowered its graph-submit threshold for this.
- Independent evidence that consecutive dispatches never overlap on AGX under Honeykrisp: asahi-agx-llm-perf LAB_NOTEBOOK.md "cont9s" (lines 2628-2634): a no-barrier interleave of a BW-bound dequant with an ALU-bound GEMM ran in exactly the same 16.57 ms as the barriered version: "AGX serializes consecutive compute dispatches on its single queue even without an explicit barrier." So there is no async-compute overlap to harvest on this stack; fewer dispatches is the only route.

5.2 Device limits that shape kernels (hk_physical_device.c)
- `subgroupSize = 32`, `minSubgroupSize = maxSubgroupSize = 32` (822, 901-902); subgroup ops: basic, ballot, vote, quad, shuffle, arithmetic (825-830).
- `maxComputeSharedMemorySize = HK_MAX_SHARED_SIZE = 32 KB` (749; hk_private.h 35); `maxComputeWorkGroupInvocations = 1024` (751).
- `shaderFloat16 = true`, `shaderInt8 = true`, `shaderInt16 = true`, `storageBuffer16BitAccess = true`, `shaderInt64 = true` (282-326); `shaderFloat64 = false`; no buffer int64 atomics; `shaderIntegerDotProduct = true` but all `integerDotProduct4x8BitPacked*Accelerated = false` (911-913), so int8 dot paths gain nothing.
- No `VK_KHR_cooperative_matrix` on stock Honeykrisp. The G13 does have an 8x8x8 `simd_matrix_fmadd16/32` instruction (dougallj/applegpu `SimdMatrixFMadd32InstructionDesc`; asahi-agx-llm-perf README, GPU-verified ~1.76 TFLOPS fp32 on a 7-core M1), but Mesa main does not emit it; the out-of-tree lowering exists only in that showcase repo (`agx_nir_lower_simdmat.c`). Out of scope for a Vulkan-level backend today.

5.3 Compiler facts from the AGX ISA (dougallj/applegpu docs, lines 76-144, 807-812)
- 32 threads per SIMD-group; up to 128 32-bit GPRs per thread, addressable as 16-bit halves (`r0l/r0h`); "Using fewer registers (e.g. by using 16-bit types instead of 32-bit types) allows more SIMD-groups to fit in the physical register file (higher occupancy)".
- 256 32-bit uniform registers `u0..u255` hold per-dispatch constants (push constants land here via the preamble/`uniform_store`).
- `device_load`: "up to four aligned values, each up to 32-bits" per lane per instruction (16 B), result usable after an explicit `wait`; the register cache has `cache`/`discard` hints per operand.
- philipturner/metal-benchmarks README (fetched): M1 GPU core = 128 ALUs, 4 schedulers each issuing one instruction from one simd per cycle; "ALU utilization maxes out at 24 simds/core", ~208 KB register file, 12 KB instruction cache, 8 KB L1 data cache, 128 B global cache line, L2 768 KB on M1 8-core. Throughput table: F32 FMA 128/cycle/core (256 ops), I32 bitwise 128/cycle but I32 shifts 32/cycle and I32 multiply 32/cycle; F32 exp2 32/cycle, rsqrt 16/cycle; back-to-back dependent FP32 FMA at low occupancy costs 6-11 cycles vs ~2 cycles with ILP 4.
- Consequences already visible in the asahi-agx-llm-perf ISA dumps (LAB_NOTEBOOK.md lines 31-37 and 2299-2306): the Honeykrisp matmul used 183 GPRs (occupancy capped 576/1024 threads), ~20% of the inner loop was `ldimm + iadd` address arithmetic per shared load, and the measured bottleneck of the scalar GEMM was the register cache (1.56 cycles/op), "NOT occupancy". Apple's own Metal GEMM uses a small 16-accumulator (4x4) per-thread tile with heavy `.cache`/`.discard` hints (lines 2302-2306).
- Mesa-side wins reported there (not upstream as of the fetch): post-RA ILP scheduler +16% pp512, shared-load vectorization +3.4%, the "bit 47" long-load encoding on `lload/lstore` +13% (xingjianll fork https://gitlab.freedesktop.org/xingjianll/mesa/-/tree/agx-llm-perf). These are driver changes; they matter to us only as an explanation of why scalar GEMM sits at ~12% of peak and as a reason to keep inner loops short and address math simple.

## 6. Other Vulkan / Apple-GPU inference work

- MLC-LLM/TVM, ncnn, MNN on Apple + Vulkan: no published Asahi/Honeykrisp numbers found in this sweep (searches on 2026-09-07 returned only MoltenVK-on-macOS material). Not usable as evidence; omitted.
- MoltenVK data point (issue #10982, 2026-06-06 table): Vulkan-via-MoltenVK on macOS M1 Ultra tg128 33.2 t/s vs Metal 103.3 on the same llama.cpp shaders, i.e. the ggml Vulkan GEMV shape itself (32-thread workgroups, 2 rows, 4 B/lane/row/iter) loses ~3x to the Metal kernel shape (64 threads, 8 rows, 8 B/lane/row/iter) even with Apple's compiler. That isolates the shader shape as a first-order factor independent of Mesa codegen.
- llama.cpp's own Asahi decode tuning (LAB_NOTEBOOK.md lines 14-29): Q4_K_M -> Q4_0 raised tg128 15.3 -> 19+ ("simpler dequant suits AGX"); `rm_stdq = 2` (4 rows per 32-thread workgroup instead of 2) 18.5 -> 20.1; `rm_kq` sweeps for k-quants flat; Q8_0 slower (memory-bound). Direction agrees with MLX/ggml-Metal: more rows per weight-stream pass, simplest possible unpack.

## Ranked techniques (highest expected effect on our numbers first)

Estimates below use the parent's measurements: 10.26 ms/token at 97.5 tok/s; 537 dispatches x 3-5 us = 1.6-2.7 ms of pure launch floor per token; per-layer q4 bytes: q/o 401 KB each, k/v 57 KB each, gate/up/down 2.18 MB each (24 layers) + lm_head 68 MB; ~68 GB/s DRAM. All effect numbers are predictions [INFERENCE], not measurements.

1. Cut decode dispatch count by fusing along the lines every reference stack already fuses (target: 537 -> ~300).
   - Source patterns: ggml Vulkan GEMV epilogue fusion flags (mul_mat_vec_base.glsl lines 100-130: bias/scale folded into the reduce); ggml Metal norm+mul(+add) fusion (ggml-metal-ops.cpp lines 4115-4175) and add-chains up to 8 fused (lines 3827-3878); MLX's mlx-lm graph uses fused `fast.rope`/`fast.rms_norm`/`fast.sdpa` so its per-layer count is already ~14.
   - Concrete fusions for Qwen2 decode: (a) q/k/v in one GEMV dispatch over a row-concatenated weight (1152 rows; -2/layer); (b) gate+up+SiLU*mul in one dispatch, interleaving gate row r and up row r in the same slot so the epilogue writes silu(g)*u (-2 or -3/layer); (c) residual add in the GEMV epilogue for o_proj and down_proj (ggml `BIAS0` pattern; -2/layer); (d) rope q and rope k in one dispatch (-1/layer); (e) rms_norm folded into the following GEMV prologue: every workgroup already reads the whole 896-element x, so a subgroupAdd of x^2 plus one rsqrt per workgroup replaces the norm dispatch (-2/layer; own derivation, cost = 896 extra MACs per workgroup).
   - Why on AGX/Honeykrisp: section 5.1 - every dispatch pays a CDM barrier and dispatches never overlap; there is no cheaper barrier to use.
   - Expected: ~220 fewer dispatches x 3-5 us = 0.7-1.1 ms/token -> 97.5 -> ~105-115 tok/s before any kernel gets faster.

2. Reshape the q4 GEMV to the AGX consensus (MLX `qmv_fast` == ggml Metal `mul_mv_q4_0`): 64-thread workgroups, 2 subgroups, 4 rows per subgroup, 8 B per lane per row per K-step, x pre-scaled once and reused over 4 rows, subgroupAdd epilogue, grid = N/8.
   - Sources: quantized.h `qmv_fast_impl`; mul_mv.metal `mul_vec_q_n_f32_impl` + `N_R0_Q4_0 4 / N_SG_Q4_0 2`; corroborated by llama.cpp Vulkan's `rm_stdq=2` decode win on Asahi (LAB_NOTEBOOK.md line 23) and the MoltenVK-vs-Metal 3x gap (section 6).
   - Why: 4 rows per lane doubles x reuse and halves per-row load instructions relative to our 2-row slots; 64-thread workgroups give 4x more workgroups for the same N (better spread over 8 cores at N=896: 112 vs 56 workgroups) and keep per-thread registers low (higher occupancy per section 5.3). 8 B loads keep 4 independent loads in flight per lane (32 B) with one `wait`, the same in-flight depth we get from 2 x 16 B.
   - Corroboration from QmmBandwidth2 (hub reply, 2026-09-07): an 8-rows-per-slot variant of our kernel reached 56-57 GB/s on lm_head but lost 3-10 us on the n=128/896 shapes from the extra dependent round trips, so rows-per-subgroup is the lever and the row count must be chosen per n (MLX's fixed 4 rows x 2 subgroups is the middle of that range).
   - Expected: the 2.18 MB shapes are the ones at 32-36 GB/s; if they reach the lm_head figure (48 GB/s) the 72 gate/up/down dispatches per token drop from ~4.6 ms to ~3.3 ms -> +1.3 ms/token -> ~113 tok/s alone, ~125-130 combined with item 1. The 401 KB shapes are floor-bound (28 us vs 18 us floor) and gain little from shape changes; they gain from item 1.

3. Small-batch prefill (M < 14) through the GEMV kernel with `NUM_COLS`/`qmv_wide`-style column batching, not the tile GEMM.
   - Sources: MLX routes M < 14 (M1) to `qmv` with grid.x = M (quantized.cpp 85-140, 1804-1849); ggml Vulkan serves up to `mul_mat_vec_max_cols = 8` activation columns from one weight stream (`temp[NUM_COLS][NUM_ROWS]`, mul_mat_vec.comp).
   - Why: at small M the weight stream dominates; a 64-thread GEMV streaming each weight byte once for up to 8 columns is bandwidth-bound like decode, while a 16x16 tile GEMM at M=8 wastes half its threads. Affects our "prefill 103" number (which prefill length it is depends on the parent's harness) and speculative/batched decode.
   - Expected: for M <= 8 the cost approaches decode cost per weight byte (~1x decode time for 8 tokens instead of ~8x).

4. Prefill GEMM tile from the tuned no-coopmat reference: llama.cpp medium warptile on Apple/Honeykrisp = 128 threads, BM=BN=64, BK=32 for quantized A (16 for f16), per-thread 8x4 fp32 accumulators (TM=4, TN=2, WMITER=2, WNITER=2), f16-pair shared tiles consumed 2 k (quant) or 4 k (f16) per register step, one barrier per BK step, BK inner loop NOT unrolled on Honeykrisp; plus split-K to reach >= ~512 workgroups (MLX `qmm_splitk`, bm=bn=32, fp partials + a separate sum, no atomics).
   - Sources: ggml-vulkan.cpp 4474-4476, 7259-7297, 2971-3048 + 3149-3163; PR #24306 (+25% pp512), PR #24663 (+22.4% pp512, "12 KB instruction cache"); mul_mm.comp 173-175, 279-290, 324-351; MLX quantized.cpp 1120-1217.
   - Why: AGX has no matrix hardware exposed through Vulkan; the scalar GEMM ceiling is set by register cache and issue slots (section 5.3), so the wins are (i) more FMAs per shared load via a taller/wider register tile with vec4 shared loads, (ii) keeping the hot loop inside the 12 KB i-cache, (iii) enough workgroups to fill 8 cores x 24 simds. Our `qmm_tile.comp` uses 16x16 (or 32x16 with 4x2 per thread) tiles and a 16-wide k-tile; the reference is 64x64 with 8x4 per thread and BK=32, i.e. ~4x the FMAs per shared byte. Dequantizing weights once per tile into shared (which we already do) matches `load_b_to_shmem`.
   - Expected: llama.cpp reports 12% -> roughly 2x with these two changes on the same driver (207 -> 260 -> 375 on M1 Ultra for 7B q4_0 across #24306 + #24663; Metal on the same box 926). A 1.5-2x on our 252/289 prefill numbers is the realistic band; the remaining gap to Metal (3-5x) is compiler/matrix-hardware and is not reachable from GLSL (issue #10982 2026-06-22 comment: GLSL-level load hoisting was exactly neutral).

5. Decode attention as MLX `sdpa_vector`: one workgroup per (head, query), 32 subgroups stride over keys, each subgroup keeps online-softmax state and the o[] partial in registers, one `subgroupAdd` per key for the score, barriers only in the final combine (5 for D=64); no per-key shared-memory trees, no PARTIAL/combine dispatch for k_len < 4096 on M1.
   - Source: sdpa_vector.h lines 15-177; dispatch scaled_dot_product_attention.cpp 393-394, 878-884 (single pass on M1 below 4096 keys).
   - Why: our kernel uses shared-memory max/sum trees and a regroup phase (5-18 barriers) and splits at 256 keys into PARTIAL + combine (2 dispatches per head group). At decode lengths < 4096 the MLX shape is one dispatch per attention with ~5 barriers.
   - Expected: small per-token effect at short contexts (attention is ~2% of prefill time in the Asahi profile, LAB_NOTEBOOK line 388) but removes 24 combine dispatches per token whenever k_len > 256 (~0.1 ms) and shortens the critical path per layer.

6. Elementwise shape hygiene from the AGX numbers (section 5.3): keep nibble extraction as AND-masks (128/cycle) not shifts (32/cycle) - already true in our kernel and in MLX/ggml; avoid 32-bit integer multiplies in inner-loop addressing (32/cycle) - precompute row bases outside the K loop and advance by adds; prefer fp16 storage loads unpacked to fp32 accumulators (halves register footprint of x and raises occupancy); keep per-dispatch push-constant reads out of inner loops (they live in uniform registers but each `ldimm` costs issue slots).
   - Expected: second-order (<5%) individually; they compound with item 2.

7. Not applicable / dead ends confirmed by others (do not spend time): software cooperative-matrix emulation on AGX (~12x slower, asahi-agx-llm-perf README); int8 dot products (not accelerated on Honeykrisp); relying on pipeline-barrier granularity or async overlap between dispatches (serialized, section 5.1); k-quant style multi-scale formats (Q4_K slower than Q4_0 on AGX decode, LAB_NOTEBOOK line 21) - our affine group-64 with separate scale/bias streams is already the "simple" format.

## Directly actionable now (per kernel worker)

QmmBandwidth2 (q4 GEMV, qmm_vec_q4.comp):
- Try the MLX/ggml-Metal shape: 64-thread workgroups (2 subgroups), 4 rows per subgroup, 8 B (2 words) per lane per row per K-step, x loaded once per step as 16 pre-scaled f32 and reused across the 4 rows, `subgroupAdd` epilogue only, grid = ceil(N/8). Sources: quantized.h `qmv_fast_impl`; mul_mv.metal lines 217-303, ggml-metal-impl.h lines 30-31.
- Keep 4 loads in flight per lane before the first FMA (MLX issues all 4 row loads in the row loop before `qdot`); ggml Vulkan gets the same depth by unrolling 4 K-iterations.
- Add the ggml-style epilogue fusion hooks (bias/residual add, silu*mul for interleaved gate/up rows) so DecodeKernels' dispatch cuts land in this kernel: mul_mat_vec_base.glsl `MAT_VEC_FUSION_FLAGS_BIAS0/1`.
- For M in 2..8 (batched/speculative decode), serve multiple activation columns from one weight stream (`NUM_COLS` in mul_mat_vec.comp; `qmv_wide` in MLX) rather than re-dispatching.

PrefillGemm (qmm_tile.comp / matmul.comp):
- Adopt the llama.cpp Honeykrisp medium warptile as the starting geometry: BM=BN=64, BK=32 (quantized A), 128 threads, per-thread 8x4 fp32 accumulators (TM=4, TN=2, WMITER=2, WNITER=2), shared tiles stored as f16 pairs and consumed 2 k per register step so each step does 64 FMAs per 12 shared vec2 loads (mul_mm.comp 279-290, 324-351). Stay under 32 KB shared (hk_physical_device.c 749).
- Do NOT unroll the BK inner loop (PR #24663: +22% on Asahi; 12 KB i-cache).
- Split K to reach ~512 workgroups for small M x N tiles (MLX `qmm_splitk`: fp partials to a [split_k, M, N] scratch + one reduction dispatch; no float atomics needed on Honeykrisp).
- Route M < ~14 to the GEMV path with column batching (item 3) instead of the tile GEMM.

DecodeKernels (rope / rmsnorm / copies / preambles):
- The per-dispatch floor is a driver constant (`hk_cdm_cache_flush` after every `agx_cdm_launch`, hk_cmd_dispatch.c 56-69); consecutive dispatches never overlap (LAB_NOTEBOOK cont9s). The lever is count: fuse rope(q)+rope(k) into one dispatch; fold rms_norm into the following GEMV prologue (each workgroup recomputes sum(x^2) over 896 elements with one subgroupAdd; MLX's `rms_single_row` shape shows the reduction is one `simd_sum` + one `rsqrt`); fold residual adds into GEMV epilogues (ggml `BIAS0`); fold silu*mul into the gate/up GEMV. Target ~220 fewer dispatches per token (~0.7-1.1 ms).
- rms_norm kernel shape if it stays separate: MLX `rms_single_row` (rms_norm.metal 12-80): one threadgroup per row, `N_READS = 4` contiguous elements per thread kept in registers, `simd_sum` + threadgroup reduce, `precise::rsqrt` once, second pass from registers (no re-read).
- rope shape: MLX `rope_single`/`rope` (rope.metal 24-26, 105-107) computes `theta = L * inv_freq` and `fast::cos/sin` per pair in-kernel; no cos/sin table dispatch; q and k are separate calls in MLX only because they are separate arrays - in our backend one dispatch over both is legal.
- Command-buffer/submit: keep all of a token's dispatches in one command buffer and one `vkQueueSubmit` (each submit is a separate `drm_asahi_cmd_compute`, hk_queue.c 829-840; `max_commands_per_submit` = 1 by default, hk_queue.c 248-257); MLX Metal commits every 40 ops on base M1 but pays no per-dispatch flush; on Honeykrisp the flush is per dispatch, so the submit count is second-order and the dispatch count is first-order.

## Source list (all fetched 2026-09-07)
- MLX: mlx/backend/metal/kernels/quantized.h, quantized.cpp, device.cpp, eval.cpp, kernels/sdpa_vector.h, scaled_dot_product_attention.cpp, kernels/rms_norm.metal, kernels/rope.metal (raw.githubusercontent.com/ml-explore/mlx/main/...)
- mlx-lm: mlx_lm/generate.py, models/cache.py, models/qwen2.py (raw.githubusercontent.com/ml-explore/mlx-lm/main/...)
- llama.cpp Vulkan: ggml/src/ggml-vulkan/ggml-vulkan.cpp, vulkan-shaders/{mul_mat_vec.comp, mul_mat_vec_base.glsl, mul_mm.comp, mul_mm_funcs.glsl, dequant_funcs.glsl, flash_attn.comp, flash_attn_base.glsl, rms_norm.comp}; PRs https://github.com/ggml-org/llama.cpp/pull/24306 and /pull/24663; issue https://github.com/ggml-org/llama.cpp/issues/10982
- llama.cpp Metal: ggml/src/ggml-metal/kernels/mul_mv.metal, ggml-metal-impl.h, ggml-metal-ops.cpp
- Mesa: src/asahi/vulkan/{hk_cmd_dispatch.c, hk_cmd_buffer.c, hk_cmd_buffer.h, hk_queue.c, hk_device.c, hk_device.h, hk_physical_device.c, hk_private.h}, src/asahi/genxml/cmdbuf.xml, src/asahi/lib/agx_usc.h
- AGX ISA/microarch: https://dougallj.github.io/applegpu/docs.html, https://github.com/philipturner/metal-benchmarks (README), https://gitlab.com/grahamatgitlab/asahi-agx-llm-perf (README.md, LAB_NOTEBOOK.md), https://gitlab.freedesktop.org/xingjianll/mesa/-/tree/agx-llm-perf

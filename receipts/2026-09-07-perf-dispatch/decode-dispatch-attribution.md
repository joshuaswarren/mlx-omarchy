# Decode dispatch attribution — one steady-state token (2026-09-07)

Repo `/tmp/mlx-release-f5-final` @ `4e116e54`. Profile: `jwm1:/home/joshuawarren/benchq/morning-20260907/current-profile.jsonl` (+ markers, analysis, provenance), copied read-only into `profile/`. Profiled wheel `0.32.2.dev202609071202+diag.348919c`. Model: Qwen2.5-0.5B-Instruct-4bit (parent-stated; every observed shape matches: 24 layers, hidden 896, 14 q heads / 2 kv heads x 64, intermediate 4864, vocab 151936, 4-bit group 64). mlx-lm 0.31.3 source from the PyPI sdist (`/tmp/mlx-lm-src-0.31.3`). No source edits, no model run.

Companion data: `decode-dispatch-attribution.json` (all 585 dispatches with kernel, op, count, grid, binding sizes, gap, model op).

## Totals (token 10 of 31; every token has the identical 585-dispatch signature)

| kernel | per token |
|---|---|
| ElementwiseF16 | 193 |
| QmmVecSubgroupF16 | 169 |
| FastRmsNormF16 | 49 |
| FastRopeF16 | 48 |
| CopyGeneralF16 | 48 |
| MatmulF16 | 48 |
| SoftmaxF16 | 24 |
| TakeF16 | 2 |
| TakeU32 | 1 |
| DequantF16 | 1 |
| LogSumExpF16 | 1 |
| ArgReduceF16 | 1 |
| **total** | **585** = 4 (embedding) + 24 x 24 (layers) + 5 (norm, lm_head, logsumexp, subtract, argmax) |

Casts: **0** (f16 end to end). Pure copies: **48** (2/layer, KV-cache `slice_update` of the 128-element k and v rows; in-place, only the update is copied).

## Layer body (24 dispatches, identical in all 24 layers)

| # | kernel | model op | count | groups | bindings (bytes) |
|---|---|---|---|---|---|
| 0 | FastRmsNormF16 | input_layernorm (mx.fast.rms_norm) | 896 | 1x1x1 | [4096, 4096, 4096, 4096] |
| 1 | QmmVecSubgroupF16 | k_proj quantized_matmul | 128 | 16x1x1 | [4096, 57344, 4096, 4096, 4096] |
| 2 | QmmVecSubgroupF16 | v_proj quantized_matmul | 128 | 16x1x1 | [4096, 57344, 4096, 4096, 4096] |
| 3 | ElementwiseF16 op0 | k_proj bias add | 128 | 1x1x1 | [4096, 4096, 4096, 4096] |
| 4 | ElementwiseF16 op0 | v_proj bias add | 128 | 1x1x1 | [4096, 4096, 4096, 4096] |
| 5 | QmmVecSubgroupF16 | q_proj quantized_matmul | 896 | 112x1x1 | [4096, 401408, 28672, 28672, 4096] |
| 6 | ElementwiseF16 op0 | q_proj bias add | 896 | 4x1x1 | [4096, 4096, 4096, 4096] |
| 7 | FastRopeF16 | rope(keys) | 64 | 1x1x1 | [4096, 4096, 4096, 4096] |
| 8 | CopyGeneralF16 | cache.update_and_fetch keys slice_update | 128 | 1x1x1 | [4096, 4096, 65536, 65536] |
| 9 | CopyGeneralF16 | cache.update_and_fetch values slice_update | 128 | 1x1x1 | [4096, 4096, 65536, 65536] |
| 10 | FastRopeF16 | rope(queries) | 448 | 2x1x1 | [4096, 4096, 4096, 4096] |
| 11 | MatmulF16 | sdpa: scores = q @ k^T (scale in alpha) | 742 | 4x1x14 | [4096, 65536, 4096, 4096] |
| 12 | SoftmaxF16 | sdpa: softmax(scores) | 742 | 14x1x1 | [4096, 4096, 4096] |
| 13 | MatmulF16 | sdpa: probs @ v | 896 | 4x1x14 | [4096, 65536, 4096, 4096] |
| 14 | QmmVecSubgroupF16 | o_proj quantized_matmul | 896 | 112x1x1 | [4096, 401408, 28672, 28672, 4096] |
| 15 | ElementwiseF16 op0 | residual add h = x + attn | 896 | 4x1x1 | [4096, 4096, 4096, 4096] |
| 16 | FastRmsNormF16 | post_attention_layernorm (mx.fast.rms_norm) | 896 | 1x1x1 | [4096, 4096, 4096, 4096] |
| 17 | QmmVecSubgroupF16 | gate_proj quantized_matmul | 4864 | 608x1x1 | [4096, 2179072, 139264, 139264, 12288] |
| 18 | ElementwiseF16 op5 | silu: sigmoid(gate) | 4864 | 19x1x1 | [12288, 12288, 12288, 12288] |
| 19 | QmmVecSubgroupF16 | up_proj quantized_matmul | 4864 | 608x1x1 | [4096, 2179072, 139264, 139264, 12288] |
| 20 | ElementwiseF16 op1 | silu: gate * sigmoid(gate) | 4864 | 19x1x1 | [12288, 12288, 12288, 12288] |
| 21 | ElementwiseF16 op1 | swiglu: silu(gate) * up | 4864 | 19x1x1 | [12288, 12288, 12288, 12288] |
| 22 | QmmVecSubgroupF16 | down_proj quantized_matmul | 896 | 112x1x1 | [12288, 2179072, 139264, 139264, 4096] |
| 23 | ElementwiseF16 op0 | residual add out = h + mlp | 896 | 4x1x1 | [4096, 4096, 4096, 4096] |

Prefix: TakeF16 scales[x], TakeF16 biases[x], TakeU32 weight[x], DequantF16 (QuantizedEmbedding). Suffix: FastRmsNormF16 model.norm, QmmVecSubgroupF16 lm_head (N=151936, 68 MB weights), LogSumExpF16, ElementwiseF16 op9 subtract, ArgReduceF16 argmax.

Alignment evidence: `mlx_lm/models/qwen2.py` (Attention/MLP/TransformerBlock), `activations.py` swiglu = `nn.silu(gate) * x` (compile disabled -> sigmoid, mul, mul), `mlx/nn/layers/quantized.py` (QuantizedLinear = quantized_matmul then `x + bias`; QuantizedEmbedding = 3 takes + dequantize), omarchy `ScaledDotProductAttention::eval_gpu` f16 path = matmul(alpha=scale) -> softmax -> matmul, `RoPE::eval_gpu` count = elements/2 (448 q, 64 k). Elementwise op codes from `shaders/elementwise.comp`.

## Upstream Metal comparison (derived from upstream 0.32.2 source, [INFERENCE], not a Metal capture)

- Eager (`MLX_DISABLE_COMPILE=1`): 22/layer -> **537/token**. Only difference: `sdpa_vector` is 1 kernel where omarchy emits 3 (matmul, softmax, matmul) -> +48/token here.
- Default (compile on): swiglu is `@mx.compile` -> 1 Compiled kernel/layer -> **489/token**. Omarchy is +96 vs that.
- Everything else (qmv, bias add, rope x2, slice_update x2, residual adds, rms_norm, logsumexp/sub/argmax) is 1:1.

## Fusion ranking (dispatches saved per token; all 24 layers)

| rank | fusion | now -> after (per layer) | saved/token | exactness |
|---|---|---|---|---|
| 1 | qkv_projection_block: per layer: k_proj.qmm, v_proj.qmm, q_proj.qmm (same lhs = input_layernorm output) + 3 bias adds -> 1 dispatch | 6 -> 1 | **120** | bit-exact if the epilogue rounds the float accumulator to f16 first and then adds the f16 bias in float and rounds again (that is exactly what ElementwiseF16 op0 computes today); bit-exact only if the fused kernel keeps qmm_vec.comp's per-column lane split and subgroup reduction order; then each output column is the same float dot product and only the dispatch grouping changes |
| 2 | mlp_block: per layer: gate_proj.qmm, up_proj.qmm (same lhs), sigmoid, multiply, multiply -> 1 dispatch (dual-weight qmm with swiglu epilogue) | 5 -> 1 | **96** | bit-exact if each intermediate is rounded to f16 between ops exactly as the three separate kernels do; bit-exact (see rank 1); bit-exact with f16 rounding of gate, up, sigmoid, product, product in that order |
| 3 | sdpa_vector: per layer: MatmulF16 (q@k^T, scale in alpha), SoftmaxF16, MatmulF16 (probs@v) -> 1 vector-SDPA dispatch (upstream Metal sdpa_vector does exactly this for q_len<=8) | 3 -> 1 | **48** | NOT automatically bit-exact: today scores and probs are stored f16 between kernels; a fused kernel must round scores to f16 after the scaled dot, run softmax in float over those f16 scores, round probs to f16, then accumulate probs@v in float and round once - replicate that rounding sequence to keep token IDs identical |
| 4 | residual_add_plus_rms_norm: per layer x2: residual add (h = x + r) immediately followed by mx.fast.rms_norm(h) on the same 896-vector -> 1 dispatch with two outputs (h and norm(h)); pairs: residual_add_attn+post_attention_layernorm, residual_add_mlp+next layer input_layernorm (layer 23 pairs with model.norm) | 4 -> 2 | **48** | bit-exact: write h rounded to f16 exactly as ElementwiseF16 op0 does, then run the existing rms_norm math on that f16 h in the same workgroup |
| 5 | cache_write_redirect: per layer x2: rope(keys) -> CopyGeneralF16 slice_update into cache.keys; v_proj.bias_add -> CopyGeneralF16 slice_update into cache.values. The producer can write straight into the (donated, in-place) cache slot with an output offset/stride, deleting the copy | 4 -> 2 | **48** | bit-exact: pure copy elimination |
| 6 | rope_q_and_k_one_dispatch: per layer: FastRopeF16 (queries, 14 heads) and FastRopeF16 (keys, 2 heads) share offset/base/scale -> one dispatch over 16 head rows | 2 -> 1 | **24** | bit-exact |
| 7 | embedding_gather_dequant: per token: 3 Take + DequantF16 -> 1 gather-dequant dispatch | None -> None | **3** | bit-exact |
| 8 | logprob_tail: per token: LogSumExpF16 + subtract + ArgReduceF16 -> 1 or 2 dispatches (argmax of logits equals argmax of logits-lse; the subtract output is only used for returned logprobs) | None -> None | **2** | argmax bit-exact; logprobs array still needed by mlx-lm generate_step return value, so keep the subtract unless mlx-lm is changed |

Sub-steps for ranks 1-2: qkv = bias-into-qmm-epilogue (72) + shared-lhs 3-weight qmm (48); mlp = swiglu 3-op elementwise chain (48, `MLX_OMARCHY_FUSED_CHAIN` territory) + gate/up dual qmm (24) + swiglu epilogue (24).

| tier | content | saved | dispatches/token |
|---|---|---|---|
| A: elementwise/copy-only fusions, no new qmm variants | mlp swiglu 3->1 (48), sdpa_vector 3->1 (48), residual+rmsnorm 2->1 x2 (48), cache write redirect (48), rope q+k (24) | 216 | **369** |
| B: A + bias into qmm epilogue | A, qmm bias epilogue (72) | 288 | **297** |
| C: B + shared-lhs multi-weight qmm (qkv 3->1, gate/up 2->1 with swiglu epilogue) | B, qkv shared-lhs (48), gate/up dual (24), swiglu epilogue instead of chain kernel (24) | 384 | **201** |
| D: C + embedding + logprob tail | C, embedding (3), tail (2) | 389 | **196** |

Parent target <= 250/token: tier B (297) misses; tier C (201, 8/layer) clears it. Sum of all listed = 389 saved -> 196/token floor with this op set.

## gap_model (parent request)

Token 10: GPU busy **10.095 ms**, intra-submission gaps **41.518 ms**, inter-submission gaps 2.731 ms -> implied wall 54.343 ms vs measured instrumented inter-token interval 54.746 ms (mean 53.814 ms over 31). The instrumented timeline closes to <1%. The uninstrumented token is 14.2 ms (parent), so ~40 ms/token of instrumented gap is timestamp/drain artifact; only order, counts and relative gap structure transfer.

Gap before a dispatch (t0 - previous t1, same submission, us, tokens 5-29):

| kernel | n | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| ElementwiseF16 | 4992 | 20.8 | 22.2 | 29.8 | 86.7 |
| QmmVecSubgroupF16 | 4368 | 33.6 | 72.1 | 255.4 | 316.6 |
| MatmulF16 | 1248 | 32.7 | 36.8 | 64.2 | 122.9 |
| FastRmsNormF16 | 1222 | 20.7 | 22.7 | 43.7 | 115.8 |
| FastRopeF16 | 1222 | 18.4 | 19.8 | 32.6 | 75.0 |
| CopyGeneralF16 | 1222 | 20.6 | 22.0 | 34.9 | 99.7 |
| SoftmaxF16 | 624 | 21.2 | 23.1 | 58.5 | 128.2 |
| TakeF16 | 26 | 46.8 | 57.9 | 126.6 | 149.3 |
| TakeU32 | 26 | 48.3 | 60.2 | 156.6 | 169.7 |
| DequantF16 | 26 | 45.8 | 58.4 | 157.5 | 164.9 |
| LogSumExpF16 | 26 | 268.8 | 281.7 | 283.5 | 284.8 |
| ArgReduceF16 | 26 | 136.5 | 137.1 | 139.0 | 139.8 |

QmmVecSubgroupF16 gap-before vs bound weight bytes (`gap = 29.9 us + 101 us/MB`, n=4200):

| weight bytes | which | p50 us | p90 us |
|---|---|---|---|
| 57344 | k/v_proj | 34.6 | 41.0 |
| 401408 | q/o_proj | 62.2 | 76.2 |
| 2179072 | gate/up/down | 243.5 | 260.2 |
| 68067328 | lm_head | 6904.8 | 6908.7 |

No: QmmVecSubgroupF16 gaps are NOT the same ~22 us floor as tiny elementwise kernels. The floor for ElementwiseF16/FastRope/CopyGeneral/Softmax is p50 20-23 us. Before a QmmVec the gap scales linearly with the bound weight buffer: 57 KB (k/v_proj) 35 us, 401 KB (q/o_proj) 62 us, 2.18 MB (gate/up/down) 243 us, 68 MB (lm_head) 6905 us = 29.9 us + 101 us/MB, i.e. ~10 GB/s over the weight bytes. Group count does not explain it (down_proj 112 groups and gate_proj 608 groups both sit at ~243 us; q_proj 112 groups sits at 62 us). Per token that size term is 24*(3*0.22+2*0.04+2*0.006)+6.9 = 25.6 ms, which is MORE than the whole uninstrumented token (14.2 ms), so at least most of it is an artifact of the per-dispatch BOTTOM_OF_PIPE timestamp pair (a drain/flush whose recovery cost tracks the next kernel's footprint - TLB/page-table or cache refill at ~10 GB/s = ~1.6 us per 16 KB page). Two things follow. (1) Fusion count IS the lever for the flat part: every dispatch pays the same ~22 us instrumented floor regardless of size, and 585 of them is the count to cut. (2) The real uninstrumented budget is 14.2 ms for 247 MB of 4-bit weights per token (24 x 7.45 MB + 68 MB lm_head) = at least 17 GB/s effective weight streaming already; macOS 6.6 ms needs >= 37 GB/s. So after the dispatch count is cut, qmm_vec weight-streaming efficiency is the second lever, and the QmmVec durations in this profile (p50 12-15 us for 2.18 MB = 145 GB/s, above M1 DRAM bandwidth) prove the weight read is NOT inside the measured kernel span - it lands in the pre-dispatch gap. Decisive uninstrumented test (needs a jwm1 grant): submit 100 back-to-back QmmVecSubgroupF16 dispatches on the 68 MB lm_head weights in one command buffer with no timestamps and time the join. ~690 ms means the 10 GB/s weight stream is real and the qmm kernel/bandwidth is the primary lever; ~5-20 ms means it is a profiler artifact and dispatch count is the only lever.

Weight bytes streamed per decode token: 247.0 MB (24 x 7.45 MB + 68 MB lm_head).

## Method

- **profile_files**: {'current-profile.jsonl': 'copied from jwm1:/home/joshuawarren/benchq/morning-20260907/ via scp (read-only)', 'current-markers.jsonl': 'same', 'current-analysis.txt': 'same', 'profile-provenance.json': 'same; wheel 0.32.2.dev202609071202+diag.348919c, libmlx sha256 95a02ee3'}
- **kernel_name_source**: overlay/mlx/backend/omarchy/compute.h ComputeKernel enum order at HEAD 4e116e54 via scripts/profile_analyze.py parse_kernel_names; the 13 kernel indices present decode to exactly the names current-analysis.txt printed at capture time, so the enum order matches the profiled wheel
- **elementwise_op_source**: overlay/mlx/backend/omarchy/shaders/elementwise.comp switch(params.operation): 0 add, 1 multiply, 5 sigmoid, 9 subtract
- **window_selection**: dispatch records grouped by submission id; decode-phase submissions (submit host time between decode_start and decode_done markers) form a strict 7-submission/585-dispatch period per token; token 10 chosen (steady state, identical signature to every other token; the 24-dispatch layer body is byte-identical across all 24 layers: kernel, op, count, grid, binding ranges)
- **alignment**: layer body aligned to mlx_lm/models/qwen2.py TransformerBlock/Attention/MLP + mlx/nn/layers/quantized.py (QuantizedLinear: quantized_matmul then x + bias; QuantizedEmbedding: 3 takes + dequantize) + omarchy primitives.cpp ScaledDotProductAttention::eval_gpu (f16 path: matmul(alpha=scale) -> softmax -> matmul). Shape checks: hidden 896, kv dim 128 (2 heads x 64), q dim 896 (14 x 64), intermediate 4864, vocab 151936, k_len 53 (scores count 742 = 14 x 53), rope count = elements/2 (half_dims pairs: 448 q, 64 k), cache buffer 65536 B = 2 x 256 x 64 x 2 B (KVCache step 256)
- **model_identity**: Qwen2.5-0.5B-Instruct-4bit as stated by the parent; every observed shape (24 layers, 896/4864/14/2/64, vocab 151936, group_size 64 -> 14 scale groups per row, 4-bit -> 112 u32 words per row) is consistent with that model; no model was run in this session
- **gap_definition**: gap_before(d) = t0(d) - t1(previous dispatch in the same submission), device timestamps (period 1 ns); durations t1 - t0; percentiles over tokens 5-29 unless stated
- **no_source_edits**: True

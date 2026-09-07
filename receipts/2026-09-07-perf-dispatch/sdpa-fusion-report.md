# Fused decode SDPA (Fusion2, 2026-09-07)

Repo `/tmp/mlx-release-f5-final` @ `4e116e54` + uncommitted tree. Default ON
since the parent's snapshot-3 M1 A/B (`v2-sdpa-ab-m1-perfsnap3`: decode
96.0->100.1, 76.6->86.8, 51.1->72.7 tok/s, token IDs equal in all 3 pairs);
`MLX_OMARCHY_SDPA_FUSED=0` is the kill switch (composed matmul -> softmax ->
matmul path). `MLX_OMARCHY_SDPA_SUBGROUP=0` forces the shared-memory
reduction variant. The llvmpipe wheel rows below were built while the gate
was opt-in (`MLX_OMARCHY_SDPA_FUSED=1`, see `env-sdpa2.txt`).

## Kernel (final, v2 + follow-ups)

`shaders/sdpa_decode.comp` (plain / `-DUSE_SUBGROUP=1` / `-DPARTIAL=1`
crosses) and `shaders/sdpa_decode_combine.comp`. One workgroup per (batch,
q head, query row, 256-key block); 256 lanes each own one key: score =
scale * q.k in float (keys loaded as f16vec4), causal / additive mask
(values <= -65504 -> -1e30 floor, fully masked row -> uniform), block max
and exp-sum (subgroupMax/Add + shared pass, or shared trees), then lanes
regroup as (key split, output dim) for probs @ v. k_len <= 256: one
dispatch writes the float16 output. Longer: one partial dispatch + one
combine dispatch (2 instead of 3). Covers f16, q_len 1..8, head_dim
multiple of 4 up to 256, GQA via strides, K/V cache slices via strides,
no sinks; everything else takes the composed path unchanged.

Numerics: scores and probs stay float where the composed path stores
float16, so outputs are not bit-identical to composed; on every one of
the 12 test shapes (llvmpipe and M1) the fused max and mean error vs a
double reference is lower than the composed path's (test logs print
both; e.g. Qwen decode shape max 7.3e-5 vs 8.5e-5, mean 1.2e-5 vs 2.0e-5).
On llvmpipe the 64-token greedy run diverges from the composed IDs at
token 27 (near-tie flips); on the M1 the parent's snapshot-3 paired check
found the IDs equal in all three workloads.

The first iteration (v1: one workgroup per (batch, kv head), bit-matching
the composed rounding order) was bit-identical to composed on llvmpipe
and on the M1 for the 9 test shapes (`sdpa-fusion-m1/test-v1.log`) but
diverged in the parent's long model leg and regressed decode (2
workgroups on 8 cores); it is gone.

## Dispatches per decode token, llvmpipe, pinned Qwen2.5-0.5B-Instruct-4bit, 64 greedy tokens

`sdpa-fusion-llvmpipe/analysis-*.txt` (diagnostics wheels built from
`/tmp/f2-tree-base` and `/tmp/f2-tree-after`):

| wheel | dispatches/decode token | SDPA kernels per layer |
|---|---|---|
| base (`+diag.4e116e54.base`) | 585.0 | MatmulF16 + SoftmaxF16 + MatmulF16 |
| v1 (`+diag.4e116e54.sdpa`) | 537.0 | SdpaDecodeF16; IDs == base |
| v2 (`+diag.4e116e54.sdpa2`, `MLX_OMARCHY_SDPA_FUSED=1`) | 537.0 | SdpaDecodeF16 (k <= 256 here); IDs diverge at token 27 |

## M1 microbench (`sdpa-fusion-m1/bench-*.jsonl`, `tests/omarchy/bench_sdpa_decode.cpp`)

Qwen shape 14/2/64, q_len 1, K/V as 256-step cache slices, N back-to-back
calls per eval, wall/N, best of 3, under `benchq/gpu.lock`. us per call:

| k_len | composed | fused (final: subgroup + f16vec4) | fused_tree |
|---|---|---|---|
| 45 (200 reps, 3 runs) | 47.3 / 47.9 / 48.9 | 22.7 / 22.9 / 22.7 | 24.7 / 24.7 / 24.9 |
| 128 (200 reps) | 81.4 | 37.7 | 42.9 |
| 280 (48 reps) | 205.3 | 57.8 | 52.6 |
| 390 (48 reps) | 214.3 | 52.8 | 54.9 |
| 1024 (48 reps) | 419.7 | 75.3 | 70.7 |
| 4096 (48 reps) | 1538.5 | 237.7 | 237.2 |

(48-rep rows carry more fixed cost per call; compare within a row.)
Before the f16vec4 key loads (`bench-sg.jsonl`): k280 101, k1024 156,
k4096 586 us.

## Tests

llvmpipe (`MLX_OMARCHY_ALLOW_NON_APPLE=1`): omarchy_sdpa_decode_tests
2/2 cases 189/189, omarchy_fast_regression_tests 2/2 16/16,
omarchy_fast_ops_tests 30/30 57158/57158. M1: omarchy_sdpa_decode_tests
189/189 for v2, subgroup and f16vec4 builds (`sdpa-fusion-m1/test-*.log`).

## Also evaluated

- The two CopyGeneralF16 per layer are `SliceUpdate::eval_gpu` ->
  `copy_gpu_inplace` for mlx-lm's KV-cache `update_and_fetch` (in-place
  cache, only the 128-element row copied); upstream Metal emits the same
  two. Removing them needs the producer (rope / bias add) to write into
  the cache slot, a cross-primitive change; not done.
- eager_fuse (predecessor draft in `/tmp/fi-tree`, never in this repo):
  dropped. Deferring a node in `gpu::eval` cannot prove single-consumer
  status: consumers outside the current eval (Python-held arrays
  evaluated later, other streams) are invisible at deferral time, so
  correctness rests on every observation path flushing first - the
  "looks resolved but is still acting" hazard class. The compiled tape
  (`MLX_OMARCHY_FUSED_CHAIN`) is the sound mechanism for the swiglu chain.

## Files changed (overlay)

- `mlx/backend/omarchy/shaders/sdpa_decode.comp` (new)
- `mlx/backend/omarchy/shaders/sdpa_decode_combine.comp` (new)
- `mlx/backend/omarchy/primitives.cpp`: `sdpa_decode_fused` + call in
  `ScaledDotProductAttention::eval_gpu`
- `mlx/backend/omarchy/compute.h`: enum `SdpaDecodeF16`,
  `SdpaDecodePartialF16`, `SdpaDecodeCombineF16`, `SdpaDecodeSubgroupF16`,
  `SdpaDecodePartialSubgroupF16` (appended before `Count`)
- `mlx/backend/omarchy/compute.cpp`: includes + `shader_bytes` cases
- `mlx/backend/omarchy/CMakeLists.txt`: five `omarchy_shader` lines
- `tests/omarchy/test_sdpa_decode.cpp`, `tests/omarchy/bench_sdpa_decode.cpp`
  (new), `tests/omarchy/CMakeLists.txt` (two blocks appended)

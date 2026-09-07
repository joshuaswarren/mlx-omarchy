# Decode per-kernel device time after q4 GEMV v2 + fused SDPA (2026-09-07, DecodeKernels)

Tree: `/tmp/mlx-release-f5-final` @ `81e9f53a` plus the uncommitted sibling work as of 15:14 (per-op
specialized elementwise, q4 GEMV v2 default on, fused flash-decoding SDPA default on), BEFORE any
DecodeKernels change. Diagnostics wheel `mlx_omarchy-0.32.2.dev202609072013+diag.81e9f53.dk0` built on
jwm1 from that snapshot (`/home/joshuawarren/benchq/DecodeKernels/snap0`, `-DMLX_OMARCHY_GPU_PROFILING=ON`,
profile literal verified in the wheel), venv copy of `benchq/perfsnap1/venv` (mlx-lm 0.31.3).

Run (jwm1, Apple M1 G13G B1, Honeykrisp, under `flock benchq/gpu.lock`): pinned
`/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx`, prompt
`What is the capital of France? Answer in one word.` through the chat template, 64 greedy tokens via the
pinned-length `gen_ids.py` driver (`generate_step` + argmax, EOS ignored; the `profile_generate.py` run
stopped at 2 tokens because the model answers `Paris` then EOS - that 2-token profile is
`benchq/DecodeKernels/prof-snap0`, unused), `MLX_DISABLE_COMPILE=1`, `MLX_OMARCHY_GPU_PROFILE` set.
Token ids: `ids.json` (starts `Paris<|im_end|>`). Instrumented median inter-token 41.42 ms (the
instrument inflates wall time; only device-timestamp durations transfer). Profile
`profile.jsonl.gz` (sha256 of the uncompressed stream
`0be1b14a6edfe2926511d92c1fa7372cef35faea47f53009a99f58df94794058`), `analysis.txt` is
`scripts/profile_analyze.py`'s report; the per-token table below is `dk_decode_table.py` (dispatches whose
submission lands between the decode_start/decode_done markers, divided by the 63 inter-token intervals).

## Per decode token (device timestamps t1 - t0, 63 intervals, 537 dispatches/token)

| kernel | n/token | mean us | p50 us | total us/token | share |
|---|---|---|---|---|---|
| ElementwiseF16 | 193.0 | 23.5 | 22.4 | 4539.8 | 38.0% |
| QmmVecQ4V2SubgroupF16 | 169.0 | 18.0 | 13.0 | 3039.6 | 25.4% |
| FastRopeF16 | 48.0 | 26.8 | 21.6 | 1288.2 | 10.8% |
| CopyGeneralF16 | 48.0 | 25.4 | 19.5 | 1220.3 | 10.2% |
| FastRmsNormF16 | 49.0 | 23.7 | 19.2 | 1160.3 | 9.7% |
| SdpaDecodeSubgroupF16 | 24.0 | 17.8 | 11.3 | 427.7 | 3.6% |
| TakeF16 | 2.0 | 58.1 | 50.4 | 116.3 | 1.0% |
| TakeU32 | 1.0 | 51.6 | 48.2 | 51.6 | 0.4% |
| DequantF16 | 1.0 | 42.4 | 45.8 | 42.4 | 0.4% |
| LogSumExpF16 | 1.0 | 30.5 | 30.2 | 30.5 | 0.3% |
| ArgReduceF16 | 1.0 | 29.3 | 28.8 | 29.3 | 0.2% |

Instrumented device-busy per token: 11.95 ms (537 dispatches, 22.2 us mean). MatmulF16 (46 total) and
SoftmaxF16 (23 total) appear only in the prefill window (the mlx-lm prompt-cache/prefill path), not in
decode: the fused SDPA removed them from the decode token. Casts: 0 per token.

## Re-ranked targets for this slice (non-GEMV, non-elementwise)

1. FastRopeF16 - 48/token, 1.29 ms/token instrumented; raw dispatch cost on the M1 6.00 us against the
   3.06 us specialized-elementwise reference (shaderdb 244 instrs / 182 uniforms / 203 preamble).
2. CopyGeneralF16 (KV-cache row paste, SliceUpdate -> copy_gpu_inplace) - 48/token, 1.22 ms/token; raw
   4.59 us (97 / 54 / 41).
3. FastRmsNormF16 - 49/token, 1.16 ms/token; raw 7.73 us, the most expensive small kernel per dispatch
   (116 / 64 / 46; eleven workgroup barriers per row).
4. Casts: none in decode; CastF16F32 / CastF32F16 shaderdb 17 instrs / 28 uniforms / 24 preamble, already
   at the FillF16 floor (14 / 24 / 22). No change made.
5. MatmulF16 / SoftmaxF16: gone from the decode token (fused SDPA); prefill only. No change made.

The per-dispatch instrumented mean (~20 us for every small kernel) is 5-6x the raw microbench cost of
the same dispatch (3-4 us); the `alternate-production` bench row (a decode-like cycle of five different
pipelines/bindings/grids, 3.92 us) shows pipeline switching is not the cause. See
`decode-kernels-m1.md` for the after numbers.

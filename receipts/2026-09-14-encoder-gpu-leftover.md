# Encoder GPU leftover kernels (2026-09-14)

Attribution only. No compile, no kernel rewrite, no flag landed.

The 18.07 s encoder wall on jwm1 (`receipts/2026-09-14-parakeet-stage-times.md`) is 5.06 s ANE worker exec plus ~13 s leftover between 72 one-shot island submits. This receipt names the GPU kernels that own that leftover.

## Verdict

`ConvF32` owns the leftover. 77 dispatches, 6.321 s GPU busy (66.4% of GPU busy, 47.2% of leftover wall). 24 of those are the per-layer `(1, 2048, 375)` convs: 4.154 s, 31.0% of leftover wall.

`SoftmaxF32` is not in the stream. `vulkan_encoder.py` lowers softmax to max/sub/exp/sum/div, so those 24 MIL softmax ops land in `ReduceF32` + `ElementwiseF32`.

`MLX_OMARCHY_GATED_BARRIERS` does not win: this capture emitted 5914 dispatch barriers and skipped 0.

## Leftover wall

Unprofiled stage-times receipt (release wheel `05015a76`, SPIR-V cache hit): encoder 18070.683 ms, ANE exec 5056.347 ms, leftover 13.014 s. 96 ANE ops / 1278 GPU MIL ops.

This locked diag capture (same runner, islands ABC, same 1278/96 split):

| quantity | value |
| --- | ---: |
| wall | 18.365 s |
| ANE worker exec | 4.983 s |
| leftover wall (wall − ANE exec) | 13.382 s |
| GPU busy | 9.515 s (71.1% of leftover wall) |
| inter-submit GPU gaps | 8.616 s (196 gaps; includes the ANE waits) |
| vk compute dispatches | 5914 |
| vk submissions | 197 |
| gpu primitive dispatches | 9347 |
| MIL gpu_ops / ane_ops | 1278 / 96 |

GPU busy 9.515 s is below leftover wall 13.382 s, so the instrument is not claiming more device time than the wall. Counts match the unprofiled three-island report (`receipts/2026-09-14-encoder-parity-ane-3islands/run-report-ane3.json`).

## GPU kernels (diag capture)

Kernel names from `overlay/mlx/backend/omarchy/compute.h`. GPU ticks `period_ns=1`. Shares of leftover wall use this capture's 13.382 s, not the 13.014 s stage-times leftover.

| kernel | dispatches | GPU busy | leftover wall | GPU busy share |
| --- | ---: | ---: | ---: | ---: |
| ConvF32 | 77 | 6.321 s | 47.2% | 66.4% |
| ElementwiseF32 | 1986 | 1.383 s | 10.3% | 14.5% |
| MatmulF32Coopmat | 194 | 1.239 s | 9.3% | 13.0% |
| CastF16F32 | 1950 | 0.179 s | 1.3% | 1.9% |
| ReduceF32 | 288 | 0.110 s | 0.8% | 1.2% |
| CopyGeneralF32 | 225 | 0.099 s | 0.7% | 1.0% |
| CopyGeneralF16 | 193 | 0.078 s | 0.6% | 0.8% |
| CastF32F16 | 879 | 0.073 s | 0.5% | 0.8% |
| CopyGeneralBool | 25 | 0.023 s | 0.2% | 0.2% |
| other | 96 | 0.010 s | <0.1% | <0.1% |

`SelectF32` is 24 dispatches / 2.4 ms: the remaining all-masked-rows selects, not island B.

## ConvF32 geometries

`n` is the profile element count and matches the MIL output volume.

| n | shape | dispatches | GPU busy | leftover wall |
| ---: | --- | ---: | ---: | ---: |
| 768000 | `(1, 2048, 375)` | 24 | 4.154 s | 31.0% |
| 384000 | `(1, 1024, 375)` | 48 | 1.743 s | 13.0% |
| 6144000 | `(1, 256, 750, 32)` | 2 | 0.298 s | 2.2% |
| 1536000 | `(1, 256, 375, 16)` | 2 | 0.074 s | 0.6% |
| 24576000 | `(1, 256, 1500, 64)` | 1 | 0.051 s | 0.4% |

The 24+48 layer convs are the conformer conv module. The five stem convs are small next to them.

## MIL GPU ops (1278)

From the runtime encoder inventory (`receipts/2026-09-12-coreml-inspector/encoder-inventory.json.gz`), minus 72 attention matmuls and 24 island-B selects (ANE):

| MIL op | count | GPU kernel |
| --- | ---: | --- |
| linear | 194 | MatmulF32Coopmat (145× `(1,375,1024)`, 48× `(1,375,4096)`, 1× `(1,375,640)`) |
| add | 183 | ElementwiseF32 |
| transpose | 146 | CopyGeneral* |
| reshape | 145 | (view; copies when materialized) |
| mul | 128 | ElementwiseF32 |
| layer_norm | 120 | ReduceF32 + ElementwiseF32 (all `(1,375,1024)`) |
| conv | 77 | ConvF32 |
| silu | 72 | ElementwiseF32 (`mx.sigmoid` × x) |
| slice_by_index | 48 | CopyGeneral* |
| pad | 24 | |
| softmax | 24 | ReduceF32 + ElementwiseF32, not SoftmaxF32 |
| split | 24 | |
| sigmoid | 24 | ElementwiseF32 |
| select | 24 | SelectF32 |
| other | 46 | casts, compares, mask arithmetic |

H13 leftover table (`receipts/2026-09-14-encoder-leftover.md`) is the peeled MIL (concat 96, transpose 73, layernorm 120, silu 72, sigmoid 24, softmax 24, conv 101). Runtime encoder conv count is 77, not 101. Time follows the 77 ConvF32 dispatches, not the peeled count.

## Next attack

Compile or rewrite the 24 ConvF32 kernels at `(1, 2048, 375)` (4.15 s). That is the first speed attack that is priced. The 48 `(1, 1024, 375)` convs are second (1.74 s). The 194 linears are third (1.24 s total).

Do not start with softmax, sigmoid, or concat: they are not the GPU-time owners.

A one-line existing flag does not win. `MLX_OMARCHY_GATED_BARRIERS` skipped 0 barriers on this encoder. Q4 A/B on this host also did not move (`receipts/2026-09-14-gated-barriers-ab.md`).

The 3.6 s leftover that is not GPU busy is inter-submit gap beyond ANE exec (197 submissions vs 28 on the vulkan-only control). Resident in-process submits address that remainder; they do not move ConvF32.

## Capture

Host `jwm1-linux`, lock inode 29, `flock -w 120 /tmp/m1-gpu.lock`, not stolen, not unlinked. Nested `flock -n` free after. `/dev/accel/accel0` live. 72 ANE submits, 0 timeouts.

```text
START 2026-09-14T14:53:21-05:00
END   2026-09-14T14:53:40-05:00 rc=0
```

- Runner: `/var/tmp/ParakeetE2EAne-stage/vulkan_encoder.py` SHA-256 `cd858ed352f6b0d25e035643cef3e5bd43ad689614fe7e396a3f7af1d4edec6d` (islands ABC).
- Diag wheel: `mlx_omarchy-0.32.2.dev202609122355+diag.b41e2b74` extracted to a throwaway site; `MLX_OMARCHY_GPU_PROFILE` literal present. Release wheels compile the profiler out.
- Profile SHA-256 `f6d266b92b4c40e007cf71402f72ecc3406df870115a706a6733c42a2db67b23` (5914 `k=d` records).
- Run report SHA-256 `2e2e5322eef575d01eb4df4e9fd3f5c55d49d4b5f51b2879fa4bd7b6dee5743c`.
- Worker `762dd1de`, libane `1ab9d95d`, islands A/B/C as in the three-island receipt.

This is not a 450 ms hardware pass and not a compile of those convs.

## Resolved model

Assigned OpenAI route. Actual `xai-oauth/grok-4.6`. Fallback: true. Numbers are host-measured on jwm1-linux.

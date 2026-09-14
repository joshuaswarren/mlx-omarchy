# Encoder ConvF32 (1,2048,375) leftover (2026-09-14)

Worktree `encoder-conv-f32` at mlx-omarchy `3cd2e93f` plus the files below.
Do not merge `63c1d3cf`. No `ane-linux-experiments` edit or execute.

Assigned OpenAI route. Actual `xai-oauth/grok-4.6`. Fallback: true.

## Verdict

H13 cannot compile the 24 expand convs without inventing ISA. A digest-preserving GPU rewrite exists: unit-window 1×1 `Convolution` dispatches tiled `MatmulF32` (K in order, no coopmat). Isolated (1,1,375,1024)×(2048,1,1,1024) stayed bit-identical and dropped 174 ms → 8.55 ms. Encoder leftover wall 13.014 s → 6.829 s. `encoder_hidden` stayed `b3d60c81`.

A ConvF32 shader 1×1 special case was bit-identical and did not move wall (177 ms). It is not the cut.

## H13 hole (not taken)

`kConvTasks` is exact-match on `(kernel, stride, groups, bias, input CHW, output CHW)`. `convParityPlan` also needs rank-4 batch-1, square `[Cout, Cin/groups, k, k]` weight, unit dilations, zero explicit pad, `pad_type` `same`/`valid`. Spatial extents in the decoded table are square 1/8/16/32/64. Kernel 1 and 3. No 375, no rank-3, no kernel 9.

The 24 GPU leftovers are MIL 1D pointwise `[1,1024,375]×[2048,1024,1]` valid. Compiler pin:

```text
assert(!supportsConvParity({1, 1, 1, false, {1024, 375, 1}, {2048, 375, 1}}));
```

in mil-hwx-compiler `tests/test_h13_encoding.cpp`. `H13ConvTemplates.inc` has no 375-wide row. `supportsConvParity` is `convTemplate != nullptr`. Adding a table row without a decoded Apple program would invent ISA. Not done.

Receipts: mil-hwx-compiler `receipts/2026-09-13-h13-conv-envelope.md`; mlx-omarchy `receipts/2026-09-14-encoder-leftover.md`.

## GPU rewrite

`vulkan_encoder.py` lifts the 1D MIL conv to NHWC 1×1 `mx.conv2d`: input `[1,1,375,1024]`, weight `[2048,1,1,1024]`, pad 0, groups 1. That is GEMM `out[375,2048] = in[375,1024] @ W[2048,1024]^T`.

`overlay/mlx/backend/omarchy/primitives.cpp` `Convolution::eval_gpu` takes that unit-window case (kernel 1, stride 1, pad_lo 0, unit dilations, groups 1, no flip, output spatial = input spatial) and dispatches `MatmulF32` with rhs-transposed flag. Not `MatmulF32Coopmat`: coopmat changes reduction order. `matmul.comp` accumulates K in 16-wide tiles in order; K=1024 is a multiple of 16.

## Before / after (jwm1-linux)

Lock inode 29, `flock -w 120 /tmp/m1-gpu.lock`, not stolen, not unlinked. Nested `flock -n` free after each run. `/dev/accel/accel0` live. Kernel `7.1.6-1-1-ARCH`.

### Isolated 1×1, seed 0, 8 timed reps after 2 warmup

| wheel / libmlx | min s | median s | digest |
| --- | ---: | ---: | --- |
| `05015a76` (stage-times encoder) | 0.173978 | 0.174682 | `032f719f3705f76449f1b2b6e1ff37bf852a0a70e753a44f983b14a1b6685aaa` |
| shader 1×1 only (`3cd2e93f` + `conv.comp`) | 0.176911 | 0.177991 | same |
| unit-window → `MatmulF32` | 0.008551 | 0.008632 | same |

New SPIR-V was in the shader-only `libmlx` (`spv_in_lib` true vs old false). Address-math cut is not the leftover.

### Encoder leftover (same runner as stage-times)

Islands ABC, 72 ANE submits, 0 timeouts, 96 ANE ops / 1278 GPU ops, `cpu_tensor_events` 0.

| quantity | before (`05015a76`, `receipts/2026-09-14-parakeet-stage-times.md`) | after (`libmlx` `bb112a7a`) |
| --- | ---: | ---: |
| encoder_ane wall | 18070.683 ms | 11817.644 ms |
| ANE worker exec | 5056.347 ms | 4988.184 ms |
| leftover (wall − exec) | 13.014 s | 6.829 s |
| pipeline wall | 22268.333 ms | 14458.357 ms |
| `encoder_hidden` sha256 | `b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e` | same |

After leftover = 11817.644 − 4988.184 = 6829.460 ms.

`encoder_hidden.npy` after: `b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e`. Matches the prior ANE encoder pin.

Tokens still diverge at emission 99 (105 vs native 104). Not claimed token-exact. `k3` conv2d still runs.

## Files

- `overlay/mlx/backend/omarchy/primitives.cpp` — unit-window 1×1 → `MatmulF32`
- `overlay/mlx/backend/omarchy/shaders/conv.comp` — 1×1 shader path; measured no wall win
- `receipts/2026-09-14-encoder-conv-f32/bench_conv.py`

Measured after `libmlx.so` sha256 `bb112a7a2e2fee740f7aeb6d46ad8d12c884ddb11daad44b543fa6a8ca341f75` (cmake incremental on the `3cd2e93f.convgemm` tree). Core extension sha256 `4ad850da16c300ea9f60aa7247f034f358aa16d7ba2fd146979a01c867e3cfa5`. Before `libmlx` `65a641e4a9e6973734a6f25f810e62c7d9e60c4a4a0e44e4490fd9518a25f2e4`.

Runner `/var/tmp/ParakeetE2EAne-stage/vulkan_encoder.py` sha256 `cd858ed352f6b0d25e035643cef3e5bd43ad689614fe7e396a3f7af1d4edec6d`. Worker `762dd1de`, libane `1ab9d95d`. Encoder `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018`.

This is not a 450 ms hardware pass and not an H13 compile of those convs.

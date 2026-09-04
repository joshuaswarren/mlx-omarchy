# Compiled elementwise fusion: correct, no model speedup established

Measured on Apple M1 with Honeykrisp on 2026-09-04. Fusion is opt-in through `MLX_OMARCHY_FUSED_CHAIN=1`; it is not in the published v0.3.5 wheels.

## Implementation and correctness

`fused_chain.{h,cpp}` and `shaders/fused_chain.comp` combine dependent float32/float16 tape nodes into one dispatch, with at most eight instructions and three input buffers. Intermediate values round to their storage dtype after each instruction. Shared outputs, unsupported layouts, and unsupported dtypes retain the per-node path. The gate-off path constructs no fusion state.

The first M1 run passed 12 of 14 cases and failed two float16 cases (89 assertions). Replacing the float16 conversion round trip with `packHalf2x16` / `unpackHalf2x16` made all 14 cases and 3,000 assertions pass, without weakening comparisons. The responsible driver/compiler transformation is not established.

The combined source build `db10f538dbc90217d9229dc3caad61c80f800465` passed on M1:

- `omarchy_fused_chain_tests`: 14 cases, 3,000 assertions.
- `omarchy_compiled_tape_tests`: 11 cases, 1,747 assertions.
- Normal wheel: `mlx_omarchy-0.32.2.dev202609041714+db10f53-cp314-cp314-linux_aarch64.whl`.
- Wheel SHA-256: `638a0395534dc7c276ba4576085190cdeb8d1434c7a0f36797195018d11314f4`.
- Raw results: `~/benchq/logs/performance-final-db10f53-20260904-batch3/` on the measurement host.

## A discarded performance comparison

The first five paired model runs did not exercise compilation. Their manifest set `MLX_DISABLE_COMPILE=0`, but upstream disables compilation whenever that variable is present. Both profiler streams contained zero tape dispatches. Those files are marked `INACTIVE`; their neutral timings are not evidence for fusion speed.

With the variable entirely unset, a bounded compiled-chain probe recorded 170 dispatches with fusion off versus 120 with it on; tape-attributed dispatches changed from 150 to 100. This confirms engagement in that probe, not a model-level reduction.

## Genuine model comparison

Five alternating OFF/ON pairs used the normal `db10f538dbc90217d9229dc3caad61c80f800465` wheel, with no runtime profiler and `MLX_DISABLE_COMPILE` absent. The Qwen2.5-0.5B 4-bit model revision was `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`. Generated token IDs matched in every pair.

| Prompt / generated tokens | Decode tok/s OFF | Decode tok/s ON |
|---|---:|---:|
| 30 / 32 | 31.48 | 31.58 |
| 262 / 128 | 29.04 | 29.19 |
| 1053 / 32 | 24.32 | 24.37 |

These small median differences do not establish a speedup. Fusion remains disabled by default.

A separate diagnostics build profiled the actual model, not the synthetic chain. Across 127 decode intervals, both settings recorded exactly 585 dispatches and 72 tape-path dispatches per interval. OFF used `ElementwiseF16` for those 72; ON replaced them one-for-one with `FusedChainF16`. Thus the real path selected the new kernel but did not combine its dispatches. The synthetic probe cannot stand in for this result.

The OFF profile attributed 29.5% of GPU time to `QmmVecSubgroupF16`, 24.1% to `ElementwiseF16`, and 9.2% to `MatmulF16`. Instrumented GPU busy time was 9.10% of the measured span; profiling overhead makes that unsuitable as an uninstrumented utilization claim. Tapeless eager and bf16 runs did not emit usable GPU streams on this diagnostics build; those missing measurements are not zeros.

Raw timing and profile results: `~/benchq/logs/performance-final-db10f53-20260904-batch4/` on the measurement host. Local copies: `/tmp/performance-final-a8f243b-batch5/genuine-summary.json`, `genuine-pair{1..5}-{off,on}.json`, `census-q4-long128-{off,on}-compilon.txt`, and `analyze-q4-long128-off-compilon.txt`.

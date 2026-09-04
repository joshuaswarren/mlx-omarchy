# Compiled elementwise fusion: correctness passed, performance gate pending

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

With the variable entirely unset, a bounded compiled-chain probe recorded 170 dispatches with fusion off versus 120 with it on; tape-attributed dispatches changed from 150 to 100. This confirms engagement, not a model-level speedup. Corrected paired model measurements and profiles are required before enabling fusion by default.

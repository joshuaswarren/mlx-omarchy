# M1 prefill tiling and bf16 candidate gates

Measured on 2026-09-04 on Apple M1, Honeykrisp, 16 GiB, AC power. Every measured leg reported a clean contention check and installed binaries matching the wheel. These are development measurements, not results from the published v0.3.5 wheel.

- Source and harness: `a34cdc5a9fa99485b5228be499bd1057e6374e4f`.
- Wheel: `mlx_omarchy-0.32.2.dev202609041547+a34cdc5-cp314-cp314-linux_aarch64.whl`.
- Wheel SHA-256: `c72f45e82e6754a3ff7c93302aa8064178d583cff89ed0068d1a4655d9b77505`.
- [Per-run metrics and binary provenance](2026-09-04-m1-performance-gates.json): 46 measured legs across 21 invocations.
- Raw commands, manifests, and logs remain on the measurement host at `~/benchq/logs/performance-gates-m1-a34cdc5-20260904-batch1/`.

The full default matrix measured both Qwen2.5-0.5B variants at all three default workloads. The optional 7B and 14B models were absent from the local cache and were skipped, not downloaded or counted as passes. Actual first-response prompt counts were 30, 262, and 1053; the earlier reports containing two prompt tokens cannot support prefill-throughput claims.

## Quantized prefill tiling: accepted

Five paired runs alternated OFF/ON order for `MLX_OMARCHY_QMM_TILE`. Compilation was disabled on both sides, and both bf16 candidate flags stayed off. Model revision: `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`. Greedy decoding used temperature 0, seed 0, four warmup tokens, and EOS suppression. Each side used the same wheel and prompts.

| Prompt tokens / generated tokens | Median prefill seconds OFF → ON | Median prefill tok/s OFF → ON | Throughput ratio | Median decode tok/s OFF → ON |
|---|---|---|---|---|
| 30 / 32 | 0.832 → 0.378 | 36.058 → 79.365 | 2.20× | 30.96 → 31.07 |
| 262 / 128 | 7.161 → 1.397 | 36.587 → 187.545 | 5.13× | 28.41 → 28.32 |
| 1053 / 32 | 29.280 → 5.331 | 35.963 → 197.524 | 5.49× | 23.53 → 23.53 |

Ratios divide the two median throughput values; they are not medians of per-pair ratios. Decode excludes the first generated token, which is timed with prefill. No decode improvement is claimed.

All 30 legs produced identical token-ID digests across both sides and all repeats, also matching the default-matrix control:

- Short: `7fd25a869ff21678`.
- Long: `4cc08910089477fd`.
- Longer context: `7da83f06ec9f001d`.

The complete QMM numerical sweep also passed on M1: 2,928 assertions, one case, zero failures (`m1-qmm-full-sweep-f66a144`). Checks reject nonfinite outputs before comparing against storage-rounded references. Together these results support enabling tiled prefill; single-row decode keeps its existing GEMV path.

## bf16 RoPE and SDPA: keep disabled

The kernel-level checks passed on the same combined wheel: `sdpa_equivalence.py --require-gates` reported `ALL PASS`; `omarchy_fast_ops_tests` passed 21 cases and 27,936 assertions. The model-level identity gate did not pass.

Default-on integration smoke: a 2×32 by 3×32 quantized multiplication with
the flag unset selected kernel 220 and returned six exact values of 32 on
llvmpipe. The broader local sweep did not finish: after 358 passing assertions,
its 1023×4864×4864 fp16 case hit the 10-second no-progress watchdog. This is
recorded as a failed local sweep, not a pass; the M1 full-sweep result above
remains the hardware qualification. No watchdog bypass was used.

Model revision: `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`. Compilation was enabled, QMM tiling disabled, and every leg processed the same 262-token prompt and generated 128 tokens. Five baseline runs gave the same digest, also matching the compile-disabled matrix control.

| `MLX_OMARCHY_ROPE_BF16_DIRECT` | `MLX_OMARCHY_SDPA_BF16_FAST` | Generated-ID digest | Decision |
|---|---|---|---|
| 0 | 0 | `def3c2c9b544eadc` | Stable baseline, five runs |
| 1 | 0 | `65daf2395eeb40c3` | Diverged; stop this candidate |
| 0 | 1 | `41215d2b1ff339be` | Diverged; stop this candidate |
| 1 | 1 | `7a407d9ad7ec91c8` | Diverged; stop this candidate |

The 4-bit control retained digest `4cc08910089477fd` with both flags off and on. Neither bf16 flag is enabled by default. No timing claim is made for a candidate that failed identity. This result does not establish the cause of the separate bf16 readiness defect; its synchronization guard also remains in place.

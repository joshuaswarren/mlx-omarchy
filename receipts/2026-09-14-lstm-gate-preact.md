# Gate-preact capture: native gates measured; fused GEMM not reproducible (2026-09-14)
Verdict: **decoder stays unlanded; no live e101 run** — no candidate rounding mode
exists to implement, so the acceptance gate (emission 101, token 7892) was not
re-tested. `vulkan_decoder.py` on origin/main remains the correctly-rounded
contract (`5a163be6…`); `/tmp/vulkan-tdt-142/coreml/` on jwm1 untouched.

## What this closes
The impl2 named operator
`parakeet.tdt.decoder-lstm.gate-preact-bnns-vs-vulkan` asked for the native
gate preacts. Answer, in three parts:

1. **Core ML will not expose the fused preacts.** Four decomposition twins of
   the fused `ios18.lstm` (wide-fp32 bodies: separate-then-bias, packed single
   GEMM, bias-in-GEMM, hidden-first; plus an fp16 per-op control) score
   **0/19200** h and c bit-exact against the fused op, while the twins' own
   preacts match the jwm1 GPU dump on **76717/76800 (99.89%)** fp16 lanes
   (83 lanes ±1 ulp). Plain BNNS GEMM ≈ Vulkan `mx.matmul`; both differ from
   the fused kernel's internal arithmetic, so a cut-open model cannot emit it.
2. **Native gate values (i,f,o,g after σ/tanh, before the products) are
   captured exactly** by probing the fused op itself: per target gate, one
   gate carries the real native-input trace while the other three are
   bias-saturated to known constants; the exposure is read from the directly
   returned cell (`c1 = r16(gate16 · multiplier)` for i/f/g) or the hidden
   product (`h = r16(σ16(z_o) · tanh16(c0))`, cell identity asserted) with
   three multiplier settings per target for candidate intersection.
   Resolution: 76733/76800 lanes (99.91%) resolve to a single exact fp16
   value; anchors (zero-input rows = real bias slice) reproduce LUT[bias]
   everywhere.
3. **The pre-GEMM rounding mode is not reproducible.** Twelve implementable
   accumulation forms (f64 exact, f32 pairwise/sequential, f16
   pairwise/sequential, two-GEMM variants, the shipped decoder's
   3-op Vulkan form, bias-in-GEMM, packed order, hidden-first, split-K4)
   scored against the captured native gates: none reproduces the fused
   kernel. Best form per gate/layer ≤ GPU dump; on the recurrent-bearing
   layer 1n, exact arithmetic matches only 0.1–5% of lanes while the shipped
   GPU preacts match 49–60% — the BNNS fused kernel accumulates in its own
   low-precision pattern (NEON-fp16-fmla-like), further from exact
   arithmetic than our GPU path is.

## Measured native-vs-GPU gate offsets (dense macstudio unaries, all 15
## ran_decoder traces × both layers × 640 lanes; singleton-resolved lanes)
| stage | exact | within ±2 ulp | 3–16 ulp | >16 ulp | unresolved |
| --- | --- | --- | --- | --- | --- |
| L0 i | 6057 (63.3%) | 72.2% | 2001 | 663 | 33 |
| L0 f | 3652 (43.4%) | 54.2% | 2731 | 1122 | 1183 |
| L0 o | 4581 (49.4%) | 59.7% | 2857 | 885 | 320 |
| L0 g | 7838 (81.7%) | 89.5% | 656 | 352 | 1 |
| L1n i | 5724 (59.6%) | 80.9% | 1823 | 11 | 1 |
| L1n f | 4703 (49.0%) | 64.1% | 3373 | 76 | 1 |
| L1n o | 5139 (53.6%) | 73.3% | 2548 | 11 | 4 |
| L1n g | 5349 (55.7%) | 83.8% | 1254 | 301 | 0 |

This confirms impl2's inferred "exact at 0 for a minority, bulk ±1–2 ulp"
and supersedes the inference: the real distribution has **heavy tails**
(layer 0 f alone has 1122 lanes beyond ±16 gate-ulps — the ±16 box scan was
too narrow). The shipped GPU preacts are the best available approximation of
the native kernel among all forms tested.

## Contract correction found by the capture
The v5 unary contract fills 17 ambiguous σ args (raw read NaN) with
correctly-rounded values. At least one fill is wrong: arg 1.09765625
(bits 0x3C64) — the op behaves as **0.74951171875** (captured through three
independent multipliers), not the cr-fill 0.75. `unary_tables2.npz` inherits
this; the captured `native_gates.npz` supersedes the LUT at measured args.

## Probe validity (receipts for the method)
- Fused probe with real decoder weights + native trace inputs reproduces the
  capture bit-exactly: 15/15 cases × (hidden, cell) × both layers —
  28800/28800 lanes (reproduces 2917e650 on macOS 26.6.2, coremltools 9.0,
  MLCPUComputeDevice).
- Anchor: zero-input lanes reduce the whole exposure chain to
  `gate16 = LUT[bias]` — exact on 639/640 checked lanes per gate/layer
  (the 1 lane/gate exemption is the ambiguous-fill set above; L0_i lane 382
  is the falsified fill, now measured).

## Named remaining operator
`parakeet.tdt.decoder-lstm.bnnS-fused-kernel-accumulation` (refined): the
residual is inside the BNNS fused LSTM kernel. Plain-form emulation is
refuted on captured ground truth; closing it needs kernel-level reverse
engineering of the BNNS LSTM filter's accumulation (Espresso/BNNS
disassembly) or accepting the measured gate-space deltas as the parity
bound.

## Artifacts
`receipts/2026-09-14-lstm-gate-preact/`:
`run_gate_preact.py` (v6 decomposition refutation), `run_gate_preact7.py`
(v7 structured exposure capture), `analyze_gate_preact7.py` (candidate
intersection + offset histograms), `modehunt.py` (anchor + 12-form hunt),
`traces.npz` (native inputs/outputs per trace), `native_gates.npz`
(captured gate values + resolution), `analysis_v7.json`, `final_stats.json`,
`modehunt.json`, `results_v6.json`, `results_v7.json`,
`z_A_sep_biafter.npz` (twin preacts vs GPU evidence). Weights and GPU gate
dump are the impl2 artifacts (`2026-09-14-lstm-fused-impl2/`).

# Fused LSTM impl 2: dense exact unaries, all native-input traces (2026-09-14)

Verdict: **decoder not landed** — named stop. Free decode first-divergence
moved back to emission 101 but picks 8029, not the native 7892. Token-exact
E2E not met (not 104/104). Overlay `vulkan_decoder.py` on origin/main stays
the correctly-rounded contract (`5a163be6…`); `/tmp/vulkan-tdt-142/coreml/`
on jwm1 was restored to that exact file after the run.

## What changed vs impl 1

Impl 1 interpolated between knot samples of band-limited exposure fits and
proved only on the reduction-free token-8192 slice (c0 = 0, gates = bias);
live it diverged at emission 23. Impl 2 throws the interpolation away:

1. **Unaries are the dense macstudio direct reads** (`sigma_full.npz`
   40704 args, `tanh_full.npz` 37896 args — every fp16 argument in range),
   shipped as `unary_tables2.npz` fp16-bit-pattern LUTs. No interpolation.
   Fills from the macstudio contract: σ(±0) = 0.5; σ |arg| > 30 saturates
   1.0 / 0.0; the 17 ambiguous σ args fall back to correctly-rounded;
   tanh |arg| > 12 clamps ±(1 − 2⁻¹⁰); tanh(−0) collapses to +0.0.
   Trace gates reach ±36.9, so saturation is load-bearing.
2. **Fit over ALL 15 ran_decoder native-input traces** (both layers,
   19200 lanes per stage). Gate preacts are the jwm1 GPU values
   (`dump_gates.py`, deterministic double-run), layer 1 injected with the
   native layer-0 output (`t????_i1n`).
3. **Cell algebra pinned wider than fp16**:
   `c1 = round16(f16·c0 + i16·tanh16(g))`, single rounding at exposure —
   confirmed by elimination against fp16-pairwise / fp32 / inner-rounded
   variants.
4. **Hidden form pinned**: `h = round16(o16·tanh16(round16(c1)))`. On
   trace 0 layer 0 the rounded-cell form scores 637/640 vs ~400 for
   raw-tanh / wide-c1 variants.

## Host numbers (all 15 traces × 2 layers × 640 lanes)

| stage | fused dense | CR baseline |
| --- | --- | --- |
| next_cell bit-exact | **8082/19200** | 4001/19200 |
| next_hidden bit-exact | **5901/19200** | 3684/19200 |
| next_cell mean_abs | **0.000538** | 0.000731 |
| next_hidden mean_abs | **0.000196** | 0.000221 |
| next_cell max_abs | 0.02753 | 0.02748 |

Trace 0 layer 0 (zero recurrent term): 639/640 cell lanes bit-exact —
the composition, tables and algebra reproduce native exactly there.

## Live run (jwm1-linux)

`flock -x -w 120 /tmp/m1-gpu.lock` (never stolen) +
`PYTHONPATH=/tmp/vulkan-tdt-142 control-venv/bin/python
/tmp/vulkan-tdt-142/diagnose_decision_142.py`, mlx
`0.32.2.dev202609131130+e4d079c`, `Apple M1 (G13G B1)` Honeykrisp,
wall 3.56 s. `vulkan_decoder.py` sha `7a790b37…`, `unary_tables2.npz`
sha `7c72e775…`.

- `free_decode.first_divergence`: emission **101**, actual token 8029 vs
  native 7892 (frame 373, duration 0 both). Same emission and tokens as the
  correctly-rounded contract; decision-142 logit gap unchanged at 0.2734375
  (actual −12.5625 vs native −12.8359).
- `decoder_on_native_inputs`: 0/15 traces cell bit-exact, max cell_abs
  0.0352 (CR: worst 0.0313); trace 0 improves 0.0028 → 0.0023, trace 10
  degrades 0.0313 → 0.0352.
- `joint_on_native_decoder_state` still 16/16 token and duration matches.
- Acceptance "land only if first_divergence is None or emission ≥ 101 with
  token 7892 at 101": **not met** (token at 101 is 8029).

## Named remaining operator

`parakeet.tdt.decoder-lstm.gate-preact-bnns-vs-vulkan`. With unaries now
exact-dense and the algebra pinned, the residual is the gate preacts
themselves: wherever the recurrent term `h0 @ whh` is nonzero, our GPU
preacts differ from native by ±1–2 ulp on ~60% of lanes (offset histogram
at trace 0 layer 1: 264/640 exact, bulk of the rest ±1–2 ulp; only 13/640
unexplained within ±16 ulp). At trace 0 layer 0 (emb row only) our preacts
match native on 637/640. Forms tested and rejected: packed single fp32
matmul (230/640), single vs double bias rounding (230 vs 265),
hidden-first packed order, wide (unrounded) h0 into layer 1, sequential
fp32 and FMA-chain AMX-style accumulation (all 230/640). The BNNS kernel's
accumulation pattern is not reproducible by any of these; closing the hole
needs the native gate preacts (authenticated capture) or a bit-exact BNNS
GEMM emulation.

## Artifacts

`receipts/2026-09-14-lstm-fused-impl2/`:
`dump_gates.py` (gate dump, deterministic), `dump_gates2.py` + 
`gate_dump2.npz` (gate-form comparison evidence), `gate_dump.npz`,
`weights.npz`, `fit_final.py` (fit + table emission), `fit_report2.json`
(per-trace detail), `unary_tables2.npz` `7c72e775…`, `vulkan_decoder.py`
(fused v2) `7a790b37…`, `result2.json` (live run).

resolved_model: this session (`zai/glm-5.3-flash`).

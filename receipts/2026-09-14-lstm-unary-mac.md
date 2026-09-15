# 2026-09-14 Fused-LSTM sigmoid/tanh identified on macstudio (fused ios18.lstm, not the standalone op)

## Verdict

**The capture's fused-LSTM unary is identified, and it is not correctly-rounded.**
Per-gate sigmoid/tanh outputs were extracted *through the fused `ios18.lstm` op*
(no standalone `ios18.sigmoid`/`tanh` op anywhere in the probe), on the exact
1280 reduction-free sigmoid arguments and 640 tanh arguments from
`lstm_unary_fit_args.npz` (sha256 `3bd08730…856`). The decoder package on
macstudio reproduces the original capture host's golden bit-exact (layer-0
`next_cell` **640/640** and `next_hidden` **640/640**, max_abs_diff **0.0**,
repeat-identical, under all four compute-unit settings), so macstudio's fused
LSTM *is* the capture implementation. `MLComputePlan` names the evaluation:
op `ios18.lstm`, preferred device `MLCPUComputeDevice` (supported: CPU+GPU
under `ALL`; CPU only under `CPU_AND_NE`; never NE).

Measured signatures (fp16, versus correctly-rounded fp32→fp16 reference):

| unary | offset vs CR | lanes |
|---|---|---|
| gate sigmoid, 1280 fit args | −1 ulp: 732, −2: 368, −3: 96, −4: 5, **0: 79** (1201/1280 mismatch) |
| gate sigmoid, 1282-lane grid | −1: 506, −2: 488, −3: 214, −4: 1, 0: 73 |
| output tanh, at actual argument | **+1 ulp: 329, +2: 317, +3: 43, 0: 294** (689/983 resolved lanes mismatch) |

Key fixed points: `sigma(0) = 0.5 − 1 ulp = 0.49951171875`; sigmoid hard
ceiling **`1 − 2^-10 = 0.9990234375`** (σ(8) = σ(12) = σ(20) = σ(30) — a clamp,
not saturation); `sigma(-30) = 0.0` exactly; `tanh(0) = +0.0`; tanh saturates
to exactly **1.0** (proved: the c0 = 20 calibration lane reads exactly
`sigma(20)`, and the only fp16 `u` with `round16(σ(20)·u) = σ(20)` is `u = 1.0`).

**Decisive extra finding:** the fused LSTM's internal cell arithmetic is *not*
a chain of fp16-rounded stages — the exposed cell state satisfies
`c1 ≠ round16(fp16(σ(20))·c0)` for 1279/1282 lanes (internal higher-precision
multiply-add, single rounding at exposure). This explains why 4760 fp16 unary
LUT pairs cannot flip the miss: per-op fp16 LUT composition cannot reproduce
the fused pipeline's arithmetic.

## Contract for LstmUnaryFit

Bit-level LUTs of the capture implementation, plus the arithmetic caveat:

- `sigma_args.npz` — fused σ at the 1280 reduction-free arguments
  (`sigma_input_gate` = lanes 0-639, `sigma_output_gate` = 640-1279).
- `sigma_grid.npz` — fused σ over `linspace(−8, 8, 1270)` + specials
  (±12, ±16, ±20, ±25, ±27, ±30).
- `tanh_extracted.npz` / `tanh_raw_runs.npz` — fused tanh at the 640 cell-gate
  args + 20 pads + 622 grid lanes, with raw `h`/`c` runs for re-derivation.
- To reproduce the capture end-to-end, model the gate preact→gate multiply and
  the `c1 = f·c0 + i·g` update at higher-than-fp16 precision with one rounding
  at state exposure; fp16-exact stage composition will not land.

## Method

Probe models are ML Programs (spec 9, `CoreML8` specialization) with a single
`ios18.lstm` (H = 1282, I = 640, W = R = 0, one timestep, `direction=forward`,
`recurrent_activation=sigmoid`, `cell_activation=tanh`, `activation=tanh`,
peephole off), bias layout ifoz with `b_i = −30` (i = 0), `b_f = +20`
(f = 1), `b_g = 0`, and the probe values on `b_o`:

- Sigmoid: `c0 = +20` ⇒ `c1 ≈ 19.98`, tanh(c1) = 1.0 (proved unique), so the
  exposed `h` lane *is* `σ(b_o_lane)` — no inversion, no candidate ambiguity.
  Calibration lanes at `b_o = +20` and `b_o = 0` read σ(20) and σ(0) directly.
- Tanh: `b_o = +20` (multiplier K = σ(20), read exactly) with compensated cell
  inputs, plus a second model with `b_o = 0.75` (multiplier m\* = σ(0.75) =
  0.6787109375, read exactly); per-lane `tanh` is the intersection of the two
  fp16-multiply preimage sets. 983/1282 lanes resolved uniquely, 299 ambiguous
  (small-|t| lanes where the multiply compresses), 1 preimage gap. Because the
  internal c1 is computed above fp16, values are reported at the *actual*
  exposed argument `c` (saved per lane), carrying ±1 ulp candidate uncertainty.

v1 (`results.json`, `run_fused_lstm_unary_probe.py`) first established the
bias discovery, the U20 = 1.0 proof route, the decoder golden reproduction,
and compute plans; v2 (`results_v2.json`, `run_probe2.py`) added the proof,
the σ grid, the m\* second multiplier, and per-lane cell-output verification.

## Host and provenance

- Host: MacStudio.local, Mac13,2, Apple M1 Ultra, macOS 26.6.2 (25G83),
  CoreML framework per `results*.json`, coremltools 9.0, Python 3.11 venv.
- Fixture: `lstm_unary_fit_args.npz` sha256
  `3bd0873054a0918d4f234b571b6bba6603ed999e46b9a09aa4a7f216e7725856`
  (from LstmUnaryFit; args are layer-0 transition-0 gate biases — token 8192,
  zero embedding row, zero entry state — so fused gate preacts equal bias bits).
- Decoder package: `~/.cache/mlx-omarchy/parakeet-reference/tdt-staging/
  b650695c-75aec2a-ane.dpiNlU/models/decoder.mlpackage` (sha256 of
  model.mlmodel and weight.bin recorded in `results.json`).
- Compute plan (decoder, all four settings): both `ios18.lstm` ops preferred
  `MLCPUComputeDevice`; probe models identical (`plan_cpu_only` in
  `results_v2.json`).

## Artifacts

- This directory:
  `results.json` (v1), `results_v2.json`, `postprocess_summary.json`,
  `sigma_args.npz` `4a0111ef…`, `sigma_grid.npz` `32dc750b…`,
  `tanh_extracted.npz` `2dee046d…`, `tanh_raw_runs.npz` `d03a952c…`,
  `run_fused_lstm_unary_probe.py`, `run_probe2.py`.
  JSON sha256: v1 `264e5d1f…`, v2 `62756fe0…`.
- Remote working copy: macstudio `/private/tmp/mac-activation-capture/`
  (`out/`, `out2/`, `probe.log`, `probe2.log`).

## Caveats

- Tanh values are exact only where the candidate intersection is unique and
  the product semantics hold; 299 lanes are ambiguous and listed in
  `results_v2.json` (`tanh_extraction.ambiguous`).
- The sigmoid direct reads have no such ambiguity (single rounding at output,
  multiplier proven exactly 1.0).
- Negative-zero sign is not observable through the fused path: the
  `f·c0 + i·g` add collapses `−0.0` to `+0.0` before the output tanh.

## Cross-check delta (2026-09-14, LstmUnaryFit)

LstmUnaryFit independently reproduced the `sigma_args.npz` ulp histogram
({-4: 5, -3: 96, -2: 368, -1: 732, 0: 79}) from a fresh macstudio pull and
flagged a conflict: at the stages/e101 arc-consistency "pin" arguments, the
constraint-solve "pins" disagree with these direct fused reads —
`sigma(0.314453125)`: direct 0.5771484375 (−2 ulp vs CR) vs solved 0.577636719
(−1); `sigma(0.56640625)`: direct = CR (0 ulp) vs solved +1. Since the direct
reads provably reproduce the capture golden 640/640 on all four unit
settings, the stages "pins" were artifacts of the fp16-product constraint
model, not native sigmoid values. Do not re-derive unary pins from that
constraint graph; the fp16-unary + fp16-algebra composition it assumes is
not what the fused LSTM computes (see the cell-arithmetic finding above).
LstmUnaryFit receipt 99d418ca remains accurate for what it measured (the
fp16-catalogue space cannot flip e101).

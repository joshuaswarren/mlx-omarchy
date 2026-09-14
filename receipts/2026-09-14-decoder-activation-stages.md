# 2026-09-14 Parakeet decoder activations: the staged form, tested

Status: the `sigmoid` half is identified far more strongly than before and the
`tanh` half is refuted far more strongly, and still nothing is bit-exact, so no
implementation change lands.

- Still open: `parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`.
- Newly runnable: Apple's own fp16 lookup tables for `sigmoid`, `tanh` and `exp`,
  decoded out of the checked-in mil-hwx-compiler oracle records and evaluated as
  candidate activations. The hardware's table is now tested rather than guessed at.
- Newly refuted, each on its own evidence: linear interpolation in those tables,
  the ANE activation exponential that divides by eight and squares three times, the
  stable positive and negative `sigmoid` halves, and every form that builds `tanh`
  as `1 - 2 * reciprocal` or `2 * sigmoid - 1` with an fp16 intermediate.
- Newly measured: 949 native `tanh` values, where the previous receipt pinned 8.
  The magnitude-upward bias it reported from those 8 pins survives at
  `+0.133 ± 0.016` fp16 ulps over 949 arguments, and it is demonstrably not
  constant in the argument.
- Best contract found: unchanged. `sigmoid` as the fp16 chain `1 / (1 + exp(-x))`
  paired with a `tanh` indistinguishable from correct rounding, 986 of 1280 lanes.
  The `sigmoid` side is now refuted on 15 of 1198 arguments rather than 2 of 42
  groups, and nothing reaches 0.

Not Phase 7. No ANE execute, no GPU, no device execution of any kind. No CPU
tensor fallback in the shipped decoder: nothing in the shipped path changed.

## Execution

Host-only arithmetic study. Every native value comes from the authenticated
capture; every candidate is NumPy at fp64 with explicit fp16 and fp32 rounding
stages, so each candidate names one exact arithmetic contract.

- host: `omp-studio-local`, kernel `6.17.2-1-pve`
- python `3.11.2`, numpy `1.26.4`
- no MLX import, no GPU, no ANE, no `accel0` open
- `/tmp/m1-gpu.lock` never taken, never unlinked, and jwm1 never contacted.
  `ExportGeometryFix` and `ParakeetE2E` hold that queue; this study needs no device.
- no file under `ane-linux-experiments` was edited or executed. Its `THEORY.MD` is
  read as the source of the staged-form hypothesis and nothing else.

## Sources

Pinned worktree `CoremlTdtStream` at `1f8804cd087e60b7ab34a2a730510977f5aca0c3`,
the commit the decision-142 receipt pins, unchanged from the previous receipt.

| file | sha256 |
| --- | --- |
| `overlay/tools/coreml/vulkan_decoder.py` | `5a163be6f78b1e1314040175ac768a4dbf3c270a0a0fa5b2fe5c88595fd46fb4` |
| `overlay/tools/coreml/pinned_component.py` | `25730aa8de275e3431112f2f205b2dde75b87f873210ff0ef6c1eafae8f6ea20` |
| `receipts/2026-09-14-decoder-activations/probe.py` | `617cf0e291cd980bea25586933c45d66d2d80b5317f4ccfc2605b82934e4cb0d` |
| `receipts/2026-09-14-decoder-activations/result.json` | `89baf47e12c3460de30c053f2ef9e65330376f4759a508b252feef2080bef0b1` |
| `receipts/2026-09-14-decoder-activation-stages/stages.py` | `f89d0184b04f828821812c1b7eb21d0d480b49cb42aa4e252e61042fd655b01a` |
| `receipts/2026-09-14-decoder-activation-stages/result.json` | `36f3853b1c4657621fb23e6a1d2e319afdfe418abd9cfecc82675661653b2e75` |

`stages.py` imports the previous receipt's `probe.py` for the capture loader and
its contract catalogue, so the two receipts share one loader and one baseline.
The 11 `sigmoid` and 8 `tanh` pins are read out of the previous `result.json`
above rather than re-derived, and the file is hashed at read time.

Capture: authenticated `20260913T105550Z-librispeech-tdt-tensors`, `ane` leg,
`tdt_tensors.json` sha256
`6eaf2bfe801584af570cd66f42e8dc17efd67776fb181f691585d2bcab590e0a`, the hash
every earlier native comparison pins. Model revision
`b650695c2322ee5281dff48d7345b2f3a58ff018`. 15 comparable transitions.

Tables: mil-hwx-compiler `main` at `29318129810aad851bcc0e35764b65dde9b2dd69`,
records under `research/oracles`, campaign source commit
`9483fe699b9416a2bd0537606b36f5b5014638b9`, produced on `MacStudio.local` by
Apple's `ane-compile-hwx`. Both targets are read and the H13/H14 prefix identity
is checked rather than assumed.

| record | sha256 | 128-byte prefix sha256 |
| --- | --- | --- |
| `h14/unary_sigmoid_c64.json` | `91a7eb6e954b576bdefa3f7ff1d2e7cfb8af15cfe2cd8c9d2e416b6da55f72f4` | `73f5680aa5f7b3833479e0ecd5a9dd0e3ec221e3aad5170e9db1e40e7a0c7469` |
| `h13/unary_sigmoid_c64.json` | `cc931d42ac06c72457549cb193bf9a03f5452afc1b5c4d3c3f83bef0d0dc9a42` | same |
| `h14/unary_tanh_c64.json` | `9fbff813e26c00ae680ec55f9ef71d7ce235acfb7a1fc6de683a3e0e14dd07ce` | `f4a38468b2a29430c8ffa4bb04cef84b8347394bbe2d6eb1081093ed4eaa57c6` |
| `h13/unary_tanh_c64.json` | `8c6492e0e5929422ef9513b445ad265f54fedd514fa8e71b1e3dae9be5f1448d` | same |
| `h14/unary_exp_c64.json` | `c79e64af497492f0d2ebde56bb523c856bbf881fbc06382bbe387e6598c6a278` | `b7b6085a1edc7def0f0bb2fc1fe345f1ba9d9a1a47e55b4516273149323b54d2` |
| `h13/unary_exp_c64.json` | `7421bc14106b66e0c28d2ee4fc9eb4dcbb425bf276c7d1734ce64c954127fe28` | same |

## Command

```text
python3 receipts/2026-09-14-decoder-activation-stages/stages.py \
  --capture-dir ~/.cache/mlx-omarchy/parakeet-reference/captures/\
b650695c-75aec2a/20260913T105550Z-librispeech-tdt-tensors/ane \
  --package ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/\
parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018/decoder.mlpackage \
  --overlay <CoremlTdtStream>/overlay \
  --oracles ~/src/mil-hwx-compiler/research/oracles \
  --output receipts/2026-09-14-decoder-activation-stages/result.json
```

Wall 7.2 s, and the JSON output is byte-identical across reruns.

## The hardware table, decoded

Each of the seven nonlinear unary operations carries a 128-byte fp16 block, which
this study reads as four header words followed by 33 knots. The header is an input
clamp with its two saturated outputs, and the knots sit on a uniform grid:

| operation | clamp low | clamp high | output below | output above | knot grid |
| --- | --- | --- | --- | --- | --- |
| `sigmoid` | -9.9375 | 8.3203125 | 0.0 | 1.0 | `x = -8 + 0.5k`, k = 0..32 |
| `tanh` | 0.0 | 4.0 | 0.0 | 1.0 | `|x| = 0.125k`, k = 0..32 |
| `exp` | -25.0 | 16.0 | 0.0 | +inf | `2**(k/32)`, k = 0..32 |

Three of those readings are self-checking. The `sigmoid` clamp high, 8.3203125, is
the fp16 number nearest the argument above which `sigmoid` rounds to exactly 1.0,
which is what the header's fourth word says it emits there. The `tanh` clamp low of
0.0 with a magnitude grid is the signature of a magnitude-and-sign evaluation. The
`exp` table stores `2**f` on 32 intervals of 1/32 rather than `exp(x)`, so `exp` is
an exponent split with a mantissa table, and its clamps sit outside the range where
fp16 `exp` is finite and nonzero, which spans -17.3287 to 11.0901.

All 99 knots of the three tables are the exact fp16 rounding of their function at
their grid point, with zero differing. The `silu` and `gelu` tables are the control
and they are not: 24 of 33 and 25 of 33 knots differ from the exact samples. So the
block is a piecewise-linear approximation contract whose knots are sometimes fitted,
and the three tables this study needs happen to be exact samples. That is measured
here, not assumed.

## What the table refutes

Linear interpolation in those tables, with the header's clamps applied and the
interpolation rounded at fp16 or fp32, is not what the decoder's LSTM evaluates.
It is not close:

| contract | pins | refuting arguments | lanes of 1280 |
| --- | --- | --- | --- |
| `sigmoid` table, linear interpolation | 0 of 11 | 172 of 1198 | 45 with the `tanh` table |
| `tanh` table, linear interpolation | 1 of 8 | 196 of 1172 | 105 with the chain `sigmoid` |

The single `tanh` pin it reproduces is a coincidence, since the same contract misses
the other seven and is refuted on 196 arguments. The reason is arithmetic rather
than mysterious: the 0.5-wide `sigmoid` knot spacing carries up to 9.947 fp16 ulps
of interpolation error for arguments inside [-1, 1], where these lanes live, and the
native values are within two ulps of correct rounding everywhere. The compiler emits
that table for a standalone elementwise `sigmoid` op; the recurrent `lstm` op is a
different lowering, and this is direct evidence that it does not reuse the unary
table.

The `exp` table is a different story, and it is the one that survives. Substituted
into the chain, `1 / (1 + exp_table(-x))`, it reproduces all 11 pinned `sigmoid`
values and earns 950 of 1280 lanes. It is worse than an `exp` accurate to fp16, 29
refuting arguments against 15, so the `exp` stage inside the native `sigmoid` is
more accurate than a 32-interval linear interpolation of `2**f`. It is not refuted
as the seed of a more accurate sequence.

## The staged forms from THEORY.MD, tested

`ane-linux-experiments` `THEORY.MD` records three staged behaviours for this family:
activation exponentials divide by eight and square three times, `sigmoid` and `silu`
use stable positive and negative halves, and the reciprocal is a Newton iteration
from a small seed. Each is built here with an explicit rounding width per stage.

| `sigmoid` contract | pins | refuting arguments of 1198 | lanes of 1280 |
| --- | --- | --- | --- |
| fp16 chain `1 / (1 + exp(-x))` | 11 of 11 | 15 | 986 |
| same, Newton reciprocal at fp32, 12 steps from 1/128 | 11 of 11 | 15 | 986 |
| same, `exp` from the hardware `2**f` table | 11 of 11 | 29 | 950 |
| same, stable positive and negative halves | 10 of 11 | 43 | 865 |
| correctly rounded, the shipped form | 9 of 11 | 67 | 816 |
| same chain, Newton reciprocal at fp16 | 8 of 11 | 76 | 750 |
| divide by eight, square three times | 4 of 11 | 110 | 534 |
| `sigmoid` table, linear interpolation | 0 of 11 | 172 | 45 |

The activation exponential that divides by eight and squares three times is refuted
outright: three fp16 squarings multiply the seed's relative error by eight, and the
result misses seven of the eleven pins and is refuted on 110 arguments. The stable
halves are refuted more narrowly, on the one negative pin and on 43 arguments
against the plain form's 15, so the native `sigmoid` evaluates one expression over
the whole line rather than switching at zero. The Newton reciprocal splits on its
width. At fp16 it is refuted: twelve steps from 1/128 differ from a divide on 391 of
the 1280 denominators and it is refuted on 76 arguments. At fp32 it is bit-identical
to the fp16 divide on every one of the 1280 lanes, so this capture cannot separate a
divide from a converged Newton iteration, only from a Newton iteration carried at
fp16.

The refutation column is the instrument that makes those comparisons sharp, and it
is new. Fix a candidate on one side, then solve each lane for the other activation's
fp16 value: the lane says the product rounds to the captured number, which is an
interval, and the fp16 numbers inside that interval are the feasible partner values.
Intersect over the lanes an argument appears on. An empty set refutes the candidate
assuming only that the partner is a deterministic fp16 function and that the lane is
one fp16 multiply, with no partner contract and no repeated arguments needed. That
scores 1198 or 1172 arguments where the previous receipt's group test scored 42, and
it agrees with it: the forms that receipt refuted are refuted here, with two orders
of magnitude more counterexamples.

Zero refuting arguments is necessary, not sufficient, and no contract on either side
reaches zero.

## The tanh axis does not move

| `tanh` contract | pins | refuting arguments of 1172 |
| --- | --- | --- |
| correctly rounded | 7 of 8 | 49 |
| `(e - 1) / (e + 1)`, fp32 `exp`, fp16 out | 7 of 8 | 49 |
| `(e - 1) * reciprocal(e + 1)`, magnitude and sign | 8 of 8 | 83 |
| `tanh:mulrcppos:321616`, the previous receipt's 8-pin form | 8 of 8 | 100 |
| `(e - 1) / (e + 1)` entirely in fp16 | 6 of 8 | 125 |
| `tanh` table, linear interpolation | 1 of 8 | 196 |

Of 911 `tanh` contracts, 82 tie at the minimum of 49 and none beats it. Every one of
those 82 is a form whose error stays inside correct rounding on these arguments, so
the axis is flat for the same reason the previous receipt found it flat, now measured
against 1172 arguments.

That also settles the previous receipt's honest caveat about its 8-pin `tanh`. The
form reproducing all 8 is refuted on 100 arguments where correct rounding is refuted
on 49, and across the 15 transitions it is not even numerically viable: `exp(2x)`
overflows fp16 in the subtraction, and 22370 of its 38400 lanes are non-finite, so
its worst case is not a number at all. Eight pins were too few, and this is the
measurement that shows it.

## 949 native tanh values, conditional on the sigmoid

Holding the leading `sigmoid` contract fixed turns almost every lane into a solved
equation for one native `tanh` value. Of 1198 distinct `tanh` arguments, 15 have no
feasible fp16 value at all, which is the refutation above, and 949 are determined
uniquely. That is 949 measurements where the partner-free method pinned 8.

The measurement is conditional, and only in one specific way: the level of the error
curve is degenerate with the `sigmoid`'s own mean bias, exactly as the previous
receipt's product decomposition was, because each lane sees only the sum of the two.
The shape is not degenerate, because each `tanh` argument pairs with an unrelated
`sigmoid` argument, so `sigmoid` error enters as scatter rather than as a function
of the `tanh` argument.

Signed offsets from correct rounding, over the 949:

| offset in fp16 ulps | -3 | -2 | -1 | 0 | +1 | +2 | +3 | +4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arguments | 1 | 3 | 124 | 703 | 107 | 9 | 1 | 1 |

Read as magnitude rather than sign, 94 of the 124 at -1 are negative arguments and
87 of the 107 at +1 are positive, so the deviation is upward in magnitude. The
mean is `+0.133 ± 0.016` fp16 ulps, 8.3 sigma, which lands inside the `+0.124` to
`+0.267` window the previous receipt derived from 8 pins. That window is now
confirmed by a measurement 118 times larger and it is the one claim from the
previous receipt that this study strengthens rather than sharpens or refutes.

It is not a constant. Binned on `|x|` in steps of 0.0625, the magnitude bias runs
from `-0.127 ± 0.054` ulps below 0.0625, through `+0.362 ± 0.048` on
`[0.3125, 0.375)`, back to `+0.019 ± 0.132` on `[0.75, 0.8125)`. The spread across
bins is several times the per-bin standard error, so the native `tanh` error varies
with its argument, with no period matching the 0.125 table grid. A single magnitude
offset therefore describes the average and not the function, which is why the
previous receipt's fitted `+0.20` ulp correction recovered 1024 lanes and not 1280.

Against those 949 measured values, the best agreement any of the 911 `tanh`
contracts reaches is 703, which is exactly what correct rounding reaches. The
`tanh` evaluation remains unidentified, and it is now unidentified against 949
measured values rather than 8.

## Two structural refutations that need no formula

A measured value is enough to kill a whole family when that family's output lands on
a coarser grid than fp16. Where `|tanh|` is in `[0.25, 0.5)`, a result computed as
`1 - 2r` with an fp16 reciprocal `r` can only land on every other fp16 number, and a
result computed as `2s - 1` with an fp16 `s` on every fourth.

| family | measured values on its grid, of 291 |
| --- | --- |
| `1 - 2 * reciprocal(e + 1)` | 151 |
| `2 * sigmoid(2x) - 1` | 73 |
| control: the value is fp16 | 291 |

Both are consistent with chance, 50% and 25%, so both families are refuted with 140
and 218 explicit counterexamples. The native `tanh` is the fp16 rounding of a value
computed at more than fp16 precision, not a cancellation against an fp16 intermediate.

The same style of test settles which argument feeds the second `tanh`. Scored on the
640 hidden lanes with the leading `sigmoid`:

| `tanh` argument | hidden lanes of 640 |
| --- | --- |
| the captured fp16 cell | 486 |
| the unrounded internal cell | 362 |
| the internal cell at fp32 | 362 |
| the internal cell rounded to fp16 | 409 |

So the cell is rounded to fp16 before it is fed to `tanh`, and the captured tensor is
that value rather than a rounded copy of a wider one. Together with the previous
receipt's finding that rounding each activation to fp16 before the product beats
carrying them wide, the native shape is settled: two fp16 activations, one fp16
multiply, and an fp16 cell that is itself the next `tanh`'s argument.

## Whether it carries across a reduction

The other 14 transitions need the gate reduction, which is the other open hole, so
these counts are not a clean activation measurement. With the fp32-class reduction
the previous study measured as least-bad:

| variant | `next_cell` of 19200 | `next_hidden` of 19200 |
| --- | --- | --- |
| shipped pair | 3369 | 3185 |
| fp16 chain `sigmoid`, correctly rounded `tanh` | 3821 | 2890 |
| stable halves `sigmoid`, correctly rounded `tanh` | 3740 | 3165 |
| hardware `exp` table chain, correctly rounded `tanh` | 3703 | 2462 |
| divide by eight and square three times | 3300 | 1961 |
| both unary tables, linear interpolation | 1716 | 842 |
| chain with the fp32 Newton reciprocal | 1136 | 780 |
| chain with the fp16 Newton reciprocal | 1021 | 581 |

The first two rows reproduce the previous receipt exactly. Two rows are worth
naming. The stable-halves form is refuted on the reduction-free evidence, on a pin
and on 43 arguments, yet it scores better than the plain chain once a reduction is
in front of it, 3740 and 3165 against 3821 and 2890. And the fp32 Newton
reciprocal, bit-identical to the divide on all 1280 reduction-free lanes, collapses
to 1136 and 780 here, because twelve steps from a 1/128 seed cannot reach the much
larger denominators these transitions produce. The first is the reduction choosing a
different activation and the second is an argument range the reduction-free
transition never exercises. Both are reasons these numbers are not evidence for or
against the elementwise result.

## Remaining hole

`parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`.

What is now established. The native `sigmoid` is a staged chain evaluating one
expression over the whole line, with an `exp` accurate to fp16 and a rounding at both
the add and the reciprocal; it reproduces all 11 pins, is the unique minimum of the
refutation metric at 15 of 1198 arguments against correct rounding's 67, and is
still refuted. Apple's unary lookup table is decoded, runnable, and refuted for this
op. The activation exponential that divides by eight and squares three times is
refuted for this op. The native `tanh` carries a magnitude-upward bias of
`+0.133 ± 0.016` ulps that varies with its argument, and no form out of 911 matches
its 949 measured values better than correct rounding does.

What is still missing is the `tanh` evaluation itself, and the shape of the gap has
changed. It is no longer argument-starved: 949 measured values are enough to
identify a form, and 911 forms fail to be it. It is now form-starved, and the two
inputs that would close it are a single-operator authenticated capture of fp16
`tanh` over an argument sweep, or the compiled compute plan for the `lstm` op naming
its lowering. The second is the more valuable of the two, because the same plan
would name whether the `sigmoid` chain's reciprocal is a divide or a Newton
iteration converged at more than fp16 width, which this capture cannot separate.

Since no variant is bit-exact on 1280 lanes, out of 153,959 combinations of 169
`sigmoid` and 911 `tanh` contracts, `_lstm` is unchanged per the change gate. No
test accompanies this receipt because no shipped code changed; `stages.py` is the
runnable check and regenerates every number above.

One numerical note carried forward. In the fp16 chain, `exp(-x)` saturates to fp16
infinity once `-x` reaches ln(65520), so for x below -11.0901, and that yields
`sigmoid` 0, which is the correct fp16 result. It is part of the contract's
definition; the overflow warning is suppressed and no candidate lane is non-finite
except in the `mulrcp` `tanh` forms noted above, where the overflow is the
refutation rather than an artifact.

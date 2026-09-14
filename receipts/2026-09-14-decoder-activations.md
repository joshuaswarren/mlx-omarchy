# 2026-09-14 Parakeet decoder elementwise activations

Status: the named hole narrows sharply and does not close. No implementation
change lands, because no candidate activation pair is bit-exact on the captured
lanes.

- Still open: `parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`,
  now with both halves separated and measured rather than only localized.
- Newly established, without assuming anything about the other activation: the
  native `sigmoid` is not correctly rounded, and neither is the native `tanh`.
  Both are refuted on arguments that repeat, where the shared activation value
  cancels exactly.
- Newly measured: 11 native `sigmoid` values and 8 native `tanh` values, read
  out of the capture at named arguments. The previous receipt held that per-lane
  evidence of the native elementwise functions was unobtainable from this
  capture. It is obtainable, in small quantity, and it is below.
- Best contract found: `sigmoid` as a full fp16 chain `1 / (1 + exp(-x))`, which
  reproduces every pinned native `sigmoid` value and lifts the reduction-free
  lanes from 816 to 986 of 1280. It is still refuted, on 2 groups of 42.

Not Phase 7. No ANE encoder execute. No device execution of any kind. No CPU
tensor fallback in the shipped decoder: nothing in the shipped path changed.

## Execution

Host-only arithmetic study. Every native value comes from the authenticated
capture; every candidate value is NumPy at fp64 with explicit fp16 and fp32
rounding stages, so each candidate names one exact arithmetic contract instead
of a backend's incidental behaviour.

- host: `omp-studio-local`, kernel `6.17.2-1-pve`
- python `3.11.2`, numpy `1.26.4`
- no MLX import, no GPU, no ANE, no `accel0` open
- `/tmp/m1-gpu.lock` never taken, never unlinked, and jwm1 never contacted.
  `DecoderProjectorKernel` holds that GPU for the projector kernel and
  `MesaWaitsPatch` is sequencing behind it; both asked, both were told this
  study needs no device.

Inputs are injected per transition from the capture, so nothing accumulates
across transitions and every number below is a single-step contract test.

## Sources

Pinned worktree `CoremlTdtStream` at `1f8804cd087e60b7ab34a2a730510977f5aca0c3`,
the commit the decision-142 receipt pins.

| file | sha256 |
| --- | --- |
| `overlay/tools/coreml/vulkan_decoder.py` | `5a163be6f78b1e1314040175ac768a4dbf3c270a0a0fa5b2fe5c88595fd46fb4` |
| `overlay/tools/coreml/pinned_component.py` | `25730aa8de275e3431112f2f205b2dde75b87f873210ff0ef6c1eafae8f6ea20` |
| `receipts/2026-09-14-decoder-activations/probe.py` | `617cf0e291cd980bea25586933c45d66d2d80b5317f4ccfc2605b82934e4cb0d` |
| `receipts/2026-09-14-decoder-activations/result.json` | `89baf47e12c3460de30c053f2ef9e65330376f4759a508b252feef2080bef0b1` |

Capture: authenticated `20260913T105550Z-librispeech-tdt-tensors`, `ane` leg,
`tdt_tensors.json` sha256
`6eaf2bfe801584af570cd66f42e8dc17efd67776fb181f691585d2bcab590e0a`, the hash the
earlier native comparisons pin. Model revision
`b650695c2322ee5281dff48d7345b2f3a58ff018`. 16 traces, of which trace 1 is a
decoder-reuse step with no recurrent transition, leaving 15 comparable
transitions: 0 and 2 through 15.

## Command

```text
python3 receipts/2026-09-14-decoder-activations/probe.py \
  --capture-dir ~/.cache/mlx-omarchy/parakeet-reference/captures/\
b650695c-75aec2a/20260913T105550Z-librispeech-tdt-tensors/ane \
  --package ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/\
parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018/decoder.mlpackage \
  --overlay <CoremlTdtStream>/overlay \
  --output receipts/2026-09-14-decoder-activations/result.json
```

Wall 5.9 s.

## The declared contract, and the reproduction check

Both `lstm` ops in the package declare `recurrent_activation = "sigmoid"`,
`cell_activation = "tanh"` and `activation = "tanh"`, with no cell clip and no
peephole. So the divergence is in how those two functions are evaluated, not in
which functions they are.

The probe reproduces the pinned localization before testing anything. The shipped
`_lstm` semantics, `sigmoid` and `tanh` each correctly rounded to fp16, give 413
of 640 bit-exact cell lanes on transition 0 with max abs 0.0009765625 and the
signed fp16 ulp histogram `{-3: 1, -2: 21, -1: 91, 0: 413, +1: 91, +2: 23}`;
across all 15 transitions with the fp32-class gate reduction they give 3369 and
3185 of 19200. Those are the previous receipt's numbers, unchanged.

## 1280 lanes of pure elementwise evidence, not 640

Transition 0 decodes token 8192, the only identically-zero embedding row, and its
entry hidden and entry cell are zero. Layer 0 therefore carries no reduction: the
gates equal the pinned fp16 bias bit for bit, and both layer-0 outputs are
elementwise,

```text
next_cell   = sigmoid(bias_i) * tanh(bias_c)
next_hidden = sigmoid(bias_o) * tanh(next_cell)
```

with `sigmoid(bias_f) * 0` vanishing exactly. The previous study used the first
of these. The second is equally reduction-free and was unused, so the evidence
is 1280 lanes, not 640. It is not a restatement of the first: it probes `sigmoid`
at the output-gate bias and `tanh` at the native cell, and it is the independent
check on every result below. The pinned baseline scores 403 of 640 there.

## The native activations are not correctly rounded, each shown alone

A single lane is one product of two unknown function values, so it determines
neither. Arguments that repeat break that. Both layer-0 equations read the same
gate vector, so all lanes sharing a `sigmoid` argument share one native `sigmoid`
value. The input-gate and output-gate slots share 46 arguments outright, and 34
more repeat inside the input-gate slot, which together with the output-gate slot's
own repeats gives 101 groups over 209 lanes. A group is then a constraint on the
*other* activation with no assumption about this one:
intersect, over the group, the set of values the shared activation could take,
and an empty intersection refutes the contract outright.

| test | contract | groups with no real value | groups with no fp16 value |
| --- | --- | --- | --- |
| sigmoid-free, 101 groups | `tanh` correctly rounded | 17 | 25 |
| tanh-free, 42 groups | `sigmoid` correctly rounded | 12 | 18 |
| tanh-free, 42 groups | `sigmoid` as an fp16 chain | 2 | 4 |

Correct rounding is refuted on both sides. The first column assumes only that the
activation is a deterministic function of its argument; the second adds that it
emits fp16. The `tanh` groups counted here are same-op only, 42 of them over 85
lanes: the two `tanh` instances are separate ops, so the 36 further groups that
would pair `cell_activation` against `activation` are excluded from the verdict.
Including them changes no verdict.

Every contract in the catalogue is refuted this way: 0 of 268 `tanh` contracts
and 0 of 38 `sigmoid` contracts survive. The fp16 chain is not right either, but
it fails 2 groups where correct rounding fails 12.

## Measured native values

The 1280 lanes form a bipartite constraint graph: nodes are distinct arguments,
edges are lanes, and each lane says that the product of its two node values
rounds to the captured fp16 number. Interval narrowing is useless here, because
one lane pins a product to half an ulp while each factor is unknown by more than
that, so the domains are kept discrete: each node ranges over the fp16 numbers
within a radius of correct rounding, and arc consistency deletes any value with
no partner support. A value deleted this way cannot appear in any solution, so a
node reduced to one value is a native activation value, read out of the capture.

Radius 1 is refuted: one lane cannot be satisfied, and five constraints cannot
once ordering is included. Radius 2 and radius 3 are both fully satisfiable, and
only values pinned identically under every satisfiable radius are reported. That
alone pins 2 `sigmoid` and 1 `tanh` value. Adding the
structural assumption that each native activation is non-decreasing in its
argument chains all arguments together instead of leaving most nodes with a
single lane, and pins 11 and 8.

| argument | native sigmoid | correctly rounded | ulps |
| --- | --- | --- | --- |
| -0.288086 | 0.428466797 | 0.428466797 | 0 |
| +0.090027 | 0.522460938 | 0.522460938 | 0 |
| +0.154419 | 0.538574219 | 0.538574219 | 0 |
| +0.154663 | 0.538574219 | 0.538574219 | 0 |
| +0.261230 | 0.564941406 | 0.564941406 | 0 |
| **+0.314453** | **0.577636719** | **0.578125000** | **-1** |
| +0.438965 | 0.607910156 | 0.607910156 | 0 |
| +0.529785 | 0.629394531 | 0.629394531 | 0 |
| **+0.566406** | **0.638183594** | **0.637695312** | **+1** |
| +0.566895 | 0.638183594 | 0.638183594 | 0 |
| +0.567871 | 0.638183594 | 0.638183594 | 0 |

| argument | native tanh | correctly rounded | ulps |
| --- | --- | --- | --- |
| -1.099609 | -0.800292969 | -0.800292969 | 0 |
| -0.681641 | -0.592773438 | -0.592773438 | 0 |
| -0.560547 | -0.508300781 | -0.508300781 | 0 |
| -0.113953 | -0.113464355 | -0.113464355 | 0 |
| +0.282715 | 0.275390625 | 0.275390625 | 0 |
| **+0.319092** | **0.308837891** | **0.308593750** | **+1** |
| +0.458984 | 0.429199219 | 0.429199219 | 0 |
| +0.622070 | 0.552734375 | 0.552734375 | 0 |

Those bold rows are the whole hole in three lines. The native `sigmoid` at
+0.314453 is one fp16 ulp below the correctly rounded value and at +0.566406 one
ulp above; the native `tanh` at +0.319092 is one ulp above. The 11 pinned
`sigmoid` values are reproduced exactly by four contracts, every one of them a
staged chain that rounds to fp16 at both the add and the reciprocal, with
`sig:recip:161616` among them and also the best contract on lanes. That is a
coherent identification of the `sigmoid` family and an incoherent one
for `tanh`: `tanh:mulrcppos:321616` reproduces all 8 pinned `tanh` values and
earns only 713 of 1280 lanes against the fp16 chain `sigmoid`, so 8 pins are too
few to identify `tanh`, and the receipt does not claim they do.

## How large the error is, and how it is shaped

A pinned value bounds the native evaluation error, because the emitted fp16
number is the rounding of some internal value. Measured toward larger magnitude,
in fp16 ulps:

| activation | one constant magnitude offset fitting every pin |
| --- | --- |
| sigmoid | none exists |
| tanh | +0.124 to +0.267 ulps |

The `sigmoid` window is empty: +0.314453 needs at least 0.186 ulps below exact
and +0.566406 needs at least 0.012 ulps above, so the native `sigmoid` error
demonstrably changes sign with the argument, which is the signature of rounding
inside a chain and not of a smooth approximation bias. The `tanh` window is not
empty and is narrow, a magnitude-upward bias of a fifth of an ulp.

Independently, over all 640 cell lanes, the native product exceeds exact
`sigmoid * tanh` by a relative `9.548e-05`, standard error `1.736e-05`, so 5.5
sigma, and by `5.039e-05` at 2.6 sigma on the 640 hidden lanes. A binned additive
fit splits the residual into one curve per argument; only the sum of the two
levels is identifiable from a product, but the shapes are not, and the `tanh`
curve's sign follows its argument's sign in 18 of 20 bins, which is a magnitude
bias. The per-lane scatter beyond product rounding is a relative `3.841e-04`,
about 0.38 fp16 ulps per activation if the two are independent and equal.

Applying exactly that bias, exact `tanh` scaled toward larger magnitude and
paired with the fp16 chain `sigmoid`, recovers lanes at the size the pins
predict:

| tanh treatment | lanes of 1280 |
| --- | --- |
| correctly rounded | 986 |
| plus 0.20 magnitude ulps | 1024 |
| times 1 + 1.0e-04 | 1018 |

The peak sits inside the +0.124 to +0.267 window derived from the pins, which is
the cross-check. It is a fitted correction, not a contract, and it is recorded
as a measurement of the bias, not as a candidate implementation.

## Catalogue

10184 combinations: 38 `sigmoid` contracts against 268 `tanh` contracts, each a
named formula with a rounding width per stage. Formulas cover `1/(1+exp(-x))`,
`exp(x)/(1+exp(x))`, `exp2` with log2(e) at fp16, fp32 and fp64 width,
`0.5*tanh(x/2)+0.5`; and for `tanh`, the direct function, `(1-e)/(1+e)`,
`(e-1)/(e+1)`, `1-2/(e+1)`, the magnitude-and-sign variant, multiply-by-
reciprocal variants, the classic `x(27+x^2)/(27+9x^2)` rational, and five ways of
building `tanh` out of each `sigmoid` contract, including `2*sigmoid(2x)-1` and
the odd-by-construction `sigmoid(2x)-sigmoid(-2x)`.

| sigmoid | tanh | cell | hidden | joint of 1280 |
| --- | --- | --- | --- | --- |
| fp16 chain `1/(1+exp(-x))` | correctly rounded | 500 | 486 | 986 |
| fp16 chain | `sigmoid(2x)-sigmoid(-2x)`, fp32 sigmoid | 500 | 486 | 986 |
| fp16 chain | `2*sigmoid(2x)-1`, fp32 sigmoid | 500 | 486 | 986 |
| fp16 chain | `1-2/(e+1)` in fp32, rounded out | 500 | 485 | 985 |
| fp16 `exp2`, fp64 log2(e) | correctly rounded | 492 | 481 | 973 |
| correctly rounded, the shipped pair | correctly rounded | 413 | 403 | 816 |

Nothing is bit-exact on 1280. The `sigmoid` axis moves 170 joint lanes and the
`tanh` axis moves nothing: every `tanh` contract that reaches the top of the
table ties with correct rounding, which is consistent with the pins, where a
sign-changing chain error fits `sigmoid` and a uniform magnitude bias fits
`tanh`.

## Whether the gain survives a reduction

The elementwise finding cannot be tested on the other 14 transitions without a
gate reduction, and the reduction is the other open hole. With the fp32-class
reduction the previous study measured as least-bad:

| variant | `next_cell` of 19200 | `next_cell` max abs | `next_hidden` of 19200 |
| --- | --- | --- | --- |
| shipped pair | 3369 | 0.03125 | 3185 |
| fp16 chain sigmoid, correctly rounded tanh | 3821 | 0.027515411376953125 | 2890 |

Both rows reproduce the previous receipt exactly. The chain buys 452 cell lanes,
cuts the `next_cell` worst case from 0.03125 to 0.0275, and gives back 295 hidden
lanes. That is a trade, and it is a trade measured on top of an unresolved
reduction, so it is not evidence for or against the elementwise result.

## Remaining hole

`parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`.

What is now established: the native `sigmoid` and `tanh` are both within 2 fp16
ulps of correct rounding on all 1280 lanes and neither is correctly rounded;
radius 1 is refuted; the native `sigmoid` error changes sign with its argument
and the leading fp16 chains reproduce all 11 pinned values; the native `tanh`
carries a magnitude-upward bias of +0.124 to +0.267 ulps that no catalogued form
produces.

What is still missing is the `tanh` evaluation itself. Closing it needs either a
single-operator authenticated capture of fp16 `tanh` over an argument sweep, or
the compiled compute plan that names the evaluation. The pinned-value machinery
here is the cheap alternative and it is argument-starved, not method-starved: it
pins 8 `tanh` values because most `tanh` arguments appear on exactly one lane. A
capture with more reduction-free transitions, or any capture whose gate biases
repeat more, would pin proportionally more, and a form that reproduces 50 pinned
values would be an identification rather than a coincidence.

Since no variant is bit-exact, `_lstm` is unchanged, per the change gate. No test
accompanies this receipt because no shipped code changed; `probe.py` is the
runnable check and regenerates every number above.

In the fp16 sigmoid chain, `exp(-x)` saturates to fp16 infinity once `-x` reaches
ln(65520), so for x below -11.0901, and that yields sigmoid 0, which is the
correct fp16 result. It is part of the contract's definition; NumPy's overflow
warning is suppressed in the probe and no candidate lane is non-finite.

## Not claimed

Decision 142 is not closed and no token-exactness is claimed. No device ran:
there is no GPU, ANE, Honeykrisp, or Apple-silicon execution behind any number
here, and no claim about which backend served the capture. The capture requests
`compute_units=ane` and exclusive backend execution remains unestablished, so
"native Core ML" here means the captured backend, whichever unit that was.

The 11 and 8 pinned values are conditional and the conditions are stated: all of
them on the activations emitting fp16 and on lying within 2 ulps of correct
rounding, which radius 2 shows is consistent with all 1280 lanes; and the step
from 3 values to 19 on the activations being non-decreasing in their argument.
The 3 values pinned without monotonicity are listed separately in `result.json`.
No claim that the fp16 chain is the native `sigmoid`: it is refuted on 2 groups,
and reproducing 11 pinned values makes it the leading family, not the answer. No
claim that the +0.2 ulp `tanh` bias is implementable or implemented; it is the
measured size of a bias. No claim about the gate reduction, which stays open.

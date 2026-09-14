# 2026-09-14 Parakeet decoder LSTM semantics

Status: the named hole moves, and it does not close. No implementation change
lands, because no LSTM variant is bit-exact on the captured transitions.

- Relocated: `parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml`
  becomes `parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`.
  The first divergence carries no reduction at all, so no accumulator
  hypothesis can reach it.
- Closed: `parakeet.tdt.decoder-projector.fp16-linear-vs-native-coreml`. The
  native linear is an fp16 accumulator over unrounded products, reduced in
  strictly ascending index order, with the bias added after the reduction.
  That is bit-exact on all 9600 captured projector lanes.

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
- `/tmp/m1-gpu.lock` never taken, never unlinked, and jwm1 never contacted:
  a concurrent sibling holds priority on that GPU and this study needs no
  device

Inputs are injected per transition from the capture, so nothing accumulates
across transitions and every number below is a single-step contract test.

## Sources

Pinned worktree `CoremlTdtStream` at `1f8804cd087e60b7ab34a2a730510977f5aca0c3`
(the same commit the decision-142 receipt pins).

| file | sha256 |
| --- | --- |
| `overlay/tools/coreml/vulkan_decoder.py` | `5a163be6f78b1e1314040175ac768a4dbf3c270a0a0fa5b2fe5c88595fd46fb4` |
| `overlay/tools/coreml/pinned_component.py` | `25730aa8de275e3431112f2f205b2dde75b87f873210ff0ef6c1eafae8f6ea20` |
| `receipts/2026-09-14-decoder-lstm-semantics/probe.py` | `05e385f7da2f69c3967c1e71f798fa1e2475b59545b71072300714b232b20051` |
| `receipts/2026-09-14-decoder-lstm-semantics/result.json` | `27735035de2ab60470fc88d85dd6d5677caa19fb32a38b6e9fdf062f31b66e0e` |

The `vulkan_decoder.py` hash is the one the decision-142 receipt pins, so this
study analyses exactly that source.

Capture: authenticated `20260913T105550Z-librispeech-tdt-tensors`, `ane` leg,
`tdt_tensors.json` sha256
`6eaf2bfe801584af570cd66f42e8dc17efd67776fb181f691585d2bcab590e0a` (the hash
the earlier native comparison pins). Model revision
`b650695c2322ee5281dff48d7345b2f3a58ff018`. 16 traces, of which trace 1 is a
decoder-reuse step with no recurrent transition, leaving 15 comparable
transitions: 0 and 2 through 15.

## Command

```text
python3 receipts/2026-09-14-decoder-lstm-semantics/probe.py \
  --capture-dir ~/.cache/mlx-omarchy/parakeet-reference/captures/\
b650695c-75aec2a/20260913T105550Z-librispeech-tdt-tensors/ane \
  --package ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/\
parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018/decoder.mlpackage \
  --overlay <CoremlTdtStream>/overlay \
  --output receipts/2026-09-14-decoder-lstm-semantics/result.json
```

Wall 29.7 s.

## The host model reproduces the pinned divergence

Before any hypothesis, the NumPy model of the shipped `_lstm` was checked
against the decision-142 localization table. It reproduces every published
figure exactly:

| output | trace 0 max abs | worst max abs over 15 |
| --- | --- | --- |
| `next_cell` | 0.0028076171875 | 0.03125 |
| `next_hidden` | 0.000732421875 | 0.019287109375 |
| `decoder_hidden` | 0.01953125 | 0.01953125 |

Those are the decision-142 numbers, reproduced on the host with no device, so
the study below measures the same divergence the GPU run measured.

## The first divergence carries no reduction

Transition 0 decodes token 8192. That token's embedding row is identically
zero, and it is the only all-zero row in the 8193-row table. The captured
entry hidden and entry cell are both zero. Under all six gate reductions
tested, layer-0 gates therefore come out exactly equal to the pinned fp16 bias
constant, bit for bit: there is no dot product left to accumulate.

Layer 0 of transition 0 is consequently a pure elementwise measurement,
`next_cell = sigmoid(bias_i) * tanh(bias_c)` over known fp16 inputs. Native
still disagrees: 413 of 640 lanes bit-exact, max abs 0.0009765625, with a
symmetric fp16 ulp error histogram.

| signed fp16 ulp | -3 | -2 | -1 | 0 | +1 | +2 |
| --- | --- | --- | --- | --- | --- | --- |
| lanes | 1 | 21 | 91 | 413 | 91 | 23 |

Every native decoder output in the capture is fp16-representable, so the native
state is stored in fp16 and this is a genuine one-to-three-ulp elementwise
disagreement, not a widening artifact.

This is the load-bearing result. The candidate list for this hole was fp32
accumulation of the gate matmuls with fp16 rounding at the state, fp16 versus
fp32 activations, gate ordering, and an `ih` plus `hh` bias split. The first
divergence has no matmul, no accumulation, and no bias split, so none of those
axes can explain it. The named hole is elementwise `sigmoid` and `tanh`
evaluation, not fp16 recurrence.

## Gate packing is IFOC, and it is not the miss

Transition 0 also pins the packing, since the gates are the bias constants
there. Sweeping which quarter of the bias vector feeds which gate:

| assignment | bit-exact lanes of 640 | max abs |
| --- | --- | --- |
| input = slot 0, cell = slot 3 | 413 | 0.0009765625 |
| every other input/cell pair | 0, except input = slot 3, cell = slot 0 at 1 | up to 1.60791015625 |
| output = slot 2 | 403 | 0.00048828125 |
| output = slot 0, 1, 3 | 0, 0, 1 | 0.557, 0.426, 0.414 |

Wrong assignments do not degrade, they collapse. IFOC packing as documented in
`vulkan_decoder.py` is correct and is removed from the candidate list.

## Native linear semantics, established bit-exactly

The projector is the one activation-free operator in the package, and the
capture holds both its input (`next_hidden` layer 1) and its output
(`decoder_output_hidden`). It therefore isolates the native accumulator and
rounding contract of a 640-term fp16 dot product with an fp16 bias. Over all
15 transitions, 9600 lanes:

| scheme | bit-exact lanes of 9600 | max abs |
| --- | --- | --- |
| fp16 accumulator, unrounded products, ascending, bias trailing | 9600 | 0 |
| same, but each product rounded to fp16 first | 4535 | 0.00390625 |
| same, but descending reduction order | 1407 | 0.0390625 |
| fp32 dot rounded once, then fp16 bias | 1859 | 0.01953125 |
| fp32 sequential accumulation | 1567 | 0.0234375 |
| exact sum | 1568 | 0.0234375 |
| fp16 chain with the bias as the accumulator seed | 768 | 0.009765625 |

The winner is exact and unique: an fp16 accumulator, products left unrounded
(a fused multiply-add), a strictly ascending reduction index, and the bias
added after the reduction. Order matters, product rounding matters, and bias
placement matters, so this is a real contract rather than a coincidence at one
operating point. This is the first bit-exact native arithmetic contract
recovered for this package, and it closes
`parakeet.tdt.decoder-projector.fp16-linear-vs-native-coreml`.

It does not make a code change advisable on its own. The contract is serial in
the reduction index, so it cannot be expressed as an `mx` matmul; honouring it
needs a custom kernel that walks k in order with an fp16 accumulator, against a
decoder that runs 145 times per utterance. That is a performance and design
call for whoever owns the Vulkan kernel set, not a quiet edit here.

## The LSTM does not share the linear operator's reduction

The matrix crosses seven gate reductions with three activation models and two
state-algebra placements, 42 variants, each scored on all 15 transitions.
Feeding the projector's own fp16 FMA chain into the gate matmuls fits the LSTM
far worse than fp32 accumulation, so the `lstm` operator and the `linear`
operator do not share a reduction primitive. Holding the activation at the
better of the two fp16-state models:

| gate reduction | `next_cell` lanes of 19200 | `next_hidden` lanes of 19200 |
| --- | --- | --- |
| fp32 dots, fp32 add, gate rounded before fp16 bias | 3857 | 2883 |
| fp32 dots, each rounded to fp16, fp16 add, fp16 bias | 3821 | 2890 |
| exact dots, exact add, rounded once | 3807 | 2814 |
| fp32 dots, fp32 add, fp32 bias | 3807 | 2809 |
| fp32 dots, fp32 bias, gate left unrounded | 3784 | 2801 |
| fp16 FMA chains per weight block, then fp16 bias | 2794 | 1889 |
| fp16 FMA chain over concatenated x then h | 2515 | 1710 |

At a fixed activation model the four fp16-rounded fp32-class orderings sit
within 1.3 percent of each other in `next_cell` lane count, 3857 down to 3807
with the fp16 sigmoid chain and 3405 down to 3369 with exact activations, and
none is bit-exact. The gate reduction is therefore fp32-class, which is what
the shipped implementation already does, and its remaining detail is not
resolvable against this capture while the elementwise residual is present.

## Best variant, the shipped one, and 8289b1bc

Best over the matrix: fp32 gate dots, gate vector rounded to fp16 before the
fp16 bias add, `sigmoid` as a full fp16 chain `1 / (1 + exp(-x))` with fp16
rounding after negate, exp, add and reciprocal, `tanh` exact and rounded to
fp16, and the elementwise state algebra in fp16.

| variant | `next_cell` exact | `next_cell` max abs | `next_hidden` exact | `next_hidden` max abs |
| --- | --- | --- | --- | --- |
| shipped `_lstm` | 3369 / 19200 | 0.03125 | 3185 / 19200 | 0.019287109375 |
| best found | 3857 / 19200 | 0.02779007 | 2883 / 19200 | 0.01904297 |
| `8289b1bc` replica | 3246 / 19200 | 0.03125 | 2708 / 19200 | 0.01904296875 |

The best variant buys 488 cell lanes and gives back 302 hidden lanes. That is
a trade, not an establishment: the fp16 sigmoid chain is closer on the cell
path and worse on the output path, which is itself evidence that the native
elementwise functions are not any of the three models tried. Per transition,
the best variant recovers 698 of 1280 cell lanes at transition 0, where no
reduction runs, and 166 to 269 lanes on transitions 2 through 15, where the
reduction does run. The gap between those two regimes is the unresolved gate
reduction detail sitting on top of the elementwise residual.

`8289b1bc` is now measured rather than assumed. Read from its diff, it upcasts
the gate constants to fp32, runs the gates, activations and cell update in
fp32, and rounds only the two state outputs at the fp16 boundary. Replicated
here as fp32 dots with an unrounded fp32 gate vector, fp32 `sigmoid` and
`tanh`, and `state=fp32-boundary`, it lands at 3246 and 2708 lanes: worse than
the shipped implementation on both outputs, and its `next_cell` worst case is
unchanged at 0.03125. Widening the arithmetic moves away from native, which is
the expected sign if native is narrower than fp32 rather than wider, and the
exclusion in the decision-142 receipt holds on measurement.

Since no variant is bit-exact on all 15 transitions, `_lstm` is unchanged, per
the change gate. No test accompanies this receipt because no shipped code
changed; `probe.py` is the runnable check and regenerates every number above.

In the fp16 sigmoid chain, `exp(-x)` saturates to fp16 infinity once `-x`
reaches ln(65520), so for x below -11.0901, and that yields sigmoid 0, which is
the correct fp16 result. NumPy reports the saturation as an overflow warning
during the run. It is part of the variant's definition, and no candidate lane
is non-finite: `non_finite_candidate_lanes` is 0 for all 42 variants in
`result.json`.

## Remaining hole

`parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`.

The decoder LSTM disagrees with native Core ML by one to three fp16 ulps in the
elementwise `sigmoid` and `tanh` evaluation, shown on a reduction-free
transition. Closing it needs per-lane evidence of the native elementwise
functions on this backend, which this capture cannot supply: a single-operator
authenticated capture of fp16 `sigmoid` and `tanh` over a known fp16 argument
sweep, or the compiled compute plan that names the evaluation. Fitting it from
the LSTM outputs alone is under-determined, because each captured lane is one
fp16 product of two unknown fp16 function values.

## Not claimed

Decision 142 is not closed and no token-exactness is claimed. No device ran:
there is no GPU, ANE, Honeykrisp, or Apple-silicon execution behind any number
here, and no claim about which backend served the capture. The capture requests
`compute_units=ane` and exclusive backend execution remains unestablished, so
"native Core ML" here means the captured backend, whichever unit that was, and
the projector and LSTM results differing is consistent with two different
units. `8289b1bc` stays excluded on the measurement above, not on assumption.
No claim that the projector contract is implementable as an `mx` matmul, and no
shipped code changed.

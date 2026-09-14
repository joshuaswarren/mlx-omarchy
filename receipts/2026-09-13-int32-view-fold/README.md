# int32 view fold — Phase 5 fail-fast receipt (2026-09-13)

## Assignment

Peel or fold non-fp16/non-bool shape aliases in the frontend (same
spirit as `output_peel`) so H13 never sees int32 views; re-emit MIL;
the compiler compiles again. Host-only.

## Inventory

The integrated encoder MIL contains exactly **one** view op whose
output dtype is neither fp16 nor bool:

```text
tensor<int32, [1, 1]> var_283 = expand_dims(axes = var_283_axes_0, x = lengths_cast_fp16_to_int32)
```

All other views (12 fp16 `expand_dims`, 145 fp16 `reshape`) stay
untouched — fp16 views are the compiler's alias business. All 10 bool
views (`expand_dims` ×6, `tile`, `transpose`) stay untouched — they
belong to the compiler lane's bool-surface work.

`var_283`'s input (`lengths_cast_fp16_to_int32`, the subsampled audio
length) is a runtime value, so const-folding does not apply. Its only
consumer is `less(x = var_281 [375 const], y = var_283 [1,1])` producing
`output_mask` bool [1,375].

## The fold (unit-expand absorption)

`overlay/tools/coreml/fold_unit_views.py`:

1. Relabel the const sibling `var_281` [375] → [1,375] under a fresh
   name (`var_281_fold_const`), identical blob bytes and offset.
2. Rewire the less to `y = lengths_cast_fp16_to_int32` ([1]).
3. Delete the `var_283` expand_dims line.

Soundness, checked mechanically per consumer before rewriting:
`less([1,375], [1])` broadcasts to `[1,375]` — the declared output
shape is unchanged — and every lane compares the same pair of values
as before (leading unit axes preserve flat order; the size-1 runtime
operand contributes its single value everywhere in both forms).
Anything else (non-elementwise consumer, output-shape change,
dynamic shapes, graph-return views) is refused with a named reason;
leftovers are reported, never raised, so the compiler names the next
reject.

General mechanism, not a one-off: const-fed relabels (P1) and
unit-expand absorption (P2) over `expand_dims`/`reshape`/`squeeze`,
with a numpy broadcast checker. The module is planner + rewriter,
mirroring the lane's `output_peel`/`mask_lowering` split.

## Verification

- `overlay/tests/omarchy/coreml/test_fold_unit_views.py`: 7/7 —
  broadcast rules, absorption into `less` (line-level + numpy
  value semantics over lengths 0/1/200/374/375), const-fed relabel
  byte preservation, matmul-consumer refusal by name, fp16/bool
  views unclassified, graph-return refusal.
- Real graph: plan finds exactly the 1 op (`FOLDED, unit-expand
  absorption into 1 consumer(s)`); post-fold the `less` line reads
  `less(x = var_281_fold_const, y = lengths_cast_fp16_to_int32)` with
  declared output still `tensor<bool, [1, 375]>`; all other lines
  byte-identical.
- Eligibility impact: none on dispositions (`expand_dims` is
  SUPPORTED-class; the fold is a MIL-text-stage delta of one op).

## Re-run (paste)

Applied on top of the fp16-output peel
(`receipts/2026-09-13-fp16-output-peel/model-fp16.mil`):

```text
build/mil-hwxc --mil model-fp16-noview.mil --model-root .../model-root \
  --output encoder-out-fp16-noview --target H13 --format anec
162:5: error [h13.invalid-transpose-parameters]: H13 transpose requires x and an exact rank-matching tensor<int32,[rank]> perm constant over a positive-rank static fp16 input
exit=65
```

Line 162 is the mask-chain bool transpose:

```text
tensor<bool, [1, 375, 375]> var_289 = transpose(perm = var_289_perm_0, x = attention_mask_3)
```

The perm IS exact rank-matching int32 `[0,2,1]` — the rejection is
purely the bool input: the fp16-only transpose contract. This is the
transpose-absorb receipt's "fp16-only contract" class (its 1-bool case),
now observed as the live first transpose reject rather than only as a
predicted class.

## What this means for the peel order

The predicted nonfoldable-transpose was not reached either — but
narrowly: the very first transpose in program order is the bool
mask-chain one, and it gates on dtype before any geometric
foldability question arises. After the compiler lane covers bool
transpose inputs (or bool views generally), the next expected gate is
the rank-4 middle-swap / rank-3 fast-axis classes from the
transpose-absorb receipt. Both remaining gaps are compiler-side; the
frontend has no sound move for a bool permutation (no reshape/split
algebra expresses it, and inventing one is exactly what the
transpose-absorb receipt forbids).

## Files

- `model-fp16-noview.mil` — peel + fold applied (returns fp16 +
  bool, zero int32 views)
- `fold-report.json` — `{"folded": ["var_283"], "refused": []}`
- `runs.log` — the compiler invocation above, verbatim

## Verification

- No hardware, no device, no model-pin change; host-only throughout.
- `test_fold_unit_views.py` 7/7; full coreml suite green (see commit).

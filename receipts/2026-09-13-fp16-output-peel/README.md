# fp16-output boundary peel — Phase 5 fail-fast receipt (2026-09-13)

## Baseline (before the peel)

`mil-hwxc` at compiler commit `366eb15` (`feature/h13-boolean-and-round2`,
built in-tree, host-only) on the integrated encoder MIL:

```text
build/mil-hwxc --mil /tmp/ane-runtime-adapter-integrated-20260913/model.mil \
  --model-root /tmp/ane-runtime-adapter-integrated-20260913/model-root \
  --output encoder-out --target H13 --format anec
3353:5: error [h13.unsupported-logical-result-conversion]: H13 logical result conversions require explicit hardware or GPU coverage
exit=65
```

The gate fired on the function returns: `encoder_hidden` is produced by
a trailing `cast` fp16→fp32 and `encoder_mask` by a trailing `cast`
bool→int32. The H13 backend cannot materialize non-fp16 logical
results.

## The peel (frontend, this lane)

`overlay/tools/coreml/output_peel.py::peel_fp16_outputs` removed the
two trailing boundary casts from the MIL text and rewired the footer:

- `encoder_hidden = cast(fp32, linear_217_cast_fp16)` →
  return `linear_217_cast_fp16` (fp16 [1,375,640])
- `encoder_mask = cast(int32, output_mask)` →
  return `output_mask` (bool [1,375], 1-byte surface)

Only the two cast lines changed; all 3354 other lines are
byte-identical (the two orphaned dtype consts are left in place as
valid, inert MIL). `gpu-epilogue.json` (schema
`mlx-omarchy.gpu-boundary-epilogue.v1`) records the two casts the
Vulkan side must re-apply; both are exact, injective widenings, so the
epilogue reproduces the original `encoder_hidden`/`encoder_mask`
bit-for-bit. This is the §3.2/§32 boundary: ANE owns the fp16 graph,
casts stay GPU-side. No fp32→fp16 narrowing exists at the boundary,
and the peeler refuses any non-widening trailing conversion by name.

## Re-run 1 — full peel (both outputs)

```text
build/mil-hwxc --mil receipts/2026-09-13-fp16-output-peel/model-fp16.mil \
  --model-root /tmp/ane-runtime-adapter-integrated-20260913/model-root \
  --output encoder-out-peeled --target H13 --format anec
156:5: error [h13.unsupported-logical-result-conversion]: H13 logical result conversions require explicit hardware or GPU coverage
exit=65
```

Line 156 is `output_mask = less(...)` — the bool `less` that now feeds
a graph output directly. The bool task-stream encoder (in4) exists on
the branch, but emitting a bool surface as a logical result (the out6
half of the bool-surface contract) has no coverage yet. **For the
compiler lane: bool logical-result binding is the next missing piece
after the encoder itself.**

## Re-run 2 — diagnostic, hidden output only

To isolate the out6 gap from the rest of the graph, a second variant
returns only `linear_217_cast_fp16` (same file, footer `} ->
(linear_217_cast_fp16);`, nothing else changed):

```text
36:5: error [h13.invalid-shape-alias]: H13 shape aliases require static fp16 input and result shapes with equal element counts and constant shape parameters
exit=65
```

Line 36 is `var_114 = expand_dims(x = channel_mask_1_cast_fp16)` —
bool [1,1500] → bool [1,1,1500]: static shapes, equal counts, constant
axes. Everything checks out except the tensor is bool, so the free
shape-alias (view) path, which requires fp16, rejects it. **For the
compiler lane: bool views need the same alias treatment as fp16
views** (no data moves; the descriptor already carries the geometry).

## What this means for the peel order

Main's predicted next reject was nonfoldable-transpose. The observed
order is more useful: the bool-surface gaps (logical-result binding,
then bool views) gate *before* any transpose is reached. The transpose
rejects documented in `receipts/2026-09-13-h13-transpose-absorb.md`
remain the expected third gate. Nothing on the frontend side blocks
either of the two new rejects — both are compiler-side bool-surface
coverage, consistent with the leaf-6 ABI decision.

## Files

- `model.mil` — input copy (source:
  `/tmp/ane-runtime-adapter-integrated-20260913/model.mil`, read-only
  sibling artifact)
- `model-fp16.mil` — peeled graph (returns fp16 + bool)
- `model-fp16-hidden-only.mil` — diagnostic single-return variant
- `gpu-epilogue.json` — the two Vulkan-side casts
- `runs.log` — the three compiler invocations above, verbatim

## Verification

- `overlay/tests/omarchy/coreml/test_output_peel.py`: 4/4 (peel both
  widenings; refuse fp32→fp16 narrowing; refuse missing footer;
  byte-identity of all non-rewritten lines).
- The peel removed exactly 2 lines; `model-fp16.mil` returns
  `(linear_217_cast_fp16, output_mask)`.
- No hardware, no device, no model-pin change; host-only throughout.

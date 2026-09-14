# Slice layout rewrite — per-head plane then drop-first (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Assignment

Eliminate the next encoder compile reject after heads_layout `8a569ea0`:
`h13.noncontiguous-slice` on
`var_367_cast_fp16 = slice_by_index(...)` fp16 `[1, 8, 749, 375]`,
8 chunks of 280875 elements spaced 281250 apart.

## Baseline (heads-layout rewrite)

```text
255:5: error [h13.noncontiguous-slice]: H13 slice_by_index over non-unit head dimensions reads 8 chunks of 280875 elements spaced 281250 apart, and one binding slice cannot represent interleaved chunks: a chunked consumer decomposition has no MIL-expressible direct reference to byte-prove against, because no view op exposes mid-range head slices
exit=65
```

Source is `attention_scores_5_cast_fp16` `[1, 8, 750, 375]`. begin
`[0, 0, 1, 0]`, end_mask all-true, so the slice is `x[:, :, 1:, :]`.
280875 = 749*375; 281250 = 750*375.

## The rewrite

`overlay/tools/coreml/slice_layout.py` replaces that slice with eight
head planes (`[:, h:h+1, :, :]`, one dim) then `[:, :, 1:, :]` on each
plane (one dim) and concats on axis 1.

A reshape of the same buffer to `[8, 750, 375]` then `[:, 1:, :]` is
the same eight-chunk layout. Dropping the last row
(`[:, :, :749, :]`) has the same output shape and the wrong values.

A single two-dim slice (`[:, h:h+1, 1:, :]`) fail-fasted as
`h13.noncontiguous-slice`: "H13 slice_by_index can slice at most one
dimension because interleaved output chunks exceed the one-slice
binding ABI". Two one-dim slices per head is the form H13 already
names as in-bounds.

24 encoder layers rewritten. 24 last-dim `matrix_bd_*` slices
(`[1, 8, 375, 749] -> [1, 8, 375, 375]`, 3000 chunks of 375 spaced
749) stay in the graph: two wrapping dims, not this rewrite.

## Numpy proof

`numpy_relpos_drop_first` compares `x[:, :, 1:, :]` against concat of
`x[:, h:h+1, 1:, :]` on encoder-shaped `[1, 8, 750, 375]` fp16
(random, zeros, and `-0`). Bit-exact (`uint16` view). Two-step
`x[:, h:h+1, :, :][:, :, 1:, :]` matches. No counterexample.

`numpy_relpos_drop_last` raises `SliceLayoutError` with the first
mismatched lane.

## Re-run

Same `mil-hwxc` as heads-layout (`1f67253`,
sha256 `a3b03ecab14d70fe5e262c2f4066b9c5d3b102a6a92a09f0adb7d6b693a397dd`)
on the rewritten heads-layout MIL. `--model-root` is the heads-layout
copy (no new BLOBFILE headers).

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-slice-layout-rewrite/model-fp16-noslice.mil \
  --model-root /tmp/heads-layout-rewrite-20260913/model-root \
  --output /tmp/slice-layout-rewrite-20260913/anec-out \
  --target H13 --format anec
296:5: error [h13.noncontiguous-slice]: H13 slice_by_index over non-unit head dimensions reads 3000 chunks of 375 elements spaced 749 apart, and one binding slice cannot represent interleaved chunks: a chunked consumer decomposition has no MIL-expressible direct reference to byte-prove against, because no view op exposes mid-range head slices
exit=65
```

Line 296 is **not** the `var_367` slice. `var_367_cast_fp16` is a
`concat` of eight `[1, 1, 749, 375]` heads. The new fail-fast is
`matrix_bd_3_cast_fp16` fp16 `[1, 8, 375, 375]` — last-dim slice of
the reshaped `[1, 8, 375, 749]` rel-pos scores.

## Files

- `model-fp16-noslice.mil` — heads-layout MIL with the 749-slice decomposed
- `rewrite-report.json` — schema `mlx-omarchy.slice-layout-rewrite.v1`
- `var-367-snippet.txt` — rewritten var_367 plane/inner/concat lines
- `runs.log` — the compiler invocation above, verbatim

## Verification

- `overlay/tests/omarchy/coreml/test_slice_layout.py`: 5/5
- No hardware, no device, no model-pin change; host-only throughout.

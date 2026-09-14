# Mask layout rewrite — drop the bool tail-swap transpose (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Assignment

Eliminate the first remaining encoder compile reject from
`mil-hwx-compiler` `receipts/2026-09-13-encoder-compile-attempt-3.md`:
`h13.nonfoldable-transpose` on
`transpose(perm=[0,2,1], x=attention_mask_3)` bool `[1,375,375]`,
consumer `mul` with no transpose flag.

## Baseline (attempt 3)

```text
162:5: error [h13.nonfoldable-transpose]: H13 a tail-swap transpose moves the storage-fastest axis, so its view reads rows whose elements sit a non-unit storage stride apart: the NCHW descriptor carries contiguous rows and no surface interpretation expresses the permutation at any size; a consuming matmul's transpose flag is the only decoded mechanism that absorbs one, and this view has a 'mul' consumer with no such flag
exit=65
```

Line 162 was:

```text
tensor<bool, [1, 375, 375]> var_289 = transpose(perm = var_289_perm_0, x = attention_mask_3)
```

The perm is exact rank-matching `[0, 2, 1]`. Bool transpose is accepted
as a perm; the reject is geometric: a fast-axis tail-swap that `mul`
cannot absorb.

## The rewrite

`overlay/tools/coreml/mask_layout.py` emits the transposed operand
already in the mul's layout.

The producer is a vector mask `output_mask` bool `[1, 375]`:

```text
var_285 = expand_dims(axes=1, x=output_mask)          # [1, 1, 375]
attention_mask_3 = tile(reps=[1, 375, 1], x=var_285)  # [1, 375, 375]
                                                     # out[b,i,j] = m[j]
var_289 = transpose(perm=[0,2,1], x=attention_mask_3) # out[b,i,j] = m[i]
attention_mask_5 = mul(attention_mask_3, var_289)     # out[b,i,j] = m[j]*m[i]
```

Complementary expand+tile, no transpose:

```text
var_289_layout_exp = expand_dims(axes=2, x=output_mask)     # [1, 375, 1]
var_289 = tile(reps=[1, 1, 375], x=var_289_layout_exp)      # out[b,i,j] = m[i]
attention_mask_5 = mul(attention_mask_3, var_289)           # unchanged
```

Bool `expand_dims` and `tile` already compiled through on this graph
in attempt 3. The inert `var_289_perm_0` const is left in place.

`mul`/`logical_and` both accepted as the consumer (the compiler folds
`logical_and` to `mul`; attempt 3's emitted MIL already has `mul`).

Anything else (matmul consumer, non-tile producer, non-square shape,
graph return) is refused by name and left for the compiler.

## Numpy proof

`numpy_tail_swap_mul` compares `col * col.transpose(0,2,1)` against
`col * row` on encoder-shaped `[1, 375]` masks (prefix lengths
0/1/200/374/375 and a random 0/1 mask). Bit-exact on fp16 0/1
(`uint16` view). No counterexample.

## Re-run

Same `mil-hwxc` as attempt 3 (`1f67253`,
sha256 `a3b03ecab14d70fe5e262c2f4066b9c5d3b102a6a92a09f0adb7d6b693a397dd`)
on the rewritten attempt-3 MIL, same `--model-root`.

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-mask-layout-rewrite/model-fp16-nolayout.mil \
  --model-root /tmp/encoder-compile-attempt-3-20260913/emitted/model-root \
  --output /tmp/mask-layout-rewrite-20260913/anec-out \
  --target H13 --format anec
204:5: error [h13.nonfoldable-transpose]: H13 transpose with several consumers or a returned value needs a materialized transposed surface, and the decoded corpus holds no data-movement encoder
exit=65
```

Line 204 is **not** the mask tail-swap. `var_289` is now a `tile`.
The new fail-fast is fp16 `query_states_1_cast_fp16 = transpose(...)`
`[1, 8, 375, 128]` (`transpose_143`), two `add` consumers — the
multi-consumer class from the transpose-absorb receipt.

## Files

- `model-fp16-nolayout.mil` — attempt-3 folded MIL with the mask
  tail-swap rewritten (source sha256 `b75801f75e357479…`)
- `rewrite-report.json` — schema `mlx-omarchy.mask-layout-rewrite.v1`
- `mask-chain-snippet.txt` — rewritten lines 157–170
- `runs.log` — the compiler invocation above, verbatim

## Verification

- `overlay/tests/omarchy/coreml/test_mask_layout.py`: 7/7
- No hardware, no device, no model-pin change; host-only throughout.

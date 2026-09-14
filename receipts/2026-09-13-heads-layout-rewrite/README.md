# Heads layout rewrite — birth Q/K/V as [B,H,T,D] (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Assignment

Eliminate the next encoder compile reject from `mil-hwxc`
`receipts/2026-09-13-encoder-compile-attempt-4.md`:
`h13.nonfoldable-transpose` on
`query_states_1_cast_fp16 = transpose(perm=[0,2,1,3], x=var_332_cast_fp16)`
fp16 `[1,8,375,128]`, two `add` consumers.

## Baseline (attempt 4)

```text
204:5: error [h13.nonfoldable-transpose]: H13 transpose with several consumers or a returned value needs a materialized transposed surface, and the decoded corpus holds no data-movement encoder
exit=65
```

Input of that transpose is `[1,375,8,128]` (`[B,T,H,D]`). The perm is
exact rank-matching `[0,2,1,3]`. Fast axis 128 is preserved; the reject
is multi-consumer (two `add`s), not a tail-swap.

## The rewrite

`overlay/tools/coreml/heads_layout.py` splits each
`linear → reshape [B,T,H,D] → transpose [B,H,T,D]` projection into `H`
linears of width `D` and concats the unit-expanded heads on axis 1.

Naive reshape of the same linear to `[B,H,T,D]` is a different packing
(token 0's `H*D` values would be read as head 0 across `T`). A
weight-row permute cannot fix it: each linear row still belongs to one
token. Both are refused with a named counterexample.

Per-head weights are contiguous fp16 row slices of the original
BLOBFILE. BLOBFILE offsets are 24-byte `DEADBEEF` headers, not payload
bytes, so a compile probe copies `--model-root` and appends one header
per slice pointing at `payloadOffset + head * stride`.

K/V use perm `[0,2,-3,-1]`, the same permutation. 24 Q + 48 K/V = 72
rewritten. 25 other rank-4 `[0,2,1,3]` transposes (not linear
projections) stay in the graph.

The original linear+reshape, whose only consumer was the transpose,
are deleted so H13 does not see unused results.

## Numpy proof

`numpy_heads_stack` compares `transpose(reshape([B,T,H,D]), (0,2,1,3))`
against `stack` of last-axis slices on encoder-shaped `[1,375,1024]`
fp16 (random, zeros, and `-0`). Bit-exact (`uint16` view). No
counterexample.

`numpy_naive_reshape` and `numpy_weight_permute_reshape` raise
`HeadsLayoutError` with the first mismatched lane.

## Re-run

Same `mil-hwxc` as attempt 4 (`1f67253`,
sha256 `a3b03ecab14d70fe5e262c2f4066b9c5d3b102a6a92a09f0adb7d6b693a397dd`)
on the rewritten attempt-4 MIL. `--model-root` is a copy of attempt 3's
root with cloned slice headers.

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-heads-layout-rewrite/model-fp16-noheads.mil \
  --model-root /tmp/heads-layout-rewrite-20260913/model-root \
  --output /tmp/heads-layout-rewrite-20260913/anec-out \
  --target H13 --format anec
255:5: error [h13.noncontiguous-slice]: H13 slice_by_index over non-unit head dimensions reads 8 chunks of 280875 elements spaced 281250 apart, and one binding slice cannot represent interleaved chunks: a chunked consumer decomposition has no MIL-expressible direct reference to byte-prove against, because no view op exposes mid-range head slices
exit=65
```

Line 255 is **not** the query_states transpose. `query_states_1_cast_fp16`
is a `concat` of eight `[1,1,375,128]` heads. The new fail-fast is
`slice_by_index` `var_367_cast_fp16` fp16 `[1,8,749,375]` — the
rel-pos scores slice, the chunked-slice class from the channel-plane
receipt.

## Files

- `model-fp16-noheads.mil` — attempt-4 MIL with Q/K/V heads born `[B,H,T,D]`
- `rewrite-report.json` — schema `mlx-omarchy.heads-layout-rewrite.v1`
- `q-chain-snippet.txt` — rewritten query_states_1 lines
- `runs.log` — the compiler invocation above, verbatim

## Verification

- `overlay/tests/omarchy/coreml/test_heads_layout.py`: 7/7
- No hardware, no device, no model-pin change; host-only throughout.

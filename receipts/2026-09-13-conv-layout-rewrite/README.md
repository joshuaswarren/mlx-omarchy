# Conv layout rewrite — 1×1 as rank-4 corpus form (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Assignment

Rewrite encoder 1×1 valid `[1, 256, 750, 32]` (and sibling
`[1, 256, 375, 16]`) so H13 sees the in-envelope rank-4 form
(`receipts/2026-09-13-h13-conv-envelope.md` in mil-hwx-compiler,
`c2cf32e4`). Do not edit `heads_layout` / `mask_layout` /
`slice_layout` / `attention_layout` / `mask_lowering`.

## The leftover

After slice layout, the encoder still has two groups-1 1×1 convs:

```text
conv k1 st1 valid bias g1
  [1, 256, 750, 32] × [256, 256, 1, 1] → [1, 256, 750, 32]
conv k1 st1 valid bias g1
  [1, 256, 375, 16] × [256, 256, 1, 1] → [1, 256, 375, 16]
```

`convParityPlan` needs `string` `pad_type` and `int32` scalar
`groups`. The encoder spells `tensor<string, []>` and
`tensor<int32, []>`. Corpus probes use `string` / `int32`. That
spelling miss fires first. After corpus spelling, CHW `{256, 750, 32}`
and `{256, 375, 16}` remain table misses. Nearby `{256, 32, 32}`
compiles. `750/32` is not an integer; this rewrite does not tile.

## The rewrite

`overlay/tools/coreml/conv_layout.py` rewrites those two ops' 
`pad_type` / `groups` consts to corpus scalars. k1 / stride-1 /
valid is the same arithmetic as `same`. Depthwise padconv, k3
subsample, and rank-3 1D convs are not this class.

A reshape of `[N, C, H, W]` to `[N, H, W, C]` then a last-axis mix
is a different packing. `numpy_naive_nhwc_reshape` is that
counterexample. The NCHW flatten through `[N, C, H*W]` is a view.

Plan of the noslice encoder MIL is 2/0.

## Numpy proof

`numpy_nchw_flatten_view` compares `[N, C, H, W]` against reshape
through `[N, C, H*W]` on `[1, 256, 6, 4]` fp16 (random, zeros, and
`-0`). Bit-exact (`uint16` view). No counterexample.

`numpy_naive_nhwc_reshape` is inexact on the same shape.

## Standalone compile

Same `mil-hwxc` as the envelope pin (`c2cf32e4`,
sha256 `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`)
on explicit `[1, 256, 32, 32]` k1 / st1 / valid / bias / g1, corpus
spelling.

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-conv-layout-rewrite/conv-r4.mil \
  --model-root receipts/2026-09-13-conv-layout-rewrite \
  --target H13 --output receipts/2026-09-13-conv-layout-rewrite/conv-r4
compiled target=H13 artifacts=1 format=anec output=.../conv-r4
```

One program: `apple-parity-conv`, `taskDescriptors = 1`. Exit 0.

The same spelling at encoder CHW `{256, 750, 32}` still fail-fasts
`h13.conv-outside-envelope` (exit 65, no output dir). That is the
table miss the envelope pin named. Do not tile it into 32×32.

## Files

- `conv-r4.mil` — standalone in-envelope 1×1
- `conv-r4/manifest.json` — compiler package (1 program, 1 TD)
- `enc-1x1-750.mil` — same spelling at encoder CHW; still a hole
- `rewrite-report.json` — encoder plan 2/0
- `rewritten-1x1-snippet.txt` — corpus-spelled pad_type/groups
- `runs.log` — both compiler invocations, verbatim

## Verification

- `overlay/tests/omarchy/coreml/test_conv_layout.py`: 5/5
- No hardware, no device, no model-pin change; host-only throughout.

Did not edit `heads_layout.py`, `mask_layout.py`, `slice_layout.py`,
`attention_layout.py`, `mask_lowering.py`, Compiler.mm, or anything
under `ane-linux-experiments` outside this worktree.

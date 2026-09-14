# Attention layout rewrite — birth K as [B,H,D,T] (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Assignment

Frontend rewrite so encoder attention uses the one-program H13
contraction `[1, 8, 375, 128] × [1, 8, 128, 375]`, `tx=0`, `ty=0`
(`receipts/2026-09-13-h13-matmul-envelope.md` in mil-hwx-compiler,
`c2cf32e4`). The leftover encoder scores were
`[1, 8, 375, 128] × [1, 8, 375, 128]^T`.

Do not edit `heads_layout` / `mask_layout` / `slice_layout` /
`mask_lowering`. Compiler.mm was not edited; the template already
exists.

## The leftover

After heads+slice layout, each encoder layer still has:

```text
matmul(transpose_x=false, transpose_y=true,
       x=mul_N [1,8,375,128],
       y=hidden_states_* [1,8,375,128])
→ [1,8,375,375]
```

A trailing-two-dim transpose of K folds back into `ty=true`. That is
the leftover. Rel-pos (`[1,8,375,128] × [1,8,128,749]`, `ty=false`)
and PV (`[1,8,375,375] × [1,8,375,128]`, `ty=false`) are other
batched entries and are not this rewrite.

## The rewrite

`overlay/tools/coreml/attention_layout.py` finds those `ty=true`
score matmuls. When K is the heads concat of `expand_dims(linear
[B,T,D], axes=1)`, each head linear is reborn as `[B,1,D,T]` by
stacking the contiguous token rows (`[B,1,D]` slice, reshape
`[B,D,1]`, concat on the last axis, expand unit head axis). Concat
on axis 1 then emits `[B,H,D,T]`. `ty` becomes false.

A reshape of `[B,H,T,D]` to `[B,H,D,T]` is a different packing.
`numpy_k_naive_reshape` is that named counterexample.

Expands shared with another consumer are not mutated; K is converted
through a new `[B,H,D,T]` tensor instead.

A 24-layer token-row concat of T=375 is the same scale as the
last-dim slice peel (99 MB) and was not kept. Plan of the noslice
encoder MIL is 24/0.

## Numpy proof

`numpy_k_token_stack` compares `transpose(K, (0,1,3,2))` against
concat of per-token last-axis rows on encoder-shaped `[1,8,375,128]`
fp16 (random, zeros, and `-0`). Bit-exact (`uint16` view). No
counterexample.

`numpy_qk_contraction` compares `Q @ K^T` against `Q @ token_stack(K)`
on a small `[1,2,3,4]` case. Bit-exact.

## Standalone compile

Same `mil-hwxc` as the envelope pin (`c2cf32e4`,
sha256 `8406755b5033e06cc542b1a41af028d4f6ad835f707b3880dc10e0a2818cfeb0`)
on explicit `[1,8,375,128] × [1,8,128,375]`, `tx=0`, `ty=0`.

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-attention-layout-rewrite/attn-r4.mil \
  --model-root receipts/2026-09-13-attention-layout-rewrite \
  --target H13 --output receipts/2026-09-13-attention-layout-rewrite/attn-r4
compiled target=H13 artifacts=1 format=anec output=.../attn-r4
```

One program: `apple-parity-batched-matmul`, `taskDescriptors = 208`.
Inputs `[1,8,128,375]` (K) and `[1,8,375,128]` (Q). Output
`[1,8,375,375]`.

## Files

- `attn-r4.mil` — standalone one-program score GEMM
- `attn-r4/manifest.json` — compiler package (1 program, 208 TDs)
- `rewrite-report.json` — encoder plan 24/0 plus born-fixture rewrite
- `born-snippet.txt` — small heads-concat K reborn `[B,H,D,T]`, `ty=false`
- `runs.log` — the compiler invocation above, verbatim

## Verification

- `overlay/tests/omarchy/coreml/test_attention_layout.py`: 8/8
- No hardware, no device, no model-pin change; host-only throughout.

Did not edit `heads_layout.py`, `mask_layout.py`, `slice_layout.py`,
`mask_lowering.py`, Compiler.mm, or anything under
`ane-linux-experiments` outside this worktree.

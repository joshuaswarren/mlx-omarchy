# Conv as matmul — 1D pointwise GEMM (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Assignment

Encoder leftover 24× 1D pointwise conv `[1,1024,375] × [2048,1024,1]`
valid. That is GEMM `(M=375, K=1024, N=2048)`. Encoder matmuls of
other shapes compile. Probe whether rewriting this conv to matmul
hits an H13 matmul envelope or the same linear hole.

Do not edit other layout modules.

## The leftover

After slice layout, each of 24 encoder layers still has:

```text
conv k1 st1 valid g1 no-bias
  [1, 1024, 375] × [2048, 1024, 1] → [1, 2048, 375]
```

Rank-3 1D never reaches `convParityPlan`. Nearby rank-4 1×1
`{256,32,32}` compiles. The 24 sibling k=1 convs
`[1,1024,375] × [1024,1024,1]` are GEMM `(375,1024,1024)` and are
not this probe. Depthwise k9 + bias is a different class.

## The rewrite

`overlay/tools/coreml/conv_as_matmul.py` classifies those 24 as
this GEMM. `x_nlc = transpose(x, (0,2,1))`, `W = weight[:,:,0]`,
`y_nlc = x_nlc @ W^T`. Standalone MIL is the encoder-linear form:
`[1,375,1024] × [2048,1024]^T`, `tx=0`, `ty=1`, const W.

A reshape of `[1, Cin, L]` to `[1, L, Cin]` is a different packing.
`numpy_naive_ncl_reshape` is that counterexample.

Plan of the noslice encoder MIL is 24 rewritten / 77 refused.

## Numpy proof

`numpy_conv1d_as_gemm` compares 1D k=1 NCL conv against the GEMM
on `[1,8,6] × [4,8,1]` fp16 (random, zeros, and `-0`). Bit-exact
(`uint16` view). No counterexample.

`numpy_naive_ncl_reshape` is inexact on the same shape.

## Standalone compile

Same `mil-hwxc` as the envelope pin (`c2cf32e4`,
sha256 `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`)
on explicit `[1,375,1024] × [2048,1024]^T`, `tx=0`, `ty=1`, const W.

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-conv-as-matmul/gemm-r2.mil \
  --model-root receipts/2026-09-13-conv-as-matmul \
  --target H13 --output receipts/2026-09-13-conv-as-matmul/gemm-r2
compiled target=H13 artifacts=15000 format=anec output=.../gemm-r2
```

Exit 0. Not a one-program envelope. 15000 programs
(`3000` matmul + `12000` add), all `h13-source-qualified`,
`taskDescriptors = 1`. Package 1671797682 bytes.

That is the M=1 native matvec fallback: K=1024 splits into two
512-chunks, N=2048 into four 512-tiles, 375 rows, `375 × 8 = 3000`
matmuls. Same linear hole as const-W `(375,1024,128)`
(`receipts/2026-09-13-h13-linear-tiling.md` in mil-hwx-compiler),
more N-tiles because N=2048. Not `apple-parity-batched-matmul`.

Deleted `gemm-r2/` after the summary. 15000 ANEC files are the
tiling, not an envelope pin.

## Files

- `gemm-r2.mil` — standalone GEMM `(375,1024,2048)` ty=1
- `weights.bin` — const W blob (`[2048,1024]` fp16)
- `rewrite-report.json` — encoder plan 24/77
- `rewritten-gemm-snippet.txt` — one leftover conv as the GEMM
- `package-summary.json` — 15000-program tiling, not the ANEC tree
- `runs.log` — the compiler invocation above, verbatim

## Verification

- `python3 overlay/tools/coreml/conv_as_matmul.py`: `conv-as-matmul numpy: PASS`
- Compile rc=0, verdict `same-linear-hole`
- No hardware, no device, no model-pin change; host-only throughout.

Did not edit other layout modules, Compiler.mm, or anything under
`ane-linux-experiments` outside this worktree.

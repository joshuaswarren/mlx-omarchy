# Select runtime rewrite — const `-inf` a becomes runtime fill (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Assignment

Rewrite encoder selects so `a` is a runtime constant input (materialized
fill), not a MIL const. Envelope pin:
`mil-hwx-compiler` `receipts/2026-09-13-h13-select-envelope.md`
(`c2cf32e4`). Do not edit other layout modules. Do not emit the ninf packing.

## The leftover

After `lower_mask_ops` rewrote the 24 finite `+0.0` selects, 24
attention-bias selects remain. Each binds scalar fp16 const `var_8_to_fp16`
(`0xFC00` `-inf`) as `a`, scores `[1, 8, 375, 375]` as `b`, and bool
`var_373` as `cond`. Geometry is inside the envelope. The MIL gate
refuses every constant-a form before table lookup:

```text
h13.select-needs-decoded-encoder
```

exit 65, no output dir. Scalar `-inf` and a full tensor blob `a` fail
the same way. `supportsBooleanOp({Select, true, 8, 375, 375})` is true.
The ninf capture's constant section is not a packing for an arbitrary
fill. Do not emit it.

## The rewrite

`overlay/tools/coreml/select_runtime.py` promotes each in-envelope
const-a select to a function input of the select output shape. The host
fills that tensor (`materialize_fill`). One shared fill covers the 24
encoder ops that read `var_8_to_fp16`. The unused scalar const is
deleted. Rank-3 `+0.0` selects (`[1, 1024, 375]`) stay for
`mask_lowering`. fp16-cond forms are refused by name.

Plan of the peeled encoder MIL is 24 rewritten / 24 refused / 1 fill.

## Numpy proof

`numpy_const_vs_runtime_fill` compares `select(cond, -inf, x)` against
`select(cond, runtime_fill, x)` on `[1, 8, 375, 375]` fp16 (random,
`-0`, broadcast cond `[1, 1, 375, 375]`). Bit-exact (`uint16` view).
Masked lane `0xFC00`; unmasked `-0` stays `0x8000`.

## Standalone compile

Same `mil-hwxc` as the envelope pin (`c2cf32e4`,
sha256 `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`)
on explicit `[1, 8, 375, 375]` runtime-a / runtime-b / bool cond.

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-select-runtime-rewrite/select-rrb.mil \
  --model-root receipts/2026-09-13-select-runtime-rewrite \
  --target H13 --output receipts/2026-09-13-select-runtime-rewrite/select-rrb
compiled target=H13 artifacts=1 format=anec output=.../select-rrb
```

One program: `apple-parity-boolean`, `taskDescriptors = 5`. Exit 0.

The same shape with const `-inf` `a` still fail-fasts
`h13.select-needs-decoded-encoder` (exit 65, no output dir). The
frontend rewrite of that program compiles as one
`apple-parity-boolean` program, 5 TDs, exit 0.

## Files

- `select-rrb.mil` — standalone runtime-a / bool cond
- `select-rrb/manifest.json` — compiler package (1 program, 5 TDs)
- `select-const-a.mil` — scalar const `-inf` a; still the hole
- `select-runtime-a.mil` — rewrite of that program
- `select-runtime-a/manifest.json` — 1 program, 5 TDs
- `rewrite-report.json` — encoder plan 24/24, one fill
- `rewritten-select-snippet.txt` — `a = var_8_to_fp16_rt`
- `runs.log` — compiler invocations, verbatim
- `test_select_runtime.log` — host tests 6/6

## Verification

- `overlay/tests/omarchy/coreml/test_select_runtime.py`: 6/6
- No hardware, no device, no model-pin change; host-only throughout.

Did not edit `conv_layout.py`, `heads_layout.py`, `mask_layout.py`,
`slice_layout.py`, `attention_layout.py`, `mask_lowering.py`,
Compiler.mm, or anything under `ane-linux-experiments` outside this
worktree. Did not emit the 2.2 MiB ninf packing. Did not compile the
full encoder.

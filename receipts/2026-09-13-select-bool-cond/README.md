# Select bool cond — fp16 0/1 cond becomes bool (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Assignment

Cast/materialize encoder select cond as bool (bit-exact `+0.0`/`1.0` →
false/true), then keep runtime-a. Envelope pin:
`mil-hwx-compiler` `receipts/2026-09-13-h13-select-envelope.md`
(`c2cf32e4`). Do not insert a device cast. Do not emit the ninf packing.

## The leftover

After `select_runtime` promoted const `-inf` `a`, the noslice encoder
still bound fp16 `var_373` (`1 + (-1 * mask)`) as cond. The MIL gate
refuses fp16-cond select before table lookup:

```text
h13.boolean-outside-envelope
```

exit 65, no output dir. `select_runtime` rewrote 0/24. Runtime-a plus a
bool cond at CHW `{8,375,375}` compiles. Apple's tool rejects the fp16
cond form; H13 has no device cast encoder.

## The rewrite

`overlay/tools/coreml/select_runtime.py` materializes an fp16 cond as
bool. A cond that is already a function input and is only consumed by
rewritten selects has its param dtype changed in place. An in-graph cond
(the encoder's `var_373`, still needed as fp16 by `cast_19`) is promoted
to a bool function input of the same shape; the host fills it with
`materialize_bool_cond` (`0x0000` → false, `0x3C00` → true; any other
bit pattern is a named error). Const `-inf` `a` still becomes runtime-a.
Do not emit a MIL `cast`.

Plan of the noslice encoder MIL is 24 rewritten / 0 refused / 1 fill /
1 bool cond (`var_373_bool`).

## Numpy proof

`materialize_bool_cond` maps bit-exact fp16 `+0.0`/`1.0` to bool. A
`0.5` lane raises `SelectRuntimeError`. `numpy_const_vs_runtime_fill`
on that bool cond stays bit-exact with runtime `-inf` fill.

## Standalone compile

Same `mil-hwxc` as the envelope pin (`c2cf32e4`,
sha256 `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`).

Runtime-a / runtime-b / **fp16** cond at `[1, 8, 375, 375]`:

```text
h13.boolean-outside-envelope
rc=65
```

Rewrite of const `-inf` `a` plus fp16 cond (param dtype → bool, fill →
`ninf_rt`):

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-select-bool-cond/select-bool-cond.mil \
  --model-root receipts/2026-09-13-select-bool-cond \
  --target H13 --output receipts/2026-09-13-select-bool-cond/select-bool-cond
compiled target=H13 artifacts=1 format=anec output=.../select-bool-cond
```

One program: `apple-parity-boolean`, `taskDescriptors = 5`. Exit 0.
Inputs: `ninf_rt` fp16, `b` fp16, `cond` bool.

## Files

- `select-fp16-cond.mil` — rewrite input: scalar `-inf` a, fp16 cond
- `select-bool-cond.mil` — rewrite product: runtime-a, bool cond
- `select-bool-cond/manifest.json` — compiler package (1 program, 5 TDs)
- `select-fp16-rrb.mil` — runtime-a / fp16 cond; still the hole
- `select-rrb.mil` — standalone runtime-a / bool cond
- `select-rrb/manifest.json` — 1 program, 5 TDs
- `rewrite-report.json` — noslice plan 24/0, one fill, one bool cond
- `rewritten-select-snippet.txt` — `cond = var_373_bool`, `a = var_8_to_fp16_rt`
- `runs.log` — compiler invocations, verbatim
- `test_select_runtime.log` — host tests 9/9

## Verification

- `overlay/tests/omarchy/coreml/test_select_runtime.py`: 9/9
- No hardware, no device, no model-pin change; host-only throughout.

Did not edit `conv_layout.py`, `heads_layout.py`, `mask_layout.py`,
`slice_layout.py`, `attention_layout.py`, `mask_lowering.py`,
Compiler.mm, or anything under `ane-linux-experiments` outside this
worktree. Did not insert a device cast. Did not emit the 2.2 MiB ninf
packing. Did not compile the full encoder.

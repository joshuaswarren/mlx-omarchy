# The exporter calls the fixed converter, and hands it the weight blob

`overlay/tools/ane-export/ane_export.py` ran its own converter fork. Both
defects fixed in `ane-linux-experiments` `52a3211`
(`receipts/2026-09-14-1x896-export-fix.json`) would have come back on the next
export, silently: the `.anec` would validate, ship, and compute the wrong
thing.

## Before

```
convert = run_tool(
    [sys.executable, "hwxv2-to-anec-patched.py",
     str(hwx_output / "model.hwx"), str(anec_path),
     str(in_elems), str(out_elems)],
    cwd=tools_dir, stage="hwxv2-to-anec",
)
```

`hwxv2-to-anec-patched.py` is a scratch-dir fork carried since
`receipts/2026-08-31-mil-oneop-proof.md` (deviation 1) for one reason: the
widened `0x4401F800` TD flag word. No `--weights`. Two consequences:

- the converter read `__TEXT,__const` from byte 0, so 64 words of blob header
  became coefficients and the real payload tail was lost;
- nchw plane and row bytes came from the convention, not from the task's
  tile-DMA totals and run counts.

## After

```
CONVERTER = "hwxv2-to-anec.py"
CONVERTER_MARKERS = ("--weights", "derive_strides")
...
    check_converter(tools_dir / CONVERTER)
...
    [sys.executable, CONVERTER,
     str(hwx_output / "model.hwx"), str(anec_path),
     str(in_elems), str(out_elems), "--weights", str(weights_path)],
```

The fork is gone; the exporter runs the canonical
`ane-linux-experiments tools/hwxv2-to-anec.py`, copied into the mac scratch
dir unmodified, and passes the `capture/weights.bin` it already had in hand.
No third copy of the converter exists. The fork's only reason to exist went
away in `a29a395`: the canonical converter matches the task record by register
and takes `0xF401F800` and `0x4401F800` alike.

Canonical converter at the time of this change:
`6c8b5266de3fe6fc9eef7863b5a03309355ceb5a388b04c2b09d637e62e4ccb8`
(`ane-linux-experiments` `52a3211`, `tools/hwxv2-to-anec.py`).

## The drift guard

Pinning a hash would break on every legitimate converter change, so the guard
is a capability check on the two names `52a3211` introduced. `check_converter`
refuses a converter that is missing, or that carries neither `--weights` nor
`derive_strides`, and says where to get a current one. A stale copy now fails
the export before the compiler runs instead of producing a quiet, wrong
bundle.

`overlay/tools/ane-export/test_ane_export.py` runs the exporter end to end on
Linux against a stub `ane-compile-hwx` and a stub converter - no mac, no
ANECompiler, no device:

- `test_converter_is_called_with_the_weight_blob` - the converter argv carries
  `--weights` and the real `capture/weights.bin` path.
- `test_export_is_refused_when_the_weight_blob_is_missing` - a converter that
  refuses a blob-backed object without its `weights.bin` fails the export; no
  `model.anec` is written.
- `test_a_converter_predating_the_fix_is_refused` - a pre-`52a3211` copy is
  rejected by name.
- `test_a_missing_converter_names_its_source` - the error points at
  ane-linux-experiments.
- `test_the_canonical_converter_satisfies_the_markers` - the guard tracks the
  real converter (skips when `$ANE_LINUX_EXPERIMENTS` /
  `~/src/ane-linux-experiments` is absent).

```
$ cd overlay/tools/ane-export && python3 -m pytest test_ane_export.py -q
.....                                                                    [100%]
5 passed in 0.23s
```

Mutation check - reverting both halves of the change (drop `--weights`, drop
the `check_converter` call) and re-running:

```
3 failed, 2 passed in 0.29s
FAILED test_ane_export.py::test_converter_is_called_with_the_weight_blob
FAILED test_ane_export.py::test_a_converter_predating_the_fix_is_refused
FAILED test_ane_export.py::test_a_missing_converter_names_its_source
```

## What is not proven here

No export was run. Both macs were unreachable this session (16m1mbp and jw14m2
time out, macstudio has no export tree), so there is no new `.hwx`, no new
`.anec`, and no on-device execution behind this change. The on-device evidence
that the converter behaviour is right remains
`ane-linux-experiments receipts/2026-09-14-1x896-export-fix.json` (64-element
control, re-exported 1x512 and 1x896, all EXACT on jwm1). What this receipt
proves is only that the exporter now invokes that converter, with the weight
blob, and refuses anything older.

Also untracked `overlay/tools/ane-export/__pycache__/ane_export.cpython-311.pyc`,
a committed build artifact already covered by `.gitignore`.

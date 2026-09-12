# Core ML package inspector

Phase 2 of `docs/plans/2026-09-12-coreml-parakeet-ane-plan.md`: a
complete, public `.mlpackage` inspector for Linux. It replaces the
hand-written wire-format parser (1126 lines of guessed message layouts
plus a 178-line varint codec) whose dtype tables were wrong — for
example it labeled MIL `BOOL=1` as `float16`. Nothing was patched:
the guessed code is deleted and the official schema is now the single
source of truth.

## Usage

```bash
python3 overlay/tools/coreml/inspect_mlpackage.py inspect PATH [options]
# or, with overlay/tools on PYTHONPATH:
python3 -m coreml.inspect_mlpackage inspect PATH [options]
```

Options:

- `--json` — machine-readable inventory (schema
  `mlx-omarchy.coreml.inventory/1`)
- `--strict` — exit 2 on unresolved weight blob references, unset
  model type, or opset inconsistencies (parse validity itself is
  unaffected)

Exit codes: `0` inspected, `1` invalid package/usage,
`2` strict findings.

Inspection never opens the ANE device, never imports `coremltools`
or MLX, and never reads or computes tensor data. It is pure file and
protobuf reading and runs device-free on any Linux host (x86_64 and
aarch64).

## What it reports

Everything on the plan's section-14 checklist, and nothing invented:

- package format and `fileFormatVersion`,
- model/spec type (which `Model.Type` oneof is set, e.g.
  `mlProgram`) and `specificationVersion`,
- per function: `opset` name, program version, block
  specializations, every operation with its parameter bindings
  (name references and typed inline values), typed outputs,
  attributes, and nested blocks,
- input/output/state names with **FeatureType** dtypes and shapes,
  including shape flexibility (`enumeratedShapes` / `shapeRange`,
  unbounded ranges stay unbounded),
- MIL-level tensor dtypes in the distinct **MIL** enum, including
  symbolic (`unknown`) dimensions and rank,
- external weight files with size and sha-256, every blob reference
  (file + offset) and which values it resolves to,
- compression representation as the schema expresses it:
  `constexpr_*` operations (`constexpr_lut_to_dense`,
  `constexpr_affine_to_dense`, ...) with their typed inputs,
- control flow (operations carrying nested blocks),
- operation histogram and totals,
- parse validity and confidence notes (unset model type, opset
  inconsistencies, missing files — always named, never blank), plus a
  recursive scan counting fields written by a schema newer than the
  vendored one (`unknown_schema_fields`: the forward-compatibility
  limit is reported, never silently hidden),
- compiler eligibility: explicitly **not assessed**. Eligibility
  claims require the target compiler's coverage data; this inspector
  refuses to guess deployment targets or compiler support. This
  separation is deliberate: parse validity and eligibility are
  different claims and never mixed.


## Official schema, vendored

`overlay/tools/coreml/schema/` vendors the official Core ML protobuf
schema: the 33 generated `*_pb2.py` bindings and the 33 `.proto`
sources from Apple's `coremltools` **9.0** release (PyPI sdist,
sha-256 `4ff346b2…f4926687`; GitHub tag `9.0`), plus the Apple
BSD-3-Clause license text. Provenance and per-file hashes are pinned
in `schema/VENDORED.json` and verified by
`overlay/tests/omarchy/coreml/test_schema.py`.

The only legacy file in the official package, `NamedParameters_pb2.py`,
is excluded: it is pre-3.x generated code that fails to import under
the pinned protobuf runtime and nothing in `Model_pb2` references it.
The exclusion is recorded in `VENDORED.json`.

Runtime dependency: `protobuf` (tested with 7.36.0). Linux aarch64 is
covered by upstream wheels: `protobuf-7.36.0-cp310-abi3-manylinux2014_aarch64.whl`
plus the pure-Python `py3-none-any` wheel on PyPI. Importing the
vendored bindings directly avoids the heavyweight `coremltools` root
import (which drags TensorFlow into the process) entirely.

## Dtype semantics: two enums, never conflated

Core ML has two independent dtype numberings and the inspector
preserves both exactly (from the generated descriptors, never from
hand-written tables):

| value | FeatureType (`ArrayFeatureType`) | MIL (`MILSpec.DataType`) |
|------:|---------------------------------|--------------------------|
| 1     | —                               | BOOL                     |
| 2     | —                               | STRING                   |
| 10    | —                               | FLOAT16                  |
| 11    | —                               | FLOAT32                  |
| 23    | —                               | INT32                    |
| 35    | —                               | UINT4                    |
| 65552 | FLOAT16                         | —                        |
| 65568 | FLOAT32                         | —                        |
| 131104| INT32                           | —                        |

A Parakeet encoder therefore legitimately reads `FLOAT32` mel input at
the model boundary and `FLOAT16` tensors inside the MIL body. The
inspector reports both as-is; it never forces, coerces, or assumes
fp16. Unknown enum values are reported as `UNRECOGNIZED(n)`, never
blank.

## Validation behavior

Structural problems raise `coreml.mlpackage.MlPackageError` with a
specific message: missing/unparsable manifest, empty itemInfoEntries,
absolute or `..`-traversing manifest paths, missing or ambiguous
model entries, missing model bytes. Corrupt or truncated model
protobufs raise `coreml.proto.ModelSpecError` naming `model.mlmodel`.
Weight blob references that do not resolve inside the package are
reported explicitly (`resolved: false`) and fail under `--strict`.

## Tests

```bash
python3 -m unittest discover -s overlay/tests/omarchy/coreml \
    -t overlay/tests/omarchy/coreml
```

Coverage: vendored-manifest hash verification, official enum
semantics, malformed/truncated protobuf, manifest path escapes,
missing weights, boundary type/shape ground truth for the three real
Parakeet packages (skipped when the download cache is absent), and a
consumer contract test in which a phase-3-style SSA walk over the
inventory fails with a precise error naming the op, parameter, and
undefined value.

## Receipts

`receipts/2026-09-12-coreml-inspector/` holds the exact per-op typed
inventories for the three pinned public packages (encoder gzipped;
contents verified byte-identical after decompression), a verdict
with environment and weight hashes, and a comparison against the old
parser's artifact: op counts/histograms agree (the old 3351/69/21
histogram is now independently verified), while the old artifact's
boundary dtype/shape entries were all `None` — blank values presented
as data, which the new inspector refuses to emit.

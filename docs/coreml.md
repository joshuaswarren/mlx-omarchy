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
- `--compiler-coverage REPORT.json` — attach a source-matched static H13
  coverage report; does not run a classifier or compiler.

Exit codes: `0` inspected, `1` invalid package/coverage report,
`2` strict findings or argument usage errors.

Inspection never opens the ANE device or imports Core ML or MLX. Weight
files are streamed for hashing; tensor values are not decoded or computed.
The inspector runs device-free on Linux x86_64 and aarch64.

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
- compiler eligibility: unassessed without an explicit coverage report.
  With one, the CLI verifies its model SHA-256, complete weight SHA-256
  map, exact H13 compiler identity, and nonnegative classification totals
  against the parsed inventory. It reports the supplied counts and source,
  while `compilable` stays `null`: static coverage is not a compile or
  execution qualification. The report source is declared, not authenticated.

The pinned public encoder report is
[`2026-09-12-parakeet-h13-coverage.json`](../receipts/2026-09-12-parakeet-h13-coverage.json).
Use it with the downloaded encoder:

```bash
python3 overlay/tools/coreml/inspect_mlpackage.py inspect /path/to/encoder.mlpackage \
  --compiler-coverage receipts/2026-09-12-parakeet-h13-coverage.json --strict
```

Its 3,351 classified operations include 427 requiring normalization and
689 missing semantics or supported shapes. The source CSV is linked by
immutable commit in the JSON. The report format contains `model_sha256`,
`weights` (package-relative path to hash), `compiler` (repository, full
commit, H13 target), `source`, and `counts`. Counts must contain all six
categories shown in that report and sum to the inventory operation count.
Different model or weight bytes reject the report with exit 1.


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
absolute, traversing, or escaping symlink paths, an invalid
rootModelIdentifier, or missing model bytes. The manifest root selects
the model regardless of its filename. Corrupt or truncated model
protobufs raise coreml.proto.ModelSpecError. Weight references are collected
from attributes and input bindings at every nesting depth, resolved exactly
relative to the model file, and checked against the file size. Missing
files or out-of-range offsets report resolved: false and fail --strict.

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

## Compiler preparation and execution boundary

`scripts/prepare-ane-compiler.sh` downloads and hash-verifies the external
compiler pinned in `ane-compiler.lock`. `scripts/verify-ane-compiler.sh all`
checks archive rejection, Linux compilation, compiler tests, and emission
of a known H13 graph. These are host checks, not ANE execution proof.

The [integration review](../receipts/2026-09-12-coreml-integration-review.json)
records the executed checks and links the complete encoder coverage table.
The public encoder remains unqualified: required H13 semantics and shapes
are missing, and the mixed-type, two-output contract is not represented by
the current compiler program model. Execution work stops at plan section 62;
no CPU fallback or altered model contract substitutes for these gaps.

The [known-H13 adapter review](../receipts/2026-09-12-h13-adapter-review.json)
finds a separate bundle-v2 blocker. All 4,097 known-graph programs have no
workspace tiles, but the schema requires positive workspace geometry, size,
and stride. Removing only the byte-size check would not resolve the other
constraints. No workspace or firmware identity is fabricated; the adapter
remains unimplemented pending an explicit absent-workspace contract.


The [licensed reference receipt](../receipts/2026-09-12-licensed-parakeet-reference.json)
replaces the undocumented-rights JFK clip with a byte-verified CC-BY-4.0
LibriSpeech utterance. ANE and GPU emit the same 104 tokens, including the
pinned reference's erroneous suffix; CPU emits 100. Encoder tensors pass the
unchanged tolerances. This is reference capture, not Linux ANE execution or
evidence of clean transcription. No Core ML feature release is claimed.

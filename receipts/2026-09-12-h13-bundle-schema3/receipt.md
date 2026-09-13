# H13 bundle schema 3 adapter

This change replaces the rejected bundle formats with schema 3. It does not
retain a version 1 or version 2 parser. Schema 3 represents every compiler
program, dispatch order, intermediate tensor, tensor slice, channel, NCHW
geometry, allocation size, and scratch allocation. It requires
`compiler.target: h13` and unsigned `driver_abi_major: 1`. Dotted firmware
fields and the old workspace field are not accepted.

The Linux adapter consumes only explicit `mil-hwxc.h13-anec-package.v1`
packages produced with `--format anec`. Before writing a bundle, it verifies
the generation receipt, the exact compiler-manifest and graph-source digests,
the payload set, and every payload digest. Both exporters derive
`release_asset.model_sha256` from the same canonical payload-record byte
domain. The C++ loader verifies this identity, every payload size and digest,
ANEC-header geometry, slice coverage, and initialization order before it
assembles programs in declared dispatch order.

## Source identity

- mlx-omarchy base: `309cd745d40117b689e8936d3ef62fcea262b232`
- Compiler repository receipt: `f4ad09066b560818a9dcc1f5f4f53273d941bd1e`
- Explicit ANEC executable source: `aa688df66cbc2110e0df94f0d50fb72c7fa30a18`
- Generation-time compiler binary SHA-256:
  `45fa6cb33e86ac9ec5e35421d338e07c9cd2cfb55b91e26242d9aa2ebeadffec`

The explicit compiler command was:

```console
$COMPILER/build/mil-hwxc --target H13 --format anec \
  --mil $WORK/model.mil --model-root $WORK/models \
  --output $WORK/pkg-explicit
```

The committed package is
[`../fixtures/h13-explicit-chain-add-mul`](../fixtures/h13-explicit-chain-add-mul).
Its graph source SHA-256 is
`5584d0fd8d40027229890408e924e6f7930cd5f02516466a442193408482ce76`.
Its compiler manifest SHA-256 is
`d3be01476fa2a3b0c6d24ca3227fd88885e6370807eda6c16d0a93775ba20e19`.
Its two ANEC payload SHA-256 values are:

- `program-0.anec`: `9a6a6a9a701ce207d4a8ea80ee1ca6d5ade076be2115b73a9aed6f4236aa56ab`
- `program-1.anec`: `62595e4a61db24066b83a4f7c077d459c69a6a740109098ee0122b4b46dcd3cc`

## Verification

The final host-only C++ run produced:

```text
[doctest] test cases:   19 |   19 passed | 0 failed | 0 skipped
[doctest] assertions: 1706 | 1706 passed | 0 failed |
[doctest] Status: SUCCESS!
```

The adapter unit command
`python3 -m unittest -v overlay.tests.omarchy.ane.test_h13_package_to_bundle`
passed all 7 tests. Python bytecode compilation also passed for
`h13_package_to_bundle.py`, `ane_export.py`, and
`bundle_payload_identity.py`.

A fresh `scripts/prepare-mlx.sh` copy matched the edited source before the
normal CMake build:

```text
[receipt] prepared-source-match mlx/backend/omarchy/ane/manifest.cpp sha256=002a4d59ae90938acc6ec53cc5a7ad21b23a6883739a982a55b420823862db8d
[receipt] prepared-source-match mlx/backend/omarchy/ane/bundle.cpp sha256=c988558738d12dda5d728313d691c92cd0023f8517a51cb35798a33b77ba89d3
[receipt] prepared-source-match tests/omarchy/ane/test_bundle.cpp sha256=570efe571aa2f3b867e061fbdf93a6f2e525ff135bfdd498bf7277dec60c191a
```

A fresh CLI adaptation of the committed compiler package, followed by the real
`mlx-omarchy-info --check-bundle` consumer, reported:

```text
h13_package_to_bundle: PASS programs=2 payloads=2 output=/tmp/mlx-omarchy-h13-final.BJlUdN/bundle
[receipt] graph: h13-first-run-chain-add-mul
[receipt] task_descriptors: 2
[receipt] dispatch_plan: 0 1
[receipt] dispatch 0: program=0 payload=program-0.anec operation=add encoder=h13-oracle-parity scratch_bytes=0 payload_size=16896 td_size=504 td_count=1 sources=2 destinations=1
[receipt] dispatch 1: program=1 payload=program-1.anec operation=mul encoder=h13-oracle-parity scratch_bytes=0 payload_size=16896 td_size=504 td_count=1 sources=2 destinations=1
[receipt] compiler: host_build=Linux 6.17.2-1-pve x86_64 toolchain=mil-hwxc aa688df66cbc2110e0df94f0d50fb72c7fa30a18 sha256:45fa6cb33e86ac9ec5e35421d338e07c9cd2cfb55b91e26242d9aa2ebeadffec target=h13
[receipt] driver_abi_major: 1
[receipt] provenance: repo=joshuaswarren/mlx-omarchy commit=b3f199aa172aee48047a73198fd21c8f44fb799d
[receipt] OK: bundle valid
```

The receipt-bound compiler manifest and generated release identity also
matched their independent recomputations:

```text
[receipt] compiler_manifest_sha256 expected=d3be01476fa2a3b0c6d24ca3227fd88885e6370807eda6c16d0a93775ba20e19 actual=d3be01476fa2a3b0c6d24ca3227fd88885e6370807eda6c16d0a93775ba20e19
[receipt] release_asset.model_sha256 declared=eddf1f251990ee72d24f53f9bf72eb1e8f71966af5551118fe4c2d0c1e32a5a5 canonical=eddf1f251990ee72d24f53f9bf72eb1e8f71966af5551118fe4c2d0c1e32a5a5
```

The same CLI accepted all four migrated repository fixtures:

- `receipts/fixtures/exported/ane-add-fp16-1x512`
- `receipts/fixtures/exported/ane-add-fp16-1x896`
- `receipts/fixtures/exported/ane-mul-fp16-1x512`
- `receipts/fixtures/mil-oneop-bundle`

The negative C++ cases cover old schema rejection, removed firmware fields,
missing, signed, and wrong ABI values, invalid dispatch plans, dispatch
use-before-write, incomplete final output ranges, incomplete payload mappings,
unknown tensors, invalid slices, physical/NCHW mismatches, packed-tile and
allocation mismatches, release-identity mismatches, exact ANEC file-length
mismatches after refreshed metadata and identity, task and scratch mismatches,
payload digest ordering, unknown files, missing files, and absent bundle
directories. Positive cases cover a legitimate two-program union of output
ranges, a two-program input/output chain, the independent slice coordinate
case top-level 1024/offset 512/count 384/physical 512, and the independently
pinned canonical digest for a non-ASCII payload path. Python producer tests
cover the output-union rule, the same independent slice coordinates, and the
generation-receipt digest boundary.

## Documentation cutover

This slice migrates `docs/ane-bundles.md`, `docs/compatibility.md`, and
`docs/architecture.md`, plus the exporter README. The parallel
`docs/coreml.md` schema 3 migration is commit `e81f45b6`, included in the
root-verified joint integration target `b7ff5c2c`. Its receipt is
`receipts/2026-09-12-parakeet-mel-frontend/receipt.json`. Root integration
must use `b7ff5c2c`. Runtime-generated bundles belong in the cache, not
in source control.

## Qualification boundary

No device was opened and no ANE program was executed. These results establish
schema, provenance, payload, ANEC-header, and CLI-consumer compatibility only.
They do not establish device eligibility, numerical correctness, physical
T8103 or T6000 identity, or compiler-wide qualification. The compiler's current
unrelated HWX extraction regression remains outside this adapter result.

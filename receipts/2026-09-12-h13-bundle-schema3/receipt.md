# H13 bundle schema 3 adapter

This change replaces the rejected bundle formats with schema 3. It does not
retain a version 1 or version 2 parser. Schema 3 represents every compiler
program, dispatch order, intermediate tensor, tensor slice, channel, NCHW
geometry, allocation size, and scratch allocation. It requires
`compiler.target: h13` and unsigned `driver_abi_major: 1`. Dotted firmware
fields and the old workspace field are not accepted.

The Linux adapter consumes only explicit `mil-hwxc.h13-anec-package.v1`
packages produced with `--format anec`. It verifies the generation receipt,
graph source digest, complete payload set, and every payload digest before it
writes a bundle. The C++ loader verifies every payload size and digest before
it parses any ANEC header. It then validates each program against its header
and assembles programs in declared dispatch order.

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
Its two ANEC payload SHA-256 values are:

- `program-0.anec`: `9a6a6a9a701ce207d4a8ea80ee1ca6d5ade076be2115b73a9aed6f4236aa56ab`
- `program-1.anec`: `62595e4a61db24066b83a4f7c077d459c69a6a740109098ee0122b4b46dcd3cc`

## Verification

The final host-only C++ run produced:

```text
[doctest] test cases:   15 |   15 passed | 0 failed | 0 skipped
[doctest] assertions: 1186 | 1186 passed | 0 failed |
[doctest] Status: SUCCESS!
```

The adapter unit command
`python3 -m unittest overlay.tests.omarchy.ane.test_h13_package_to_bundle`
passed 4 tests. Python bytecode compilation also passed for
`h13_package_to_bundle.py` and `ane_export.py`.

A fresh CLI adaptation of the committed compiler package reported:

```text
h13_package_to_bundle: PASS programs=2 payloads=2
[receipt] graph: h13-explicit-chain-add-mul
[receipt] task_descriptors: 2
[receipt] dispatch_plan: 0 1
[receipt] dispatch 0: program=0 payload=program-0.anec operation=add encoder=h13-oracle-parity scratch_bytes=0 payload_size=16896 td_size=504 td_count=1 sources=2 destinations=1
[receipt] dispatch 1: program=1 payload=program-1.anec operation=mul encoder=h13-oracle-parity scratch_bytes=0 payload_size=16896 td_size=504 td_count=1 sources=2 destinations=1
[receipt] compiler: host_build=Linux 6.17.2-1-pve x86_64 toolchain=mil-hwxc aa688df66cbc2110e0df94f0d50fb72c7fa30a18 sha256:45fa6cb33e86ac9ec5e35421d338e07c9cd2cfb55b91e26242d9aa2ebeadffec target=h13
[receipt] driver_abi_major: 1
[receipt] OK: bundle valid
```

The same CLI accepted all four migrated repository fixtures:

- `receipts/fixtures/exported/ane-add-fp16-1x512`
- `receipts/fixtures/exported/ane-add-fp16-1x896`
- `receipts/fixtures/exported/ane-mul-fp16-1x512`
- `receipts/fixtures/mil-oneop-bundle`

The negative C++ cases cover old schema rejection, removed firmware fields,
missing, signed, and wrong ABI values, invalid dispatch plans, dispatch
use-before-write, incomplete payload mappings, unknown tensors, invalid slices,
zero-sized ordinary tensors, channel and allocation mismatches, task and
scratch mismatches, malformed ANEC headers, payload digest ordering, unknown
files, missing files, and absent bundle directories.

## Documentation cutover

This slice migrates `docs/ane-bundles.md`, `docs/compatibility.md`, and
`docs/architecture.md`, plus the exporter README. The parallel
`docs/coreml.md` schema 3 migration is commit `e81f45b6` on
`wave/CoremlFrontendIntegration`; its receipt is
`receipts/2026-09-12-parakeet-mel-frontend/receipt.json`. The root integration
must include that commit. Runtime-generated bundles belong in the cache, not
in source control.

## Qualification boundary

No device was opened and no ANE program was executed. These results establish
schema, provenance, payload, ANEC-header, and CLI-consumer compatibility only.
They do not establish device eligibility, numerical correctness, physical
T8103 or T6000 identity, or compiler-wide qualification. The compiler's current
unrelated HWX extraction regression remains outside this adapter result.

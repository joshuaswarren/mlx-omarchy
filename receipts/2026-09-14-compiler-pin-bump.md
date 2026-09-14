# Compiler pin bump: mil-hwx-compiler 83a4434 → 417554c (2026-09-14)

Every island bundle executed on hardware today was compiled from
`feature/h13-concat`, while `ane-compiler.lock` still pinned `83a4434`.
This lands that branch on the compiler's `main`, publishes the immutable
archive, and moves the pin so the product build reproduces today's
bundles from its own lock.

Host: omp-studio-local (x86_64 Linux 6.17.2-1-pve). No device access.

## mil-hwx-compiler

- Merge: fast-forward. `main` `83a4434` → `417554c3e22de9124623765717acee25e9860f19`
  (`git merge --ff-only feature/h13-concat`, no merge commit, no squash; the
  27 code commits cited elsewhere by hash are unchanged, plus three
  receipts-only commits `5769476`, `c083a90`, `2931812` that the branch
  gained after `7ab3eb5`, and the THEORY.MD commit below).
- `417554c` "theory: select binds a to template 4; polarity is the operand
  swap" commits the previously uncommitted THEORY.MD edit; it documents
  `37fb29e` and cites receipts that exist in-tree. `lib/`, `plugins/`,
  `tools/`, `tests/` are byte-identical to `7ab3eb5`.
- Gates on the merged tree, before the push:
  - `make test-h13`: `H13_ENCODING_OK`, `test_h13_anec`, and all 15 CLI
    suites `PASS` (cli, split, layout, composite, batched, registry,
    boolean, tile, linear tiling, matmul/silu/norm/conv/select/chain
    envelope).
  - `python3 tests/test_h13_parity.py build/mil-hwxc`:
    `H13 oracle parity: PASS (846 cases, 182 matmul, 79 broadcast,
    105 softmax/layer_norm, 114 reduction, 284 convolution, 1692 artifacts)`.
- Pushed: `main 83a4434..417554c`, `feature/h13-concat 2931812..417554c`.
- Tag: `ane-parity-417554c` (annotated) → `417554c3e22de9124623765717acee25e9860f19`.
- Archive: `git archive --format=tar.gz --prefix=mil-hwx-compiler-417554c/ 417554c…`,
  3780 entries, 34,369,240 bytes (the branch adds captures and receipts;
  the 83a4434 archive was 4,454,171 bytes).
- Release: https://github.com/joshuaswarren/mil-hwx-compiler/releases/tag/ane-parity-417554c
  - `mil-hwx-compiler-417554c.tar.gz` 34369240 bytes, GitHub digest
    `sha256:16046efc43921cc3d55defc5f5fdc17aa7b6bcaba90b836cae9985d2154a720d`
  - `mil-hwx-compiler-417554c.tar.gz.sha256` sidecar, 98 bytes.

## mlx-omarchy lock (`ane-compiler.lock`)

```
ANE_COMPILER_REPOSITORY=https://github.com/joshuaswarren/mil-hwx-compiler
ANE_COMPILER_COMMIT=417554c3e22de9124623765717acee25e9860f19
ANE_COMPILER_ARCHIVE_URL=https://github.com/joshuaswarren/mil-hwx-compiler/releases/download/ane-parity-417554c/mil-hwx-compiler-417554c.tar.gz
ANE_COMPILER_ARCHIVE_SHA256=16046efc43921cc3d55defc5f5fdc17aa7b6bcaba90b836cae9985d2154a720d
ANE_COMPILER_PACKAGE_SCHEMA=mil-hwxc.h13-anec-package.v2
ANE_COMPILER_QUALIFIED_TARGET=H13
```

Schema and target are unchanged. `docs/dependency-licenses.md` and the
stale "pin still names the v1 archive" paragraph in `docs/ane-bundles.md`
now name the new release.

## Gates on the bumped tree (worktree from `origin/main` `b79a4b68`)

- `scripts/verify-ane-compiler.sh all` (2m42s wall, log at
  `/tmp/relpin/verify-all.log` on this host): wrong archive rejected before
  extraction and build `PASS`; prepare downloaded the locked URL, archive
  SHA-256 `16046efc…` matched, built `.work/ane-compiler/bin/mil-hwxc`
  (binary SHA-256 `1b8bf9dc0c3644806ccbd5789c3a6bd9c0a0d9f9e06ccd536ca74cd5d5145053`);
  operation graph, HWX object writer, program composition, H13 CLI, all
  h13 cli suites, HWX inspection, linux compiler software tests and hygiene
  `PASS`; conv_relu H13 qualification emitted 4097 programs, tile program
  `3d73787c1fc04842da72bb6e1a1930ffa1d3c7b7280ab7d838add2da2edfba22`
  (same as the 83a4434 pin); `verify-ane-compiler: PASS`.
- `python3 overlay/tests/omarchy/ane/test_h13_package_to_bundle.py`:
  `Ran 13 tests in 0.081s OK`.
- `python3 -m unittest discover -s overlay/tests/omarchy/coreml -t overlay/tests/omarchy/coreml`:
  `Ran 149 tests in 16.054s OK (skipped=1)`.
- `.work/build/tools/mlx-omarchy-info --check-bundle` (built from this tree,
  cmake targets `mlx-omarchy-info omarchy_ane_bundle_tests`), all
  `[receipt] OK: bundle valid` EXIT:0:
  `receipts/fixtures/exported/ane-add-fp16-1x512`,
  `receipts/fixtures/exported/ane-add-fp16-1x896`,
  `receipts/fixtures/exported/ane-mul-fp16-1x512`,
  `receipts/fixtures/mil-oneop-bundle`.
- `.work/build/tests/omarchy/omarchy_ane_bundle_tests`:
  `24 passed | 0 failed`, `3183 assertions`, `Status: SUCCESS!`.

## Not done here

- `overlay/tools/coreml/eligibility.py` still freezes the H13 lowering
  surface extracted from `83a4434` and lists `less`, `floor`, `floor_div`
  as `SUPPORTED-PENDING-COMPILER-RELEASE` citing `b4f4da9`. `b4f4da9` is
  now an ancestor of the pinned commit, so that table is due for
  re-extraction against `417554c` (`plugins/H13/ANEH13Compiler.mm`), with
  `test_eligibility.py` updated with it. That is a coverage re-derivation,
  not a pin edit, and was not part of this change.
- No ANE submission, no bundle re-execution: the pin now matches the source
  today's hardware bundles were compiled from; it does not re-qualify them.

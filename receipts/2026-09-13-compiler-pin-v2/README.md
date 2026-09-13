# Compiler pin upgrade to the v2 package schema (2026-09-13)

Repoints `ane-compiler.lock` from the v1 archive (a0ce354,
`mil-hwxc.h13-anec-package.v1`) to the ordered-MIL-result merge on
`mil-hwx-compiler` main, which emits `mil-hwxc.h13-anec-package.v2`.
Closes the stale-pin half of Phase 3 / gate 46; the scripts
(`scripts/prepare-ane-compiler.sh`, `scripts/verify-ane-compiler.sh`)
already source the lock and needed no changes. The strict v2 package
adapter in `overlay/tools/ane-export/` was not touched and not weakened.

Scope: host only. No device access, no bundle adaptation (`load_bundle`),
no ANE submission, no runtime qualification, and no claim of either.

## Emission verification at source (commit 83a4434)

- `plugins/H13/ANEH13Compiler.mm:2488` writes
  `"schema": "mil-hwxc.h13-anec-package.v2"` into every H13 package
  manifest; the manifest now records physical outputs and logical
  results (ordered-result support) on top of the v1 fields.
- `tools/mil-hwxc.mm:59` defaults `artifactFormat` to `anec` for the
  qualification path; `hwx` stays an explicit opt-in.
- In-tree tests pin both: `tests/test_h13_cli.py:395` asserts
  schema v2 + `artifactFormat: anec` + byte-identical recompilation;
  `tests/test_h13_split_cli.py:195` and `research/inspect_anec.py:463`
  require v2; `tests/hardware/run_h13.mm:192` rejects anything else
  before hardware use.

## New immutable release

- tag: `ane-parity-83a4434`, target commit
  `83a4434810b0981d1764e6231f6a679df6f35241`
  ("Merge origin/main with ordered MIL result support")
- archive: `git archive --format=tar.gz --prefix=mil-hwx-compiler-83a4434/`
  from the exact commit; single top-level directory, 3565 entries,
  4,454,171 bytes
- archive URL:
  `https://github.com/joshuaswarren/mil-hwx-compiler/releases/download/ane-parity-83a4434/mil-hwx-compiler-83a4434.tar.gz`
- archive SHA-256:
  `2fbb0e2bc21d729ab114cf9623d31cbccee73fdb84b0ef8d4c078c91d1a05b57`
  (matches GitHub's asset digest `sha256:2fbb0e2b…` exactly)
- a `.tar.gz.sha256` sidecar asset is published next to the tarball

## Lock change (`ane-compiler.lock`)

- `ANE_COMPILER_COMMIT` a0ce354… → `83a4434810b0981d1764e6231f6a679df6f35241`
- `ANE_COMPILER_ARCHIVE_URL` / `_SHA256` → the new release asset
- `ANE_COMPILER_PACKAGE_SCHEMA` → `mil-hwxc.h13-anec-package.v2`
- `ANE_COMPILER_QUALIFIED_TARGET` stays `H13`
- the v1 archive is superseded; the adapter must not re-accept it

## Exact commands and results

`scripts/verify-ane-compiler.sh all` from a clean work area (full log in
`verify-all.log`, 64.91 s wall):

1. Wrong-archive rejection — valid gzip with wrong bytes placed at the
   archive path; prepare refuses with `archive SHA-256 mismatch; refusing
   to build` before extraction or build. PASS.
2. Prepare — downloads the new release asset end-to-end from the locked
   URL, verifies SHA-256, extracts, builds `build/mil-hwxc` against the
   existing GNUstep prefix with `ldd` linking assertions, runs the
   compiler's host checks. PASS. Provenance in `provenance.txt`.
3. Compiler host checks: operation graph PASS, HWX object writer PASS,
   program composition PASS, H13 MIL-to-ANEC/HWX CLI PASS (device-free),
   h13 split cli PASS, HWX inspection PASS, linux compiler software
   tests PASS, linux compiler hygiene PASS.
4. Known H13 graph — `tests/fixtures/conv_relu.mil` with in-tree weights,
   `--target H13`: manifest asserts
   `schema == mil-hwxc.h13-anec-package.v2`, `target == H13`,
   `artifactFormat == anec`; 4097 programs, all non-empty;
   `research/inspect_anec.py` passes device-free.
5. Package hash summary (`h13-package-sha256.txt`): 4096 identical
   conv-tile programs
   (`3d73787c1fc04842da72bb6e1a1930ffa1d3c7b7280ab7d838add2da2edfba22`),
   one distinct tail program
   (`73754c91606704bbd3cd23e3265529d7281e8a5cd92c58aa53c5dd310503cfaa`),
   manifest
   (`47db317d2c18d3f708c383587d76f9734b55d9c7f19d5e8f8815b1d1de112268`).
6. Determinism cross-check: a second independent prepare from the same
   archive bytes rebuilt a byte-identical compiler binary (SHA-256
   `ab3d1d4e8d69b43f45b2031c1a638bff59ca5e2ad55aa9f48f9713438dafc7ad`);
   recorded in `binary-determinism.txt`.

## Host

omp-studio-local, x86_64 Linux, kernel 6.17.2-1-pve. GNUstep prefix
`/home/joshuawarren/.local/mil-hwx-gnustep` (existing install, reused;
dependency pins live in the locked source's
`scripts/bootstrap-linux-toolchain.sh`).

## Not established by this receipt

- ANE device execution of the produced package.
- Bundle adaptation into `mlx-omarchy` (`load_bundle` path).
- Runtime qualification of H13 on M1, or Parakeet encoder compiler
  coverage (plan sections 47 onward; the encoder inventory comparison
  against compiler coverage is the next open leaf).

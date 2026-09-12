# Compiler lock, preparation, and known-H13 host qualification (2026-09-12)

Implements plan sections 7-9 (compiler integration strategy, compiler lock
and preparation, no vendoring) and the host-compile portion of Phase 3 /
gate 46 of `docs/plans/2026-09-12-coreml-parakeet-ane-plan.md`.

Scope: host only. No device access, no bundle adaptation (`load_bundle`),
no ANE submission, no runtime qualification, and no claim of either.

## Artifacts

- `ane-compiler.lock` — pinned repository, commit, archive URL, release
  tag, archive SHA-256, package schema, qualified target.
- `scripts/prepare-ane-compiler.sh` — deterministic download, hash
  verification before any build, extraction into the ignored
  `.work/ane-compiler` subtree, Linux build, the compiler's required
  host-side tests, a deterministic wrapper exposing the binary with the
  correct GNUstep runtime environment, and provenance output.
- `scripts/verify-ane-compiler.sh` — clean-area verification: source hash
  rejection before build, prepare, known H13 graph compilation, and
  device-free package validation.

## Provenance (from the lock, printed by prepare)

- repository: `https://github.com/joshuaswarren/mil-hwx-compiler`
- commit: `a0ce354cf800011a84420da4e12013eb8140b2a5` (branch/tag
  `ane-parity`, tag `ane-parity-a0ce354`)
- archive URL: `https://github.com/joshuawarren/mil-hwx-compiler/releases/download/ane-parity-a0ce354/mil-hwx-compiler-a0ce354.tar.gz`
- archive SHA-256: `12bfbcd0049849dba18ee5fd3556762bf3313459c8ca5a65ae962f507a5b6d64`
- package schema: `mil-hwxc.h13-anec-package.v1`
- qualified target: `H13`

The external compiler is not vendored; no compiler internals were copied
into `mlx-omarchy` (the compiler's own hygiene check enforces this on the
source, and this repository carries none of it).

## Host

- omp-studio-local, x86_64 Linux, kernel 6.17.2-1-pve
- clang++-14 (`/usr/bin/clang++-14`)
- GNUstep prefix: `/home/joshuawarren/.local/mil-hwx-gnustep` (existing
  install, reused). Dependency pin story verified in the locked source's
  `scripts/bootstrap-linux-toolchain.sh`: exact commits, not branches —
  libobjc2 `a1faad66cee79f29c4ca268b004111edc33a2ae2`,
  gnustep-make `d0349cc83a23367acdc53df364f73504c281b26c`,
  gnustep-base `3d7013e144b153abcd6c088dc8d6c64ebd43bc4b`.

## Exact commands and results

Run from a clean work area (`rm -rf .work/ane-compiler` equivalent, then
`scripts/verify-ane-compiler.sh all`): see `verify-all.log`.

1. Source hash rejection before build — a valid gzip archive with wrong
   bytes is placed at the archive path; prepare refuses with
   `archive SHA-256 mismatch; refusing to build`, no source tree is
   extracted, no binary is produced. Result: PASS (rejection precedes
   extraction and build).
2. Prepare — download (curl, with authenticated `gh release download`
   fallback for the same immutable asset), SHA-256 check, extraction,
   build of `build/mil-hwxc` with `ldd` linking assertions against the
   GNUstep prefix. Result: PASS; binary SHA-256 in the log.
3. Compiler host checks (the compiler's own required host-side tests):
   - `operation graph: PASS`
   - `HWX object writer: PASS`
   - `program composition: PASS`
   - `H13_ENCODING_OK` (`test_h13_encoding`)
   - `H13 MIL-to-ANEC/HWX CLI: PASS (device-free)` (`test_h13_cli`)
   - `HWX inspection: PASS (multi-task H13 plus source-derived H14 fixtures)`
   - `linux compiler software tests: PASS`
   - `linux compiler hygiene: PASS`
4. Known H13 graph — `tests/fixtures/conv_relu.mil` with in-tree model
   weights compiled with `--target H13` by the prepared binary:
   - manifest asserts `schema == mil-hwxc.h13-anec-package.v1`,
     `target == H13`, `artifactFormat == anec`, all program files
     non-empty;
   - `research/inspect_anec.py` passes device-free
     (`container, tensor, binding, slice, and dispatch consistency only`);
   - package SHA-256 summary is in `h13-package-sha256.txt` (the full
     per-file listing was trimmed from `verify-all.log`).
5. Scope statement — device access: none; bundle adaptation and runtime
   qualification not exercised.

## Reproducibility

Nothing in the flow reads `~/src/mil-hwx-compiler`. Fresh clones resolve
the pinned source only through `ane-compiler.lock`. The release asset is
immutable and was verified to hash to the locked SHA-256 on download.

Determinism cross-check: two independent prepare runs from the same locked
archive bytes produced a byte-identical compiler binary (SHA-256
`fff9f8fde5eb97bc08098980ec647594013ee3dec0e2893ec2e3bb572ab7b9dd`); the
4096 conv-tile programs in the H13 package share one hash
(`3d73787c1fc04842da72bb6e1a1930ffa1d3c7b7280ab7d838add2da2edfba22`) with a
single distinct tail program (`63462dece46ebcd100b3785be09979786c969d7add75b212902344418fcb7123`).
No script or lock in this change references `~/src/mil-hwx-compiler`.

## GitHub notes (2026-09-12)

- codeload/GitHub archive generation for this fork was unavailable at pin
  time (persistent 404 while the parent repository's archives served).
  The pinned commit `a0ce354` was pushed as branch `ane-parity` and tagged
  `ane-parity-a0ce354`, and the locked archive was published as the
  immutable release asset above.
- The initial archive URL misspelled the repository owner as
  `joshuawarren`. The canonical public owner is `joshuaswarren`; its
  anonymous release URL answers successfully. The incorrect authenticated
  fallback was removed. The locked SHA-256 still gates every build.

## Not established by this receipt

- ANE device execution of the produced package.
- Bundle adaptation into `mlx-omarchy` (`load_bundle` path).
- Runtime qualification of H13 on M1, firmware interaction, or Parakeet
  compiler coverage (plan sections 47 onward).

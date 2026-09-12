# Release freshness audit, 2026-09-12

Scope: is the currently shipping release healthy, and is a new release
required? This audit verifies the UPLOADED release bytes, the install pin,
and the delta versus qualified main. It does not cut a release.

## Verdict

The project has a recent release: v0.4.2, published September 10. Its
uploaded artifacts pass the release-asset verifier. No replacement is
needed solely to satisfy release recency. This does not establish full
correctness or qualify current main; the affine-offset repair remains open.

## Shipped release: v0.4.2

- URL: https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.4.2
- Published: 2026-09-10T03:59:10Z, stable (not prerelease), marked Latest
- Tag `v0.4.2` (annotated `22d404b0`) -> commit `b3e977b4` ("Release v0.4.2")
- Assets downloaded fresh this session and re-hashed; all three match the
  release notes, the `SHA256SUMS` asset, and GitHub's reported digests:

| asset | bytes | sha256 |
|---|---:|---|
| `mlx_omarchy-0.32.2.dev202609100353+b3e977b-cp314-cp314-linux_aarch64.whl` | 7445801 | `c59e06037861be4e6fa0c0e3ded5c831f35765092572ed71ac105f012cf3bbc1` |
| `mlx_omarchy-0.32.2.dev202609100356+b3e977b4-cp311-cp311-linux_x86_64.whl` | 7501989 | `a9f64520c195d88a147913c1251423d09d691e69d21ca937f5734a7e404f3145` |
| `SHA256SUMS` | 278 | `a311cd903ec0e9b3646bb4dc1467cf4c3a569216fe80859de8c176286d512e24` |

- Release gate on the UPLOADED bytes
  (`python3 scripts/verify-release-assets.py v0.4.2`, this session):
  **VERIFIED** - sha256, version consistency
  (filename/dist-info/METADATA), stamped build commit equals the tag
  commit, stable feature strings (profiler absent, `MLX_DISABLE_COMPILE`
  control present), platform tags agree, both required platforms present.

## Install pin is current

`install.sh` on `main` (fetched live this session; the one-liner in the
release notes fetches it from `main`):

- default `VERSION="${MLX_OMARCHY_VERSION:-v0.4.2}"` - pinned to the
  current Latest, overridable by env;
- downloads `$base/SHA256SUMS` from the tag URL and enforces
  `sha256sum -c --ignore-missing` before installing the wheel;
- requires cp314, matching the aarch64 wheel;
- installs `lapack blas openblas` - the aarch64 wheel's `libmlx.so`
  declares `NEEDED libopenblas.so.0` (checked with `readelf -d` on the
  extracted binary this session), so without `openblas` installed
  `import mlx.core` fails on a fresh install. Commit `286ef729` (after
  the release): fresh installs between 2026-09-10 03:59Z and 08:05-0500
  would have failed to import; the live installer fixes that today
  without a new release.

## GitHub Actions

Only the scheduled community-data mirror (green, latest run 2026-09-12)
and gh-pages deployment. Releases are hand-cut per `docs/release.md`;
there is no release workflow to inspect.

## Delta versus measured main

`v0.4.2..f365d5b5` contains 195 commits. The latest composed hardware
receipt, `receipts/2026-09-12-parity-status`, measures `a2e38c3e` and
records battery 29/30. The failing affine-offset case was not rerun in
this audit. Main is not fully qualified.

The audit checked named post-release fixes and their ancestry, not every
change for incidental correctness effects. An ancestor check proves
whether a commit shipped; it does not prove whether an older release
contains the defect that commit fixes.

## Open defects affecting v0.4.2 users

The current known-defect ledger lists these unresolved behaviors:

1. Affine quantized matmul produces a wrong value in the non-zero-offset
   m=1 test. Applications using affine scale/bias views can exercise this
   contract. Passing canonical model digests does not make it harmless or
   unreachable. The root cause and fix are being investigated separately.
2. Cooperative-matrix prefill output depends on which Mesa build provides
   the extension. Releases are digest-checked on the installed driver,
   which produces the required values; noted "not yet fixed" with a
   candidate fix.

## CoreML and ANE content of the stable release

Checks on the v0.4.2 artifacts and tagged source, performed this session:

- Wheel file inventories (both wheels, full `namelist`): only the
  documented `mlx` package - python modules, `core.*.so`, `libmlx.so`,
  `mlx/bin/mlx-omarchy-info`, 250 upstream `mlx/include/**` headers,
  dist-info. No `coreml`-named member in either wheel; the only
  `ane`-named members are `mlx/include/mlx/backend/omarchy/ane/{bundle,manifest}.h`,
  headers of the qualified ANE bundle parser.
- Tagged source tree (`git ls-tree -r v0.4.2`): zero files matching
  coreml/`core_ml`/`core-ml`; `git grep -il coreml v0.4.2 -- overlay/
  patches/` is empty.
- Byte-scan of shipped `libmlx.so`: the three `coreml` sequences are
  substrings of mangled `mlx::core::ml*` symbols (namespace `core`,
  function `mul` - verified with surrounding context), not Apple
  CoreML. The two `libane` sequences are the ANE bundle validator's
  error strings ("ANEC file is smaller than libane header", ...),
  compiled from `overlay/mlx/backend/omarchy/ane/`, which is qualified
  pre-release work covered by the standing battery's
  `omarchy_ane_bundle_tests` - per `AGENTS.md` ANE is the internal
  graph-region accelerator this repository owns, not the experimental
  CoreML compiler.
- Dynamic dependencies (`readelf -d` on both extracted binaries): only
  `libopenblas.so.0` (aarch64) / `liblapack.so.3`+`libblas.so.3`
  (x86_64) plus base system libs. No `libane`, no Apple framework, no
  CoreML linkage.

Limits: this establishes no CoreML files, no CoreML-referencing source
in the stamped tree, and no CoreML-linked dependency in the shipped
binaries. It does not prove absence of every conceivable compiled-in
byte pattern inside `libmlx.so`; the authoritative statement is that
the wheels' gate-verified stamped commit (`b3e977b4`) contains no
CoreML code, and the binaries link nothing beyond the libraries above.

## Next release decision

A new GPU release must build the exact candidate, pass the standing
battery and canonical model checks, and verify the uploaded artifacts.
The affine-offset correction is a reason to prepare that release once
verified. Unqualified Mesa experiments and Core ML work must not be
presented as stable shipped capabilities.

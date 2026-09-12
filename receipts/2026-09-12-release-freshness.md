# Release freshness audit, 2026-09-12

Scope: is the currently shipping release healthy, and is a new release
required? This audit verifies the UPLOADED release bytes, the install pin,
and the delta versus qualified main. It does not cut a release.

## Verdict

**v0.4.2 remains the adequate shipping release. No new release is
required.** By the observed release triggers below, no post-release
commit qualifies as a repair of a defect shipped in v0.4.2's wheels;
the two known-issue ledger entries affecting v0.4.2 users are unchanged
in qualified main. This rests on ancestor checks plus a read of each
named fix commit (limits stated in the delta section), not an exhaustive
causal audit of all 195 commits.

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

## Delta versus qualified main

`v0.4.2..f365d5b5` is 195 commits; qualified main per
`receipts/2026-09-12-parity-status` is `a2e38c3e` (battery 29/30 - the
one failure is the documented pre-existing affine m=1 offset defect; not
rerun here). Lineage of every post-release fix commit, checked with
`git merge-base --is-ancestor` - **none is in v0.4.2**. Limits: the
"not a shipped defect" column rests on each commit's own message and
diff read together with the ancestor check. A defect that exists in
v0.4.2 and was fixed incidentally inside an unrelated post-release
commit would not be excluded by this method; no such commit was
observed, and the known-issue ledger names no other open wrong-value
defect on shipped paths:

| commit | subject | role |
|---|---|---|
| `6eaad2b0` | native-order decode SDPA kernel | performance (fork driver) |
| `286ef729` | install: add openblas | installer only; already live on main |
| `e189610f` | scalar-FMA prefill qualification receipts | main-only broken window (conflict markers), pushed and repaired 2026-09-12, never reachable from the tag |
| `f03e7ee1` | land qualified FMA route state, repair `e189610f` | repairs `e189610f`; not a shipped defect |
| `f433007e` | fill unpublished GPU scalars by broadcast | fixes `e189610f`-generation code; not in v0.4.2 |
| `a2e38c3e` | register three FMA prefill shaders | fixes `f03e7ee1`'s route; not in v0.4.2 |

The remaining delta is performance work (native decode SDPA, grouped
GEMV follow-ups), tests, bench harnesses, and receipts/docs.

## Open defects affecting v0.4.2 users

Documented in `docs/known-defects.md`; both present equally in
qualified main. Non-blocking in the observed-shipping sense only -
neither blocks the release pipeline, and #1 is a real wrong-value
defect an arbitrary caller can hit:

1. Affine quantized matmul mis-composes a non-zero storage offset at m=1
   (the battery's 29/30 failure): a silent wrong value, not a refusal,
   on the m=1 vector route whenever a caller binds affine scale/bias
   streams at a non-zero storage offset. Current canonical models pass
   offsets at zero, so shipped workflows do not hit it and every
   canonical digest holds - that is why it does not trigger a release,
   not evidence the defect is unreachable. Reproduces back through
   `bddc061f`; expected disposition per the ledger is deriving the view
   offset per stream.
2. Cooperative-matrix prefill output depends on which Mesa build provides
   the extension. Releases are digest-checked on the installed driver,
   which produces the required values; noted "not yet fixed" with a
   candidate fix.

## CoreML and ANE content of the stable release

The experimental CoreML/MIL compiler work lives outside this repository
(`ane-linux-experiments`, `mil-hwx-compiler`). Checks on the shipped
artifacts and their tagged source this session:

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

## Why no new release

Cutting one now would ship main's performance work, which per protocol
requires a fresh aarch64 wheel built on the M1 from the new tag, the full
gate, and a pinned decode receipt - a deliberate release cut, not a
freshness repair. By the observed triggers, nothing in the delta repairs
v0.4.2 for shipped users, and the documented failure modes are unchanged.
Triggers that WOULD require a new release: any correctness fix landing in
a wheel-consumed path, a broken install pin, or a gate failure on the
live assets.

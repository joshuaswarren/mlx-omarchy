# Release freshness audit, 2026-09-12

Scope: is the currently shipping release healthy, and is a new release
required? This audit verifies the UPLOADED release bytes, the install pin,
and the delta versus qualified main. It does not cut a release.

## Verdict

**v0.4.2 remains the adequate shipping release. No new release is
required.** Every post-release commit is an improvement or a main-only
repair; none fixes a defect shipped in v0.4.2's wheels.

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
- installs `lapack blas openblas` - `openblas` provides
  `libopenblas.so.0` which the wheel's BLAS calls resolve at import time
  (commit `286ef729`, after the release: fresh installs between
  2026-09-10 03:59Z and 08:05-0500 would have failed to import; the live
  installer fixes that today without a new release).

## GitHub Actions

Only the scheduled community-data mirror (green, latest run 2026-09-12)
and gh-pages deployment. Releases are hand-cut per `docs/release.md`;
there is no release workflow to inspect.

## Delta versus qualified main

`v0.4.2..f365d5b5` is 195 commits; qualified main per
`receipts/2026-09-12-parity-status` is `a2e38c3e` (battery 29/30 - the
one failure is the documented pre-existing affine m=1 offset defect; not
rerun here). Lineage of every post-release fix commit, checked with
`git merge-base --is-ancestor` - **none is in v0.4.2**:

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

Both are documented in `docs/known-defects.md` and present equally in
qualified main - neither is release-blocking:

1. Affine quantized matmul mis-composes a non-zero storage offset at m=1
   (the battery's 29/30 failure). Reproduces back through `bddc061f`;
   impact synthetic - real models pass offsets at zero and every
   canonical digest holds.
2. Cooperative-matrix prefill output depends on which Mesa build provides
   the extension. Releases are digest-checked on the installed driver,
   which produces the required values; noted "not yet fixed" with a
   candidate fix.

## Why no new release

Cutting one now would ship main's performance work, which per protocol
requires a fresh aarch64 wheel built on the M1 from the new tag, the full
gate, and a pinned decode receipt - a deliberate release cut, not a
freshness repair. Nothing in the delta repairs v0.4.2 for shipped users,
and the documented failure modes are unchanged. Triggers that WOULD
require a new release: any correctness fix landing in a wheel-consumed
path, a broken install pin, or a gate failure on the live assets.

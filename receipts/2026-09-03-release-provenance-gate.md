# 2026-09-03 — Installed-binary provenance gate, pin guard, commit stamping, and the uploaded-asset gate

Scope correction of record: the morning assignment assumed the published
v0.3.3-diag.1 asset lacked the profiling harness. That premise was
WITHDRAWN before any release was touched: direct inspection shows
`MLX_OMARCHY_GPU_PROFILE` present in its `libmlx.so`, and the new asset
gate below independently confirms it (see its output: "profile present").
Nothing about any release, tag, asset, or contributor doc was changed in
this work. The real damage today came from measurement environments, not
from uploaded assets.

## The three defects, as they now stand

1. **Stale binary under a correct-looking environment (root cause named).**
   A support-requirements freeze carried a direct-URL pin of the form
   `mlx-omarchy @ file:///.../mlx_omarchy-0.32.2.dev20260903-...whl#sha256=6e54ab2b...`.
   An exclusion filter that only understood `==` did not match it, so pip
   silently reinstalled the stale 5,300,664-byte-libmlx wheel OVER the
   correct 20.9 MB one in every fresh venv. This one trap produced both
   false-measurement episodes: the retracted byte-identity claim and the
   withdrawn "profiling absent" reading.
2. **v0.3.2 predates `ceae628` (stands).** Tag cut at `b17cc86` 05:02;
   `ceae628` ("+9 percent tokens per second") landed 05:18 and is not in
   the release. Users of the published asset are 8-11% slower than main.
   Evidence: `receipts/2026-09-03-wheel-delta-v0.3.2-vs-local.md`. The
   stable release itself was NOT touched (owner-gated); the fix ships
   with the next stable cut from main.
3. **Diag asset missing profiling (WITHDRAWN).** The published
   v0.3.3-diag.1 asset is healthy. The reading came from the reporting
   agent's own venv provisioning - in other words, defect 1 again.

## What shipped

### 1. Installed-binary provenance gate (primary)

`scripts/mlx_provenance.py` resolves the binaries actually loaded by the
interpreter - the `mlx.core` extension (`mlx.core.__spec__.origin`; `mlx`
installs as a namespace package, so `mlx.__file__` is None) and
`mlx/lib/libmlx.so` - hashes them, and compares against the installed
wheel's RECORD plus the compiled `mx.__version__` against the installed
distribution version. On mismatch it refuses with found-vs-expected.

Wired so a number and its provenance cannot be separated:

- `scripts/bench_decode.py`: runs the gate BEFORE any mlx import, prints
  the `provenance:` line beside every rate, and exits 3 with
  `REFUSING TO EMIT NUMBERS: ...` on mismatch. `--wheel FILE` adds a
  claim check against a named wheel's members.
- `scripts/collect_deep.py`: the correctness and benchmark probes now
  embed a `provenance` object in their JSON. A RECORD mismatch flips the
  section to `available: false` with the refusal as the error, so a
  lying environment cannot publish numbers as data.

Proof, all against the real published v0.3.2 x86 wheel installed into a
fresh venv (`/tmp/relgate/venv-prov`, Python 3.11):

Clean install (`python3 scripts/mlx_provenance.py`): `verified: match`,
both binaries' hashes equal their RECORD entries, `dist_version ==
mx_version == 0.32.2.dev202609030502`.

Scenario A - the real incident replayed, stale-generation `libmlx.so`
(5,718,552 bytes, from the 2026-09-02 wheel) copied over the installed
v0.3.2 one, then `python3 scripts/bench_decode.py --model /nonexistent`:

```
REFUSING TO EMIT NUMBERS: mlx/lib/libmlx.so at .../mlx/lib/libmlx.so: on disk sha256 c1922e21d07a148224ff9f493248766efe0f6384c042bdf645e29c886af82d04 but the installed wheel's RECORD says 364d13f0e76f10f3abf17845dfcae6d95840d64e4ea47bf3dfc183b4f8d9802b. The runtime binary is not the wheel this environment claims to have installed - do not quote any number from this process. Fix: reinstall the exact wheel under test with pip --force-reinstall, then re-run scripts/mlx_provenance.py. ; compiled mx.__version__ '0.32.2.dev20260902' != installed distribution version '0.32.2.dev202609030502'; the loaded library is not the installed wheel.
```
exit 3, before any model load.

Scenario B - same version string, one appended byte:
`REFUSING TO EMIT NUMBERS: ... on disk sha256 6d7a05f6... but the installed wheel's RECORD says 364d13f0...` exit 3.

Scenario C - `--wheel <fresh v0.3.2 wheel>` against the tampered binary:
`REFUSING TO EMIT NUMBERS: loaded mlx/lib/libmlx.so sha256 6d7a05f6... != claimed wheel ... member mlx/lib/libmlx.so sha256 364d13f0... Refusing to emit numbers against the claimed wheel.` exit 3.

`scripts/collect_deep.py` benchmark probe against the tampered install:
`available: false`, `error: "refusing to emit timing numbers: mlx/lib/libmlx.so ... on disk sha256 6d7a05f6..."`.

After `pip install --force-reinstall`, the same bench_decode prints
`provenance: mlx-omarchy 0.32.2.dev202609030502 mx=0.32.2.dev202609030502 verified=match core.cpython-311-x86_64-linux-gnu.so=sha256:a53f742538df765a libmlx.so=sha256:364d13f0e76f10f3` and proceeds.

Source builds without dist metadata are reported `verified: "no-metadata"`
and flagged in the provenance line, not silently trusted.

### 2. Requirements pin guard

`scripts/check-wheel-pins.py` rejects mlx-omarchy pins in ANY form in
requirements inputs: `==`, ranges, `@ file://`, `@ https://...`, and bare
direct-URL wheel lines; `-r`/`-c` includes are followed. Editable
installs warn. Each violation names the line and the 2026-09-03 trap.
Exit 1 on violation. Any provisioning step can call it as a gate.
Tests: `tests/test_wheel_pins.py` (8 cases, stdlib unittest, offline).

### 3. Build stamps the source commit (build changed - YES)

`scripts/build-wheel.sh` changed. Release builds (operator exports
`DEV_RELEASE=1`) and diagnostics builds now stamp the source commit into
the version's local segment via the existing
`patches/mlx-version-time.patch` mechanism
(`MLX_OMARCHY_LOCAL_VERSION`): release `0.32.2.dev<ts>+<short7>`,
diagnostics `+diag.<short7>`. `MLX_OMARCHY_SOURCE_COMMIT` overrides;
building outside a git checkout fails with a named error instead of
shipping an untraceable wheel. Default dev builds are unchanged
(setup.py already appended the enclosing checkout's short hash).

No byte-reproducibility requirement exists for these wheels - the dev
segment already carries minute-level timestamps - so the stamp adds no
new nondeterminism and touches no reproducible surface.

Proof (staged tree, `get_version()` A/B, the same method the diag.1
receipt used):

```
default dev (no env):            0.32.2.dev202609031608+cf09c1f
release path (DEV_RELEASE=1):    0.32.2.dev202609031608+cf09c1f
diagnostics path:                0.32.2.dev202609031608+diag.cf09c1f
repo HEAD:                       cf09c1f
```

### 4. Uploaded-asset gate (secondary, kept)

`scripts/verify-release-assets.py <tag>` downloads every wheel asset from
the GitHub release and checks: sha256 against the release notes or this
repo's receipts; version identity across filename / dist-info / METADATA
and any recorded version; the stamped build commit against the tag's
commit (a DIFFERENT stamped commit is a FAILURE; no stamp is a FINDING);
feature strings with a positive control - `MLX_DISABLE_COMPILE` (baked
into every `libmlx.so` by `overlay/mlx/backend/omarchy/compiled.cpp`)
must be present or the check reads as broken, `MLX_OMARCHY_GPU_PROFILE`
must be present for `-diag` releases and absent for stable ones;
`mlx/bin/mlx-omarchy-info` required for diag releases.

Verbatim output, v0.3.3-diag.1 (the healthy case; exit 0):

```
release v0.3.3-diag.1 (prerelease) tag commit e82f6fb78b777be03e6e868683f80b4004157d7b

== mlx_omarchy-0.32.2.dev202609031348+diag-cp314-cp314-linux_aarch64.whl (6803505 bytes)
   recorded sha256: 1939cd388f8be630a3192c6b88ad1fc0d425b758f811e4bd50d3d3a3eb8094f5 (release notes)
   PASS sha256 matches recorded (1939cd388f8be630a3192c6b88ad1fc0d425b758f811e4bd50d3d3a3eb8094f5)
   PASS version consistent across filename/dist-info/METADATA: 0.32.2.dev202609031348+diag
   PASS feature strings correct for a diagnostics build (control MLX_DISABLE_COMPILE present; profile present)
   FINDING records no build commit anywhere in the artifact: it cannot be traced to tag v0.3.3-diag.1 (e82f6fb78b777be03e6e868683f80b4004157d7b); builds since the stamping change carry the commit in the version's local segment

== mlx-omarchy-info (recorded in release notes)
   PASS sha256 matches recorded (7f498db92bb6ef96da05d9f714f59b9981595728927fb9363993c15fab9b24f9)

== summary
FINDING records no build commit anywhere in the artifact: it cannot be traced to tag v0.3.3-diag.1 (e82f6fb78b777be03e6e868683f80b4004157d7b); builds since the stamping change carry the commit in the version's local segment
VERIFIED WITH FINDINGS: 0 failures, 1 finding(s)
```

"profile present" in the UPLOADED asset is the independent confirmation
behind the defect-3 withdrawal.

Verbatim output, v0.3.2 (exit 0; hashes sourced from this repo's
`2026-09-03-release-v0.3.2.md` because the release notes record none):

```
release v0.3.2 (stable) tag commit b17cc86c9923cd992e6b2f16535f75600203c7f2

== mlx_omarchy-0.32.2.dev202609030502-cp311-cp311-linux_x86_64.whl (6821827 bytes)
   recorded sha256: 9e00463b2106f4c4605ffa289fe521c5813a7a7fa343c6727ff8b9a529cffbae (2026-09-03-release-v0.3.2.md)
   PASS sha256 matches recorded (9e00463b2106f4c4605ffa289fe521c5813a7a7fa343c6727ff8b9a529cffbae)
   PASS version consistent across filename/dist-info/METADATA: 0.32.2.dev202609030502
   PASS feature strings correct for a stable build (control MLX_DISABLE_COMPILE present; profile absent)
   FINDING records no build commit anywhere in the artifact: it cannot be traced to tag v0.3.2 (b17cc86c9923cd992e6b2f16535f75600203c7f2); builds since the stamping change carry the commit in the version's local segment

== mlx_omarchy-0.32.2.dev202609030512-cp314-cp314-linux_aarch64.whl (6789976 bytes)
   recorded sha256: 61424114d983c8f3c02b6ae83125be9fe709a13a2fcb8b27ff0b0d02fb64e370 (2026-09-03-release-v0.3.2.md)
   PASS sha256 matches recorded (61424114d983c8f3c02b6ae83125be9fe709a13a2fcb8b27ff0b0d02fb64e370)
   PASS version consistent across filename/dist-info/METADATA: 0.32.2.dev202609030512
   PASS feature strings correct for a stable build (control MLX_DISABLE_COMPILE present; profile absent)
   FINDING records no build commit anywhere in the artifact: it cannot be traced to tag v0.3.2 (b17cc86c9923cd992e6b2f16535f75600203c7f2); builds since the stamping change carry the commit in the version's local segment

== summary
VERIFIED WITH FINDINGS: 0 failures, 2 finding(s)
```

The finding IS the v0.3.2 defect shape: an asset that cannot be traced
to a commit. Defect 2 itself (source gap vs `ceae628`) is not an asset
integrity failure - the asset exactly matches its tag's build - which is
why the gate reports it and the fix is the next stable cut from main.

Mismatch detection proof: a doctored copy of the v0.3.2 x86 wheel
(filename unchanged, METADATA Version rewritten to
`0.32.2.dev202609030928+ceae628`) fails with three named causes:

```
FAIL version identity disagreement: filename says '0.32.2.dev202609030502', dist-info directory says '0.32.2.dev202609030502', METADATA says '0.32.2.dev202609030928+ceae628'; this wheel is named one generation and carries another's metadata
FAIL artifact records build commit ceae628 but tag v0.3.2 is b17cc86; the asset was not built from the tagged commit
FAIL metadata version '0.32.2.dev202609030928+ceae628' != the version the release records for this asset ('0.32.2.dev202609030502')
```

### 5. Release procedure

Releases are cut by hand; the steps now live in `docs/release.md`. The
gate is the step before announcing: `build-wheel.sh` ends its receipt
with the operator instruction (`NEXT STEP - do not skip: ... python3
scripts/verify-release-assets.py <tag> ... It must print VERIFIED before
the release is announced as usable. See docs/release.md.`).

### 6. Tests

`python3 -m unittest tests.test_wheel_pins tests.test_mlx_provenance`:
`Ran 11 tests ... OK`. Script self-tests:
`mlx_provenance.py --self-test` -> `self-test: OK (5 checks)`;
`check-wheel-pins.py --self-test` -> `self-test: OK (6 trap forms
rejected, clean input passes)`. `bench_decode.py --self-test` still
passes unchanged.

## Not touched

- Release v0.3.3-diag.1, its tag, and both assets: untouched and verified
  healthy (https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.3.3-diag.1).
- Release v0.3.2, its tag, and both assets: untouched
  (https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.3.2).
- `CONTRIBUTING.md`: unchanged; the v0.3.3-diag.1 install line it carries
  remains correct.

## Verdict

Measurement can no longer silently disagree with the wheel under test:
the loaded binary is hashed against RECORD before any number is emitted,
pins cannot smuggle a stale generation past provisioning, future wheels
carry their source commit, and uploaded assets are checked against their
release's claims before announcement. The withdrawn defect-3 premise is
recorded here so nobody re-investigates the diag release.

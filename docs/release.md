# Cutting a release

Releases are cut by hand. The rule that produced this procedure: nothing
verifies the ARTIFACT AFTER UPLOAD by default, and a local check does not
verify what users download. Three releases in one day (2026-09-03) needed
manual static-diff investigations because of it. The same day, v0.3.3 was
first published with only the x86_64 wheel - correct metadata, gate-green,
and unusable on every Apple Silicon machine the project serves - so the
procedure now names the required platforms, and the gate enforces them.

1. Pick the commit. Tag only what you verified. If a performance or
   correctness commit lands after your target commit, re-cut from the new
   commit or accept a slower release on the record.
2. Build EVERY required platform. A stable release carries both:

   | platform | wheel | built on |
   |---|---|---|
   | linux_x86_64 | `cp311` | the dev box, `scripts/build-wheel.sh` |
   | linux_aarch64 | `cp314` | the M1 (`jwm1`), same script from a detached worktree at the tag commit |

   Cutting on the dev box alone produces only the x86_64 wheel, which
   does not run on Apple Silicon - the project's target hardware. scp
   the aarch64 wheel to the dev box for upload and re-verify its sha256
   after transfer. Build with `scripts/build-wheel.sh` (add
   `--diagnostics` only for a `-diag.*` prerelease, which requires only
   the aarch64 wheel). The receipt prints the source commit the wheel is
   stamped with; record wheel name, bytes, sha256, and the installed
   version string in the receipt.
3. Upload the assets to the release, then run the gate against the
   UPLOADED bytes:

   ```bash
   python3 scripts/verify-release-assets.py <tag>
   ```

   The gate downloads every wheel asset from the release and checks the
   sha256 against the release notes or `receipts/`, the version metadata
   against the filename and any recorded version, the stamped build
   commit against the tag's commit, and the feature strings (a `-diag`
   release must carry the profiling harness in `libmlx.so`; a stable
   release must not; a positive control keeps an empty result from
   reading as a pass). The gate also FAILS a release whose wheels do not
   cover the required platforms above, and checks each wheel's filename
   platform against its `dist-info/WHEEL` Tag.
4. The release is announced as usable only after the gate prints
   `VERIFIED`. `VERIFIED WITH FINDINGS` is acceptable only for the
   traceability finding on assets built before the stamping change.
5. Run one pinned 4-bit decode on the UPLOADED aarch64 asset, installed
   from the release URL on the M1, and put that number in the notes.
   The gate checks hashes, platforms and feature strings; none of that
   sees speed. v0.3.4 passed the gate and shipped 4-bit decode at 0.21
   tok/s, 70x below the number in its notes, which had been measured on
   the commit before the one that was tagged
   (`receipts/2026-09-04-v0.3.4-decode-regression.md`).
6. Write the dated receipt under `receipts/` with the gate output and
   the decode number.
7. For runtime measurement work, install the exact wheel under test and
   run `scripts/mlx_provenance.py` (or let `scripts/bench_decode.py` do
   it): a run whose loaded `libmlx.so` does not match the installed
   wheel's RECORD refuses to emit a number. Never pass a requirements
   input containing an mlx-omarchy pin to provisioning;
   `scripts/check-wheel-pins.py` rejects them in any form.

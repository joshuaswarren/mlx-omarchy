# Cutting a release

Releases are cut by hand. The rule that produced this procedure: nothing
verifies the ARTIFACT AFTER UPLOAD by default, and a local check does not
verify what users download. Three releases in one day (2026-09-03) needed
manual static-diff investigations because of it.

1. Pick the commit. Tag only what you verified. If a performance or
   correctness commit lands after your target commit, re-cut from the new
   commit or accept a slower release on the record.
2. Build with `scripts/build-wheel.sh` (add `--diagnostics` only for a
   `-diag.*` prerelease). The receipt now prints the source commit the
   wheel is stamped with. Record wheel name, bytes, sha256, and the
   installed version string in the receipt.
3. Upload the asset to the release, then run the gate against the
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
   reading as a pass).

4. The release is announced as usable only after the gate prints
   `VERIFIED`. `VERIFIED WITH FINDINGS` is acceptable only for the
   traceability finding on assets built before the stamping change.
5. Write the dated receipt under `receipts/` with the gate output.
6. For runtime measurement work, install the exact wheel under test and
   run `scripts/mlx_provenance.py` (or let `scripts/bench_decode.py` do
   it): a run whose loaded `libmlx.so` does not match the installed
   wheel's RECORD refuses to emit a number. Never pass a requirements
   input containing an mlx-omarchy pin to provisioning;
   `scripts/check-wheel-pins.py` rejects them in any form.

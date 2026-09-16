# Wheel libane pin bumped to omarchy-ane 6fa243a with STRICT_BIND (2026-09-15)

Result: **PASS**. The product wheel's private ANE runtime now qualifies
`joshuaswarren/omarchy-ane` at `6fa243ac7241119a9eb229abbf8cb4dd8949f915`
and compiles libane with `LIBANE_CONFIG_STRICT_BIND`, so a positional
channel map is refused at load — the same contract the strict bundle
validator (bundle.cpp) already enforces.

## Change

Branch `libane/strict-pin-6fa243a` (two commits on origin/main
`8558064a`):

- `e45935bd` ane: pin the wheel's libane at omarchy-ane 6fa243a with
  STRICT_BIND
  - `overlay/mlx/backend/omarchy/CMakeLists.txt`: the two
    `MLX_OMARCHY_ANE_SOURCE_DIR` qualification strings move to
    `6fa243ac…`; `mlx_omarchy_libane` gains
    `LIBANE_CONFIG_STRICT_BIND`.
  - `overlay/mlx/backend/omarchy/ane/runtime_worker.cpp`:
    `kQualifiedLibaneCommit` moves to `6fa243ac…`.
  - `docs/ane-runtime.md`: pin moved; STRICT_BIND contract documented.
- `46bb0700` test: runtime synthetic positional ANEC now refused under
  STRICT_BIND. With strict libane the synthetic no-task-stream ANEC
  fixture is refused at load before the diagnostic-path guard is
  reachable; the test asserts the strict refusal instead of the lenient
  fallthrough.

Not merged (per assignment): `63c1d3cf` was not touched.

## Wheel build (jwm1)

- Worktree `/var/tmp/LibaneStrictPin-6fa243a` at `46bb0700`; ANE source
  qualified at `~/src/omarchy-ane-lifecycle-rebase` (clean,
  `fix/driver-lifecycle-rebase`, HEAD `6fa243ac…`).
- `DEV_RELEASE=1 scripts/build-wheel.sh` with
  `MLX_OMARCHY_ANE_SOURCE_DIR` → wheel
  `mlx_omarchy-0.32.2.dev202609160346+e45935bd62a4e194da4988dd68eb8bcbce0f6763-cp314-cp314-linux_aarch64.whl`,
  7 857 045 bytes, sha256
  `32b8130bf797508cadbb18fd7ae573319d7e8b1a965d101712494c8fbff457ba`.
- `scripts/mlx_provenance.py --expect-wheel` → **`verified: "match"`**,
  dist/mx version `0.32.2.dev202609160346+e45935bd…`, `version_match:
  true`.
- Wheel identity shas: installed `mlx/bin/mlx-omarchy-ane-worker`
  (runtime worker, statically carries strict libane via
  `mlx_omarchy_libane`)
  `49bf4d399955ba22efdb923590565dc468d9fbc612c9a00f5e5d1546e12c8c7a`.

## Gates

1. **check-bundle OK ×6** with the wheel's `mlx-omarchy-info` on the
   re-adapted paths: `receipts/fixtures/exported/{ane-add-fp16-1x512,
   ane-add-fp16-1x896,ane-mul-fp16-1x512}` (mil-hwxc `b61de467…`) and
   `receipts/2026-09-14-encoder-split-plan/bundles/{island-pv,
   island-attn-a-kt,island-select-8head}` (mil-hwxc `b61de468…`).
2. **Runtime/bundle tests green** (device-mode + runtime-mode builds
   from this tree, omarchy-ane 6fa243a headers):
   `omarchy_ane_runtime_tests` 14/14 cases, 431/431 assertions
   (includes the new strict-refusal subcase);
   `omarchy_ane_bundle_tests` 27/27, 4732/4732;
   `omarchy_ane_worker_tests` 16/16, 199/199;
   `omarchy_ane_tile_layout_tests` 78/78.
3. **Negative control (device)**: the pre-fix positional bundle
   (`receipts/fixtures/exported/ane-add-fp16-1x512` at `8558064a~1`,
   old compiler ANEC) run by the pinned-tree worker
   (`f171a61ecfc89942b009dc379d6636b17b0dda6f583841470a1d789853f5952e`,
   identical to the worker quoted in 8558064a's gates) with
   `libane-strict.so`
   (`56b462346128b04139cb7c9397b06ca8d513c680c972d8914395ddd3aa978ba7`,
   built from 6fa243a `libane/` with `-DLIBANE_CONFIG_STRICT_BIND`) is
   **refused**: exit 1, `error: [omarchy-ane] bundle: program 0 task
   stream does not name every surface; channel map is positional.`
   Under the old f261a6cb libane this bundle loaded.
4. **Positive device control**: the re-adapted b61de46
   `ane-add-fp16-1x512` bundle loads and executes under the same strict
   libane; two runs byte-identical (`--save` then `--expect` →
   `verified output t2 exact`).
5. **Parakeet E2E 104/104 on the wheel's python + pinned-tree worker**:
   `/var/tmp/ParakeetE2E/parakeet_e2e.py` under `flock /tmp/m1-gpu.lock`
   (never stolen, quarantine 0 before/after, lock released after), with
   `PYTHONPATH` loading the wheel above, worker sha `f171a61e…`,
   `libane-strict.so` `56b46234…`, and the three re-adapted island
   bundles. Report (`e2e-out/e2e-report.json`): `status: "match"`,
   emissions **104/104**, `matching_prefix_length` 104,
   `encoder_bounds_pass: true`, `mel_bit_exact: true`, 72 ANE
   submissions, `cpu_tensor_events: 0`.
   - transcript sha256
     `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790`
     (pin),
   - `encoder_hidden.npy` sha256
     `38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`
     (pin) — byte-identical to the accepted ANE run.

## Notes

- The old `island-select-8head` bundle at `8558064a~1` still loads under
  strict libane (its ANEC names its surfaces); the positional-refusal
  negative control is proven on the old exported `ane-add-fp16-1x512`
  fixture instead.
- Host intermediate: a jwm1 ANE module wedge was recovered by
  AneTmRecovery mid-run; all gates above ran after the recovery, on
  healthy hardware (quarantine 0, module loaded).

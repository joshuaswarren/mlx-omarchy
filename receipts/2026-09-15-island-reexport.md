# Island re-export: strict-worker device pass, compiler pin bump, re-adapted bundles (2026-09-15)

Closes the known issue from `receipts/2026-09-15-v0.5.0-release.md`: every pinned
bundle now passes the strict bundle loader, and the Parakeet E2E reproduces
**104/104 with `encoder_hidden` = pin `38c73261…`** through a worker whose
libane refuses positional channel maps.

## Compiler: mil-hwx-compiler `ane-parity-b61de46`

- `main` fast-forwarded `eff0faa` → `5ac214c` (branch
  `feature/strict-bind-surface-naming`, also pushed), then `5ac214c` →
  `b61de468d1ca37b687d5fa371e489be16cd84ed1` on the branch and `main`
  together. No merge commit, no squash.
- `5ac214c` "h13: name every declared surface in the task stream": runtime
  batched-matmul staging TDs carry the second-source selector (`ch6`) and an
  s2 DMA config, so a bind walk derives `src=[5,6]` (select: `[5,6,7]`)
  instead of undercounting to `[5]`; ElementwiseKind::BinaryConstant lowers
  add [1,512]/[1,896] and mul [1,512] as one whole-tensor oracle-parity
  program (no constantInputs).
- `b61de46` "h13: stage BinaryConstant surfaces contiguously across the TD
  span" — device-forced fix, first hardware execution of this family
  (jwm1 2026-09-15): the decoded task87/88/89 descriptors DMA exactly
  `2*elements` bytes (0x400 / 0x700); with the shared one-element-per-64-byte-row
  layout that span reached only the first 16 staged rows (lanes 0..15 exact,
  16..511 unwritten). The constant-blob surfaces now stage contiguously
  (nchw `[1,1,1,elements,row,row]`, row = `2*elements`), so the TD's whole
  span covers every element. Island ANEC bytes are unchanged by this commit
  (`island-pv` program-0 `3ae36f21…`, `island-attn-a-kt` `cf0ecac2…`/`b801f621…`
  byte-identical across `5ac214c` and `b61de46` builds).
- Gates before push: `make test-h13` 15/15 suites PASS;
  `tests/test_h13_parity.py` 846 cases PASS (1692 artifacts).
- Tag `ane-parity-b61de46` (annotated), pushed. Archive
  `mil-hwx-compiler-b61de46.tar.gz` 34,371,964 bytes, sha256
  `3e2b27350f48810b2aa2749129ed8e5a6eae62568772adca3fd1585b67bc3271`,
  published at https://github.com/joshuaswarren/mil-hwx-compiler/releases/tag/ane-parity-b61de46
  with a `.sha256` sidecar.

## jwm1 device pass (STRICT worker)

Worker: `mlx-omarchy-ane-worker` built on jwm1 from `origin/main` `3aa4f348`
(`/var/tmp/mlx-main-strict`), sha256
`f171a61ecfc89942b009dc379d6636b17b0dda6f583841470a1d789853f5952e`;
`omarchy_ane_worker_tests` 16/16 cases, 199 assertions PASS.
libane: built from `omarchy-ane` `6fa243a` (the loaded module's source
version) with `-DLIBANE_CONFIG_STRICT_BIND`, sha256
`56b462346128b04139cb7c9397b06ca8d513c680c972d8914395ddd3aa978ba7`.
Note: the earlier runtime pin `f261a6cb` predates `5113ef6` (STRICT_BIND
refusal), so a strict libane must come from `5113ef6` or later; the wheel
cmake pin (`f261a6cb`) still selects the non-strict fallback build and is
untouched here. Negative control: the pre-fix positional `island-pv` bundle
REFUSES under this worker ("task stream does not name every surface; channel
map is positional", exit 1) — the strict gate is live on device.

Lock `/tmp/m1-gpu.lock`, `flock -w 900`, never stolen. A module wedge
("tm completion failed: -110 … preserving resources until reboot", 21:36:35
local, attributed on Main's authority to the preceding E2E island stage
teardown) forced a reboot (boot `700eaeea`); every device claim below is
from the recovered device, re-run from scratch (22:16 local).

- Fixtures under STRICT bind, all **exact** against numpy fp16 vectors
  (`t1[i] = i*0.25 - 8`; expect `t2 = t1 + 0.25` / `t1 * 0.25`):
  `ane-add-fp16-1x512`, `ane-add-fp16-1x896`, `ane-mul-fp16-1x512` —
  "verified output t2 exact", exit 0.
- Islands under STRICT bind, 1 submit each, 0 timeouts:
  - B `island-select-8head` (5 TDs): exit 0, "verified output
    attention_mask_9 exact" against the split-plan reference
    `B_ref_attention_mask_9.bin`. (The old pinned B output —
    `276a23d6…` — differs from this reference; it is the numerically
    rejected pre-scratch417 run, superseded by the 6832128-byte scratch
    declaration the new bundle carries.)
  - C `island-pv` (208 TDs): exit 0; `attn_output_1` **byte-identical** to
    the prior accepted run (`e1de9eec…`).
  - A `island-attn-a-kt` (416 TDs): exit 0; `attention_scores_1`
    (`f1f18eee…`) and `matmul_0` (`c1c53be9…`) **byte-identical** to the
    prior accepted run.

## Parakeet E2E on the STRICT worker (jwm1-linux, recovered device)

`/var/tmp/ParakeetE2E/parakeet_e2e.py` with the fused origin/main encoder
runner `6c175adf…`, golden capture `/var/tmp/EncoderParityAne/capture`,
model `mweinbach1/parakeet-tdt-0.6b-v3-coreml` `b650695c…`, deadline
20000 ms/submit:

| quantity | value |
| --- | --- |
| status | **match** |
| emissions actual / native | **104 / 104**, matching_prefix 104 |
| transcript sha256 | `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` (= native) |
| encoder_hidden sha256 | **`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`** — the pin holds; no bit shift, no new pin |
| encoder bounds | all PASS (layer 5 `all_bounds_pass`) |
| mel | bit-exact |
| ANE | islands A+B+C, 72 submits, 0 timeouts, exec 2910.4 ms |
| worker / libane | `f171a61e…` / `56b46234…` (STRICT) |

Artifacts: `/var/tmp/island-reexport/out-e2e/` on jwm1
(`e2e-report.json`, `encoder_hidden.npy`, `transcript.txt`).

## mlx-omarchy: pin bump and re-adapted bundles

Branch worktree off `origin/main` `3aa4f348`.

- `ane-compiler.lock`: COMMIT `b61de468d1ca37b687d5fa371e489be16cd84ed1`,
  archive URL `…/ane-parity-b61de46/mil-hwx-compiler-b61de46.tar.gz`,
  ARCHIVE_SHA256 `3e2b2735…`; schema `mil-hwxc.h13-anec-package.v2` and
  target `H13` unchanged. `docs/dependency-licenses.md` and
  `docs/ane-bundles.md` name the new release.
- Gates on the bumped tree: `scripts/verify-ane-compiler.sh all` PASS
  (locked archive sha verified before extraction; full suite log
  `/tmp/verify-b61de46.log` on the build host);
  `test_h13_package_to_bundle.py` 15/15; coreml discover 159 OK
  (skipped=1); `test_ane_export.py` PASS.
- Re-adapted pinned bundles (replace, not additive; old positional copies
  deleted):
  - `receipts/fixtures/exported/ane-add-fp16-1x512`,
    `ane-add-fp16-1x896`, `ane-mul-fp16-1x512` — mil-hwxc `b61de46`
    packages adapted to schema 4 (constant baked in the ANEC; no
    weights.bin payload — the old Apple exports carried one).
  - `receipts/2026-09-14-encoder-split-plan/bundles/{island-pv,
    island-attn-a-kt,island-select-8head}` — `b61de46` recompiles;
    `island-select-8head` declares scratch 6832128 bytes (417 tiles), so
    the `island-select-8head-scratch417` alias is retired: one name, one
    bundle.
  - Island fixture `receipts/2026-09-13-attn-select-island/attn-select-island`
    recompiled at `b61de46` (the old c2cf32e4 ANEC is positional and the
    adapter now refuses it by design); `test_h13_v2_to_schema4` payload
    digests updated to `b801f621…` / `0879c627…`. The select program ANEC
    is byte-identical to the 7ab3eb5-era `scratch417` compile
    (`0879c627…`) — compiler stability corroboration.
  - Host gate on all six: `/tmp/check_bundle_dir` (load_bundle from the
    fb4dfa86 gate sources, bundle.cpp sha `7c4716a4…`) ACCEPT ×6; control
    pre-fix `island-pv` REFUSE.
- `omarchy_ane_bundle_tests` + `mlx-omarchy-info --check-bundle` on the
  replaced bundles: see the landing commit message for the final numbers.

## Not claimed here

- The wheel-side cmake pin (`MLX_OMARCHY_ANE_SOURCE_DIR` HEAD check)
  still selects omarchy-ane `f261a6cb`; STRICT Bind semantics live in
  `5113ef6`+. Product wheels therefore still ship the non-strict
  fallback until that pin moves — the device pass above ran the
  standalone worker with a `6fa243a` strict libane, which is the
  configuration the E2E runner names explicitly.
- `eligibility.py` H13 lowering surface re-extraction (carried over from
  the 2026-09-14 pin-bump receipt).

Resolved model: see the landing session; this leaf ran on
`zai/glm-5.3-flash`.

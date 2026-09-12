# H13 known-graph bundle-adapter boundary (plan §46/§62) — 2026-09-12

Branch `h13/bundle-adapter` (isolated worktree from pinned main
`a12eafd994af881dbc2b39c0b9807523f27b171a`). Assignment: adapt the known-H13
graph package (gate 46) into the existing strict `load_bundle` contract,
host-only; or, if the compiler output cannot reconcile safely, deliver a
precise demonstrated blocker receipt and no partial-success adapter.

## Determination

The pinned-compiler conv_relu package reconciles **losslessly into every
bundle-v2 field and every ANEC cross-check for all 4097 programs — except
exactly one manifest requirement**:

> v2 requires exactly one workspace tensor with a **positive** `byte_size`
> (`parse_tensor` → `require_positive_integer`, manifest.cpp:160-161). The
> honest package-wide value is **0**: `tiles[3] == 0` in every one of the
> 4097 payloads, and the package schema `mil-hwxc.h13-anec-package.v1`
> (ANEH13Compiler.mm:2101-2105) has no workspace/scratch concept at all —
> the H13 writer allocates channels {0, 4, 5+i} only.

The only v2 bundle ever accepted (`receipts/fixtures/mil-oneop-bundle/`)
carries workspace 16384 with `tiles[3] = 0` — the fabricated-positive
pattern discovery receipt §8.6 already ruled out. Fabricating a value here
is forbidden by assignment; weakening the validator is forbidden by
assignment. Therefore per plan §62 ("compiler output cannot be reconciled
with runtime safely"): the adapter is **not implemented**, and this receipt
demonstrates the boundary precisely and independently.

## Inputs (all real, read-only)

| item | value |
|---|---|
| mlx-omarchy main | `a12eafd994af881dbc2b39c0b9807523f27b171a` |
| mil-hwx-compiler | `a0ce354cf800011a84420da4e12013eb8140b2a5` (lock-pinned; archive sha256 `12bfbcd0049849dba18ee5fd3556762bf3313459c8ca5a65ae962f507a5b6d64`) |
| package | `/dev/shm/mlx-compiler-review-20260912/ane-compiler/h13-qualification/conv_relu` (root ran `scripts/verify-ane-compiler.sh all`) |
| package shape | 4097 programs: 1 `conv` (encoder `apple-parity-conv`) + 4096 `maximum` 64-element slices (encoder `h13-source-qualified`), `dispatchPlan == [0..4096]`, slices cover y fully in plan order (elementOffset 0..262080 step 64) |
| `conv_relu.mil` sha256 | `cc204fd0252b16fe514970fcd82abde15600303f8355718984ae8ba4ddab7cf8` |
| `program-0.anec` sha256 | `63462dece46ebcd100b3785be09979786c969d7add75b212902344418fcb7123` (12928 B) |
| `program-1.anec` sha256 | `3d73787c1fc04842da72bb6e1a1930ffa1d3c7b7280ab7d838add2da2edfba22` (4736 B) |
| committed fixtures | `receipts/fixtures/h13-conv-relu-boundary/` (both payloads + source MIL, byte-exact) |

## Reconciliation evidence (executable)

1. **C++ host contract suite** `omarchy_ane_h13_boundary_tests`
   (`overlay/tests/omarchy/ane/test_h13_boundary.cpp`, real bytes embedded;
   staging-proof, no device anywhere):
   - real payloads pass the full libane ANEC header contract; header fields
     equal the compiler records (td 628/1, tsk 628, krn 8192/0, src 1..2,
     dst 1, tiles {0,4(,5,6)});
   - kernel-placement guard §8.8 holds: `align16(628) == 640 ==
     constantOffset` for both program shapes;
   - **the honest manifest (workspace byte_size 0) is rejected by
     `parse_ane_manifest` at exactly `field 'byte_size' must be positive`**,
     before any payload access, for both program shapes;
   - fail-closed identity: with the workspace satisfiable, an empty firmware
     window still rejects (`field 'min' must not be empty`);
   - **boundary diagnostic**: the same manifest with a fabricated positive
     workspace passes the ENTIRE strict pipeline (payload hashes, header
     envelope, channel bindings, tile/NCHW geometry, compiler/provenance
     identity) — isolating the workspace wording as the only gap. The
     diagnostic also pins that the validator cannot check firmware
     semantics (§8.7), which is why firmware stays an operator-verified
     required input, never a default;
   - payload digest/size mutations fail closed before header parse.
   Result: `test cases: 5 | 5 passed`, `assertions: 76 | 76 passed`.
   Existing `omarchy_ane_bundle_tests` unchanged and green: 25/25 cases,
   2719 assertions.

2. **All-package sweep** `demonstrate_blocker.py` (this directory) against
   the real 4097-program package:
   - per program: file sizes, td/src/dst counts, channel allocations,
     tile/NCHW geometry, kernel placement, zero 0x800..0x1000 page —
     ALL PASS;
   - `tiles[3]` across all payloads: `{0}`;
   - emits 4097 honest per-program v2 manifests
     (workspace byte_size 0) + `dispatch_descriptor.json` preserving
     dispatchPlan ordering, per-program tensor slices, and physical buffer
     spans (compiler keys stay outside bundle dirs).

## Identity and firmware (no invented values)

- The package carries no identity fields (writer source verified), so
  `name`, `graph_hash`, provenance, `release_asset`, and compiler identity
  are **required verified inputs**: the demonstration sources them from the
  real pinned artifacts (`graph_hash`/`model_sha256` = sha256 of
  `conv_relu.mil` per `ane_export.py` precedent; `source_commit` =
  compiler commit `a0ce354…`; `exported_at` = 2026-09-12).
- Firmware: no device firmware exists to version (omarchy-ane loads none;
  discovery receipt §7.3; root's jwm1 probe this date found no ANE
  module/device). No measured window exists for this never-device-executed
  package. The adapter design therefore takes firmware min/max as explicit
  required flags with **no defaults**, unlike `ane_export.py`'s 26.0/26.6
  defaults (which produced the fixture's invented applicability). The
  `0.0/0.0` range used in labeled diagnostics demonstrates the validator
  cannot check range semantics — it is not an identity claim.

## Scope statements

- Compiler, driver, bundle validation, and public model: unmodified
  (`git diff` on this branch touches only tests/CMake/fixtures/receipts).
- No hardware/device access; host load validation is NOT ANE execution and
  nothing here claims device output.
- Not in scope (unchanged): full Parakeet compilation (blocked on
  unsupported semantics/shapes/multi-output mixed types), device gate.

## Unblock path (owner decision owed, discovery receipt §8.6)

Representing an absent workspace is NOT a one-line positivity flip. v2
currently pins absence out through five coherent checks, and any unblock
must address all of them as one representation-of-absent-workspace
contract decision:

- `require_positive_shape` (manifest.cpp:159): every shape dim must be
  positive, so a zero-size tensor cannot even be shaped;
- `require_positive_integer("byte_size")` (manifest.cpp:160);
- `require_positive_integer("stride")` (manifest.cpp:161);
- dtype geometry `byte_size == element_size * prod(shape)`
  (manifest.cpp:163-173): with all dims >= 1, byte_size >= 1 is forced,
  so shape [1] + byte_size 0 is geometrically impossible under v2;
- the exactly-one workspace rule (manifest.cpp:355-357): the cleanest
  absence representation, an empty `workspace: []`, is rejected outright.

The owner decision is therefore "how does v2 represent an absent
workspace": either an explicit zero-size tensor form (relaxing shape,
byte_size, stride, and geometry together for exactly that form) or an
empty-list form (dropping the exactly-one rule) — with every other check
retained. When that lands:

1. the honest manifests emitted by `demonstrate_blocker.py` pass
   `load_bundle` unchanged;
2. `test_h13_boundary.cpp`'s boundary case flips green and becomes the
   adapter's acceptance test;
3. the smallest adapter is a thin packager over the already-proven
   emission path (per-program v2 bundles + enclosing dispatch descriptor),
   with firmware/run identity as verified required inputs.

Until then, any "passing" H13 bundle would have to fabricate the workspace
value — the exact defect the accepted fixture already carries.

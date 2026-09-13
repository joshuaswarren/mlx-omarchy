# Bounded ANE worker — jwm1 hardware validation (2026-09-13)

Phase 4 (plan sections 22-27): the in-tree worker boundary's single
bounded hardware window on jwm1-linux. Result: **PASS**.

## What ran

`run-once.sh /var/tmp/AneWorkerValidation-c05ba1df` at commit
`c05ba1df` (plus the fixes landed in this same change):

1. Preflight: `/dev/accel/accel0` character device, `ane` module loaded,
   GPU lock free, quarantine 0.
2. Build on-target: `scripts/prepare-mlx.sh` + CMake
   (`MLX_BUILD_OMARCHY=ON`, `MLX_OMARCHY_ANE_DEVICE=ON`, include dir =
   the pinned libane checkout `~/src/omarchy-ane-h13-f261a6c`,
   f261a6cb) → `mlx-omarchy-ane-worker` + host tests.
3. Host lifecycle tests on-target: 6/6 cases, 32/32 assertions —
   spawn/complete, deadline-kill + quarantine + refusal, signal death,
   named failure without quarantine, missing-input, config validation.
4. `libane.so` built from the pinned checkout (flags mirror its
   Makefile: gnu99, libdrm, the ane UAPI include) and dlopen'd by the
   worker — the mlx library never links libane.
5. `h13_package_to_bundle.py` adapted the repo fixture
   `h13-explicit-chain-add-mul` (add, mul; graph
   `5584d0fd…`; payload hashes `9a6a6a9a…`, `6259594e…`) into a strict
   schema-4 bundle.
6. Deterministic fp16 fixtures: `a[i] = i·0.25 − 8`,
   `b[i] = (i mod 7)·1.5 − 4.5`; expected `y = fp16(fp16(a+b) · b)`.
7. One bounded worker run: `--deadline-ms 5000 --iterations 2
   --input a --input b --expect y`.
8. Post-verify: device present, module loaded, quarantine 0, lock
   released.

## Result

- worker status **Completed**, 2 iterations, 2 programs released,
  2 ms elapsed, no quarantine
- **verified output y exact**: all 64 fp16 lanes match the expected
  values. The only bit-level difference class anywhere in the fixture
  is `+0.0` (device) vs `−0.0` (mathematical product) on lanes where
  `b = 0` — equal by IEEE value comparison, and exactly the compiler's
  unsigned-zero-product model (mil-hwx-compiler `aa688df` "model
  unsigned zero products"). The tool's `--expect` compares fp16 values
  with `±0` equivalence for this reason.
- prohibited actions: none (no unload, no reboot, no 1×896)

## Fixes the window itself forced (all landed with this change)

- Tile staging replaced with the hardware-proven stride layout
  (`NCHW[6] = [N, C, H, W, plane_stride, row_stride]`; element →
  `plane·plane_stride + row·row_stride + column·2` bytes), matching the
  validated a9f14124 smoke runtime; my first attempt used libane's
  `ane_tile`, which scrambled the lanes.
- `--input/--expect/--save NAME=FILE` argument parsing.
- fp16 value comparison for `--expect` (see above).
- Idempotent bundle-output clearing in the runner.
- Power probe paths: the `ane_sys` genpd domain reports
  `runtime_status=unsupported`; module-loaded plus successful
  execution is the power evidence recorded.

Debug runs used the same bounded tool invocations (deadline 5 s, actual
2 ms each) inside the window; the device stayed healthy throughout
(quarantine 0 at every check).

Full log: `hardware-window.log` (final successful segment; earlier
same-day attempts are described above). Receipt: `receipt.json`.

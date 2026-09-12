# Parity status — 2026-09-12 (composed main, authoritative)

Qualifies the composed `origin/main` tree as a single unit and replaces the
stale fractions in the README with measured values. **Headline finding: the
wheel this receipt supersedes could not have existed — `f433007e` did not
build.** The assignment's step-1 tripwire fired exactly: `f03e7ee1` (the
repair of the conflict markers pushed at `e189610f`) landed the FMA route's
`compute.cpp` includes (`qmm_fma_f16.h`, `qmm_fma_precise_f16.h`,
`matmul_fma_bf16.h`) **without** the three matching `omarchy_shader()`
registrations in `overlay/mlx/backend/omarchy/CMakeLists.txt` — the
registrations existed only on `origin/wave/PrefillFmaQualify` (lines
276–278) and were never merged. Both `scripts/build-wheel.sh` and the test
build failed identically (`fatal error: qmm_fma_precise_f16.h: No such file
or directory` at `compute.cpp:282`). Cause: file-level integration of a
branch instead of taking the branch's commits. Main author fixed it on main
at `a2e38c3e`; everything below measures `a2e38c3e`.

## Wheel identity

- `mlx_omarchy-0.32.2.dev202609121038+a2e38c3-cp314-cp314-linux_aarch64.whl`
- sha256 `99664e11d51cd1ab375645992f971a2ea3c902c5d6e4ab2b22682d4b5b84fbe2`
- source commit `a2e38c3e5237c88b5e1c7cf5aee7a1d82d0c633d` (stamped in the
  wheel version, in every leg's provenance line; asserted for all 36 cells)
- `scripts/prepare-mlx.sh` re-run before both builds (fresh `.work/mlx`
  from `overlay/` + patches)
- host placeholder `jwm1` (Apple M1 G13G B1, aarch64), Omarchy Linux kernel
  `7.1.6-1-1-ARCH`, Python 3.14.7, fork driver
  `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`; stock runs via the
  private stock Mesa ICD (`stock-icd.json` sha256 `97bda66c…`,
  `drivers.txt`)
- raw artifacts: `fork-warmup`/`stock-warmup` (discarded), `fork-r{1,2,3}`,
  `stock-r{1,2,3}` JSON + logs, `digest-matrix.json`, `perf-fractions.json`,
  `battery/`, `window.sh`, `window.log`
- one GPU lock window (`/tmp/m1-gpu.lock`, acquired 11:20:01 UTC, released
  11:25:47 UTC, 346 s matrix + battery inside the lock). An earlier window
  attempt was aborted before measurement: jwm1 sat at load ~22 for ~50 min
  (parallel Mesa rebuild) and the quiet gate refused to pass — no numbers
  were taken under load. Two aborted no-measurement runs (path bug, load)
  are visible in `window.log` history only; nothing above came from them.

## Digest matrix — zero movement (36/36 held)

Six canonical legs, fork and stock, three repetitions per driver after a
discarded warmup per driver, every repetition asserted against the
committed pins. Every cell's provenance stamps `a2e38c3`.

| leg (prompt/gen) | fork | stock |
|---|---|---|
| Q4 short (30/32) | `7fd25a869ff21678` | same |
| Q4 long (262/128) | `4cc08910089477fd` | same |
| Q4 1K ctx (1053/32) | `7da83f06ec9f001d` | same |
| BF16 short (30/32) | `f26175202f3dabe9` | `7fc0f968789b1882` (= native) |
| BF16 long (262/128) | `8690dc83246b39f8` | `46108ad71157cb4d` |
| BF16 1K ctx (1053/32) | `ff502900d2a179a5` | same |

## Performance — fork (README driver table)

Fractions of the committed native baseline
(`receipts/native-baseline-2026-09-06/native-2026-09-06-summary.json`,
med of 5). Median of 3 fresh-process repetitions on this wheel.

| leg | decode tok/s (frac) | prefill tok/s (frac) |
|---|---|---|
| Q4 short | 111.3 (0.739) | 331.4 (**1.127**) |
| Q4 long | 108.0 (0.736) | 967.4 (0.797) |
| Q4 1K ctx | 96.4 (0.687) | 1110.4 (0.603) |
| BF16 short | 31.7 (0.561) | 131.6 (0.566) |
| BF16 long | 30.4 (0.545) | 588.5 (0.584) |
| BF16 1K ctx | 24.9 (0.456) | 677.4 (0.409) |

## Performance — stock (same wheel, same protocol)

The FMA prefill route engages only on drivers without cooperative matrix,
so stock changed materially tonight. First published stock table:

| leg | decode tok/s (frac) | prefill tok/s (frac) |
|---|---|---|
| Q4 short | 101.1 (0.671) | 170.4 (0.579) |
| Q4 long | 81.3 (0.554) | 632.4 (0.521) |
| Q4 1K ctx | 52.9 (0.377) | 656.7 (0.357) |
| BF16 short | 32.1 (0.569) | 132.8 (0.571) |
| BF16 long | 30.6 (0.549) | 484.4 (0.481) |
| BF16 1K ctx | 25.1 (0.459) | 492.8 (0.298) |

## Movement vs the ab08be8b composed qualification

Session-to-session noise floor is about 5 percent; differences inside ±0.05
fraction points are **not resolvable across sessions** and are marked
"(noise)".

Fork:

| cell | ab08be8b | now | delta | |
|---|---|---|---|---|
| Q4 short prefill | 0.739 | 1.127 | +0.388 | resolved — recovered to the individually-claimed value (1.13 at `711327ce`); the composed regression is gone at `a2e38c3e` |
| Q4 long prefill | 0.641 | 0.797 | +0.156 | resolved — matches the 0.80 claim |
| Q4 1K prefill | 0.577 | 0.603 | +0.026 | noise |
| Q4 short decode | 0.730 | 0.739 | +0.009 | noise |
| Q4 long decode | 0.729 | 0.736 | +0.007 | noise |
| Q4 1K decode | 0.692 | 0.687 | −0.005 | noise |
| BF16 short prefill | 0.492 | 0.566 | +0.074 | resolved (just outside floor) |
| BF16 long prefill | 0.531 | 0.584 | +0.053 | noise boundary |
| BF16 1K prefill | 0.396 | 0.409 | +0.013 | noise |
| BF16 short decode | 0.566 | 0.561 | −0.005 | noise |
| BF16 long decode | 0.557 | 0.545 | −0.012 | noise |
| BF16 1K decode | 0.458 | 0.456 | −0.002 | noise |

Stock (first composition-level stock comparison; ab08be8b → now):

| cell | ab08be8b | now | delta | |
|---|---|---|---|---|
| Q4 short prefill | 0.462 | 0.579 | +0.117 | FMA route |
| Q4 long prefill | 0.246 | 0.521 | +0.275 | FMA route |
| Q4 1K prefill | 0.189 | 0.357 | +0.168 | FMA route |
| BF16 short prefill | 0.498 | 0.571 | +0.073 | resolved (just outside floor) |
| BF16 long prefill | 0.225 | 0.481 | +0.256 | FMA route |
| BF16 1K prefill | 0.137 | 0.298 | +0.161 | FMA route |
| Q4 short decode | 0.670 | 0.671 | +0.001 | noise |
| Q4 long decode | 0.552 | 0.554 | +0.002 | noise |
| Q4 1K decode | 0.375 | 0.377 | +0.002 | noise |
| BF16 short decode | 0.572 | 0.569 | −0.003 | noise |
| BF16 long decode | 0.557 | 0.549 | −0.008 | noise |
| BF16 1K decode | 0.460 | 0.459 | −0.001 | noise |

## Standing M1 battery — 29 pass, 1 fail (pre-existing)

Full suite list from AGENTS.md (all suites under `overlay/tests/omarchy/`)
plus the capability-sim profile matrix, on the test build of the same
prepared tree (`-DMLX_BUILD_TESTS=ON`, Release). Logs in `battery/`.

- `omarchy_primitive_tests` — **FAIL** (1 of 103 cases):
  `"quantized matmul binds affine streams at storage offsets"`
  (`test_primitives.cpp:5514`, m=1): 0.416748 vs expected 0.409468 at
  epsilon 4e-3. **Byte-identical to the failure bisected in
  `receipts/2026-09-12-composed-main-qualification`** — pre-existing Q4
  GEMV affine-stream binding wrong value, not a new regression and not
  touched by anything landing since.
- All other 25 targets pass: runtime, matmul_family, fast_ops, kv_ops,
  indexing, reduce, shape, linalg, **copy_offset** (the scalar-fill
  ordering test that failed at `ab08be8b` now passes — fixed by
  `f433007e`'s device-side broadcast), distributed, compiled_tape,
  fft_ops, fft_general, eig, take_fill, conv, complex_ops, select_layout,
  fast_regression, scatter_determinism, eq_math, fused_chain,
  error_contract, ane_bundle, plus capability-sim profiles
  `m1-honeykrisp-fork`, `m1-stock-no-coopmat`, `subgroup-size-64`,
  `small-shared-memory`, `no-cooperative-matrix`.

## Distance to parity (goal ≥ 0.95 of native on every canonical leg)

Fork (README driver):

| leg | fraction | gap to 0.95 |
|---|---|---|
| Q4 short prefill | 1.127 | **past parity** (+0.177) |
| Q4 long prefill | 0.797 | 0.153 |
| Q4 1K prefill | 0.603 | 0.347 |
| Q4 short decode | 0.739 | 0.211 |
| Q4 long decode | 0.736 | 0.214 |
| Q4 1K decode | 0.687 | 0.263 |
| BF16 short prefill | 0.566 | 0.384 |
| BF16 long prefill | 0.584 | 0.366 |
| BF16 1K prefill | 0.409 | 0.541 |
| BF16 short decode | 0.561 | 0.389 |
| BF16 long decode | 0.545 | 0.405 |
| BF16 1K decode | 0.456 | 0.494 |

Stock:

| leg | fraction | gap to 0.95 |
|---|---|---|
| Q4 short prefill | 0.579 | 0.371 |
| Q4 long prefill | 0.521 | 0.429 |
| Q4 1K prefill | 0.357 | 0.593 |
| Q4 short decode | 0.671 | 0.279 |
| Q4 long decode | 0.554 | 0.396 |
| Q4 1K decode | 0.377 | 0.573 |
| BF16 short prefill | 0.571 | 0.379 |
| BF16 long prefill | 0.481 | 0.469 |
| BF16 1K prefill | 0.298 | 0.652 |
| BF16 short decode | 0.569 | 0.381 |
| BF16 long decode | 0.549 | 0.401 |
| BF16 1K decode | 0.459 | 0.491 |

The frontier is now BF16 everywhere (all eight BF16 cells are 0.41–0.58 on
fork) plus the Q4 decode row; Q4 prefill is closed on fork (one leg past
native) and halved on stock.

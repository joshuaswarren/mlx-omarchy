# Composed-main qualification — 2026-09-12

Every landing of 2026-09-11 carried its own green receipt, measured on its
own predecessor-commit wheel; no wheel had ever been built from the
composed tree those landings produced. This receipt qualifies that tree as
a single unit: one release wheel from `origin/main` `ab08be8b`, one GPU
lock window (`/tmp/m1-gpu.lock`, 00:51:07–00:58:41 UTC, 384 s) holding the
canonical digest matrix on both drivers, the twelve-cell performance
matrix, and the standing M1 battery.

## Wheel identity

- `mlx_omarchy-0.32.2.dev202609120039+ab08be8b-cp314-cp314-linux_aarch64.whl`
- sha256 `2584d5f1b8dc947fe6b5bca2e47b119c5810f830520e73b343e0535b23a988b7`
- source commit `ab08be8bb5a8cb232330b2a69229ea26e0b0cf32` (stamped in the
  wheel version and in every leg's provenance line; asserted in-window)
- `scripts/prepare-mlx.sh` re-run immediately before the build (fresh
  clone; the prepared `.work/mlx` tree is rebuilt from `overlay/` + patches)
- host placeholder `jwm1` (Apple M1 G13G B1, aarch64), Omarchy Linux
  kernel `7.1.6-1-1-ARCH`, Python 3.14.7, fork driver
  `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`; stock runs via a
  private stock Mesa ICD (`stock-icd.json` sha256 `97bda66c…`, recorded in
  `drivers.txt`)
- raw artifacts: `fork-warmup/stock-warmup` (discarded), `fork-r{1,2,3}`,
  `stock-r{1,2,3}` JSON + logs, `digest-matrix.json`, `perf-fractions.json`,
  `battery/`

## Digest matrix — zero movement (36/36 held)

All six canonical legs, fork and stock, three repetitions per driver after
a discarded warmup per driver, every repetition asserted against the
committed pins. Every leg's provenance line stamps `ab08be8`.

| leg (prompt/gen) | fork | stock |
|---|---|---|
| Q4 short (30/32) | `7fd25a869ff21678` | same |
| Q4 long (262/128) | `4cc08910089477fd` | same |
| Q4 1K ctx (1053/32) | `7da83f06ec9f001d` | same |
| BF16 short (30/32) | `f26175202f3dabe9` | `7fc0f968789b1882` (= native) |
| BF16 long (262/128) | `8690dc83246b39f8` | `46108ad71157cb4d` |
| BF16 1K ctx (1053/32) | `ff502900d2a179a5` | same |

This also closes the one check left open in
`receipts/2026-09-11-bf16-decode-fused-land` ("a short confirmatory digest
window on the f425f46f-based wheel"): the window ran on a wheel built from
`ab08be8b`, which contains `f425f46f`, and no digest moved on either
driver.

## Performance — composed tree vs the individually-claimed cells

Fractions of the committed native baseline
(`receipts/native-baseline-2026-09-06/native-2026-09-06-summary.json`,
med of 5). Composed = median of 3 fresh-process repetitions on this wheel;
`claim` = the value published in README at `ab08be8b` from the individual
landing measurements. **The two Q4 prefill cells regressed on the composed
tree; the README parity table was NOT moved to these numbers** — a
regression is not a correction, and the regression bisect is routed
separately (peer `ComposedRegressions`).

| leg | fork decode tok/s (frac, claim) | fork prefill tok/s (frac, claim) |
|---|---|---|
| Q4 short | 110.0 (0.730, claim 0.75) | **217.4 (0.739, claim 1.13)** |
| Q4 long | 106.9 (0.729, claim 0.74) | **777.4 (0.641, claim 0.80)** |
| Q4 1K ctx | 97.2 (0.692, claim 0.68) | 1062.6 (0.577, claim 0.60) |
| BF16 short | 32.0 (0.566, claim 0.57) | 114.5 (0.492, claim 0.57) |
| BF16 long | 31.0 (0.557, claim 0.52) | 534.7 (0.531, claim 0.58) |
| BF16 1K ctx | 25.0 (0.458, claim 0.48) | 655.7 (0.396, claim 0.41) |

Stock (same wheel, same protocol, no claim to compare — README tracks fork):

| leg | stock decode tok/s (frac) | stock prefill tok/s (frac) |
|---|---|---|
| Q4 short | 100.9 (0.670) | 135.7 (0.462) |
| Q4 long | 81.0 (0.552) | 298.4 (0.246) |
| Q4 1K ctx | 52.6 (0.375) | 348.6 (0.189) |
| BF16 short | 32.3 (0.572) | 115.8 (0.498) |
| BF16 long | 31.0 (0.557) | 226.3 (0.225) |
| BF16 1K ctx | 25.0 (0.460) | 227.5 (0.137) |

Findings:

1. **Q4 short prefill lost a third of the claimed throughput** (333.3 tok/s
   = 1.133 at `711327ce`, per the 2026-09-11 12-matrix verdict, vs 217.4
   tok/s composed; reps 217.4/220.6/209.8, tight). Q4 long prefill lost
   ~20% (966.8 → 777.4). Decode cells sit within ~±0.04 fraction points of
   their claims except BF16 long, which improved (+0.037, the fused decode
   arm helping its best leg). BF16 prefill cells read 0.49/0.53/0.40 vs
   claims 0.57/0.58/0.41.
2. The "one leg already past native" README statement (Q4 short prefill
   1.13) no longer holds on the composed tree. The README table at
   `ab08be8b` therefore still shows pre-composition measurements and is
   known-stale in both directions; correcting it is deferred until the
   regression is attributed and fixed, so the regressed numbers are not
   baked in as the new reference.

## Standing M1 battery — 24 of 26 pass, two failures

Full suite list from AGENTS.md (all suites under `overlay/tests/omarchy/`)
plus the capability-sim profile matrix, run on the test build of the same
prepared tree (`-DMLX_BUILD_TESTS=ON`, Release). Logs in `battery/`.

- `omarchy_copy_offset_tests` — **FAIL**:
  `"back-to-back scalar fills stay ordered in one submission"`
  (`test_copy_offsets.cpp:181`): expected the two scalar fills + add +
  multiply to land in one vk submission, got 6 submissions (values still
  correct: 22/121 checks pass). This test was added this window-generation
  (`e5072671`, the f16-sdpa-ragged wave) and pins the batching contract
  the wrong-value sweep then traded away.
- `omarchy_primitive_tests` — **FAIL**:
  `"quantized matmul binds affine streams at storage offsets"`
  (`test_primitives.cpp:5514`, m=1): one element 0.416748 vs expected
  0.409468 at epsilon 4e-3 — a wrong value on hardware in the Q4 GEMV
  affine-stream binding path (scales/biases bound at byte offsets 1/3 of
  padded buffers, group 64, 4-bit, transpose).
- All other 24 targets pass: runtime, matmul_family, fast_ops, kv_ops,
  indexing, reduce, shape, linalg, distributed, compiled_tape, fft_ops,
  fft_general, eig, take_fill, conv, complex_ops, select_layout,
  fast_regression, scatter_determinism, eq_math, fused_chain,
  error_contract, ane_bundle, primitive subset as noted, plus
  capability-sim profiles `m1-honeykrisp-fork`, `m1-stock-no-coopmat`,
  `subgroup-size-64`, `small-shared-memory`, `no-cooperative-matrix`.

## Failure attribution bisect (same box, same protocol)

Both failing suites were rebuilt and run at seven checkpoints of the
first-parent stack. Each recorded run is a same-checkout build+run pair
(the primitive-suite case count printed per run identifies the binary);
early batched loops that mixed binaries were discarded and re-run in
isolation. Per-run evidence: `bisect/bisect-observations.md`.

| checkpoint | copy_offset scalar-fill | primitive affine | prim cases |
|---|---|---|---|
| `bddc061f` pre-tonight main | PASS ×5 | FAIL ×3 | 100 |
| `5aab281f` qmm-layout merge | PASS | FAIL | 100 |
| `8fd294ed` Q4FusedProjections merge | PASS | FAIL | 103 |
| `de780aa2` scalar-fill drain restore | **FAIL ×3** | FAIL | 103 |
| `485404cf` recycled-storage drain gate | **FAIL ×3** | FAIL | 103 |
| `f425f46f` narrow recycled-drain receipt | **FAIL** | FAIL | 103 |
| `ab08be8b` composed main | **FAIL ×3 cold + window** | FAIL | 103 |

- The scalar-fill batching break is **introduced by `de780aa2`** ("restore
  the scalar-fill host drain the batching commit removed") — deterministic
  and cold-reproducible from that commit to the tip. `485404cf`'s
  recycled-storage gate did not recover the contract: the signature
  (expected 6 submissions, got 7; value checks still pass) is identical
  before and after it. The wrong-value motivation for the drain and the
  one-submission batching invariant pinned earlier in this stack
  (`e5072671`) are in direct conflict as landed.
- The affine storage-offset wrong value is **pre-existing on main before
  tonight's landings**: the case exists and fails at `bddc061f`
  (deterministic, 99/100) and at every checkpoint since, unchanged. It is
  not introduced or worsened by any of tonight's five. This matches the
  qmm-layout worker's "pre-existing on `bddc061f`" report; any claim that
  this case is green on M1 hardware does not reproduce.

## Protocol

One top-level `flock /tmp/m1-gpu.lock` for the whole window; quiet-CPU
gate (1-min loadavg < 1.0, three checks) before measurements; load sampler
every 10 s (loadavg stayed < 1.2); fresh `bench_matrix.py --mode run`
process per matrix run, fork/stock alternating; greedy generation (temp 0,
seed 0), EOS suppressed, pinned lengths 32/128/32, 4 warmup tokens per
leg, `MLX_DISABLE_COMPILE=1`, pinned model revisions asserted
(`qwen25-0.5b-4bit=a5339a41…`, `qwen25-0.5b-bf16=56d07e76…`); wall-anchored
timings only; per-leg provenance gate (wheel RECORD vs loaded `libmlx.so`)
green on every run. Digests re-asserted after the window over all six rep
JSONs (`digest-matrix.json`).

## Provenance

- Host placeholder `jwm1`; no private addresses, serials, or service
  inventory in this receipt or its artifacts.
- Bisect checkpoint suites were run from a scratch clone
  (`mlx-composed-bisect`) so the qualification tree's `.work/` stayed the
  wheel-build evidence.

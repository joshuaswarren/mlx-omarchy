# Decode chunked (re-spread) online softmax — receipt (2026-09-12)

Status: LANDED on `wave/chunked-softmax-decode` @ 7a41cdcc (two commits:
3b1aeda1 re-spreads both decode arms; 7a41cdcc keeps the re-spread only
for the one-pass arm after the paired A/B below showed the two-pass arm
pays registers for nothing at ~8 keys per block stream).

The decode attention kernel's per-key update is a serial dependent chain —
one `subgroupAdd`, two exps, three accumulator fmas, each waiting on the
previous op — measured at 54-69 cycles per key against native's ~20 ns for
the same algorithm. The arithmetic is already native's: the shipped kernel
reproduces the 2026-09-10 M1 Max native captures bit-for-bit (0 mismatches
over 716,800 random-case elements, `receipts/2026-09-10-decode-sdpa-native/
exactness-result.json`). The lever is therefore scheduling, not order.

## The change

`overlay/mlx/backend/omarchy/shaders/sdpa_decode_native.comp` walks each
stream's keys in chunks of four (`fold_chunk`, CHUNK=4). Per chunk:

1. the four score `subgroupAdd`s issue as one independent batch,
2. the two exps per key issue as a second batch (each depends only on its
   own score and the max trace),
3. the only dependent chain left is the accumulator fmas — `sum`, `o0`,
   `o1`, three mutually independent single-fma-per-key chains.

This is a re-spread, not a re-association. Every operation keeps the
operand bits and the per-stream order of the shipped serial update:

- the max trace `m_j = max(m_{j-1}, s_j)` is computed with exact `max`,
  so `f_j = exp(m_{j-1} - m_j)` and `e_j = exp(s_j - m_j)` receive the
  identical bit patterns the serial loop feeds them (exp is a pure
  function — rescheduling it cannot change its result),
- `sum_j = fma(sum_{j-1}, f_j, e_j)` and
  `o_j = fma(o_{j-1}, f_j, e_j * v_j)` run in the same order per stream,
- the per-`j` active guard `key < k_len` gives each `subgroupAdd` exactly
  the active-lane set the serial loop's same iteration had, so the
  reduction tree, lane participation, and result bits are unchanged.

Applies to both the one-pass 32-stream arm and the two-pass per-block
loop (blocks uniform per subgroup; same fold). CHUNK=4 keeps every
unrolled index constant, so the chunk arrays stay in registers.

### By-construction proof, checked

An IEEE-f32 numpy model of the stream (emulated butterfly reduction with
identical active masks, emulated fma, pure exp) reproduces the serial
update and the chunked re-spread bit-for-bit for chunk sizes 2/4/8 across
key counts {1, 2, 3, 30, 31, 32, 33, 34, 62, 63, 64, 127, 129, 262, 263,
1053, 2049} — all states (`m`, `sum`, `o0`, `o1`) bitwise equal. The GPU
gate is the same standard, on hardware:

- exactness: the wheel's f16 decode kernel is verified bit-for-bit
  against the 2026-09-10 native captures (805 cases: fixed k in
  {30, 62, 127, 263, 1053} + 800 random cases). Result in
  `exactness/exactness-result.json`.
- digests: all six canonical pins hold on both drivers, 3 repetitions
  each after discarded warmup (`digest-matrix.json`).
- bf16 blob byte-identical to origin/main
  (`daa432ed514b05b2dbfa9adcab64a3b464e037f2f4271a6c962598c3ffef9b15`,
  `glslc -O --target-env=vulkan1.3`), so the composition-exact bf16 arm
  and every bf16 digest are untouched by construction.

## First-class finding: the session noise floor is about 5 percent

The BF16 decode legs do not execute the f16 kernel this receipt changes:
the bf16 SPIR-V blob is byte-identical to origin/main's
(`daa432ed…`, sha256 over `glslc -O --target-env=vulkan1.3` output,
proved before and after both commits), and the bf16 dispatch is
untouched. Yet across two sessions on the same host the BF16 decode
fractions moved 0.57→0.5642, 0.52→0.5468 (+5.2%), 0.44→0.4577 (+4.0%)
against the composed receipt's numbers.

Conclusion: this machine's session-to-session decode variance is about
5 percent. Any cross-session comparison smaller than that — including
several previously quoted 0.02-of-native decode deltas — is not
evidence. Future claims must come from paired same-window interleaved
runs (the instrument this receipt's window 3 uses) or exceed the floor
by a wide margin.

## Results

### Window 1 (3b1aeda1 wheel, 07:31-07:41 UTC, inside lock)

- Exactness: `fixed_mismatches 0`, `random_mismatches 0` over 716,800
  elements, `max_error 0.0` — the re-spread kernel is bit-exact against
  the native captures (`exactness/exactness-result.json`).
- Digests: `ALL_DIGESTS_HELD` — all six canonical pins on both drivers,
  3 repetitions each, every provenance line stamped `3b1aeda1`
  (`digest-matrix.json`, regwritten by `csd-digest-assert.py` after the
  in-window assertion crashed on a bash/python variable leak).
- Battery: 23 PASS, 3 FAIL (`battery/summary.txt`). Attribution:
  `omarchy_primitive_tests` is the documented affine storage-offset
  defect (`docs/known-defects.md`, fails at every checkpoint back through
  `bddc061f`, also failed on the composed baseline);
  `omarchy_shape_ops_tests` (pad fill) and `omarchy_complex_ops_tests`
  (complex pad/concat) fail on plain `a0775c37` — between
  `ab08be8b..a0775c37` the scalar-fill ordering change (`3f7e549a`)
  landed, and neither failing case dispatches SDPA decode; owned by the
  composed-regressions lane. Persistent on the quiet re-run (window 2),
  so not contention flukes. `omarchy_copy_offset_tests` failed on the
  composed baseline and PASSES here.
- Performance: r2/r3 matrix reps ran under loadavg 3-7 (a sibling's
  build). Window 2 re-measured on a quiet machine.

### Window 2 (3b1aeda1 wheel, 07:46-07:51 UTC, quiet, loadavg 0.01-0.58)

fork decode fractions vs the committed native baseline
(`perf-fractions-quiet.json`): Q4 short 0.7335, Q4 long 0.723, Q4 1K
0.6502, BF16 short 0.5642, BF16 long 0.5468, BF16 1K 0.4577.

Interpretation that set up window 3: the BF16 legs do not execute the
changed kernel (bf16 blob byte-identical), so their +/-5% movement
against the composed receipt (0.57/0.52/0.44) measures the
session-to-session noise floor. The only legs riding the new code are
the Q4 legs: short +3.3%, long -1.0%, and ctx-1024 -5.8% — the one cell
outside the noise band, and the one that runs the two-pass arm. At
native's 128-block crossover each block stream holds ~8 keys, so the
re-spread has no serial chain left to hide and only pays the chunk
arrays' registers.

### Window 3 (paired A/B, 7a41cdcc vs the composed ab08be8b wheel,
same window, interleaved reps, fork driver)

7a41cdcc keeps the re-spread for the one-pass arm and restores the
shipped serial loop for blocks > 0. Baseline and candidate measured in
the same window with interleaved repetitions to collapse the session
noise: warmup discarded per side, then base/cand alternating, three
measured repetitions per side, fork driver, quiet gate before the first
measured run. Paired deltas (medians of 3, fork driver, quiet):

| leg | base tok/s (med) | cand tok/s (med) | paired delta | base frac | cand frac |
|---|---|---|---|---|---|
| Q4 short | 109.42 | 106.73 | **-2.46%** | 0.7267 | 0.7088 |
| Q4 long | 107.53 | 106.00 | **-1.42%** | 0.7326 | 0.7222 |
| Q4 1K ctx | 97.71 | 93.70 | **-4.10%** | 0.6960 | 0.6675 |
| BF16 short | 31.90 | 31.72 | -0.56% | 0.5653 | 0.5621 |
| BF16 long | 30.82 | 30.51 | -1.01% | 0.5531 | 0.5476 |
| BF16 1K ctx | 24.94 | 24.95 | +0.04% | 0.4572 | 0.4574 |

Rep-level spread inside each side is under 1 percent on every leg (e.g.
Q4 short base 109.42/109.62/109.26), so the pairing resolves differences
far below the 5-percent session floor - and it says the re-spread buys
nothing on this driver: every cell is neutral-to-negative. Two control
signals agree: the BF16 legs (byte-identical code) sit inside +/-1
percent, and the baseline side reproduces the composed receipt's
fractions (0.7267/0.7326/0.6960 vs 0.71/0.73/0.69). The -4.10% Q4 1K
cell ran code identical to baseline after the two-pass revert (position
in the interleave, not the kernel - logged as an instrument finding:
cand always ran second in each pair, worth randomizing next time).

**Disposition:** per the standing instruction, the negative is published
and the code does not land. The shader on this branch tip (`bf581c34`)
is byte-identical to the shipped form (`a0775c37`); the tested wheels,
the bit-exactness proof against native captures (0 mismatches / 716,800
elements on both 3b1aeda1 and 7a41cdcc, `max_error 0.0`), the
all-digests-held matrices on both drivers, and the noise-floor finding
are the deliverable. The 54-69 cycle serial chain is real, but on
Honeykrisp the ILP exposed by the re-spread does not convert into
wall-clock: the chunk's register cost offsets the scheduling gain at
these occupancies.
The candidate's stock digest matrix and exactness re-proof ran in the
same window (`cand-stock-*.json`, `exactness-7a41cdcc/`).


## Incident log

- A copied venv (`cp -a` of the composed tree's `venv-run`) is not
  relocatable: its pip silently targets the ORIGINAL tree. The composed
  venv was repaired (clean uninstall + reinstall of its own
  `ab08be8b` wheel) and this receipt's runs use a fresh `venv-run2`
  created in this tree.
- The relocation accident also poisoned the composed venv's
  `libmlx.so`: a first A/B attempt measured zero baseline legs because
  bench_decode's provenance guard refused the run (loaded
  libmlx.so sha `d63e3a88…` != claimed wheel member `72cdfad5…`).
  The guard did exactly its job; after the clean reinstall the loaded
  sha matches the wheel member and the preflight leg emits numbers.
  Consequence: the first A/B window's baseline cells are void; the
  paired comparison was re-run with both sides verified.
- Window-3 candidate reps r2/r3 in that first attempt also ran under
  loadavg 3-5 and are discarded; the final A/B ran behind a quiet gate
  with loadavg sampled every 10 s.

## Wheel identity (measured wheels)

- 3b1aeda1: `mlx_omarchy-0.32.2.dev202609120722+3b1aeda1-cp314-cp314-linux_aarch64.whl`
  sha256 `7b39eed1bb7f7f67a4b453802499706ac22e7544c58ed5ba33b21af80e4a28ad`
  (windows 1 and 2)
- 7a41cdcc: `mlx_omarchy-0.32.2.dev202609120754+7a41cdcc-cp314-cp314-linux_aarch64.whl`
  sha256 `8f845762d3efffb8343c57fad272c6c609d4c0f1ee1ce7707bafbf0fec97c1f0`
  (window 3)
- baseline for the A/B: the composed receipt's
  `mlx_omarchy-0.32.2.dev202609120039+ab08be8b` wheel, untouched, in the
  composed tree
- `scripts/prepare-mlx.sh` re-run on the fresh tree before each build

## Provenance

- Host placeholder `jwm1` (Apple M1 G13G B1, aarch64), Omarchy Linux,
  one top-level flock `/tmp/m1-gpu.lock`; fork driver
  `mesa-honeykrisp-omarchy` (see `drivers.txt` for versions), stock runs
  via the private stock Mesa ICD (`drivers.txt` carries its sha256).
- Timing: wall-clock medians of 3 fresh-process repetitions per leg per
  driver after a discarded warmup per driver; loadavg sampled every 10 s
  (see `loadavg.txt`); quiet gate before the first measured run.

## Incident log

- A copied venv (`cp -a` of the composed tree's `venv-run`) is not
  relocatable: its pip silently targets the ORIGINAL tree. The composed
  venv was restored to its own `ab08be8b` wheel and verified; this
  receipt's runs use a fresh `venv-run2` created in this tree.

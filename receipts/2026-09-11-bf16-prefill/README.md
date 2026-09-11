# BF16 prefill close — qualification receipt (2026-09-11)

Decision and paired numbers: `verdict.json`. Raw stage logs, suite logs,
attribution NDJSON, and the full paired matrix live in this directory.

## What was qualified

The dense BF16 prefill coopmat restructure (`wave/Bf16PrefillClose`):
64-lane workgroup, two 16x32 subgroup halves, 16-wide k steps, word-pair
bf16 loads, 8 accumulators per lane, k%16==8 tail, 4 KiB shared-memory
dispatch gate. Bit-identical to the production shader on the fork driver
at every screened shape; per-layer projection speedups from window A2
(1.25-2.25x per dispatch, per-layer chain 74.5 -> 52.8 ms at m=1053).

## The bug the qualification caught (and fixed)

Window B's production family suite caught the candidate shipping a real
wrong-results defect: the qmm-shaped restructure (commit 93581a93)
computed scalar-orientation staging addresses from step-local k and
dropped `k_base` (`pack_pair` call sites), so every k step after the
first re-staged step-0 data. Any dispatch with A col-major or B row-major
and k>16 was wrong (family suite tile32 max_abs_err up to 145); word-path
cells stayed exact, which is why the flags=1-only probe screen had passed
36/36 while production shapes were wrong. Fixed in commit 1918135e; the
broken window was killed before its matrix, and the candidate wheel was
rebuilt. Evidence: `window-b-broken-run1.log`,
`window-b2-stage1-attempt1.log`, `m1-logs/probe-flags-sweep-invalid.ndjson`.

A follow-up lesson: the bench probe's own flags sweep was an invalid
instrument (out-of-bounds layouts for non-production flags; reads past
the operand buffers compare uninitialized-memory luck), so correctness
cells stay at the production path (commit 1f53de21) and
multi-orientation staging coverage belongs to the family suite, which
fills valid layouts.

## Method (split per Main's re-sequencing; builds outside the lock)

1. stage1: candidate wheel, venvs, and two sets of suite binaries built
   outside the lock (base shader from frozen mlx-main-b6d662a8 vs cand
   shader; binaries differ by construction, sha-stamped).
2. stage1b (flock, 41 s): driver gate + probe bit-identity, 36/36 exact.
3. Q4GemvNativeMapping and PythonHostParity took their windows.
4. stage2 (flock): family + runtime suites on both binaries (rc=0 x4,
   zero row mismatches — includes the new tail-k rows and tile32 in all
   four orientations), attribution probes, 13-leg paired matrix
   (warmup + 3 reps x {fork,stock} x {base,cand}).
5. stage2b (flock): attribution re-run with MLX_DISABLE_COMPILE=1
   (Honeykrisp refuses compiled bf16 tapes by design) after two stale-mlx
   API fixes in the probe (rope offset, rms_norm signature).

Digest gates: `../2026-09-10-bf16-prefill-close/summarize_prefill.py`
against the collected matrix — 0 failures. Six canonical Q4 digests
unchanged on both drivers; BF16 fork f26175202f3dabe9 / 8690dc83246b39f8,
stock 7fc0f968789b1882 / 46108ad71157cb4d, 1K ff502900d2a179a5 on both;
base cells reproduce pre-landing canon. All six paired BF16 prefill
medians repeat the gain on both drivers.

## Files

- `verdict.json` — decision, digest gates, suites, paired medians, fractions.
- `provenance.json` — commits, wheel/binary hashes, host, lock windows.
- `window-b2-stage*.log`, `window-b-broken-run1.log` — raw stage logs.
- `m1-logs/` — suite logs, probe sweeps, attribution NDJSON, hashes.
- `matrix/` — 13 paired matrix runs + `summary.json` (summarizer output).
- `verdict_gen.py` + `summary.json` + `suite-status.json` — verdict inputs.

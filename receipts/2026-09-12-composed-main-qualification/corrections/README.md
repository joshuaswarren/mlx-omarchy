# Corrected-instrument re-measurement — 2026-09-12

Re-measurement of the twelve cells on the same `ab08be8b` wheel
(`mlx_omarchy-0.32.2.dev202609120039+ab08be8b`, sha256 `2584d5f1…`) after
fixing the reporting defect in the prefill pipeline. One lock window
(02:04:54–02:10:30 UTC, 336 s), same protocol as the qualification window
(warmup matrices discarded, 3 reps per driver, quiet gate + sampler, pins
asserted, digests re-asserted: ALL_DIGESTS_HELD).

## The instrument defect

- The timer was never coarse: `bench_decode.py` stamps
  `time.monotonic_ns()` at t0 and at the first-token yield
  (`prefill_ns = times[0] - t0`), a nanosecond monotonic span with load
  and warmup excluded.
- The resolution loss was in reporting: `bench_decode` prints only
  `prefill %.3fs` (1 ms granularity), and `bench_matrix` regex-parsed
  that printed line (`PREFILL_RE`) instead of the machine-readable
  result line, which carries `prefill_s` at 6 decimals. Every published
  prefill tok/s derived from the printed line therefore carries up to
  ±0.5 ms of pure rounding.
- Fix: `bench_matrix.parse_bench_output` now consumes `prefill_s` from
  the JSON result line (regex fallback kept for output that predates the
  result line); self-test extended to pin the override.
  **Rule for every future consumer: the JSON field is the instrument;
  never re-derive timings from printed output.**

## Per-leg inflation bounds (printed-line instrument)

Bound = ±0.5 ms on the span, at the OLD published spans (the numbers the
README table was built from) and at this wheel's corrected spans:

| leg | old span (published) | bound at old span | corrected span (this wheel) | bound at corrected span |
|---|---|---|---|---|
| Q4 short | 0.090 s | **0.556 %** | 0.133929 s | 0.37 % |
| Q4 long | 0.275 s | 0.182 % | 0.332986 s | 0.15 % |
| Q4 1K | 0.945 s | 0.053 % | 0.975126 s | 0.05 % |
| BF16 short | 0.226 s | 0.221 % | 0.241984 s | 0.21 % |
| BF16 long | 0.448 s | 0.112 % | 0.451981 s | 0.11 % |
| BF16 1K | 1.555 s | 0.032 % | 1.619100 s | 0.03 % |

The short legs are the only ones where the rounding is even visible
(≥ 0.2 %). At every leg the bound is one to two orders of magnitude
below the Q4-short tree gap (1.13 vs 0.74–0.76): the quantization cannot
explain that gap, and the regression finding stands.

## Corrected twelve cells (this wheel, med of 3, full-precision spans)

| leg | fork decode tok/s (frac) | fork prefill tok/s (frac) |
|---|---|---|
| Q4 short | 106.2 (0.705) | 224.0 (0.762) |
| Q4 long | 107.7 (0.734) | 786.8 (0.649) |
| Q4 1K | 97.3 (0.693) | 1079.9 (0.587) |
| BF16 short | 32.0 (0.567) | 124.0 (0.533) |
| BF16 long | 28.9 (0.519) | 579.7 (0.575) |
| BF16 1K | 24.1 (0.441) | 650.4 (0.393) |

| leg | stock decode tok/s (frac) | stock prefill tok/s (frac) |
|---|---|---|
| Q4 short | 97.6 (0.648) | 142.5 (0.485) |
| Q4 long | 78.1 (0.532) | 302.3 (0.249) |
| Q4 1K | 50.7 (0.361) | 346.3 (0.188) |
| BF16 short | 29.7 (0.526) | 117.7 (0.506) |
| BF16 long | 29.0 (0.521) | 225.9 (0.224) |
| BF16 1K | 23.9 (0.437) | 227.0 (0.137) |

## Between-window drift, observed

The two windows on the identical wheel are ~1 h apart. Cell deltas
between them run from 0.8 % (BF16 1K prefill) to 8.4 % (BF16 long
prefill; decode moved −6.7 % on the same leg) — machine-state drift
between sessions that is 4–20× larger than the printed-line quantization
on the long legs. Consequences:

- Single-window 3-rep fractions carry roughly ±5 % session uncertainty on
  the BF16 legs; the corrected BF16 long prefill (0.575) sits within that
  of the published 0.58 claim, and BF16 1K (0.393 vs 0.41) likewise.
- The Q4 short prefill gap (published 1.133, measured 0.739–0.762 across
  two windows and two instruments) is ~33 %, far outside quantization
  (≤ 0.56 %) and session drift (≤ 8.4 %). It is a property of the
  composed tree, not of the instrument or the session.

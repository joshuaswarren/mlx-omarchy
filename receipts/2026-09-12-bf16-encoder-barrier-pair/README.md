# Deferred post-dispatch barriers: no measured decode gain

The gate-on candidate preserved all 72 digest cells in the matched M1 window
but showed no repeatable decode gain. Main receives this receipt only. The
experimental encoder flag and trace ABI remain on
[wave/Bf16EncoderBoundary](https://github.com/joshuaswarren/mlx-omarchy/tree/wave/Bf16EncoderBoundary).
This result rejects this optimization for the measured workloads; it does
not establish an upper bound on all barrier cost or locate every remaining
source of the native-performance gap.

## Candidate and baseline

- Baseline: `a12eafd994af881dbc2b39c0b9807523f27b171a`.
- Candidate: `f5649936f68a2262e6f1e51ad9b146ec035808cb`.
- Baseline wheel SHA-256: `9e601cebadd578657e678a173dd2efc993679ea3469c563f63d2b6c5725aef94`.
- Candidate wheel SHA-256: `3f90904520d5441b2b1133aada10160b4f9cfb58bca79a378e7420c62e042df0`.
- Host: jwm1, Apple M1 G13G B1; fork driver
  `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`.
- Stock smoke uses the existing private stock ICD override. No driver,
  kernel, release fallback, or digest-policy change was made.

The candidate keeps the pre-dispatch HOST|TRANSFER|COMPUTE → COMPUTE
barrier. It defers the post-dispatch COMPUTE → COMPUTE|TRANSFER|HOST
barrier until copy, fill, diagnostic full barrier, or batch close. A later
compute dispatch's pre-barrier covers the earlier compute read/write
hazards. The deferred barrier retains the old access masks before a
non-compute consumer; its position in the command buffer changes.

Review caught an incompatible expansion of the eight-field
`MlxOmarchyTraceSnapshot` consumed by `scripts/fragmentation_probe.py`.
The measured candidate preserves that layout and exposes barrier counters
through a separate symbol. Neither change is integrated into main.

## Engagement

The isolated 200-dispatch chain emitted 400 barriers with the gate off;
with the gate on it emitted 300 and counted 200 deferred post-barriers
(`window3/probe-cand-{off,on}.txt`).

The separate 12-token generation probe includes prompt processing and
completion work, not just steady-state decode. Both arms recorded 790.5
compute dispatches and 4.333 submissions per generated token. The candidate
counted 790.5 deferred post-barriers and 846.75 emitted barriers per token.
Its measured emitted count is about 46.4% below the baseline's inferred
two-per-dispatch count of 1581; the baseline has no barrier counter symbol.

Subtracting the 790.5 pre-barriers leaves 56.25 deferred flushes per token.
There were 52 copies and 4.333 submissions per token; those counts do not
sum exactly to the flush count. A copy or close flushes only when a barrier
is pending. The aggregate counters do not identify which boundary did not
flush. See `window4/decode-{base,cand-on}.txt` and
`instruments/decode_probe.py`.

## Matched timing and correctness

Window 3 held one `flock /tmp/m1-gpu.lock`, passed load <1.0 on three
checks 20 seconds apart, discarded a full warmup per arm, measured an A/A
baseline pair, then ran interleaved b-c-c-b-b-c repetitions. Both stock
smoke arms followed in the same lock window. The wrapper completed at
17:02:33Z after 364 seconds with checker exit 0. Source/wheel provenance
reported `verified=match` in each measured cell. Raw logs report
`harness=unknown`; instrument source is retained here.

Twelve runs × six required legs held their existing fork or stock digest
pins. This includes warmups, the A/A pair, six paired runs, and two stock
smokes. Timing below is fork-driver decode throughput, median of three;
stock was a correctness smoke, not a repeated performance comparison.

| Leg | Baseline tok/s | Candidate tok/s | Candidate / baseline |
|---|---:|---:|---:|
| Q4 short | 111.59 | 111.69 | 1.0009 |
| Q4 long | 107.79 | 107.29 | 0.9954 |
| Q4 1K context | 96.27 | 96.08 | 0.9980 |
| BF16 short | 31.86 | 31.86 | 1.0000 |
| BF16 long | 30.48 | 30.45 | 0.9990 |
| BF16 1K context | 24.98 | 24.95 | 0.9988 |

The A/A pair differed by 0.22%, 0.79%, and 0.04% on BF16, and 4.43%,
0.09%, and 0.64% on Q4. Q4 long and BF16 1K candidate decreases exceed
their own A/A pair spread; not every delta is inside its per-leg spread.
A single A/A pair is not a confidence interval. These measurements support
no repeatable gain, not a statistical upper bound such as 0.3 ms/token.

The counter reduction did not improve these decode medians. It does not
prove that required dependency stalls, dispatch launches, or submission
cadence are the exclusive remaining costs. Native performance parity
remains open; the published composed-main baseline is unchanged.

## Invalid and limited earlier windows

- Window 1 is invalid: fork legs exited 127 because an environment
  assignment became the command; the checker accepted an empty leg set.
  Its printed success is not correctness or performance evidence.
- Window 2 measured gate-off versus gate-off. The probe recorded zero
  deferrals. A launcher error later repeated that window. Neither run
  measures this mechanism's performance.
- Window 3 is the gate-on comparison retained here.
- Window 4 is the generation-counter probe, not a timing qualification.

Historical instruments remain on the diagnostic branch. Main retains
only the successful-window instruments and raw evidence.

## Local semantic checks and limits

The candidate's runtime and fast-ops suites ran on software Vulkan with
`MLX_OMARCHY_ALLOW_NON_APPLE=1`. The worker recorded both gate states;
the parent independently repeated gate-on: runtime 41 cases / 22,691
assertions, fast ops 35 cases / 1,116,299 assertions, no failed assertions.
Software Vulkan cannot exercise independent GPU execution or the M1
cooperative-matrix route; those paths printed skip messages. An initial
parent run without the development-device opt-in returned early from GPU
tests and is not semantic evidence. These checks are not the full standing
M1 battery, which was not run for this unmerged negative experiment.

Raw measurements are in `window3/` and `window4/`; the commands and
probes are in `instruments/`. Production encoder code is unchanged.

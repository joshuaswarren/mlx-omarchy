# 2026-09-03 — dispatcher-poll and ordering bakeoff, compiled-path corruption proof, README column replacement (jwm1, 8 cores)

Same machine and session as
`receipts/2026-09-03-decode-ab-and-affinity-jwm1.md`; read that receipt for
the wheel-generation correction and the EOS-truncation caveat that motivated
this work. All runs on jwm1-linux, M1 T8103, real Honeykrisp device,
`MLX_OMARCHY_ALLOW_NON_APPLE` unset.

## Legs 1 and 2: one-constant bakeoff, eager mode

Protocol per TinyWriteFix: both sides built on the box with
`scripts/build-wheel.sh`, each wheel into its own fresh venv, prompt
"What is the capital of France?" for every leg, `MLX_DISABLE_COMPILE=1`,
`--max-tokens 32 --temp 0 --seed 0`, 5 runs, median.

| leg | wheel sha256 | bf16 decode | 4-bit decode |
|---|---|---|---|
| base `3fdcc20` | `5d2e185e1d2253626768028bcd17a6ca2ac57305bcd0243b5d66bc9a0c8545f1` | 2.045 | 2.225 |
| fix `4ea2f47` (1 ms in-flight poll) | `061dfd36ddae74285ff920b8f14d58013e75945cd433d12a85793a940d566dd5` | 2.065 | 2.222 |
| ordering `ff4b05a` (device-side timeline wait) | `f8e35d7b12f0c33fa56b861dc7b9e915afbab9ec48a5cae6b8ccc9961f2b366c` | 2.134 | 2.217 |

- Leg 1 verdict: the dispatcher wakeup is **not** on the decode critical
  path. bf16 +1.0%, 4-bit -0.1% — noise. The hypothesis died cleanly.
- Leg 2 verdict: the ordering fix is **free** in eager mode. bf16 +3.3%
  (inside run spread), 4-bit -0.2%.
- Caveat carried from the decode-metric fix: these are EOS-truncated
  short-burst rates (8-10 generated tokens) and are comparable only within
  the table, which holds prompt and length fixed.

## Leg 3: compiled path at `ff4b05a` — silently corrupt, not an abort

`env -u MLX_DISABLE_COMPILE`, 4-bit snapshot
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, same prompt, 20 runs requested.
Run 1 failed and the leg stopped per protocol. Verbatim output
(`jwm1:~/benchq/logs/ABORT.run1.log`):

```
==========
<|endoftext|><|endoftext|><|endoftext|>avs<|endoftext|>StrictEqual identical胬逐111! .  :      !  的帮助  auditing
==========
Prompt: 36 tokens, 16.307 tokens-per-sec
Generation: 32 tokens, 1.977 tokens-per-sec
Peak memory: 0.292 GB
```

rc=0, all 32 tokens generated at normal speed. Every eager run at the same
seed produced clean `The capital of France is Paris.`. Two observations are
recorded WITHOUT asserting a shared cause: (a) the compiled path previously
aborted hard at a trigonometric gate on a value near 9.1e8; (b) with
`ff4b05a` it no longer aborts and instead returns wrong values. Leg 4
(compiled speed) is cancelled: the speed of a wrong answer is meaningless.

## C++ batteries on real hardware (ff4b05a tree)

Built with `-DMLX_BUILD_TESTS=ON -DMLX_BUILD_OMARCHY=ON` and run on the
Honeykrisp device. Logs: `jwm1:~/benchq/logs/battery-omarchy_*.log`.

| battery | result |
|---|---|
| `omarchy_compiled_tape_tests` | rc=0, 8/8 cases, 343/343 assertions |
| `omarchy_eq_math_tests` | rc=0, 7/7 cases, 116/116 assertions |
| `omarchy_runtime_tests` | rc=1, 22/23 cases, 6179/6181 assertions |

The one runtime failure, verbatim: TEST CASE "independent streams measure
against the serialized sum" (`test_runtime.cpp:896`, CHECK at `:997`):
`[receipt] iters=400 serialized_median=0.380567s concurrent_median=0.314865s
ratio=1.20867` / `CHECK( ratio < 1.10 ) is NOT correct! values:
CHECK( 1.20867 < 1.1 )`. That is a stream-overlap timing assertion on a
single-queue device, not a values test. The compiled-tape battery passing
8/8 on the hardware that corrupts means the model-level defect needs a
graph larger than the unit tests build; the differential harness against
the model is the reproducer path.

## README 8-core column replacement (current main, pinned-length decode)

Wheel built from main `e7a6542e2162c6e0f9b5a58cf40630ab1e9bbfb5`:
`mlx_omarchy-0.32.2.dev202609031509+e7a6542-cp314-cp314-linux_aarch64.whl`,
6797038 bytes, sha256
`ba0afdb788e1e023c449f286b4d6f7831c1f4ca1d3a238008a03725875f6cf55`.
`scripts/bench_decode.py`, prompt "What is the capital of France? Answer in
one word." (36 prompt tokens), `--tokens 64` with EOS suppressed and the
token count asserted, load excluded from decode, two passes on a settled
box, medians:

| row | value | condition |
|---|---|---|
| bf16 prefill | 18.5 tok/s (36-token prompt; 1.946/1.943 s) | current-main wheel, compile off, 8 cores |
| bf16 decode | 1.85 tok/s over 63 tokens (steady-state, EOS suppressed) | same |
| 4-bit prefill | 15.5 tok/s (36-token prompt; 2.333/2.319 s) | same |
| 4-bit decode | 1.97 tok/s over 63 tokens | same |

The old 8-core decode rows (2.04 / 3.88 "tok/s") were EOS-truncated 2-10
token bursts and are not comparable; DecodeMetricFix's receipt
(`receipts/2026-09-03-decode-metric-fix.md`) annotates them. The old
prefill rows were taken on the stale-generation wheel (5.3 MB `libmlx.so`)
and are superseded by this re-measure.

## Not done here

- Compiled-path performance numbers: withheld by order until the corruption
  is fixed and the output is right.
- Profiling-wheel submissions-per-token counts (DecodeMetricFix protocol):
  parked by the parent; the v0.3.3-diag.1 asset and protocol are documented
  in `scripts/profile_generate.py` when it is re-queued.

# 2026-09-03 — Decode metric defect and pinned-length replacement

## Defect

Every decode tokens-per-second number this project published before
2026-09-03 is an EOS-truncated short-burst rate. Runs used
`--max-tokens 32`, which is only a cap: with a greedy prompt such as
"What is the capital of France?" the model stops after 2-10 tokens. A
recorded "3.85 tok/s" was a 2-token rate. Each figure therefore mixed
model load, prompt processing, fixed startup, and a couple of decode
steps. It is not a steady-state decode rate and is not comparable
across machines or wheels, which defeats the purpose of the community
benchmark table.

Affected published rows: README bf16 decode (3.56, 2.04 tok/s) and
4-bit decode (6.46, 3.88 tok/s) columns, the same rows in
`receipts/2026-09-03-eight-core-remeasure.md`,
`receipts/2026-09-03-decode-ab-and-affinity-jwm1.md`,
`receipts/2026-09-02-gpu-profile-decode.md`,
`receipts/2026-09-02-m1-qualification.md`,
`receipts/2026-09-01-m1-same-chip-parity.md`,
`receipts/2026-09-01-macos-native-mlx-baseline.md`,
`receipts/2026-08-31-m1-mlxlm-fp16-smoke.md`,
`receipts/2026-09-02-gemv-decode/README.md`, and
`docs/2026-09-01-m1-4bit-greedy-sdpa-f16-scores.md`. Each now carries an
annotation; the rows are kept for history, not restated as steady state.

## Fix

`scripts/bench_decode.py` measures steady-state decode at a pinned
generation length:

1. EOS is suppressed for the run (`tokenizer.eos_token_ids = set()`), so
   generation produces exactly `--tokens N` tokens.
2. The produced token count is asserted against N. A short burst exits
   nonzero and emits no rate.
3. Model load is excluded by construction; prompt processing (prefill)
   is timed and reported separately.
4. The rate is always printed with its token count: "decode X tok/s over
   Y tokens".

Self-test evidence (this x86 box, no GPU): `python3
scripts/bench_decode.py --self-test` proves the pinned length is honored
(64 requested, 64 produced) and that the assertion fires with exit 1
when 3 tokens arrive against a request for 32. llvmpipe timings would be
meaningless; only the assertion behavior was verified here. Timings must
be re-measured on jwm1.

## Submissions per token

The per-submission-overhead explanation is now directly measurable.
`overlay/mlx/backend/omarchy/trace.h` counts `vk_submissions` and
`vk_compute_dispatches`. Exposure paths (no new one invented):

- The `MLX_OMARCHY_GPU_PROFILE` event stream records one `{"k":"s"}`
  record per submission with a host timestamp. `scripts/profile_generate.py`
  writes decode-phase markers; `scripts/profile_analyze.py` now prints
  `submissions/decode-token` and `dispatches/decode-token` next to the
  existing per-phase report. This requires the profiling wheel
  (for example `v0.3.3-diag.1`); the default release wheel compiles the
  profiling harness out (`overlay/mlx/backend/omarchy/CMakeLists.txt`).
- `mlx-omarchy-info` now prints `vk_compute_dispatches` in its trace
  JSON and smoke output alongside `vk_submissions` (previously missing).

Run eager (`MLX_DISABLE_COMPILE=1`) versus compiled at the same pinned
length: the ratio of submissions/decode-token is the direct test of the
per-submission-overhead explanation.

## What must be re-measured on hardware (jwm1)

1. Pinned-length decode rates, bf16 and 4-bit, eager, 64 tokens.
2. The same with compiled tapes enabled.
3. Submissions/decode-token for both, via the profiling wheel.
4. New README table rows only after 1-3 exist.

Hardware protocol for BenchQueueM1: see the hub handoff of 2026-09-03.
Nothing in this change ran on jwm1.

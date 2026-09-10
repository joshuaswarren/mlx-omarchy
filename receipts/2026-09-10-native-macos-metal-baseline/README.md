# 2026-09-10 — Authoritative native macOS Metal baseline (canonical parity matrix)

Date 2026-09-10 (UTC). Purpose: replace the unprovenanced "native macOS
Metal" denominator (Q4 decode 150.84/146.57/140.25 tok/s) that survives only
as a secondary citation in `m1gpuqual-20260906-326e2ec6/HANDBACK.md:11`
(ane-linux-experiments tree) with measured, provenance-complete native
numbers taken with the SAME protocol as the committed Linux canonical
verdict `receipts/2026-09-10-main-parity-12-matrix/verdict.json`
(mlx-omarchy `main` commit `b6d662a8`, receipt commit `e154bb8f`).

## Chip equivalence — read this first

**None of the reachable macOS hosts is a base M1. The Linux canonical host
`jwm1-linux` is a base Apple M1 (T8103, 8-core GPU, ~200 GB/s). A
non-base-M1 chip is NOT a valid same-chip denominator for it.** Every
native:linux ratio below is therefore cross-chip context, never a parity
divisor. Per-chip numbers are reported unnormalized; no cross-chip
normalization was applied anywhere in this receipt.

| Host | Chip | GPU | CPU cores | Memory | macOS |
|---|---|---|---|---|---|
| 16m1mbp | Apple M1 Max (applegpu_g13s) | 32-core, Metal 4 | 10 (8P+2E) | 64 GB | 26.6.2 (25G83) |
| MacStudio | Apple M1 Ultra (applegpu_g13d) | see host JSON | 20 (16P+4E) | 128 GB | 26.6.2 (25G83) |
| laptop | Apple M2 Max | see host JSON | 12 | 96 GB | 26.6.2 (25G83) |

## Why the old denominator is rejected

`HANDBACK.md:11` quotes 150.84/146.57/140.25 tok/s as "native" ratios for
its Linux Vulkan decode numbers. The audit
(NativeBaselineScout, 2026-09-10) found: the cited raw receipt
`receipts/native-baseline-2026-09-06` was never committed; the numbers
appear nowhere else in-tree; no prefill figures, machine identity, or
software versions accompany them; the device string recorded alongside
looks copied from the Linux host. They are not reproducible and are
superseded by this receipt.

## Protocol (identical to the Linux canonical matrix)

- Engine: `scripts/bench_decode.py` at harness commit `b6d662a` (verbatim
  copies in `harness/`), fresh process per leg-run, `MLX_DISABLE_COMPILE=1`.
- Greedy temp 0 seed 0, EOS suppressed, pinned generated counts 32/128/32,
  4 warmup tokens per leg, prefill timed separately, decode rate over the
  n-1 inter-token gaps, digest = sha256(exact generated ids)[:16].
- One full 6-leg warmup matrix discarded per host before measurement.
- Prompts: `short` ("Hi", 30 chat-template tokens), `long` (262),
  `ctx1024` (numbered template, 1053) — byte-identical to
  `scripts/bench_matrix.json` expansion; exact per-leg `prompt_tokens`
  asserted (30/262/1053) or the run refuses.
- Models pinned: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`;
  `mlx-community/Qwen2.5-0.5B-Instruct-bf16`
  `56d07e766edd7159fbe12ed12d9cf114bf38bf1e` — local HF snapshot dir must
  equal the pin or the run refuses.
- Software: upstream PyPI `mlx==0.32.2` + `mlx-lm==0.31.3` (same mlx-lm as
  the Linux canonical run; upstream MLX, not the mlx-omarchy fork).
- Gates: AC power; no loaded ollama models (`ollama ps` empty); no
  standalone model servers persisting across a ~1 min rescan loop
  (transient `llama-server --version` probes seen and waited out); and a
  contention watch on the resident `omlx-server` router leg (cumulative
  CPU sampled around every leg-run): a rep is contended when the watched
  delta >= 0.10 s OR its decode_tps lands below 70% of the leg's
  best-band (upper-half median) — the second guard catches unwatched
  GPU contenders; contended reps are excluded from medians and >= 3
  clean reps are required per leg. 0.01 s heartbeat ticks were measured
  throughput-neutral (283.6-296.8 tok/s ticked vs 282.5-303.9 strictly
  clean on the same legs) and stay included.

## Results (decode median tok/s / prefill median tok/s; range min–max;
16m1mbp 12 reps, MacStudio 7 reps)

| leg (prompt/gen) | 16m1mbp — M1 Max, 32c GPU | MacStudio — M1 Ultra | laptop — M2 Max |
|---|---|---|---|
| Q4 short (30/32) | 286.96 (282.95–300.49) / 1517.55 | 299.19 (285.78–304.54) / 1410.55 | not available |
| Q4 long (262/128) | 296.25 (294.51–300.43) / 5937.91 | 287.09 (281.96–290.61) / 6754.28 | not available |
| Q4 1K ctx (1053/32) | 283.79 (283.25–285.62) / 8048.42 | 282.98 (243.02–289.60) / 11040.14 | not available |
| BF16 short (30/32) | 217.88 (215.95–220.63) / 1192.74 | 257.08 (250.77–259.21) / 1111.17 | not available |
| BF16 long (262/128) | 214.72 (210.83–216.76) / 5281.12 | 252.92 (251.76–254.29) / 6265.09 | not available |
| BF16 1K ctx (1053/32) | 207.95 (201.09–210.73) / 7751.35 | 243.39 (241.63–244.78) / 9932.54 | not available |

Clean reps: 16m1mbp 10–11 of 12 per leg; MacStudio 5–6 of 7 per leg
(strict zero-delta classifier; the 0.10 s + 70 % classifier reproduces
the same medians). Every accepted rep's digest matches the reference
native digest for its leg.

**laptop (M2 Max): no accepted matrix.** The resident `omlx-server`
router leg served sustained traffic through every attempt (12/12 reps
contended twice, per-leg CPU deltas 1–2.2 s, decode collapsed to
47–68 tok/s); attempts were refused rather than published. Raw evidence:
`att1/laptop-att1-routerbusy-rep12/`. The router was not disturbed
(standing llm-router redundancy order); a clean M2 Max window must wait
for quiet router traffic. The two accepted hosts already bracket the
cross-chip conclusion above.

Read-across: Q4 decode on M1 Max and M1 Ultra is nearly identical
(~283–299 tok/s) despite 2× memory bandwidth — 0.5B Q4 decode is
kernel-overhead-bound, not bandwidth-bound. Against the Linux base-M1
canonical legs, native Q4 decode is ~2.6–2.9× fork and ~2.8–5.3× stock;
native BF16 decode (~208–257) is ~18–22× the fork/stock BF16 decode
(~10–11.7) — the Linux fork's BF16 decode deficit, not a native
reference. Prefill ratios are larger still. All of these are cross-chip
observations; none is a same-chip parity divisor. Full table:
`gap-table.md`.

## Digest verification

All six legs on every accepted host reproduce the reference native digests
recorded in the committed Linux verdict (`verdict.md` digest table):
q4_short `7fd25a869ff21678`, q4_long `254d73fd93164b98`, q4_longctx
`7da83f06ec9f001d`, bf16_short `7fc0f968789b1882`, bf16_long
`407b7624ed1b3b29`, bf16_longctx `ff502900d2a179a5`. Digested identity was
stable across every repetition including contended ones: on native Metal +
upstream MLX, router contention changed speed only, never the generated
ids. The known Linux-vs-native digest divergences (q4_long Linux
`4cc08910089477fd` vs native `254d73fd93164b98`; bf16_long Linux
`ad964232ee67fecd`/`c5be9207833d2a26` vs native `407b7624ed1b3b29`)
reproduce exactly, confirming `docs/parity-id-policy.md` rule 2.

## Mac measurement hazards encountered (recorded honestly)

1. First 16m1mbp attempt ran before the watch existed; numbers consistent
   with its later clean reps, but ungated — kept only as
   `att1/native-baseline-16m1mbp-att1-gateless.json` corroboration, not
   authority.
2. macstudio attempt 1 was hit mid-run by the resident `omlx-server`
   router leg (58% CPU, ~31 GB resident): decode collapsed 296 -> 11
   tok/s with digests unchanged. Kept as
   `att1/native-baseline-macstudio-att1-omlx-contended.json`; the router
   leg was NOT disturbed (standing llm-router redundancy order).
3. With `ollama stop` applied to resident models and the CPU-watch
   active, macstudio produced a valid gated matrix (router busy for the
   first two reps, quiet for 5-6 clean reps per leg; medians unchanged
   under either classifier).
4. 16m1mbp attempt inventory (all raw artifacts retained on the host):
   att2 refused (sustained router burst, 1 clean rep); att3 died mid-run
   (bench_decode produced no output once; connection drop); att4 killed
   by a local timeout mistake; att5 refused (transient `llama-server
   --version` mid-scan); att6 refused under the zero-tolerance rule
   (every rep ticked 0.01 s — this evidence set motivated the 0.10 s
   threshold); att7 (12 reps, 10-11 clean per leg) accepted after
   re-summary with the final classifier; att8 refused (sustained
   ~1.1 s/rep router load). att7 is the accepted matrix.
5. laptop attempts 1-2: 12/12 reps contended each time (sustained
   1-2.2 s router deltas, decode 47-68 tok/s) — refused; no accepted
   M2 Max matrix in this window.

## Files

- `native-baseline-16m1mbp/`, `native-baseline-macstudio/` — full raw
  trees: discarded warmup matrix, per-rep per-leg `.log` (verbatim
  bench_decode output) and `.json` with per-rep watch deltas, and the
  final `native-baseline-<HOST>.json`.
- `att1/` — superseded-attempt evidence: gateless 16m1mbp att1,
  contended macstudio att1, 16m1mbp att6 all-tick rep12 (threshold
  rationale), 16m1mbp att7 pre-distguard final JSON, laptop att1 rep12
  contended sample.
- `verdict-linux-canonical.json` — copy of the committed Linux verdict
  this receipt gaps against (source of truth:
  `receipts/2026-09-10-main-parity-12-matrix/verdict.json` @ e154bb8f).
- `harness/` — byte-identical `bench_decode.py`, `mlx_provenance.py`,
  runner `run_native_matrix.py`, `prompts.json`, `resummarize.py`,
  `summarize_gap.py`.
- `gap-table.md`, `gap-table.json` — native vs Linux canonical legs with
  explicit cross-chip framing.

## Reproduction

Per host (example 16m1mbp):

```
ssh 16m1mbp
cd ~/src/mlx-bench-20260910
WATCH_PID=$(pgrep -x omlx-server | head -1) \
  ~/src/mlx-bench-20260901/venv/bin/python run_native_matrix.py 12 \
  native-baseline-16m1mbp
```

The runner refuses on: missing pinned snapshot, prompt-token drift,
generated-count shortfall, non-AC power, loaded ollama models, standalone
model servers, or < 3 clean reps for any leg.

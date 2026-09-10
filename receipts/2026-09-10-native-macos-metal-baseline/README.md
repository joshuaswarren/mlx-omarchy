# 2026-09-10 — Authoritative native macOS Metal baseline (canonical parity matrix)

Date 2026-09-10 (UTC). Purpose: independently re-measure the native macOS
Metal baseline with the SAME protocol as the committed Linux canonical
verdict `receipts/2026-09-10-main-parity-12-matrix/verdict.json`
(mlx-omarchy `main` commit `b6d662a8`, receipt commit `e154bb8f`), and
extend it with prefill + BF16 coverage on M1 Max / M1 Ultra.

**Correction during this work:** the base-M1 native denominator is NOT
missing. It is committed in this repository at `ea09eefa`
("Publish the matched native-vs-Linux M1 performance baseline", 2026-09-06):
`receipts/native-baseline-2026-09-06/native-mlx0.32.2.json` records a real
base Apple M1 Mac (Darwin arm64, 8 GPU cores, 16 GiB, macOS 14.8.9, mlx
0.32.2) and `native-2026-09-06-summary.json` carries 5 stable-repeat legs
(Q4 decode 150.57/146.77/140.38, prefill 294.1/1213.0/1840.9; BF16 decode
56.43/55.72/54.55) on the same pinned revisions, with the same native
digests this receipt reproduces. That committed denominator STANDS. The
earlier "unprovenanced" suspicion arose because
`m1gpuqual-20260906-326e2ec6/HANDBACK.md:11` (a different repository)
cited a receipt path without noting it lives here; the numbers it quotes
(150.84/146.57/140.25) match the committed receipt within noise.

## Chip equivalence — read this first

**None of the hosts measured in THIS receipt is a base M1. The same-chip
denominator for the base-M1 Linux canonical legs is the committed ea09eefa
base-M1 receipt; the M1 Max / M1 Ultra / M2 Max numbers below are
cross-chip context, never a parity divisor.** Per-chip numbers are
reported unnormalized; no cross-chip normalization was applied anywhere
in this receipt.

| Host | Chip | GPU | CPU cores | Memory | macOS | Status |
|---|---|---|---|---|---|---|
| (ea09eefa) | Apple M1, 8-core GPU, Metal 3 | 8 | 8 | 16 GiB | 14.8.9 (23J631) | committed denominator, 5 reps |
| 16m1mbp | Apple M1 Max (applegpu_g13s) | 32-core, Metal 4 | 10 (8P+2E) | 64 GB | 26.6.2 (25G83) | accepted, 12 reps |
| MacStudio | Apple M1 Ultra (applegpu_g13d) | see host JSON | 20 (16P+4E) | 128 GB | 26.6.2 (25G83) | accepted, 7 reps |
| laptop | Apple M2 Max | see host JSON | 12 | 96 GB | 26.6.2 (25G83) | not available (router busy) |

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

| leg (prompt/gen) | base M1 — committed ea09eefa | 16m1mbp — M1 Max, 32c GPU | MacStudio — M1 Ultra | laptop — M2 Max |
|---|---|---|---|---|
| Q4 short (30/32) | 150.57 (150.56–151.10) / 294.10 | 286.96 (282.95–300.49) / 1517.55 | 299.19 (285.78–304.54) / 1410.55 | not available |
| Q4 long (262/128) | 146.77 (146.34–147.03) / 1213.00 | 296.25 (294.51–300.43) / 5937.91 | 287.09 (281.96–290.61) / 6754.28 | not available |
| Q4 1K ctx (1053/32) | 140.38 (140.07–141.20) / 1840.90 | 283.79 (283.25–285.58) / 8048.42 | 282.98 (243.02–289.60) / 11040.14 | not available |
| BF16 short (30/32) | 56.43 / 232.60 | 217.88 (215.95–220.63) / 1192.74 | 257.08 (250.77–259.21) / 1111.17 | not available |
| BF16 long (262/128) | 55.72 / 1007.70 | 214.72 (210.83–216.76) / 5281.12 | 252.92 (251.76–254.29) / 6265.09 | not available |
| BF16 1K ctx (1053/32) | 54.55 / 1655.70 | 207.95 (201.09–210.73) / 7751.35 | 243.39 (241.63–244.78) / 9932.54 | not available |

Base-M1 column = the committed ea09eefa receipt (`committed-base-m1/`);
5 repeats, digests identical to this receipt's. It is the same-chip
denominator for the base-M1 Linux canonical legs.

Clean reps: 16m1mbp 10–11 of 12 per leg; MacStudio 5–6 of 7 per leg
(strict zero-delta classifier; the 0.10 s + 70 % classifier reproduces
the same medians). Every accepted rep's digest matches the reference
native digest for its leg, and the committed ea09eefa base-M1 receipt
carries the same six digests.

**laptop (M2 Max): no accepted matrix.** The resident `omlx-server`
router leg served sustained traffic through every attempt (12/12 reps
contended twice, per-leg CPU deltas 1–2.2 s, decode collapsed to
47–68 tok/s); attempts were refused rather than published. Raw evidence:
`att1/laptop-att1-routerbusy-rep12/`. The router was not disturbed
(standing llm-router redundancy order); a clean M2 Max window must wait
for quiet router traffic. The two accepted hosts already bracket the
cross-chip conclusion above.

Read-across: base-M1 → M1 Max → M1 Ultra decode scales ~150 → ~287 →
~287 tok/s (Q4) and ~56 → ~215 → ~253 (BF16): Q4 decode saturates at M1
Max (kernel-overhead-bound at 0.5B), while BF16 keeps scaling with
memory bandwidth. This is why cross-chip ratios mean nothing without the
per-chip rows. The same-chip comparison that matters for the Linux
canonical legs is base-M1 native (ea09eefa) vs base-M1 Linux
fork/stock — decode 150.57 vs 112.20 fork / 102.53 stock (1.34x /
1.47x native-faster), Q4 prefill 294.1 vs 331.5 fork / 170.9 stock, BF16
decode 56.4 vs 11.7 / 11.7 (4.8x) — full table in `gap-table.md`. The
M1 Max / M1 Ultra rows are cross-chip context only.

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

- `committed-base-m1/` — the committed base-M1 denominator (extracted
  from `ea09eefa`): identity JSON, 5-repeat summary, and its conversion
  to this receipt's schema.
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

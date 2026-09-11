# Decode in-model vs isolated gap: receipt-only — the gap was the instrument

- schema: mlx-omarchy/decode-gap/1 (CLOSED 2026-09-11)
- agent: DecodeInModelGap; branch wave/DecodeInModelGap (off main 26b746e0, never merged)
- host: jwm1-linux, Apple M1 (G13G B1), installed fork driver mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1, linux-asahi 7.1.6
- window: ONE top-level flock /tmp/m1-gpu.lock, 2026-09-11 08:02Z, quiet gate, ~30 s GPU, no driver swap, no overlay source change

## Assignment

Explain and remove the 2.4x gap between in-model (~236 us/layer) and isolated
(~98 us/layer) cost of the same Q4 decode GEMV kernels, testing the candidates
(residency eviction, dependency stalls, per-dispatch state re-emission,
allocator churn, submission structure) with an isolated interleaving
experiment, one factor at a time. Land a minimal digest-preserving change if a
gain repeats; otherwise receipt-only naming the mechanism.

## Result: receipt-only. The gap is an instrument artifact, not recoverable time.

The new `--gap` mode in tools/q4-bw-bench rebuilds the in-model decode
interleaving in isolation — 24 independent per-layer weight sets
(8,427,008 bytes/layer x 24 = 202 MB/token, the exact four Qwen2.5-0.5B
dispatch shapes through the production shader, subgroup defines, flags=0),
with per-arm ablations. Timing is host CLOCK_MONOTONIC around whole submits
only (per-dispatch timestamp brackets are a forbidden instrument since
receipts/2026-09-10-dispatch-floor). Two independent runs, two passes each,
medians of 7 rounds.

### The wall-anchored truth

| arm (one factor each)                        | us/layer | delta vs iso4 |
|---|---|---|
| iso4 — isolated calibration (1 set)          | 202.9    | — |
| iso4dep — RAW dependency chain               | 202.5    | −0.4 |
| churn — fresh x/out allocations per round    | 202.0    | −0.9 |
| fill2 — 2 MB interstitial read between GEMVs | 250.4    | +47.5 = filler's own work; GEMVs unchanged |
| sweep 1→24 sets (residency)                  | 203→207  | ≤ +6, FLAT, no cache crossover |
| wideind — 24 sets, independent               | 206.6    | +3.7 |
| widedep — 24 sets + RAW chain (in-model analog) | 206.6 | +3.7 (+1.8%) |
| sub2async — two submissions per token        | 220.5    | +13.9 vs widedep = +0.33 ms/token (upper bound) |
| sub2sync — fence between submissions         | 231.2    | +24.6 |

**The same four dispatches cost ~203 us/layer isolated and ~207 us/layer with
full in-model interleaving.** Interleaving costs 1.8%. The 138 us/layer "gap"
decomposes as:

1. **~105 us/layer — instrument artifact.** The old isolated 98 us/layer
   (bench-a, receipts/2026-09-10-q4-gemv-native-map) is a sum of per-dispatch
   device-timestamp intervals. Two independent measurements:
   - *Wall vs timestamps, same kernel, same machine, same defines:* wall gives
     2.07x the timestamp sum (202.9 vs 98.0 us/layer).
   - *Physical impossibility:* the old number implies 86.0 GB/s of weight
     streaming against a 68.25 GB/s DRAM part. The wall-anchored rate is
     41.5 GB/s. This same run's timestamp-based peak probe re-demonstrates the
     broken clock: a 512 MB copy "in 41,833 ns" = 12.8 TB/s.
2. **~33-55 us/layer — composition.** The in-model 236 us/layer (5.68 ms/token
   ablation marginal over 97 dispatches) includes the lm_head GEMV
   (151,936x896 q4 = 72.3 MB, one dispatch, ~1.0-1.3 ms/token at its
   18,992-workgroup grid's plausible rates) amortized over 24 layers. The true
   in-model per-layer projection cost is ~180-195 us/layer — at or below the
   isolated number.

### Factor attribution (assignment's candidate list)

- residency/cache eviction: **0** — sweep flat 203→209 us/layer from an
  8.4 MB (cache-fitting) footprint to the full 202 MB token working set; the
  kernels are latency-bound and insensitive to weight locality at any level.
- dependency stalls (RAW chain): **0** (−0.4 us).
- allocator churn: **0** (−0.9 us).
- per-dispatch state re-emission / pipeline switching: **0** — free per the
  dispatch-floor receipt, and in-model all four dispatches share one pipeline.
- interstitial attention/norm/rope/KV work: **no eviction effect**; its cost
  is its own streaming work (fill2's +47 us = the filler kernel itself).
- submission structure: the only real term — two async submissions per token
  bound at **+13.9 us/layer = +0.33 ms/token (+3.7% short leg)**, and the
  bench overstates it (it records submit B after submitting A; production
  records the sampler batch while the forward batch executes).

### Why no landing

In-model dispatch cost is at parity with isolated; the kernel chain runs at
its intrinsic ~41.5 GB/s (61% of the part) in every context. The only
actionable-looking term (submission boundary) is an upper bound with no
proven in-model win, and closing it means merging mlx-lm's two per-token
graph evals or speculative commit deferral — behavior-risky redesign for at
most 3.7%. The remaining kernel headroom (203 vs the 123 us/layer
full-DRAM-peak floor) is inside the kernel and already shown unreachable by
bit-pinned candidates (receipts/2026-09-10-q4-gemv-bandwidth,
receipts/2026-09-10-q4-gemv-native-map). Per the assignment's bar:
receipt-only. Digest legs not exercised — nothing landed.

### Corrections this receipt makes to prior readings (for policy owners)

- receipts/2026-09-10-q4-gemv-bandwidth "achieved 1.1-1.20 GB/s" divided one
  layer's traffic (9.36 MB, mislabeled per-token) by a full token of GPU time;
  per-token q4 traffic is ~202 MB + 72 MB lm_head → ~48 GB/s in-model.
- Its "bench_isolated_kernel_ticks" absolutes (and the "warm 9.3 MB working
  set (SLC-resident)" explanation built on them) rest on the untrustworthy
  device clock; the sweep here shows residency explains nothing at all.
- receipts/2026-09-10-decode-attribution-finish ablation marginals remain
  valid (wall-anchored); its microbench gemv row is harness-floor, as flagged.

## Provenance

- bench: 1364d077 (tools/q4-bw-bench --gap + run-gap-m1.sh; default bench
  behavior unchanged; prior-pinned shader copies untouched)
- shader refresh: 373f63da — frozen qmm_vec_base.comp 7e57b15d → 56aff1d3;
  sole delta is the producer-direct KV sum-store epilogue (commit a06bf49a,
  flags bit 12+i) which flags=0 dispatches never execute; GEMV body
  byte-identical, so measurements are about the shipped kernel
- session: gap-session-20260911T080220Z.txt (shader sha256s, commit, host)
- legs: gap-a-20260911T080220Z.ndjson, gap-b-20260911T080220Z.ndjson

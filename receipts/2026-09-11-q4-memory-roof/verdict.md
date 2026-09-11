# Q4 decode as a memory-efficiency problem: receipt-only negative — the layout is at the roof, the kernel is not, and no bit-preserving change reaches it

- schema: mlx-omarchy/q4-memory-roof/1 (CLOSED 2026-09-11)
- agent: Q4DecodeMemoryRoof; branch wave/Q4DecodeMemoryRoof (bench arms + receipt only; never merged, no production source change, no test change, digest legs not exercised)
- host: jwm1-linux, Apple M1 (G13G B1), installed fork driver mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 (verified before each window), linux-asahi 7.1.6.asahi1-1
- windows: window 1 = ONE top-level flock 08:43Z (~15 s GPU, roof legs a/b, commit 22ad3bc); window 2 = ONE top-level flock 10:04Z (~70 s GPU, corrected peaks + candidate screen, commit 6db0506); window 2b = ONE top-level flock ~10:20Z (~5 s GPU, AGX shader dumps only). Nothing nested; no driver swap.
- timing: wall-anchored only — host CLOCK_MONOTONIC around whole submits (submit_and_wait bracket). No device-timestamp number is used anywhere in this receipt.

## Assignment

Establish the real memory roof for the exact Q4 GEMV access pattern (packed
four-bit words + per-group f16 scales/biases, one row per lane group); sweep
load width, alignment, stride, rows per workgroup against a large-copy roof
measured the same way; identify the specific inefficiency; fix the access
pattern without changing arithmetic order (six canonical digests are a hard
gate); measure an order-changing faster pattern separately and do not land
it; land only a repeating gain, else push this receipt-only negative.

## 1. The roof, wall-anchored (window 2 corrected peaks + pattern probe)

| stream | GB/s (leg A / leg B) | note |
|---|---|---|
| large streaming copy, 256 MB | **58.44 / 58.52** | the roof anchor, same instrument |
| large streaming read, 256 MB | 59.54 / 59.65 | read-dominant streams run slightly above a balanced copy |
| production Q4 pattern, near-empty body (pat_w1_r8) | 60.24 / 60.09 | **the layout is AT the roof** |
| pat_w1_r8 line-aligned pitch 512 B | 59.99 / 59.98 | +1.5%, noise-level |
| pat_w1_r8 odd-phase pitch 114 words | 58.95 / 58.56 | −0.6%, noise-level |
| rows per workgroup 1 / 2 / 4 / 8 / 16 | 58.9-59.8 / 58.9-59.4 | flat; concurrency at the probe's 65k-workgroup scale saturates at any row count |
| uvec4 loads (16 B per lane) | 60.7-61.1 / 61.0-61.5 | +2.7% |
| **pair map: 2 consecutive words per lane per step (pat_w2_r8)** | **67.2-68.3 / 67.2-68.3** | **+13%; exceeds the balanced-copy roof because the stream is read-dominant** |

Arm-level conclusions the assignment asked to adjudicate:

- Partial cacheline utilization: **not the limiter** — line-aligned rows buy
  1.5%, odd-phase rows cost 0.6%; the production contiguous 448 B pitch
  streams at the roof.
- Uncoalesced scale/bias fetches: **not the limiter** — per-group f16
  scale+bias (25% traffic overhead over the words, format-inherent) are
  included in every pattern arm and the pattern still saturates.
- Read amplification from the group layout: **format-inherent only**
  (0.625 vs 0.5 B/el), priced into the 60 GB/s; nothing recoverable.
- Load width: uvec4 +2.7% at the memory system; minor.
- The only material access-pattern lever is the **lane->word map** (two
  consecutive words per lane): +13%, ABOVE the copy roof. That map changes
  each lane's word sequence, hence the accumulation order pinned
  bit-for-bit to native qmv (receipts/2026-09-09-q4-gemv-order). **Measured
  and reported here; NOT landed** — the six canonical digests would move.

Instrument correction (belongs in the record): the historic "copy peak"
readings of 9-13 TB/s (including the gap receipt's "512 MB copy in 41,833
ns = 12.8 TB/s") were an **invalid dispatch** — 1<<24 uvec4 / 256 threads =
65536 workgroups, one over maxComputeGroupCountX = 65535, which honeykrisp
mis-executes. Fixed in run_peak (clamp + account real coverage); the valid
copy roof is the 58.5 GB/s above.

## 2. Where the production kernel actually is (gap-mode, wall, matched pairs)

In the in-model-style environment (24 independent 8.4 MB sets = the
202 MB/token working set, RAW chain = `widedep`; medians of 7 wall rounds x
2 passes x 2 legs):

| side | us/layer (A / B) | GB/s | vs base |
|---|---|---|---|
| base (production shader) | 205.5 / 206.1 | **41.0 / 40.9** | 1.000 |
| pf2 (prefetch depth 2) | 225.0 / 225.7 | 37.5 / 37.3 | **1.095 / 1.095 (WORSE)** |
| pf4 (depth 4) | 242.3 / 242.5 | 34.8 / 34.8 | **1.179 / 1.177 (WORSE)** |
| pf8 (depth 8) | 256.3 / 256.3 | 32.9 / 32.9 | **1.247 / 1.244 (WORSE)** |
| wg128 (4 columns/workgroup, 2x warps) | 206.2 / 196.9 | 40.9 / 42.8 | **1.003 / 0.956 — does not repeat** |

Reference arms reproduce the prior receipts exactly on this instrument:
iso4 203.3/202.6 us/layer (41.5/41.6 GB/s) vs 202.9 in
receipts/2026-09-11-decode-gap; iso4dep/churn/wideind deltas all ~0;
sub2async +14 us/layer, sub2sync +24 (submission boundary, upper bound).

So the assignment's hypothesis is confirmed as the LOCUS — the production
kernel runs at 41 GB/s where its own byte layout delivers 60 — but the
bit-preserving levers do not recover it:

- **Prefetch depth (Main's MLP-depth ladder): monotonically WORSE.** Each
  lane's own future words were issued as clamped in-bounds register loads
  ahead of use, consumed in the exact existing order (bit-identical by
  construction; llvmpipe tree-mode eq 42/42 cells zero across all seven
  sides). Depth 2 already costs 9.5% us/layer and it degrades linearly to
  +24.5% at depth 8, on both legs. AGX IR (AGX_MESA_DEBUG=shaders):
  base 949 instructions, pf2 993 (+4.6%), pf4 1029 (+8.4%), pf8 1101
  (+16%) — instruction growth is small and cannot explain a 10-25%
  slowdown; the cost is liveness (one more live register per lane per
  depth), i.e. occupancy: fewer resident warps means LESS device-level
  memory-level parallelism. The per-warp and per-device MLP levers pull in
  opposite directions and per-warp prefetch loses. Final allocated-register
  counts are not exposed by the driver's debug flags (AGX_MESA_DEBUG
  shaders/ra dumps carry no RA stats); instruction counts and the measured
  ladder are the reported cost evidence (window2-agx-ra.log).
- **wg128 (occupancy via 2x workgroups): bimodal, no repeat.** iso4
  (hot 8.4 MB set): −3.8%/−4.0%, consistent. But the decision environment
  (widedep, the 202 MB streaming set) gives +0.34% on leg A and −4.43% on
  leg B — each leg internally tight (passes within 0.1%), legs disagree.
  Fails the >=3%-repeating bar; landing it would risk a real regression.
- **The 41 -> 60 GB/s kernel gap is therefore measured-unreachable within
  the pinned arithmetic order**: not by per-lane prefetch (occupancy tax),
  not by workgroup remapping (bimodal), not by load width (+2.7% at best),
  not by alignment/stride/rows (layout already at roof). The order-changing
  route (pair map, +13% over roof, ~68 GB/s) exists and is reported but is
  digest-forbidden.

Perspective for the fraction bookkeeping: in-model the kernel chain already
runs at 41 GB/s, which is native macOS's effective whole-token rate
(150.57 tok/s x 278.8 MB/token = 41.9 GB/s if GEMV were 100% of runtime).
Even a hypothetical at-roof kernel would cut ~1.5 ms/token off the short
leg (~14%) — the remaining fork-vs-native deficit lives outside the q4 GEMV
chain. The ladder result also explains WHY native does not beat this by
more: 8.4 MB/layer streaming with per-word FMA chains is latency- and
occupancy-bound on this part, not roof-bound.

## 3. Decision: receipt-only negative

No source change lands: the assignment's bar (gain repeating) is not met by
any bit-preserving arm, and the only faster pattern changes the
accumulation order. The six canonical Q4 digests
(short 7fd25a869ff21678, long 4cc08910089477fd, 1K 7da83f06ec9f001d, on
fork and stock) are untouched — no digest legs were needed and none were
run. Nothing is merged; wave/Q4DecodeMemoryRoof carries bench arms
(--roof mode, wall brackets, corrected peaks, pf ladder shaders) and this
receipt only. Regression test: none added — nothing landed.

## Provenance

- bench: wave/Q4DecodeMemoryRoof — 22ad3bc (roof mode + wall brackets),
  0013728 (peak grid fix + gap candidate screen), 6db0506 (pf2/pf4/pf8
  ladder + screen swap to wg128+pf arms)
- llvmpipe bit-identity screens (tree mode, before any timing): 42/42 eq
  cells zero for unroll/loadfirst/wg128/pf2/pf4/pf8 vs base
- window 1 legs: roof-{a,b}-20260911T084318Z.ndjson +
  roof-session-20260911T084318Z.txt (shader sha256s; qmm_vec_base 56aff1d3
  = production shader, diff-gated in-script)
- window 2 legs: roof-{a,b}-20260911T100428Z.ndjson (corrected peaks),
  window2-gap-{a,b}-20260911T100428Z.ndjson, window2-roof.log
- AGX dumps: window2-agx-ra.log (9 shader dumps incl. base + pf ladder)
- GPU windows: announced to CoopmatIlpChains / QmmF16Operand / Main via hub
  before each flock; M1 working clone /tmp/q4roof-m1 (shared, ephemeral)

# BF16 vector-mapping — receipt (2026-09-12)

Status: COMPLETE. Receipt-only negative. The assigned NEW bit-preserving
mechanism for `shaders/matmul_vec.comp` (`USE_BF16`) was named from
inspection, measured, and eliminated two independent ways: it is not
bit-preserving (software counterexample), and the kernel has no
load-bound headroom to convert (in-window hardware rates). Nothing lands;
no digest moves; no driver or ICD change touched.

## Question

Composed BF16 decode sits at 0.456/0.545/0.561 of native
(`receipts/2026-09-12-parity-status`). The gemv class is the dominant
ablation marginal (18.31/17.96/18.06 ms/token,
`receipts/2026-09-11-bf16-decode-attribution`), and `matmul_vec.comp`
(`USE_BF16`) owns its vec half (97 of 169 gemv dispatches/token).
Column-group spans 4/16/64/512, wider Q4 row mapping, split-K, source
de-divergence, and chunked-softmax all have published negatives, so the
assignment asked for a NEW mechanism named from the actual BF16 vector
arithmetic, access topology, generated instruction structure, and the
native counterpart — with no assumed win.

## Inspection (what the kernel actually does)

`matmul_vec.comp`, `USE_BF16` arm: one 32-thread workgroup (= one
subgroup) owns 4 output columns; thread t owns k-elements
{4t..4t+3} advancing by 128 (32 lanes x 4). Per k-iteration each thread
issues one `uvec2` A-load (its x-slice) and four `uvec2` B-loads (rows
r..r+3 at the same k-span), then 16 explicit `fma`s under `precise`
(4 serial chains of 4). Reduction is a shuffle-down tree (16/8/4/2/1);
lane 0 stores four RNE-packed bf16 outputs.

- Access topology: per (row, iteration) the 32 lanes touch 256
  consecutive bytes — full-line coalesced on all four B streams; A is
  x (1.8 KB) re-read per workgroup and served by cache. There is no
  over-fetch: the B bytes loaded are exactly the mandatory weight bytes.
- Generated instruction structure (glslc -O --target-env=vulkan1.3,
  `spirv/`): the inner loop is 5 storage-buffer `v2uint` loads + 16
  `GLSL.std450 Fma` + per-load index arithmetic; nothing scalarized, no
  pathological addressing, push constants uniform.
- Native counterpart (pinned MLX Metal `gemv.h` + `matmul.cpp`): for
  these shapes macOS uses 128-512-thread threadgroups, 4-element
  k-chunks per thread per row, threadgroup-memory cross-reduction when
  bn > 1 — a different shape, the same mandatory bytes. Native's decode
  advantage is dispatch structure, not kernel bandwidth: native streams
  the same ~988 MB token at ~56 GB/s including everything, i.e. also at
  the practical roof.

## Named mechanism and its elimination

**Mechanism: per-lane load widening — load-instruction-count reduction.**
Remap each lane from 4-element k-chunks (one `uvec2` per row per
iteration) to 8-element k-chunks (one 16-byte load per row per
iteration, stride 256), cutting total load instructions ~43% at k=896.
This is the strongest surviving candidate in the assigned space
(packing/address lowering/load count): the layout is already
byte-minimal, so only instruction count or bandwidth remain.

Eliminated, both ways the assignment allows:

1. **Not bit-preserving — software counterexample.** `model/bit_model.py`
   models the exact f32 accumulation (bf16xbf16 products are exact in
   f64, so f64-add + f32-round is the exact fma; the same standard as
   the chunked-softmax receipt). On benign random bf16 data the remap is
   association-stable — 0 of 96 columns differ at k=896/4864 — but the
   tie-rich adversarial grid yields
   `model/counterexample_K16.npz`: shipped sums to `0x0000`, the widened
   mapping to `0x4000` (ULP 16384). A bit-moving remap cannot land:
   fork short/long pins may only move to native's digest and BF16 1K
   holds at full strength (`docs/parity-id-policy.md`). A same-lane-mapping
   variant (span 2) is bit-identical by construction but changes no
   bytes and adds workgroups — no roof-level gain exists to collect.
2. **No headroom — in-window hardware rates.** See the window below:
   the engaged kernel streams lm_head at 57.6 GB/s read while the
   same-instrument copy moves 52.1 GB/s aggregate. A DRAM-bound kernel
   that already exceeds copy throughput cannot gain from fewer load
   instructions; the half-size arm's rate (53.5 GB/s, 93% of full)
   confirms the bandwidth-bound regime rather than a latency/ramp-bound
   one.

## Metric adjudication (required before use)

- **lm_head isolated 9.3 ms (attribution isolation probe): INVALIDATED.**
  In-window whole-submit median over 24 warmed rounds is **4.730 ms**
  (57.57 GB/s read); the span screen's 4.687 ms reproduces (0.9%
  agreement). 9.3 ms would mean 29.3 GB/s for a pure read stream shown
  running at 57.6 GB/s by the same instrument class — improperly scoped.
- **Isolated misaligned arm (22.36 ms): ISOLATED-EAGER ONLY.** It proves
  route discrimination — aligned 4.73 ms vs guard-refused 22.36 ms on
  the identical shape — but its 12.2 GB/s is an eager-isolation number
  with launch overhead and no pipelining. It must not be quoted as the
  in-model tile rate; the class marginal bounds in-model tile at
  >= 44.3 GB/s (below).
- Dispatch counts and elapsed time are kept separate throughout:
  169 gemv dispatches/token, and the class's above-roof remainder is
  8.4 us per dispatch — dispatch-ramp scale, not kernel work.

## Byte accounting (deterministic; `model/byte_accounting.py`)

Qwen2.5-0.5B bf16, per generated token, weights streamed once:

| class | kernels | bytes/token |
|---|---|---|
| vec (this kernel) | wq, wo, gate, up, lm_head | 767,721,472 |
| tile | wk, wv (bias), down | 220,200,960 |
| **total** | 169 dispatches | **987,922,432** |

Against the 18.31 ms class marginal (wall-anchored ablation,
attribution receipt): 54.0 GB/s effective = 92.2% of the 58.5 GB/s
same-instrument copy roof, 79.1% of the 68.25 GB/s part. The residual
above-roof bound — the maximum time ANY byte-preserving kernel change
could recover in the whole class — is **1.42 ms/token (7.8%)**, and
1.42 ms / 169 dispatches = 8.4 us/dispatch, i.e. per-dispatch ramp, not
reachable from kernel source.

This corrects the attribution receipt's byte figures: its "~370 MB
class" and "down + k + v are ~9.2 MB" dropped the x24 layer count (the
9.2 MB value is per-layer; per-token it is 220.2 MB). The corrected
total does not change that receipt's conclusion — it strengthens it.

## Window (jwm1-linux, Apple M1 G13G B1)

Driver mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1; wheel
`mlx_omarchy-0.32.2.dev202609121038+a2e38c3` (sha256 `99664e11...b84fbe2`,
the parity-status-qualified wheel; source commit `49325af7` recorded in
`window/drivers.txt`). One top-level flock `/tmp/m1-gpu.lock` (wrapper
pid 3124404, 53 s inside the lock), quiet gate passed before measuring,
host CLOCK_MONOTONIC whole-submit brackets only, 5 warmed rounds
discarded up front, medians of 24 rounds (8 for the misaligned arm).

| arm | median | rate |
|---|---|---|
| lm_head n=151936 k=896 (vec engaged) | 4.730 ms | **57.57 GB/s read** |
| same shape, lhs_offset misaligned (guard refuses, tile route) | 22.357 ms | 12.18 GB/s (isolated-eager only) |
| lm_head half, n=75968 | 2.546 ms | 53.47 GB/s read |
| 272 MB contiguous bf16 copy | 10.452 ms | 52.10 GB/s (R+W aggregate) |

Engagement discrimination is 4.7x on the identical shape. Rate
flatness at half size (93%) marks the bandwidth-bound regime. The A/A
interleaved series differ by 0.24% — the in-window noise bar.

Controls (canonical engine `bench_decode.py`, fixed prompt "Hi",
30/32 tokens, temp 0, seed 0, warmup 4 discarded, release-equivalent
installed fork wheel): three runs — `bench-warmup`, `bench-a`,
`bench-b` — every run reproduces the canonical fork BF16 short pin
`f26175202f3dabe9` (31.88 / 31.88 / 31.95 tok/s), so the composed
parity-status numbers reproduce inside this window. Raw artifacts in
`window/` (micro JSON+ndjson, bench logs, loadavg trace, drivers.txt,
wrapper log). No driver change, no experimental ICD, no wheel rebuild.

## Why nothing lands

The mechanism is eliminated (not bit-preserving; no headroom). The
class-level accounting caps ALL byte-preserving kernel-side recovery at
1.42 ms/token of dispatch ramp. The remaining BF16 decode gap versus
native is structure — ~13 ms/token of dispatch skeleton (726
dispatches at ~16 us versus native's ~2-3 us) — which lives in the
graph/encoder/driver layers, not in `matmul_vec.comp`; kernel-side
attacks on it are already published negatives (span widening,
de-divergence, chunked softmax) or landed (composition-exact fused
attention, gated >= 256 keys).

No digest legs were run against modified kernels because no kernel
change ships; per the policy the six-leg fork+stock matrix is a
landing requirement, and this receipt proposes no landing.

## Provenance

- Local (x86 dev box): `glslc` SPIR-V compile + census, `spirv-dis`
  listing (`spirv/`, spv sha256
  `b6cc4e35eb01a059b7a30f80734aa917bd1a6de0cb9971c22962dd548c515ad1`),
  bit model + counterexample (`model/`), byte accounting
  (`model/byte_accounting.out`).
- Hardware window: jwm1-linux, wrapper staged to `~/bf16vm-window/scratch`
  via ssh, detached launch with pid/log receipts; artifacts copied to
  `window/`. Bounded foreground ssh calls only; no repeated status
  chatter outside one hub announcement.

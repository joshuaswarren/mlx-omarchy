# BF16 decode barrier-pair collapse at the command-encoder boundary —
# exclusion receipt (2026-09-12)

Status: COMPLETE. Receipt-only negative. The named mechanism — deferring
the default path's redundant post-dispatch barrier until a non-compute
consumer needs it — is spec-safe, was implemented behind
`MLX_OMARCHY_DEFERRED_POST_BARRIERS` (default off, commit on
`wave/Bf16EncoderBoundary`), provably ENGAGED on hardware in the real
decode loop, held every digest on both drivers, and produced no
repeating end-to-end gain on any canonical leg. It stays an opt-in
diagnostic; nothing lands into production defaults. The measurement
eliminates "barrier-command cost" as a meaningful term of the BF16
decode skeleton and bounds it.

## Question

Composed BF16 decode sits at 0.561/0.545/0.456 of native
(`receipts/2026-09-12-parity-status`). The attribution receipt
(`receipts/2026-09-11-bf16-decode-attribution`) closed the payload
side (gemv dominates; residuals close) and named the skeleton —
11.86/14.69/21.58 ms/token over 726 dispatches/token, ~16 us/dispatch
versus native's 2-3 us — as the remaining cost, living in the
graph/encoder/driver layers. The 2026-09-08 gated-barrier screen said
"barriers don't matter", but it measured the pre-dense-GEMV regime
(BF16 decode was ~116 ms/token then, kernel-payload-bound; the dense
vec kernel brought it to ~31 ms), so its negative never covered
today's 726-dispatch regime. This slice traced the actual
command-encoding path and tested the strongest surviving encoder-bound
mechanism with full engagement proof.

## The mechanism, and why it is semantically safe

Default mode records TWO full barriers per dispatch:

- pre: src HOST|TRANSFER|COMPUTE (HOST_WRITE|TRANSFER_WRITE|SHADER_WRITE)
  -> dst COMPUTE (SHADER_READ|SHADER_WRITE)
- post: src COMPUTE (SHADER_WRITE)
  -> dst COMPUTE|TRANSFER|HOST (SHADER_READ|TRANSFER_READ|HOST_READ)

For a dispatch pair A,B the post barrier of A is redundant: pre(B)'s
EXECUTION dependency (first scope HOST/TRANSFER/COMPUTE, second scope
COMPUTE) already orders every earlier compute-stage access — reads
included, which covers WAR — before B, and its access masks make A's
shader writes available and visible to B's reads and writes (RAW, WAW).
The post barrier is only load-bearing for non-compute consumers that
record no barrier of their own: `copy_buffer`, `fill_buffer`, host
readback at batch close, and diagnostic dependency barriers. The
implementation defers the post barrier and flushes it verbatim (same
masks, same stream position) at exactly those four points
(`encoder.cpp` `flush_pending_post_barrier`: copy_buffer, fill_buffer,
`record_dependency_barrier`, submit-before-EndCommandBuffer). Interior
dispatch pairs record one barrier instead of two. Gated-barrier mode
and tape-full diagnostics are untouched; deferred counts surface in a
new `post_barriers_deferred` trace counter.

## Engagement (measured, not assumed)

ctypes probes over the new `mlx_omarchy_trace_barriers` C ABI (added
additively; the published 8-field `MlxOmarchyTraceSnapshot` is
unchanged — an initial expansion that would have broken
`scripts/fragmentation_probe.py` was caught in review and reverted):

- Isolated eager chain, 200 dispatches (probe.py, window 3):
  gate off = 400 emitted / 0 deferred; gate on = 300 emitted / 200
  deferred. The +100 are the per-submission close flushes: exact.
- REAL BF16 decode loop, 12 greedy tokens via `stream_generate`
  (decode_probe.py, window 4, gate on):
  **790.5 compute dispatches/token across 4.333 submissions/token**
  (average batch depth ~182 — the op-throttled batch is deep, not
  per-op), **790.5 post barriers deferred/token, 846.75 emitted/token**.
  Arithmetic closes exactly: emitted = 790.5 pre + 56.25 flushes
  (52.0 copies/token + 4.33 batch closes). Against the base path's
  2-per-dispatch (~1581 barriers/token) the engaged mechanism removes
  ~46% of barrier commands in the real workload. The base wheel
  rejects the barrier symbol (AttributeError, window4/decode-base.txt)
  — expected: it predates the ABI addition, and its per-token
  dispatch/submission counters are identical to the candidate's
  (790.5/4.333 both arms).

## Paired measurement vs the canonical unchanged baseline

Three jwm1-linux windows (Apple M1 G13G B1, driver
mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 verified in-window,
ONE top-level `/tmp/m1-gpu.lock` per window covering builds and runs,
quiet gate = 1-min loadavg < 1.0 on three checks 20 s apart, discarded
warmup per arm, wall-clock bench only):

- **Window 1 (invalid, kept for build provenance):** fork matrix legs
  died on an empty env-array expansion (`timeout` received
  `HF_HUB_OFFLINE=1` as the command, exit 127); probes crashed on the
  mlx namespace package; the checker accepted a vacuous leg set. Its
  `ALL_DIGEST_ASSERTIONS_PASSED` is VOID. Builds (rc=0) and wheel
  hashes are the only evidence taken from it.
- **Window 2 (valid but measures gate-off vs gate-off):** candidate
  arm never set the gate env; probe proved non-engagement
  (0 deferred / 400 emitted). A launcher error re-ran this window
  once (16:47-16:55Z); artifacts are the rerun's, same protocol.
  Value: build reproducibility + 12x6 digest cells on the candidate
  wheel with the gate compiled in but off.
- **Window 3 (the A/B):** candidate arm runs with the gate env set,
  engagement proven by probes. Base = pristine main `a12eafd9`
  wheel sha256 `9e601cebadd578657e678a173dd2efc993679ea3469c563f63d2b6c5725aef94`;
  candidate = `f5649936` wheel sha256
  `3f90904520d5441b2b1133aada10160b4f9cfb58bca79a378e7420c62e042df0`.
  Interleaved counterbalanced b-c-c-b-b-c, medians of 3, 12 runs x 6
  legs, every measured cell asserted against the committed fork pins
  (`7fd25a869ff21678`, `4cc08910089477fd`, `7da83f06ec9f001d`,
  `f26175202f3dabe9`, `8690dc83246b39f8`, `ff502900d2a179a5`) and the
  stock pins on the stock-smoke cells (`7fc0f968789b1882`,
  `46108ad71157cb4d`) — **72/72 held, provenance verified=match on
  every cell, digest-neutral on both drivers with the gate ON.**
- **Window 4:** the decode-loop structure probe above.

Paired decode tok/s, window 3 (higher is better):

| leg | base | cand (gate on) | ratio |
|---|---|---|---|
| Q4 short | 111.59 | 111.69 | 1.0009 |
| Q4 long | 107.79 | 107.29 | 0.9954 |
| Q4 1K ctx | 96.27 | 96.08 | 0.9980 |
| BF16 short | 31.86 | 31.86 | 1.0000 |
| BF16 long | 30.48 | 30.45 | 0.9990 |
| BF16 1K ctx | 24.98 | 24.95 | 0.9988 |

A/A baseline pair spread in the same window: BF16 0.22% / 0.79% /
0.04%; Q4 4.43% (one transient outlier) / 0.09% / 0.64%. Every
candidate delta is inside the in-window noise band; the session noise
floor is ~5%. **No repeating gain on any leg** — the acceptance bar
for a default flip is not met, so per the assignment nothing lands
beyond the default-off diagnostic.

## What this eliminates, exactly

1. **Barrier-command cost is not the BF16 decode skeleton.** With
   engagement proven at ~46% fewer barrier commands per token
   (1581 -> 847) and paired timing flat to <1%, the total wall-time
   value of the removed commands is bounded by the noise band:
   roughly <= 0.3 ms/token (~0.4 us per barrier command) on a
   ~31.5 ms leg. The 2026-09-08 gated-barrier screen's negative now
   holds in the modern regime too, with direct engagement
   instrumentation instead of an off/on screen.
2. **The "redundant barrier pair" as an optimization target.** The
   remaining per-dispatch structure is one pre barrier + dispatch
   launch + execution on the GPU; halving the barrier count around it
   moves nothing, so the ~16 us/dispatch skeleton lives in
   per-dispatch GPU work (launch and dependency-ordered execution at
   726 dispatches/token) and per-submission structure (4.3
   submissions/token), not in barrier commands.
3. **Encoder-boundary routes to the skeleton, with receipts.** What
   remains is owned elsewhere: dispatch COUNT is graph/kernel fusion
   territory (the composition-exact fused attention at >=256 keys is
   landed; further kernel fusion has published negatives or moved
   digests), and submission cadence is the published batching
   negative (2026-09-03: deep batching 4.5x slower; the correct
   "flush when the scheduler would block" shape is upstream work).

## Why the code stays (default off)

The gate ships as an inert-by-default diagnostic in
`wave/Bf16EncoderBoundary`: ~130 lines in encoder.{h,cpp} + additive
trace counters + docs. It is spec-proved, contract-tested
(omarchy_runtime_tests + omarchy_fast_ops_tests green in both gate
states on llvmpipe, 22,691 + 1,116,299 assertions each; Main
independently repeated the gate-on suites on the candidate),
digest-neutral proven on hardware on both drivers, and its counters
make barrier accounting observable. A future submission-structure or
fusion change that re-weights barrier cost can re-qualify it from
this receipt's baseline. Production defaults are root's decision;
this receipt proposes no landing.

## Provenance

- Branch `wave/Bf16EncoderBoundary`, worktree
  `~/.config/superpowers/worktrees/mlx-omarchy/Bf16EncoderBoundary`,
  head `f5649936f68a2262e6f1e51ad9b146ec035808cb` (single commit:
  mechanism + counters + docs; the ABI-expansion fixup was amended in
  before any wheel was measured).
- Artifacts: `window3/` (21 files: wrapper log, drivers.txt with both
  wheel sha256s, summary.json, 12 run JSONs + logs, probes, loadavg
  trace), `window4/` (decode probes + wrapper log), `instruments/`
  (window scripts, probes) — all copied verbatim from
  `jwm1:~/enc-ab-window{3,4}`.
- Hosts: dev box (x86, llvmpipe `MLX_OMARCHY_ALLOW_NON_APPLE=1`) for
  compile + contract suites; jwm1-linux for every GPU number. No
  driver or ICD change, no kernel change, no CPU fallback, no
  weakened gate; stock reached only through the private stock ICD
  override, same as prior receipts.

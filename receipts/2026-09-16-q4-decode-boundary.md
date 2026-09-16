# Q4 decode gap: boundary statement (2026-09-16)

Umbrella record for "Close Q4 decode gap vs native Metal". Every lever
inside mlx-omarchy shader/runtime scope is now either landed or measured
and rejected with a named mechanism. The residual gap lives in the
Mesa/Honeykrisp driver and is out of this repository's scope.

## Landed since v0.4.x (all digest-pinned, both hosts)

| change | dispatches/token | receipt |
| --- | ---: | --- |
| rope-pair dispatch + linear chain-settle | 249 → 225 | 2026-09-15-rope-pair-land.md |
| SwiGLU GEMV store epilogue | 225 → 201 | 2026-09-15-swiglu-epilogue.md, -jw16.md |
| QmmPrefillCoopmatF16 + tile-M floor, prefill | — | 2026-09-14-qmm-occupancy-tilem.md |
| SPIR-V disk cache (mel compile) | cold r1 only | release notes v0.5.0 |
| GEMV lane split (earlier) | 249 baseline built on it | 2026-09-09-q4-gemv-bandwidth |

Current medians at 201 dispatches (rmsnorm-qkv-ab base arms, 12-round):
jwm1 short 116.34 / ctx1024 102.35 tok/s (77.3% / 72.9% of native
150.57 / 140.38); jw16 short 190.72 / ctx1024 135.54 tok/s (66.4% /
47.8% of native 286.96 / 283.79). Short prefill is 113% of native on
jwm1.

## Measured and rejected (named mechanisms)

- RoPE into qmm_vec (201 → 177): Honeykrisp NIR lowers the exp/cos/sin
  chain differently per compilation unit; deterministic ctx digest
  `31267e7ed4c6d0dc` on both hosts. 2026-09-14-decode-epilogue-fold.md,
  2026-09-15-qmm-vec-lowering.md. **Blocker: Mesa AGX transcendental
  scheduling/approximation invariance across compilation units.**
- RMSNorm GEMV prologue (201 → 154): ctx +3.8% but short −7.3%;
  D-const ceiling proves no bit-exact fold beats −1.8% short.
  Attention-scoped variant fails the −1% short floor on both hosts and
  inverts on jwm1. 2026-09-15-rmsnorm-gemv-fix.md, -rmsnorm-qkv-ab.md.
- Gated barriers: flat. 2026-09-14-gated-barriers-ab.md. Compile ON:
  identical GPU command stream, dispatch count unchanged.
  2026-09-14-decode-compile-ab.md.
- SDPA KV load-granularity (three formulations): flat to negative.
  2026-09-16-q4-sdpa-kv.md.
- KV-head-major geometry (each KV byte read once, bit-exact by
  construction, pins 20/20): jw16 −9.6% short / −21.3% ctx — the dedup
  hypothesis is falsified, L2 already dedups the 7x GQA repeat traffic;
  binding constraint is per-subgroup serial key-walk latency.
  2026-09-16-q4-sdpa-kv-major.md.
- 2x4 GEMV row split on native-order kernel: 1–2% slower.
  2026-09-09-q4-gemv-bandwidth/verdict-rebased.json.

## The remaining deficit, attributed

Per 2026-09-14-decode-gap-attribution.md at 249 dispatches (structure
unchanged at 201): term A (context-independent) ≈ 70–79% is a
per-dispatch serial cost of ~10 µs inside the Vulkan submit path —
device-invariant across a 3.3x GPU difference; term B ≈ 21–30% is the
KV stream service rate (~11 GB/s effective on both hosts vs native 26 /
323 GB/s). Both are driver-side. No source-level change in this
repository can reduce them further without breaking the pinned
generated-ID digests.

## Next attack, when taken

Mesa/Honeykrisp AGX: (1) transcendental lowering invariance across
compilation units — unlocks the RoPE fold, priced +2.6% jwm1 ctx;
(2) the per-dispatch submit-path cost; (3) L2 sector policy for
16-bit SSBO repeat traffic. All three are outside the four project
repositories and are not scheduled here.

## Identity

Host-only record; every number quoted from the cited receipts, no new
measurement. `63c1d3cf` not merged.

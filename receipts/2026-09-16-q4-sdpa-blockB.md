# Q4 SDPA block-B: B=8 register-blocked decode twin is bit-exact (18856/18856) but the kernel is a regression — lever measured dead, no battery, ships off

Date: 2026-09-17 (windows 2026-09-17T03:36–03:38 UTC, jw16mbp1-linux).
Task: the named next lever from
`receipts/2026-09-16-termB-kv-mechanism.md` — multi-key register
blocking (B=8) of the f16 decode arm's serial key walk, "load B keys'
K rows into registers per iteration and perform one ordered
online-softmax merge per group, so the serial chain cost amortizes over
B keys".

## Verdict

**Parity proven, performance rejected at the screen gate. Nothing
lands; the branch is preserved and the env gate ships off.**

1. **Bit-exactness: PROVEN on the M1.** `omarchy_sdpa_decode_fused_tests`
   (now 4 cases): **4/4 cases, 18856/18856 assertions, 0 failed** on
   jw16 against the real Honeykrisp compiler. The new block-B case
   engages at exactly 1 dispatch/call and stores bit-identical words to
   the per-head `SdpaDecodeNativeF16` arm at keys
   {1, 5, 30, 64, 263, 320, 1024, 1053} — spanning one-pass partial
   groups (1, 5, 30, 263, 320), the 64-block regime at 1024 (two
   exactly-full 8-key groups per block walk), and the 128-block regime
   at 1053 (8+1 tail groups). bf16 inputs with the env on still ride
   the composition-exact bf16 arm (1 dispatch, bit-identical words).
   Bit-exactness is by construction, not by luck: the block walk keeps
   each lane's serial key order and every merge executes the identical
   per-key f32 chain (max → 2× native_exp → fma-sum → fma-o0 → fma-o1)
   on the same operands; only the dependency structure around the loads
   differs. Out-of-range tail slots load the last key's row and are
   predicated off before reaching any state.
2. **Performance: REJECTED at the screen gate (stop rule: <20% kernel
   win ⇒ stop, do not battery).** Batched-eval screens (50 calls per
   command buffer, 8 interleaved base/blockb cycles per k, arm exposure
   balanced across drift phases):

   | k | base med (µs) | blockb med (µs) | win | per-arm range |
   | ---: | ---: | ---: | ---: | --- |
   | 30 | 37.65 | 49.66 | **−31.9%** | 30.6% / 22.6% |
   | 1024 | 92.35 | 95.52 | **−3.4%** | 11.6% / 15.2% |
   | 1053 | 91.93 | 105.68 | **−15.0%** | 15.1% / 21.9% |

   A first single-contrast window (A/B/A with drift echo) agreed in
   direction on k=30 (−44.5%) and k=1024 (−1.0%) and showed k=1053
   inside the noise (+5.3%, echo −0.6%); the 8-cycle window is the
   decision evidence.
3. **The marginal dropped but the intercept ate it.** One-pass
   k=30→1024 marginal: base 55.0 ns/key (matches the 52–57 ns/key
   baseline) vs blockb 46.1 ns/key — a real ~16% per-key amortization.
   But blockb's intercept is +12.0 µs (49.66 vs 37.65 at k=30), so net
   kernel time is flat at 1024 and worse at 1053. The chain got
   shorter per key and something else got more expensive.
4. **Mechanism reading [INFERENCE, not measured]:** the two documented
   suspects survive. (a) The backend load-sinking that killed
   source-level prefetch likely sinks the 32 block loads into the
   consumer chain anyway, so the intended dependency restructure never
   reaches scheduling; (b) the unrolled body's register footprint
   (~40+ live values per thread across 1024-thread workgroups) plausibly
   cuts resident workgroups — the same in-flight-chain collapse that
   killed the kv-major geometry — which is exactly what the +12 µs
   intercept and the short-leg regression look like. No G13C hardware
   counters exist to separate these (termB receipt); the B-ladder
   (B=2/4) that could price occupancy vs sinking was NOT run — the stop
   rule fired first, and a variant sweep is a new lever, not this one.
5. **Land rule: unmet (no battery warranted).** The env gate
   (`MLX_OMARCHY_SDPA_BLOCK_B`) ships default-off, so the installed
   default route is unchanged: the default shader specialization
   compiles **byte-identical SPIR-V** to main's `sdpa_decode_native_f16`
   (verified by `cmp` of glslc output), and with the env unset the host
   route never reads past the getenv null-check.

## What was built

- Branch `agent/sdpa-blockb` @ `dfdb0efc` on
  `github.com/joshuaswarren/mlx-omarchy`, one commit on main
  (`403142d9`):
  - `shaders/sdpa_decode_native.comp`: `#ifdef SDPA_BLOCK_B` macros —
    `B8_LOAD` (8 slots' K/V rows, clamped indices), `B8_DOT` (8
    independent fma/subgroupAdd dots), `B8_MERGE` (8 predicated ordered
    online-softmax merges) — replacing both key loops under the define;
    default path untouched (byte-identical SPIR-V).
  - `SdpaDecodeNativeF16B8` kernel enum entry (append-only profile id),
    blob case, `-DSDPA_BLOCK_B=8` shader target,
    env-gated dispatch in primitives.cpp (refuse-and-fallback: any
    non-f16 or refused shape keeps the existing routes).
  - New test case "block-B f16 decode is bit-identical to the
    per-head route" (dispatch + bit-identity at the 8 key counts, bf16
    refusal arm).
- jw16 wheel `mlx_omarchy-0.32.2.dev202609170332+dfdb0efc-cp314-cp314-linux_aarch64.whl`
  sha256 `0ef280a8ee90daff3a654d9b0c22ff8ea70122283cef33d74c4f83febc13332b`
  (stamp +dfdb0efc = HEAD, asserted by build.sh); parity/test binary
  sha256 `8bdb96506ec9eb73cba622fcfad9124855be7a11f5f46b554d7ce743a49cdaba`.

## Protocol

- Host: jw16mbp1-linux only; GPU window granted by Main (PrefillDriver
  confirmed free); `llm-inference.service` stopped before and restarted
  after each window by `orchestrate.sh` (`active` confirmed after every
  window; three windows: 2 s, 40 s, 2 s of locked work).
- GPU lock: `/tmp/m1-gpu.lock` inode **12** before and after every
  window; nested `flock -n` refused under hold, printed in every window
  log.
- Screens: `screen_b8.py` — batched-eval GPU timing as TermBTrace's
  `screen_sweep.py` (chain 50, 8 reps, median), route engagement
  asserted per arm-row via `mlx_omarchy_trace_snapshot`
  vk_compute_dispatches (exactly 1 per call, both arms); arms interleaved
  8 cycles per k so drift cannot fake a gap. Window drift observed at
  ±9–18% per row range — the reason the first single-contrast screen
  was discarded as unreadable.
- Parity: `omarchy_sdpa_decode_fused_tests` from the cand tree's
  prepared source on jw16 — 4 cases, 18856/18856, 0 failed, twice
  (both windows).
- Battery: **not run** — screen gate failed; per the term-B stop rule
  the lever stops here.
- No driver installed; no reboot; jwm1 never touched; `63c1d3cf`
  untouched.

## Artifacts

- This directory: `q4-sdpa-blockB-artifacts/` — `screen-b8.{json,txt}`
  (8-cycle window), `sdpa_fused_tests-final.txt` (parity transcript),
  `started/finished/host/lock` stamps.
- jw16 `/var/tmp/SdpaBlockB/`: `cand-tree` (branch at dfdb0efc),
  `build.sh` (wheel + test builds, stamp receipts),
  `screen_b8.py`, `ab_decode_b8.py` (battery script, staged and
  syntax-checked, never run), `run-window1.sh`, `orchestrate.sh`,
  `jw16-out-w1/` (all three windows' logs), `build-wheel.log`,
  `build-testbin.log`, `testbuild-{configure,ninja}.log`.

## Identity

- Assignment: SdpaBlockB lane; base tree: mlx-omarchy main `403142d9`
  (word-view f16 arm as shipped), candidate `agent/sdpa-blockb`
  `dfdb0efc`.
- Receipts this lane depends on:
  `receipts/2026-09-16-termB-kv-mechanism.md` (lever definition,
  52–57 ns/key baseline, stop rule),
  `mlx-omarchy-kvmajor/receipts/2026-09-16-q4-sdpa-kv-major.md`
  (two-pass format provenance, occupancy-collapse precedent).
- Agent model: `zai/glm-5.3-flash`.

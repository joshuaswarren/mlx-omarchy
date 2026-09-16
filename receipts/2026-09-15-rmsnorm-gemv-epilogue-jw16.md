# 2026-09-15 GEMV-consumed RMSNorm prologue fold, retested on the SwiGLU line — pins 20/20, 201→154 dispatches, ctx1053 +3.8% median; LAND

Task: retest the 2026-09-14 GEMV-RMSNorm prologue fold
(`agent/gemv-rmsnorm-prologue`, 6110078e — real, firing, 249→202, but
ctx1024-pin-breaking) on top of `a21b3c81` with the statement-for-
statement ROUND_STORAGE discipline the SwiGLU store epilogue proved.
Branch: `agent/rmsnorm-gemv-epilogue` (fold commit + receipts; tip
`da523ea7`). No RoPE in qmm_vec; `63c1d3cf` not merged.

## Verdict

**All three land gates pass. LAND as `agent/rmsnorm-gemv-epilogue`.**

1. **Pins hold** — the Honeykrisp lowering hole that killed the
   2026-09-14 attempt did not reappear on this line: short
   `7fd25a869ff21678` and ctx1024 `7da83f06ec9f001d` in every arm and
   every round of the 5-round interleaved A/B (20/20 leg digests), plus
   the dedicated ctx1024 digest arm on cand, rms, and rms-with-fold-off
   (first ids `13060,498,369`, identical on all three arms). The C++
   fold test (below) proves the same at the kernel-contract level: the
   normalized prologue dot is bit-exact against per-node
   fast_norm + GEMV at k=896 on the real Honeykrisp compiler.
2. **Dispatches drop below 201** — **154** `vk_compute_dispatches` per
   decode token (−47; the planned 48 minus one pair that does not fire
   at runtime, same secondary hole the 2026-09-14 run named and did not
   diagnose). `gpu_primitive_dispatches` stays 858. Attribution:
   `MLX_OMARCHY_FUSED_GEMV_RMSNORM=0` on the same wheel restores 201;
   `MLX_OMARCHY_FUSED_GEMV=0` gives the 465 counter floor.
3. **ctx1053 median rises** (same-battery interleaved A/B against the
   a21b3c81 wheel):

| leg | cand (a21b3c81) | rms fold | Δ |
|---|---|---|---|
| ctx1024 (1053/32) | 136.89 tok/s | **142.06 tok/s** | **+3.8%** |
| short (30/32) | 190.66 tok/s | 176.77 tok/s | −7.3% |

Per-run ctx1024 cand: 141.04, 125.72, 147.86, 135.12, 136.89; rms:
134.73, 142.06, 144.92, 134.66, 143.52. The win is on the long-context
leg, where the 47 deleted dispatches live. **Named cost:** the short
leg gives back −7.3% — every workgroup of the consumer GEMV now
re-derives the RMSNorm reduction redundantly, and at a 30-token prompt
that overhead is not amortized by the deleted dispatch. The land rule
gates on ctx1053, which rises; the short regression is a real
trade-off, not noise (all five rms short runs 174.65–177.34 sit below
all five cand runs 190.21–190.79).

## What changed vs the 2026-09-14 attempt

Same fold contract (consumer group's shared x is an unclaimed f16
RMSNorm whose input Add is another planned group's epilogue; RMSNorm
buffer aliases the epilogue sum; every consumer workgroup reproduces
the fast_norm reduction — per-thread strided squares, stride-128..1
shared tree; dot x elements take the storage-rounded normalized value).
New on this line:

- The gate/up group now dispatches as the SwiGLU store epilogue
  (flags bit 16), so the prologue (bit 15) lives in that branch too:
  one dispatch can carry prologue + paired store, and the LN1→gate/up
  pair folds (the 2026-09-14 attempt predated the swiglu epilogue).
- The RMSNorm planner matcher runs AFTER the SwiGLU matcher so a
  gate/up group can carry both folds.
- Statement-for-statement ROUND_STORAGE discipline from the SwiGLU
  land: every x element entering the dot is
  `float16_t(ROUND_STORAGE(v * rms_scale * w))` — the exact f16 store
  fast_norm.comp writes and the standalone dot reads back; reduction,
  mean, and `inversesqrt` stay unqualified f32 exactly as
  fast_norm.comp writes them.

Tests: `test_fused_chain.cpp` gains the fold case — per-node 9 →
folded 4 dispatches, 5 with the gate off, projection outputs bit-exact
through the normalized prologue, and the aliased rms buffer asserted by
pointer (a fired fold's RMSNorm array aliases the epilogue sum by
design; its only readers normalize in register). C++ battery on jw16:
fused_chain 34/34, kv_ops producer-direct 1/1.

## Identity

- Branch `agent/rmsnorm-gemv-epilogue` off origin/main `a21b3c81`:
  fold `9bc62945`, confirmation receipt `260ec1c3` (this file's
  sibling), test-fix + this receipt `da523ea7` (tip).
- rms wheel `mlx_omarchy-0.32.2.dev202609152149+260ec1c3` sha256
  `45ce7f916f29b7a9bb076b6845ef687546c8b840f140315e4f08a7fb8c399503`;
  stamp `+260ec1c3` predates the tip by the test-source-only commit —
  overlay code is identical to `da523ea7`. Base arm: the task-1 cand
  wheel `+a21b3c81` sha256 `1540b11d…`, provenance `verified=match` on
  every arm and leg.
- Host: jw16mbp1-linux, Python 3.14, mlx-lm 0.31.3,
  `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`, model snapshot
  `a5339a41…`, one `flock` hold per run set on `/tmp/m1-gpu.lock`
  (inode 12 before and after; nested `flock -n` refused; never
  unlinked). Build logs: `/var/tmp/SwigluRmsJw16/{build,run-rms-ab}.log`
  — the first test-binary build caught a planning-invariant repair
  (x_row declarations in the swiglu branch) before any measurement.

## Land notes for the lander

- `260ec1c3` (SwiGLU jw16 confirmation) is independent of this fold and
  lands cleanly alone if the fold is refused.
- Follow-up candidates, out of scope here: the one non-firing fold
  pair (47 vs 48), and the short-leg regression (per-workgroup
  redundant reduction) if a cross-workgroup norm sharing scheme ever
  lands.

## Artifacts

- `receipts/2026-09-15-rmsnorm-gemv-epilogue-jw16/`: `dispatch-rms.json`,
  `dispatch-rms-off.json`, `dispatch-rms-gemvoff.json`, `digest.txt`,
  `ab.txt`, `ab.json`, `fused_chain_tests.txt`, `kv_ops_direct.txt`,
  `lock.txt`, `host.txt`, `started.txt`, `finished.txt`,
  `wheel-sha256.txt`.
- jw16: `/var/tmp/SwigluRmsJw16/` (rms worktree at the branch tip,
  wheels, venvs, `build-tests-rms` test binaries, logs).

# 2026-09-14 RMSNorm Add+Norm matcher diagnosis on jw16 — HOLE NAMED, NO-LAND

## Verdict

**The matcher cannot fire in any landable configuration; landing rule fails; nothing
landed.** Measured, not inferred:

1. **Default gates (the landable config):** the decode-tape planner sees 49 f16
   single-row RMSNorm nodes, rejects 48 because their input Add is already claimed
   as a GEMV epilogue (`add_claimed=48`), rejects 1 because its input Add is not a
   tape node, and plans **0 pairs**. Dispatches stay **249**.
2. **`MLX_OMARCHY_FUSED_GEMV=0`:** the pair scan is structurally unreachable — the
   planner early-returns at `if (!fused_gemv_enabled()) return;`
   (fused_chain.cpp ~line 1012) **before** the RMSNorm scan (~line 1158), so
   `MLX_OMARCHY_FUSED_RMSNORM` on or off is identical. This is the exact mechanism
   behind the 465/465 probe result in receipts/2026-09-14-rmsnorm-epilogue-ab.md.
   The fold was dead code in every configuration ever tested.
3. **Stealing the GEMV-claimed Adds is forbidden and arithmetically worthless:** an
   epilogue Add costs zero dispatches inside the fused GEMV dispatch; a pair that
   steals it re-adds the same sum as its own dispatch. Boundary cost is 2 dispatches
   before and after. 249 → 249.
4. Making the fold reachable (probe-only hoist of the scan above the early return)
   fires 48 pairs and drops 465 → **417** (−48), with pins held — but that config
   is +168 dispatches over default and **slower** (126.33 vs 131.50 tok/s ctx1053
   same-session; base 143.66). GEMV fusion is worth far more than 48 pair savings.

## What was run

- Host: `jw16mbp1-linux`; one `flock` hold per run set on `/tmp/m1-gpu.lock`
  (inode 12 before and after; never unlinked; nested `flock -n` refused each time).
- Model `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, `MLX_DISABLE_COMPILE=1`,
  `HF_HUB_OFFLINE=1`, `--tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4`
  (ctx1024 = 1053 prompt tokens), `--tokens 8` for dispatch counts.
- Probe wheel: `rmsnorm-epilogue` a3e9f486 + diagnostic-only patch
  (`rmsnorm-fire-probe.patch`, sha256 `3ae08d245ae69c6454577e38f6a9149425e700afede5ab1d70d138da1c9f1498`):
  - hoists the pair scan into `rmsnorm_pair_scan()` with per-reason plan counters
    and a decode-shaped-plan trace (`MLX_OMARCHY_RMSNORM_TRACE=1`),
  - adds `MLX_OMARCHY_RMSNORM_PROBE=1`, which lets the scan run while the GEMV gate
    is off (a config that cannot exist otherwise),
  - adds an un-plan trace; default-gate behavior is untouched (249 verified).
  Wheel sha256 `95f8ab8fadb9e814385dbaecc777b5b22bd6dea336cc6628040bb21ddfc0335f`,
  installed in `venv-probe` only.

## Per-token `vk_compute_dispatches` (medians, quoted)

| config | dispatches |
|---|---|
| base wheel `b79a4b68`, default gates (prior receipt) | 249 |
| a3e9f486 wheel, default gates (prior receipt) | 249 |
| probe wheel, default gates (legA) | **249** |
| a3e9f486, `FUSED_GEMV=0`, RMSNORM on or off (prior receipt) | 465 |
| probe wheel, `FUSED_GEMV=0` (legB, scan hoisted) | **417** |

Plan traces (probe wheel, decode-shaped plans, 3 plans each):

```text
legA default gates: rms_f16=49 rms_single_row=49 no_add_in_tape=1 add_claimed=48 cond_fail=0 pairs_planned=0
legB GEMV=0+PROBE:  rms_f16=49 rms_single_row=49 no_add_in_tape=1 add_claimed=0  cond_fail=0 pairs_planned=48
```

No `[rmsnorm-run]` un-plan lines appeared: all 48 planned pairs dispatched; the
runtime contract never refused.

## ctx1053 tok/s and pins (same session, one flock hold)

| leg | config | decode tok/s | pin |
|---|---|---|---|
| C | base wheel, default gates | 143.6643 | `7da83f06ec9f001d` |
| A | probe wheel, default gates (249) | 131.4966 | `7da83f06ec9f001d` |
| B | probe wheel, fold reachable (417) | 126.3349 | `7da83f06ec9f001d` |
| R | venv-rms restored-pristine wheel | 143.0273 | `7da83f06ec9f001d` |

All four legs `prompt_tokens 1053`. The fold-firing legB is digest-preserving
end-to-end — the kernel is sound; the planner reachability is the failure.

## Synthetic-probe caveat

`rms-probe.py` (f16 add→fast.rms_norm tape vs f32 control) is **insensitive to the
fold**: base wheel `b79a4b68` (no fold code) and the fold wheel both print
pattern 3 / control 5, with and without `FUSED_GEMV=0`. The 3-vs-5 delta is an
f16/f32 composition artifact, not the pair. It must not be cited as evidence of
pair firing; the real-tape plan trace and dispatch deltas above are the evidence.

## Decision

Land rule (dispatch drop AND ctx1053 median rise AND pins hold) is **not met**:
- no dispatch drop exists in any landable configuration (249 in all);
- the only config where dispatches drop (417) is GEMV-off — a regression, and its
  tok/s is the worst of the session;
- pins held everywhere, but a digest-preserving kernel with no reachable win does
  not land.

**Named hole:** on Qwen2.5-0.5B-Instruct-4bit decode, every residual Add that could
pair with a following RMSNorm is already a GEMV epilogue, and GEMV owns it for free
— so an Add+RMSNorm pair matcher has nothing to eliminate. The only route to a real
dispatch drop is folding the norm prologue into the GEMV epilogue itself (sum and
normalized row written by the GEMV dispatch), which requires a cross-workgroup row
reduction in the qmm kernel — new kernel work, not a matcher fix, and out of scope
under "do not regress GEMV fusion".

No repo state changed: `mlx-omarchy` `main` untouched, `rmsnorm-epilogue` left at
a3e9f486 with a clean tree (probe reverted; worktree stash dropped), RoPE fold lane
untouched, `63c1d3cf` still not merged. `venv-rms` was accidentally contaminated
mid-run by a copied-venv pip shebang and was repaired with a pristine rebuild
(legR re-pin above proves restoration); the original `db91990f…` wheel file had
been overwritten in `dist/` by probe builds — the pristine wheel now in `dist/` and
`jw16-out-rmsnorm-fire/pristine/` (sha256
`37613a2a27fa7a2ae7cb0e773a641b747d8e955eec33b8841d5bcd3157c87721`) is a
content-equivalent rebuild, not byte-identical to the original (wheel builds embed
varying metadata; behavior re-pinned by legR).

## Artifacts

- jw16: `/var/tmp/DecodeEpilogueFold/jw16-out-rmsnorm-fire/` — `started.txt`,
  `finished.txt`, `lock.txt`, `wheel-sha256.txt`, `probe-synth-{base,base-gemvoff,rms,rms-gemvoff}.txt`,
  `dispatch-probeA.json`, `dispatch-probeB.json`, `leg{A,B}-trace.txt`,
  `leg{A,B}-plan.txt`, `ctx-{A,B,C,R}.txt`, `probe/` (probe wheel),
  `pristine/` (restored wheel).
- jw16: `/var/tmp/DecodeEpilogueFold/rmsnorm-fire-probe.patch` (the diagnostic
  patch, sha256 above), `apply-probe.py`, `run-rmsnorm-fire.sh`, `run-legR.sh`.
- Worktree: `/var/tmp/rmsnorm-epilogue` clean at a3e9f486; `dist/` holds the
  pristine wheel.

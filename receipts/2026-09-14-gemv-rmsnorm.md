# 2026-09-14 GEMV-epilogue RMSNorm prologue fold on jw16 — NO-LAND (ctx1024 pin breaks)

## Verdict

**Named no-land.** The consumer-GEMV RMSNorm prologue fold is real, fires in
decode, and drops dispatches — but the ctx1024 digest pin breaks, so the land
rule (dispatches drop AND pins hold AND ctx1053 median rises) fails at the
pins gate. Per the assignment ("if cross-workgroup reduction cannot reproduce
FastRmsNormF16's tree, name that hole and stop"), the hole is named below.

## What was built

Branch `agent/gemv-rmsnorm-prologue` (studio + jw16 clone), commit `6110078e`
on top of `rmsnorm-epilogue` tip `10c55855`. Design: when a planned GEMV
group's shared x is an unclaimed float16 RMSNorm whose input Add is another
planned group's epilogue, the consumer dispatch fires at the RMSNorm's eval,
aliases the RMSNorm buffer to the epilogue sum, and every consumer workgroup
reproduces the fast_norm reduction (per-thread strided squares,
stride-128..1 shared-memory tree, statement-for-statement fast_norm.comp)
and normalizes its own dot's x elements in register. No cross-workgroup
reduction; the RMSNorm dispatch is deleted. Gate `MLX_OMARCHY_FUSED_GEMV_RMSNORM`
(default on). Files: `shaders/qmm_vec.comp`, `compute.h`, `fused_chain.{h,cpp}`,
`primitives.cpp`. Any runtime contract failure un-plans the whole group onto
the per-node path (fold failure leaves the RMSNorm free to dispatch
ordinarily; the alias is overwritten by its own fresh buffer).

## Measured on jw16 (one flock hold per run set on /tmp/m1-gpu.lock)

Wheel: `mlx_omarchy-0.32.2.dev202609150054+6110078e-cp314-cp314-linux_aarch64.whl`,
sha256 `b6d4c99bd0385fb214b5e30d063240db7685edb4f7dfa03cb835e5034cfb1f32`
(venv `venv-gepin`). Base arm: `dist-base` wheel `b79a4b68` (sha256
`beeeaee9…`), venv `venv-base`. Model Qwen2.5-0.5B-Instruct-4bit snapshot
`a5339a41…`, `MLX_DISABLE_COMPILE=1`, receipt protocol.

Per-token `vk_compute_dispatches` (medians, `dispatch_count.py`, tokens 8):

| config | dispatches |
|---|---|
| base wheel, default gates | 249 |
| fold wheel, default gates | **202** |
| fold wheel, `FUSED_GEMV_RMSNORM=0` | 249 |
| fold wheel, `FUSED_GEMV=0` | 465 |

The fold engages (gate-off restores 249 exactly) and drops 47 dispatches,
not the planned 48 — one fold pair did not fire at runtime (not further
diagnosed; secondary to the pin failure).

A/B (`ab_decode.py`, interleaved, pins asserted, run aborted at first pin
violation by design): short leg round 0 held `7fd25a869ff21678` on both
arms; ctx1024 round 0:

```
base  143.83 tok/s  ids 7da83f06ec9f001d   (pin held)
fold  DIGEST MISMATCH 31267e7ed4c6d0dc != 7da83f06ec9f001d
```

No ctx1053 median is quoted because the run is invalid at the pins gate.

## The hole (named)

The fold's normalized row is not bit-identical to the standalone
FastRmsNormF16 output on real hidden-state values, so the digest drifts and
greedy ids eventually diverge (short 32-token prompt masked it; ctx1024
broke within 32 tokens).

Evidence chain (all in-process on jw16, scripts in
`/var/tmp/DecodeEpilogueFold/`):

1. Synthetic boundary probe (`fold_probe.py`, random values, same shapes):
   fold arm byte-identical to base and to gate-off arm — the kernel is
   *capable* of exact reproduction for those values.
2. Real-model boundary bisect (`bisect_fold.py`): step-0 boundary sums
   identical across arms; from step 1 the residual stream diverges, entering
   through fold boundaries (`exact_map.py`): LN0→qkv fold boundaries mismatch
   their standalone recomputation while LN1→gate/up fold boundaries are
   byte-identical — same kernel code, different values. The difference is
   value-dependent and sub-f16-ulp (f16 residual adds absorb it at step 0;
   it leaks through later).
3. Source-level: the prologue reduction and y application are
   statement-for-statement fast_norm.comp (same striding, same tree, same
   rounding-through-packHalf2x16, same (v*scale)*w order), yet compile
   differently in the qmm_vec context. Conclusion: Honeykrisp/NIR lowers the
   unqualified arithmetic differently when embedded in the GEMV kernel than
   in standalone fast_norm.comp — a compiler-lowering difference, not a
   source-order difference. The proven bit-exact precedent
   (`fast_norm_add.comp`, commit `4853e18e`) is a standalone near-copy of
   fast_norm.comp; the same trick does not survive embedding in qmm_vec.

Stated plainly: the RMSNorm tree cannot currently be reproduced bit-exactly
from inside the fused GEMV kernel on this stack. Fixing it means either
forcing the standalone lowering inside qmm_vec (precise/`exact` qualifiers
carry no guarantee of matching the *baseline's* unqualified lowering, and
the baseline is what the pins hash) or compiler work — out of scope here.

## No repo state changed beyond the branch

`main` untouched (`cc3e2ff0` origin; `3db3cb9a` jw16 base); work is on
`agent/gemv-rmsnorm-prologue` (`6110078e`), pushed to the jw16 clone
(`jw16mbp1-linux:src/mlx-omarchy`) and to the jwm1 parity clone; not merged
anywhere. RoPE fold lane untouched. `63c1d3cf` still not merged. Sibling
worktree dirt (bundle/vulkan_decoder files) left untouched. Build artifacts:
jw16 `/var/tmp/gemv-rmsnorm/` (worktree, `.work`, `dist`, `build2.log`,
stale-v1 archives); run artifacts
`/var/tmp/DecodeEpilogueFold/jw16-out-gemv-rmsnorm/` (`started.txt`,
`lock.txt`, `wheel-sha256.txt`, `dispatch-{gepin-default,gepin-nofold,gepin-gemvoff,base}.json`,
partial `ab.txt`), `run-gemv-rmsnorm-run.log`; investigation scripts
`fold_probe.py`, `bisect_fold.py`, `dump_step0.py`, `exact_map.py`,
`tagged_probe.py`, `airtight2.py` + `/tmp` dumps.

## Decision

Land rule fails on pins (ctx1024 digest). NO-LAND. The dispatch-drop half of
the ticket is proven reachable (249→202, attribution-clean); the blocker is
exactly the named hole: FastRmsNormF16's tree is not reproducible bit-exactly
from inside the qmm_vec compilation context on this stack.

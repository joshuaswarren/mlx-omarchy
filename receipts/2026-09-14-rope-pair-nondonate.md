# 2026-09-14/15 Rope-pair non-detaching settle on jw16 — NO-LAND (fix works, ctx1053 tok/s does not rise)

## Verdict

**Named no-land.** The named fix from
`receipts/2026-09-14-rope-pair-settle.md` — non-detaching nested settle
evals for outer-tape-owned nodes — is implemented, and the 23 rope pairs
now fire without SEGV. Three of the four land clauses hold: dispatches
drop (249 → 225/token), both digest pins hold on all 48 A/B runs, no
SEGV. The fourth fails: ctx1053 median tok/s does **not** rise (136.4 vs
141.6 base, −3.65% over 12 interleaved rounds). Short decode rises
clearly (+5.27%, no range overlap). Per the land rule, nothing merges.

## Root cause of the SEGV, and the fix (commit `d6b87ee4`, branch tip)

The experimental pre-settle (`8f1ed713`) called public `eval(chains)` at
the pair hook. That nested `eval_impl` detached every node it ran —
including rope members still pending in the outer tape (transformer
residual/cache chains make earlier ropes ancestors of later chains). The
outer pop then held a live primitive with empty inputs, the pair
refused, and the ordinary path dispatched on the detached node (SEGV,
`eval.cpp:65`).

`mlx/transforms.cpp` (via `patches/mlx-rope-settle-tape.patch`) now
carries a `thread_local` nest depth installed at `eval_impl` entry, and
three gated sites at depth > 1:

1. **No detaching** — all three detach sites (tape-build DFS pre-detach,
   the evaluated-node skip in the loop, the post-eval detach) are
   skipped in nested passes. The outer tape pops shared nodes later,
   skips them as already-evaluated, and detaches them itself.
2. **No commit at frame end** — a nested pass records into the
   enclosing eval's open command buffer on the same stream; its tail
   `gpu::finalize` is skipped, and the enclosing frame commits once.
3. **No throttle commit, no event signal** — the in-loop task-throttle
   branch is gated (the outer frame re-checks the same condition at
   every node), and the nested synchronizer-event signal is skipped:
   `settle()` discards its synchronizer, and a signal on an encoder
   that owes a batch flush takes the queued path, **which submits**.

`settle()` (new, declared in `mlx/transforms.h`) schedules the chains
exactly like `eval()` but drops the host wait — the pair records on the
same stream immediately after, so the in-order command buffer settles
the chains before the pair reads them.

`overlay/.../fused_chain.cpp`: `try_eval_eager_fusion` marks a pending
pair done when both members are already evaluated. Each eval scope
plans the same pairs, so a pair fired inside a settle scope must not
fire again when the outer tape pops it.

## The performance debugging chain (all on jw16, decode probes with
8 generated tokens, `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`)

| build | vk_compute_dispatches/token | vk_submissions/token | short decode |
|---|---|---|---|
| base `b41e2b74` | 249 | 2 | 168–170 tok/s |
| settle commits + waits (`+5a4a9a51`) | 225 | 25 | ~65 tok/s |
| tail-finalize gated (`+ad0ff840`) | 225 | 25 | ~70 tok/s |
| throttle gated (`+383207b1`) | 225 | 25 | ~64 tok/s |
| signal gated, final (`+d6b87ee4`) | **225** | **2** | **178.6 tok/s** |

The 25→2 measurement localized the serialization to the queued-signal
submit path (encoder.h: "a signal on an encoder that owes a batch flush
takes the queued path, which submits"), not to the finalizers — gating
them alone changed nothing. The final build's long-context probe
(1053-token prompt, base vs cand) shows the identical counter shape:
225 vs 249 dispatches, 2 submissions each — no settle-induced throttle
or submission cost at any context.

## The remaining boundary: the producer-direct keys window

The fired pair keeps the producer-direct KV window planned by the
epilogue workstream: the pair copies the whole keys window
(`copy_gpu(window.base → window.node)`) and redirects cache reads,
where the ordinary path slice-updates only the new token. The copy is
the one context-scaling cost in the pair path, and it is what the
ctx1053 arm pays:

- 12-round interleaved A/B (final battery): short base median 169.63 →
  cand 178.57 (+5.27%; cand range 177.4–179.9 above base 165.8–170.5);
  ctx1053 base 141.56 → cand 136.39 (−3.65%; ranges overlap,
  130.7–145.9 vs 127.6–150.9). An earlier 12-round battery on the
  code-identical `+7c1ce62e` wheel measured +6.22% / −2.69% — the
  ctx1053 delta is reproducibly small and negative.
- Falsification test: bypassing the window (ordinary slice-update)
  breaks the generated digest immediately (`a3f7b65f4517ae15` on the
  short pin) — the A/B harness refused the run. The window is
  structural to a fired pair; fixing the copy means writing rope'd keys
  through the ring cache directly, which is new planner-kernel work,
  out of this task's scope.

## Guard rails

`qmm_vec.comp` untouched (no diff on the branch against main
`3db3cb9a`); `63c1d3cf` not merged (not present in this repository);
every GPU run under one `flock` hold on `/tmp/m1-gpu.lock` (inode
checked before/after, never unlinked, nested `flock -n` refused).

## Wheels (full sha256, stamped from a tree that then carried the next
amendment; hashes are the identity)

- `+5a4a9a51` settle commits+waits:
  `9b5db82c9825bd3e594a7f57bfbed0bf896af15d5b242da6a2b017ae70d43f44`
- `+ad0ff840` tail-finalize gated:
  `93445fcea3d834ddca57bfde403df62261ab0d4cc83712fdf3834f54df8b779d`
- `+383207b1` throttle gated:
  `f0269d27e4bf0a72afbda728c35eae3b1aae21a3cbfce82e244c95d8c2a76107`
- `+7c1ce62e` signal gated (12-round battery):
  `1ff7b8f426bf02898c06b75ab15c2fde0c2aa0e7da1c2ccbe2d5f15a30d919aa`
- final `+d6b87ee4` (branch tip content):
  `e2c0fc4e2d54651bf5c6a69b26232e49f041debead7e3fb6a0f33a4a19cd7f1e`

## Artifacts

- jw16 branch: `/var/tmp/decode-trio` (`agent/decode-standalone-epilogue`,
  tip `d6b87ee4`).
- Final battery: `/var/tmp/DecodeTrioRun/jw16-out-final/`
  (dispatch probes ×3 default + ×3 trio-off, trace probe, 12-round A/B
  `ab.json`; base arm `venv-ab` wheel `b41e2b74`, cand arm `venv-trio`
  wheel `+d6b87ee4`).
- Intermediates: `jw16-out-nondetach{,2,3,4,5}/`, `longprobe-*.json`
  (1053-token probes), `run-nondetach*.log`, `run-final.log`.
- Harness (reused, unchanged): `/var/tmp/DecodeEpilogueFold/ab_decode.py`,
  `/var/tmp/DecodeCompileAB/{dispatch_count.py,venv-ab}`,
  `/var/tmp/mlx-omarchy-prof-b41e2b74/scripts/bench_decode.py`.

## Decision

**No repo state on mlx-omarchy main changed; nothing merged.** The
branch carries the working non-detaching settle (SEGV eliminated,
23-pair dispatch win realized: 249 → 225/token decode, 1063 → 1015
prefill, both pins exact) and this receipt. Follow-up (out of scope
here): make the fired pair write rope'd keys through the ring cache
directly instead of copy-window-plus-redirect — that is the only
remaining context-scaling cost between short +5% and ctx1053.

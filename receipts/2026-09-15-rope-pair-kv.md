# 2026-09-15 Rope-pair KV-window attack on jw16 — NO-LAND (same-layer + window is structurally blocked; kvdir0 finding recorded)

## Verdict

**Named no-land.** The land rule (dispatches ≤225, pins hold, ctx1053
median tok/s rises vs the 249-dispatch base) is not met by any
configuration measured today. The branch tip is restored to the
known-good cross-layer pair state (`3e96d144` reverts `80cd4c7f`;
behavior re-verified: 225 vk/token, ctx pin `7da83f06ec9f001d` exact,
single-run ctx 139.0 tok/s).

## What was tried, in order

1. **Same-layer (keys_L, query_L) pairs with the producer-direct KV
   window on the keys member** (commit `80cd4c7f`: scan rejects a
   *query* member carrying the redirect instead of a *keys* first
   member; `dispatch_rope_pair` resolves query/keys roles by the
   redirect plan). Measured on wheel `+80cd4c7f`
   (c41c5821ad2be605e4a159d724c687fc8a1fa75314ddc64dce21123c8385b3d0):
   **364 vk/token** (not ≤225), short median 145.1 vs base 168.0
   (−13.6%), ctx median 122.3 vs base 138.2 (−11.5%) over 12
   interleaved rounds, pins exact on every run
   (`/var/tmp/DecodeTrioRun/jw16-out-kvfinal/`).

2. **Root cause chain, all instrumented on jw16** (diag traces
   `[kvplan]`, `[kvredirect]`, `[kvpair]`):
   - The decode tape evaluates each layer's k-chain first; at the
     same-layer pair fire the layer's v-side gemv group has **not run
     yet**, so `values_committed` is false (`[kvredirect] blocked
     direct=1 state=0 committed=0`) and the window refuses.
   - The falling-back pair then triggered the non-detaching pre-settle
     at every fire; the nested re-evaluation added ~46+46+25 kernels
     per token (kernel census 397/403/245) — the 364 shape.
   - The **cross-layer** design (current branch state) hides this only
     by settling the *entire intervening graph* at each of the 23
     pairs — an O(graph²) host walk that scales with the decode graph
     size (sdpa split nodes at ctx1053) — that is the ctx1053 −3.65%
     mechanism the task set out to remove.

3. **Split producer-direct commit + dropped settle + in-place ring
   writes** (uncommitted experimental tree, one wheel
   `+80cd4c7f`-stamped dirty build
   a887ba171478bdd0886eb5992c5b3dd56432f8cfb7c1cdfd22481286eb4a76dd,
   on disk in `dist/`, venv-trio had it installed during probing):
   - Split commit (keys commits at the pair, values at the v gemv,
     per-node slice skip) alone **segfaults**: without the settle the
     pair binds view arrays whose producer buffers install lazily
     (gdb: `compute_binding` → `Buffer::ptr(this=0x0)` from
     `SliceUpdate::eval_gpu`); with the settle restored it is
     functionally correct (pins held in a 12-round battery: short
     145.1/ctx 122.3 — still the degraded 364 shape).
   - In-place ring writes (alias the slice-update output onto the
     single-use live cache buffer, producers write only the new rows,
     zero full-cache copies — the receipt-follow-up design) were
     implemented on top but still measure the 364 shape in quick
     probes: the same-layer window does not come alive at rope_k time
     on this tape. **The work is left uncommitted on jw16's
     `/var/tmp/decode-trio` working notes only via this receipt; the
     branch carries none of it.**

4. **Same-layer pairs without the window** (`MLX_OMARCHY_KV_DIRECT=0`
   on `+d6b87ee4`, 12 rounds, `/var/tmp/DecodeTrioRun/jw16-out-kvab/`):
   249 vk/token, short median 170.69 vs base 168.23 (+1.5%), ctx median
   143.12 vs base 141.68 (+1.0%), pins exact. The ctx regression of the
   fired-pair state disappears — but 249 fails the dispatch clause.

## Decision

**No repo state beyond the revert + this receipt; nothing merged;**
`qmm_vec.comp` untouched; `63c1d3cf` not merged; every GPU run under
one `flock` hold on `/tmp/m1-gpu.lock` (inode 12 before/after every
battery, never unlinked, nested `flock -n` refused each time).

## Measurements (all jw16, MLX_DISABLE_COMPILE=1, HF_HUB_OFFLINE=1,
Qwen2.5-0.5B-Instruct-4bit snapshot `a5339a4131f135d0…`)

| config | vk/token | short med | ctx1053 med | source |
|---|---|---|---|---|
| base b41e2b74 | 249 | 168.0–168.2 | 141.6–141.7 | 2×12-round batteries today |
| cross-layer pairs (tip `3e96d144`) | 225 | 178.6 | 136.4 | prior receipt + re-verified counts |
| same-layer + window `80cd4c7f` | 364 | 145.1 | 122.3 | `jw16-out-kvfinal/ab.json` |
| same-layer, KV_DIRECT=0 | 249 | 170.7 | 143.1 | `jw16-out-kvab/ab.json` |

Every battery run's generated-ID digest matched both pins
(`7fd25a869ff21678` short, `7da83f06ec9f001d` ctx1024).

## Follow-up (out of scope here)

The in-place ring-write design remains the only path that removes the
context-scaling copy outright (~48 full-cache copies/token today, both
arms). It needs the values-side commit ordering solved first: either
force the layer's v gemv group to dispatch before the k-chain (planner
reorder), or move the keys write to a hook that pops after the v
commit. The per-node skip + split-commit state machine from today's
experimental tree is the skeleton to restart from.

## Artifacts

- jw16 branch `/var/tmp/decode-trio` tip `3e96d144` (revert) + this
  receipt commit; wheel `+3e96d144`
  49f55125e8aca738b81e0e3dd3cfa436cb392e9e39ccd90a3fde0f0646bd091d
  (re-verified: 225 default / 249 KV_DIRECT=0, ctx pin exact).
- Batteries: `/var/tmp/DecodeTrioRun/jw16-out-kvdiag/`,
  `jw16-out-kvab/`, `jw16-out-kvfinal/`, `jw16-out-fullcount/`,
  `jw16-out-mat/` (incl. `kt-default.err` kernel census), gdb backtrace
  in `jw16-out-mat/` session log.
- Harness reused unchanged: `/var/tmp/DecodeEpilogueFold/ab_decode.py`,
  `/var/tmp/DecodeCompileAB/{dispatch_count.py,venv-ab}`,
  `/var/tmp/mlx-omarchy-prof-b41e2b74/scripts/{bench_decode.py,bench_matrix.json}`.

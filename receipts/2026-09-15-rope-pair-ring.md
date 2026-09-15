# 2026-09-15 In-place ring writes for the rope-pair KV windows on jw16 — NO-LAND (alias is correct and pins-exact; ctx1053 still does not rise — the settle walk, not the copies, is the ctx bottleneck)

## Verdict

**Named no-land.** The land rule (dispatches ≤225, pins hold, ctx1053
median tok/s rises vs the 249-dispatch base) fails on the third clause:
the in-place ring wheel measures ctx1024 median 135.39 tok/s vs base
138.80 (−2.5%, ranges overlap) while short rises +3.3% (174.36 vs
168.80). Dispatches hold at 225 and both digest pins are exact on all
48 battery runs. Removing the context-scaling window copies did not
lift ctx1053 — which pins the remaining ctx cost on the per-pair
nested settle's O(graph²) host walk, not on GPU copy traffic. The
branch carries none of the code; the tree is restored to tip
`25f93f04` (verified byte-identical, md5s below).

## What was built (measured on wheel `+25f93f04` dirty-tree, final
## sha256 below)

`dispatch_rope_pair` and `dispatch_quantized_gemv_group` installed the
producer-direct KV window's node with `malloc` + full-chunk
`copy_gpu(base → node)` and then wrote only the new rows through the
window. The ring change replaces that with one helper,
`alias_kv_ring_window` (~16 lines, `overlay/mlx/backend/omarchy/
primitives.cpp`): `window.node.copy_shared_buffer(window.base, …, 0)`
— the updated cache copy IS the previous cache buffer. This is safe
structurally because the python KV cache is chunk-shaped
(`KVCache.step = 256`, `mlx_lm/models/cache.py`): the slice_update
output's shape equals its input's shape, the paste row lands inside
the chunk's allocated headroom, and the ring chains through
`inputs()` (each token's member holds the previous member's buffer).
Net effect: producers write only the new rows; both full-cache copies
per layer per token (~16 MB/token at ctx1053) are gone. Diff vs
`25f93f04`: 23 insertions, 17 deletions, one file.

The follow-up's second idea — read-side ring views (aliasing the
`[..., :offset, :]` cache-state Slice onto the window buffer, planned
via a single-trimmed-axis prefix contract) — was implemented and
**broke the digests in every configuration** (see bisect). It is not
in the final tree; the read-side slice copies remain, as in the
225-dispatch state.

## Root-cause chain, all instrumented on jw16

1. **Tape shape confirmed by trace** (`MLX_OMARCHY_TRIO_TRACE=1`,
   8-token decode): per layer the tape runs `… Add, Transpose, RoPE
   (pair fires here), Reshape, Transpose, Synchronize, 2×
   SliceUpdate (settled=1 — producer-installed), 2× Slice (settled=0
   — ordinary copy), RoPE (done), SDPA`. The 2× Slice are the cache
   state views and were the second context-scaling copy.
2. **Bisect harness** (env-gated build, three configs × short + ctx
   pins): `fallback` (both changes off) pins exact; `alias-only`
   pins exact; `slices-only` broke both pins, and still broke them
   after the redirect source was corrected from `window.base` to
   `window.node` (the member — the semantically right buffer).
   Different wrong digests per config (`2c88bf…`/`1ae73a…` first
   build; `237abc…`/`0dca51…`; `f52b2a…`/`85d17c…`) — a real
   read-path defect, mechanism unresolved. The redirect was deleted
   rather than shipped; a future attempt should start from the
   observation that aliasing the member is not sufficient — something
   downstream still reads the slice node as if freshly materialized.
3. **The settle is load-bearing and is the ctx cost.** With the
   copies gone, ctx1053 moved from 136.4 (cross-layer + copies) to
   135.4 (cross-layer + ring) — i.e. ~30 MB/token of removed copy
   traffic bought nothing at ctx, while short (copy-light, host-bound
   less so) gained 3.3%. That localizes the ctx1053 regression to the
   per-pair nested settle: at each of the 23 pairs, `settle(q_in,
   k_in)` BFS-walks the ancestor graph, and the residual stream makes
   every earlier layer an ancestor — O(graph²) host work per token
   that grows with sdpa split-node counts. The copies were real
   context-scaling GPU work, but they were not on the ctx critical
   path.

## Measurements (all jw16, MLX_DISABLE_COMPILE=1, HF_HUB_OFFLINE=1,
## Qwen2.5-0.5B-Instruct-4bit snapshot a5339a4131f135d0…)

| config | vk/token | short med | ctx1053 med | source |
|---|---|---|---|---|
| base b41e2b74 | 249 | 168.80 | 138.80 | this battery, base arm |
| ring wheel (alias-only) | 225 | 174.36 (+3.3%) | 135.39 (−2.5%) | `jw16-out-ring/ab.json`, 12 rounds |
| KV_DIRECT=0 on ring wheel | 249 | — | — | `dispatch-ring-kvdir0.txt` (merged fallback intact) |
| cross-layer pairs + copies (prior) | 225 | 178.6 | 136.4 | kv receipt, 2026-09-15 |

Pins exact on every run of every arm (`7fd25a869ff21678` short,
`7da83f06ec9f001d` ctx1024) — 48/48 in the battery, plus the bisect
legs that passed. Lock inode 12 before/after every stage, never
unlinked; nested `flock -n` refused each time.

## Decision

**No repo state beyond this receipt; nothing merged;** the working
tree is restored to tip `25f93f04` — `primitives.cpp`
297ddcd24beebe7183099ae1e42c5ec7, `fused_chain.cpp`
40c501769c5cc62ac6d151be3f8d261e (both byte-identical to HEAD),
`git status` clean. `qmm_vec.comp` untouched; `63c1d3cf` not merged;
every GPU run under one `flock` hold on `/tmp/m1-gpu.lock`.

## Follow-up (out of scope here)

Make ctx1053 rise by attacking the settle walk, not the copies:
replace the per-pair nested `settle(q_in, k_in)` (full eval_impl BFS
over ancestors) with a targeted linear chain-scheduler that walks
only the unscheduled producer chain (Transpose ← Reshape ← Add ←
GEMV group ← RMSNorm, ~6 nodes) and stops at the first settled
ancestor. The ring alias in this receipt (wheel sha256 below) is the
verified skeleton to re-apply once the settle cost is gone; the
read-side slice redirect needs a fresh mechanism — aliasing the
member alone provably corrupts the stream.

## Artifacts

- Final ring wheel (dirty tree stamped `+25f93f04`; content = the
  alias helper + two call sites, no gates):
  `dist/mlx_omarchy-0.32.2.dev202609151711+25f93f04-cp314-cp314-linux_aarch64.whl`
  sha256 `a1980594d4f8048d7cd8de6c0983a8f68177e8d387e93c8e351f916c2b649f31`
  (still installed in `venv-trio`; reinstall a committed-tree wheel
  before any future A/B).
- Battery: `/var/tmp/DecodeTrioRun/jw16-out-ring/` (dispatch probes
  ×3 default + kvdir0, provenance line, 12-round interleaved
  `ab.json`, lock records, started/finished stamps). Runner:
  `/var/tmp/decode-trio/run-ring-battery.sh`; log
  `/tmp/ring-battery.log`. Bisect build logs `/tmp/ring-build*.log`.
- Harness reused unchanged: `/var/tmp/DecodeEpilogueFold/ab_decode.py`,
  `/var/tmp/DecodeCompileAB/{dispatch_count.py,mlx_provenance.py,run-jw16.sh}`,
  `/var/tmp/mlx-omarchy-prof-b41e2b74/scripts/{bench_decode.py,bench_matrix.json}`.

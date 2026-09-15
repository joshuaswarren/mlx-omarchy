# 2026-09-14 Decode standalone fused RMS+RoPE+SwiGLU epilogue on jw16 — NO-LAND

## Verdict

**Named no-land.** The land rule (pins hold AND per-token
`vk_compute_dispatches` drop AND ctx1053 median tok/s rise) is not met:

1. **Dispatches never dropped.** Every landable configuration measured on
   jw16 reports **249 `vk_compute_dispatches`/token** — identical to base.
   The rope-pair fusion, the only legal adjacency in the trio, planned 23
   pairs/plan but never fired (see hole below).
2. **ctx1053 tok/s regressed on the only wheel that ran a full A/B.** The
   single mode-switched megashader form measured short **170.68 → 155.35**
   (−9.0%) and ctx1024 **142.20 → 135.34** (−4.8%) medians against the
   base wheel. The per-mode specialization (one straight-line body per
   pipeline) removes that regression mechanism but has no dispatch win to
   convert into tok/s.
3. **Pins held everywhere measured.** The megashader A/B wheel passed
   `7fd25a869ff21678` (short) and `7da83f06ec9f001d` (ctx1024) on all
   runs — the trio norm and swiglu near-copies are bit-exact on the real
   decode tape. No Honeykrisp divergence was introduced;
   `qmm_vec.comp` is untouched (verified: no diff to that file on the
   branch); `63c1d3cf` was not merged.

## What was built (branch `agent/decode-standalone-epilogue`, unmerged)

Worktree `/var/tmp/decode-trio` off `mlx-omarchy` main `3db3cb9a`; tip
`0a2370ba`. New shader `overlay/mlx/backend/omarchy/shaders/fast_trio.comp`:
one source carrying near-copies of the three standalone bit-exact kernels
(`fast_norm.comp` RMS row, `fast_rope.comp` forward scalar-offset branch,
`swiglu.comp` four-wide chain), compiled per mode
(`fast_trio_norm_f16` / `fast_trio_rope_pair_f16` / `fast_trio_swiglu_f16`).
Routing: f16 RMSNorm → trio norm; adjacent-rope pairs → one trio rope-pair
dispatch (planner scan in `fused_chain.cpp` + `dispatch_rope_pair` in
`primitives.cpp`, composing with the producer-direct KV window); fused
SwiGLU chain f16 → trio swiglu. Gate `MLX_OMARCHY_FUSED_TRIO` (default on;
`=0` restores the standalone paths — measured: 249 both ways). Traces
behind `MLX_OMARCHY_TRIO_TRACE=1` (plan buckets, refusal reasons).

## Why the rope pair never fires — named hole

Plan-time trace (decode-shaped plans, jw16): 47 rope nodes collected,
23 pairs planned, **every pair refused at runtime** across four refusal
generations. Measured facts:

- The rope subsequence in the recorded plan is **not tape-adjacent**
  (k-rope and q-rope are separated by the per-call offset `Full` and cache
  view nodes) and the keys rope **evaluates before the query rope**.
- At both rope hooks the partner GEMV output reads
  `data_shared_ptr() == nullptr` (lazy buffer install), while the ordinary
  single-rope path binds the same array correctly — the null probe was
  over-strict and was removed; the ordinary path produces the pinned
  digests from the same arrays.
- The real rope inputs are strided views, not row-contiguous: measured
  `q ndim=4 rowc=1 strides=3712,1856,64,1` vs
  `k ndim=4 rowc=0 strides=25984,64,896,1`. The k side's stride pattern
  keeps the **query width** as its time stride (`14*64=896` for a 2-head
  key), which is outside even the single RoPE kernel's no-copy head/seq
  transposed contract (`strides[0] == T*n*D` with the tensor's own `n`) —
  the production single path takes its contiguous-temporary fallback here.

Fusing the pair therefore requires either reading both GEMV outputs at a
point where the backend has not settled them (planner-level output
pre-allocation / GEMV-group cooperation — new planner-kernel work) or
folding the norm/rope work into the GEMV dispatch itself — **qmm_vec
territory, forbidden by this task** and already known to break the ctx1024
pin (`64e52943`: 249 → 202 dispatches, pin broken).

## Measurements

- Dispatch probes (dispatch_count.py, 8 tokens, MLX_DISABLE_COMPILE=1,
  HF_HUB_OFFLINE=1, model snapshot `a5339a4131f1…`): cand default gates
  249/token (median of 7 decode rows), cand `FUSED_TRIO=0` 249/token,
  token-0 prefill 1063 — identical base vs cand.
- 5-round interleaved A/B (`ab_decode.py`, one `flock` hold on
  `/tmp/m1-gpu.lock`, inode 12 before and after, never unlinked, nested
  `flock -n` refused; arm provenance `verified=match` on every run),
  wheel `+2b7d018d` (megashader form): medians above; every run's
  generated-ID digest matched both pins. The specialized wheel (`+0a2370ba`
  lineage) was measured for dispatches and refusal reasons only; a full
  A/B on it would measure a configuration that cannot pass the dispatch
  criterion.
- Base arm: `venv-ab` wheel `b41e2b74`; `git diff --stat b41e2b74..3db3cb9a`
  is one receipts JSON (108 insertions), so the base wheel is
  code-identical to main `3db3cb9a`.

## Decision

**No repo state changed on main; nothing merged.** The branch stays at
`0a2370ba` with this receipt for the record. Follow-up candidate (out of
this task's scope): extend the GEMV group planner to pre-settle partner
outputs so a standalone pair dispatch can see both rope inputs — new
planner work, not a shader near-copy.

## Artifacts

- jw16 branch: `/var/tmp/decode-trio` (`agent/decode-standalone-epilogue`).
- Wheels (full sha256): `…+2b7d018d` =
  `18608e7eb5bc153f2306c64bae7a232e7dd47721bbe369d33f871f286c7bbd71`
  (megashader A/B cand); `…+43defc4a` =
  `86de1b0ad91639ba2a26b627acf2e64ecd90a898166f5b2d5f56a8ba1bcb5716`;
  final `…+0a2370ba` =
  `fc666439debf4d2bbccaa91bb9ca59af250f6bfca8347914dc0c091db2e6ab5f`.
  Intermediate trace wheels `…+945b3a12`, `…+cdd9024f` et al.: hashes in
  the jw16 build logs (`/var/tmp/decode-trio/build-trio.log` per build).
- Runs: `/var/tmp/DecodeTrioRun/jw16-out/` (A/B + probes),
  `/var/tmp/DecodeTrioRun/jw16-out-trace/` (trace probes),
  `probe-trio.log`, `run-trio.log`.
- Harness (reused, unchanged): `/var/tmp/DecodeEpilogueFold/ab_decode.py`,
  `/var/tmp/DecodeCompileAB/{dispatch_count.py,venv-ab}`,
  `/var/tmp/mlx-omarchy-prof-b41e2b74/scripts/{bench_decode.py,bench_matrix.json}`.

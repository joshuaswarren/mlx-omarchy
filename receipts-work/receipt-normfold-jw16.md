# 2026-09-24 — t6001-norm-swarm (jw16): RMSNorm-into-GEMV prologue fold — REJECTED-CANDIDATE (gate flips; not installed)

## Verdict

**REJECTED-CANDIDATE.** The build-equivalence arm pinned (MLX_OMARCHY_FUSED_GEMV_NORM=0 →
3-pass digest `bc519c03` exact, 77.05 tok/s), but the candidate arm flipped 44/320
greedy argmaxes vs a fresh installed-stack gold. Per the abort rule the battery
stopped at gate 1; the 10-pass and contract arms never ran. NOTHING was installed;
`/var/tmp/v072-venv-fused` untouched (read-only); service restored and probed after
every window (health ok, real completion "OK", finish=stop).

## Artifacts

- Branch `agent/t6001-norm-swarm` (base `16df8b6b`), tip `4039fcfd`:
  - `465667ca` design (prior-art-aware; the two ancestor NO-LAND/LAND folds studied)
  - `74a516c4` shader NORM_PROLOGUE variant (binding 19, shared bf16 normed row,
    flags bit 15), CMake target `qmm_vec_q4_multi_subgroup_bf16_np`, append-only
    kernel enum `QmmVecQ4MultiSubgroupBF16NormPrologue` + shader table, binding
    budget 19→20
  - `f585e032` host fold: planner reader-safety match (fast::RMSNorm x, uses ==
    members, unclaimed, bf16, vector weight), norm-node deferral (group fires at
    the norm's eval turn), `GemvNormPrologue` plumbing, `MLX_OMARCHY_FUSED_GEMV_NORM` gate
  - `057a4420`/`5f9a9b5c` GLSL reserved-word fixes, `e7549835` fast::RMSNorm
    namespace fix, `4039fcfd` diagnostic scope knobs
- Release wheel `mlx_omarchy-0.32.3.dev202609241907+e7549835...whl`
  sha256 `3f44c24c367030589fac9914ebe463118acb4e7acf7cc1c8c2554ecca9da108a`;
  knobbed wheel `+4039fcfd`; diag wheel `+diag.e754983`.
- venvs: `/var/tmp/normfold-venv` (candidate clone), `/var/tmp/normfold-diag-venv`
  (diag); `/var/tmp/v072-venv-fused` untouched.
- Outputs: `/var/tmp/normfold-out/` — window logs
  (`window-20260924T191439Z.log`, `microwindow.log`, `locwindow.log`,
  `profwindow.log`), logits captures (`logits-gold.json`, `logits-cand-*.json`,
  `loc-*.json`), profiles (`micro-profile.ndjson`, `model-on.ndjson`,
  `model-off.ndjson`), analyzers (`flipmap.py`, `streamdiff.py`, `shapecmp.py`).

## Gate numbers

| arm | result |
|---|---|
| build-equivalence (cand, NORM=0), 3-pass | **bc519c03 PIN, 77.05 tok/s — PASS** |
| gates x3 cand (default), fresh ctl gold | **gate 1: 44/320 flips, max|d_top1| 10.375 — FAIL → ABORT** |
| 10-pass ctl / cand x2 | NOT RUN (abort rule) |
| paired CI contract | NOT RUN (abort rule) |

Flip map (prompt: flips@first-step): p0 0, p1 18@14, p2 0, p3 0, p4 13@19, p5 0,
p6 0, p7 13@19, p8 0, p9 0 — rare one-bf16-ulp drifts compounding into argmax
losses; 6/10 prompts never diverge.

## What was proven (the valuable part)

1. **The fold fires and the stream is structurally clean**: full-model profile
   (diag wheel, 1 prompt × 8 tokens): 6167 → 5897 dispatches (−270 = exactly the
   270 standalone FastRmsNormBF16 gx=1 launches); stream identical until the
   first early-fired group (`model-{on,off}.ndjson`, first diff index 1142).
2. **Per-class bit-exactness in isolation**: with the fold PROVEN firing (diag
   profile: zero standalone RMSNorm/qmm dispatches, unnamed group dispatches
   gx=128/88/32/32), all four group classes (trio, swiglu+prologue, single+Add,
   pair+Add) hash byte-identical fold-on vs fold-off across 4 seeds at K=2048
   (`shapes-*.json`).
3. **SPIR-V-level reduction identity**: `fast_rms_norm_bf16.spv` vs the np variant
   — identical op sequences (FMul+FAdd square accumulate, FAdd tree, FDiv mean,
   GLSL.std450 InverseSqrt); no Fma contraction of the prologue anywhere
   (`base.spvasm`/`np.spvasm`/`plain.spvasm` in /var/tmp/normfold-out).
4. **Kill switch is exact**: NORM=0 restores the unfused stream bit-for-bit
   (digest pin + 0 flips, max|d_top1| 0.125 = the documented near-tie jitter of
   the installed stack itself).

## Localization status (for the next lane)

Scope knobs show the divergence is global, not per-class: all-on 44, no-swiglu 44,
no-single 44 (those classes never norm-fold in this model — GDN has 4 in_proj
qmms chunked 3+1; MLP gate/up is the swiglu class), no-trio 36, no-epi 76
(chaotic — the flip pattern shifts with any planning perturbation). The divergence
enters at the first early-fired groups; the untested interaction is the **kv-direct
sum-window path combined with the norm prologue** (the model's qkv group carries
v's KV window — micro shapes exercised plain Adds only) and, secondarily, the
reordered firing vs the GdnConvUpdate/rope redirect chain. Recommended next: add a
kv-direct sum-window shape to `micro_shapes.py` (v member with `sum_window`
plumbing), and if exact there, diff per-token KV-cache bytes.

## Hygiene

llm-inference stopped/flocked for each of the 7 short windows, trap-restored every
time, health polled to `"status":"ok"`, completion probe run every window (key
read via sudo cat, never printed). No reboots, no /dev/shm usage, v072 untouched.
Branch pushed to origin BLOCKED by the fail-closed privacy hook (24 fleet-only
base commits carry RFC1918 test fixtures); reported to Main with options.


## ADDENDUM — bounded iteration 2 (parent-authorized): kv-direct exclusion — final-REJECT

Planner now refuses the norm prologue for ANY group containing a member with a
kv-direct sum_window (commit `bfbf7581`): those groups run the standalone norm +
group-reads-from-memory, kv-direct untouched. Wheel
`mlx_omarchy-0.32.3.dev202609242003+bfbf7581` sha256 `a5bb05e6…` installed in
/var/tmp/normfold-venv (venv import version verified). Full battery rerun
(window 20260924T200550Z):

| arm | result |
|---|---|
| build-equivalence (NORM=0) 3-pass | PASS — bc519c03 @ 77.28 |
| gate 1 cand default | **FAIL — 44/320 flips, max|d_top1| 10.375 (prompt 7), byte-identical signature to the pre-fix run** |
| remaining arms | not run (abort rule) |

The identical flip signature REFUTES the kv-direct interaction hypothesis: the
divergence source is elsewhere (suspects narrowed by elimination: the early-fired
trio in_proj/qkv groups whose members are all window-free, i.e. the norm
prologue's interaction with the GDN conv/gated-delta state chain or the
fused-chain swiglu consumers under the model's real tape — NOT kv-direct, NOT the
reduction lowering (SPIR-V-identical), NOT per-group kernel values
(micro-proven)). Per the authorization: FINAL-REJECT, service restored + probe
ok, lane closed. No further iterations.


## ADDENDUM 2 — bounded iteration 3 (parent-authorized): writeback barrier — FINAL-REJECT (lane closed)

SpirvDisc's discriminator (14c36fcd) refuted the lowering wall; the working
hypothesis moved to host/tape-domain scheduling (early-fired group vs producer
writeback). Probe used: the fix itself — a deterministic full dependency barrier
recorded before every early-fired prologue group dispatch (commits `11293b3c`
+ `7519a92c`, `record_dependency_barrier()` made public for this). Wheel
`mlx_omarchy-0.32.3.dev202609250003+7519a92c` sha256 `25e6757b…`. Battery
(window 20260925T000714Z):

| arm | result |
|---|---|
| build-equivalence (NORM=0) 3-pass | PASS — bc519c03 @ 77.31 |
| gate 1 cand default | **FAIL — 44/320 flips, max|d_top1| 10.375 (prompt 7) — third byte-identical signature** |

Three independent mitigations, one deterministic signature: baseline 44,
kv-direct-excluded 44, writeback-barrier 44 (same prompt, same max delta). The
divergence is a deterministic function of the fold's presence, invariant to
lowering (SPIR-V/NIR/AGX-IR op-identical per discriminator), invariant to
per-group values (micro 12/12 byte-identical with firing proven), and invariant
to scheduling barriers. Eliminated definitively: Honeykrisp lowering, kernel
arithmetic, kv-direct write ordering, dispatch writeback ordering. Remaining
domains for any future lane: the model tape's exact planner shapes (a group
class present in-model but absent from the micro — e.g. the GDN [qkv,z,b] trio
whose z member feeds a reshape chain, or multi-token tape re-planning), or a
buffer-identity/offset subtlety in the model's sliced residual rows.

Per the authorization: FINAL-REJECT, lane closed, service restored + probe ok.


## ADDENDUM 3 — commissioned round: SwiGLU-store+chained-prologue and RMSNormGated-fed out_proj — neither reproduces; LANE TERMINAL

Harness_swing.py: per token, na=RMSNorm(xin)+trio, nb=RMSNorm(xin)+gate/up pair with
the silu(g)*u store chain (bit 15 + bit 16 + chained input — the exact commissioned
structure), down group on the swiglu product, RMSNormGated-fed out_proj class
(real mx.fast.rms_norm_gated binding), n2 trio, chained residual, slice-update
state, 8 tokens. FIRING PROVEN (diag profile: 14 of 24 RMSNorm gx=1 dispatches
removed fold-on; the folded gate/up group keeps the same signature as its plain
trio, so the removal count is the marker). Results: swing / swing-nochain /
swing-nogated ALL byte-identical fold-on vs fold-off across 8 tokens x 13
observables.

With this round, every in-model-reachable structure expressible outside the full
model is proven bit-exact under the fold: trios, pairs, singles, swiglu-store
with chained prologue, gated-norm out_proj, epilogue-sum-chained norm inputs,
per-token replanning, in-place state. The divergence reproduces ONLY inside the
full 24-layer model — the remaining domain is full-model planner state (planner
interaction surface across the whole tape: rope-pair/kv-direct/swiglu matchers
and the allocator interleaving at 24-layer scale). LANE TERMINAL at tip
99e86ee5 + this addendum. Final disposition: FINAL-REJECT (never installed;
v072 untouched throughout).

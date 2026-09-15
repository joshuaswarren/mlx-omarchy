# SwiGLU-only GEMV store epilogue: pins hold, 225 → 201 dispatches, LAND

Date: 2026-09-15
Task: fold ONLY the SwiGLU (silu(gate) * up) into the fused gate/up
QmmVecQ4Multi store epilogue. No RoPE in qmm_vec — the Honeykrisp NIR
trig hole (digest `31267e7ed4c6d0dc`, receipts 2026-09-14/15) stays out
of scope: this path is mul/silu only, no exp/cos/sin rotation chain.

## Verdict

**All three acceptance gates pass. Land.**

1. Both pins hold — short `7fd25a869ff21678` and ctx1024
   `7da83f06ec9f001d` in every arm and every round of two 5-round
   interleaved batteries (20/20 leg digests), plus the dedicated
   ctx1024 digest arm on base, cand, and cand-with-fold-off
   (first ids `13060,498,369`, argmax never flips). The sigmoid's exp
   lowers identically inside `qmm_vec.comp` and inside
   `shaders/swiglu.comp` on Honeykrisp — unlike the RoPE trig chain,
   which is the divergence that killed the 2026-09-14 epilogue fold.
2. Dispatches drop — 225 → 201 vk compute dispatches per decode token
   (−24 = exactly the per-layer standalone swiglu), vk submissions
   2 → 1; `MLX_OMARCHY_FUSED_GEMV_SWIGLU=0` on the same cand wheel
   restores 225 / 2, so the drop is the fold. `gpu_primitive_dispatches`
   stays 858 (no host-side fixups appeared).
3. ctx1053 median decode rate rises — two batteries, 10 interleaved
   rounds:

| leg | base median | cand median | Δ |
|---|---|---|---|
| ctx1024 (1053/32) | 97.75 tok/s | 101.37 tok/s | **+3.7%** |
| short (30/32) | 112.26 tok/s | 113.90 tok/s | +1.5% |

Per-battery ctx1024 medians: battery 1 base 97.78 → cand 98.04
(cand round 2 caught a 96.34 dip), battery 2 base 97.72 → cand
102.34, where every cand round (100.88–103.13) beats every base round
(97.34–98.36). Battery 2's short leg carried one cold-start outlier
(cand round 0, 100.87); the other nine rounds put short within noise
of base. The win is on the long-context leg, which is where the
deleted 24 dispatches and their store/load traffic live.

## Change

`overlay/mlx/backend/omarchy/shaders/qmm_vec.comp` (QMM_VEC_MULTI):
flags bit 16 pairs the gate (weight 0) and up (weight 1) dots in ONE
workgroup — same Q4_ROW chain, same LANES-strided words, same reduce
pairing, so both totals are bit-identical to the weight-partitioned
layout's stores — then applies the fused_chain swiglu program
statement-for-statement (`sig = sigmoid(gate); silu = gate * sig;
out = silu * up`, every intermediate through ROUND_STORAGE exactly as
swiglu.comp rounds values loaded back from memory) and stores only the
product. The workgroup count for a folded group is weight 0's alone.
A `Q4_REDUCE` macro factors the shared workgroup reduction for both
flavors (tree + USE_SUBGROUP).

`fused_chain.cpp`: the planner records planned swiglu chains and
matches them against planned two-member epilogue-free GEMV groups of
equal length; the gate leaf's only readers must be the chain's sigmoid
and gate multiply (the gate leaf is read twice — sigmoid + gate mul —
by design), the up leaf's only reader the tail multiply. Either tape
order is accepted (jwm1's tape emits up before gate); a swapped group
is reordered because the epilogue's silu belongs to the gate weight,
and the per-slot binding/output wiring is positional, which makes the
swap a no-op for the plain multi-weight path. The chain's three nodes
move to the group's roles, so the first member's eval fires the one
dispatch and the standalone swiglu dispatch never exists. Both
projections alias the product buffer; their only readers were the
deleted dispatch. Runtime contract failure un-plans the whole group
onto the per-node path (equal-length mismatch cannot exist in a valid
swiglu graph — the multiply would not broadcast — so the runtime shape
check is belt-and-braces). Knob `MLX_OMARCHY_FUSED_GEMV_SWIGLU=0`
(default on; the `MLX_OMARCHY_FUSED_GEMV` gate also covers it).

`primitives.cpp` `dispatch_quantized_gemv_group`: fold contract
(two epilogue-free members, equal lengths, f16/bf16), halved group
count, product allocation, both member outputs aliased to it,
output0 binding carries the product, flags bit 16.

Tests (`overlay/tests/omarchy/test_fused_chain.cpp`, 33/33 local on
llvmpipe, 23,749 assertions): the gate/up residual test now expects
the fold (fused 6 → 3); new f16+bf16 bit-exact fold test (per-node 3
→ folded 1, projections alias the product); new env-gate-off test
(group + chain = 2, chain values bit-exact). The two fold tests
caught two real planning bugs pre-battery: `up` may be a tape INPUT
(lookup-null must not kill chain formation), and the strict
members[0]=gate order assumption (llvmpipe tape emits up first).

## Identity

- Branch `agent/swiglu-store-epilogue` at 7e64e41c on parent
  origin/main 1deb70f1 (rope-pair line; 225-dispatch default decode).
  Base wheel `mlx_omarchy-0.32.2.dev202609152020+1deb70f1` sha256
  `dfec430467628e700ec8d40524126ef2495154552973db2055acbb3d33f8e5e4`;
  cand wheel `mlx_omarchy-0.32.2.dev202609152037+7e64e41c` sha256
  `92e318e72eab40b15a1c0ea72113b03976c77579606ba6f0fcafd77a631f9845`;
  libmlx.so base `e272fb7e…` cand `3245289c…`, provenance
  `verified=match` on every arm and leg.
- Host: jwm1-linux (M1 Max, Honeykrisp), Python 3.14, mlx-lm 0.31.3,
  `MLX_DISABLE_COMPILE=1`, model snapshot `a5339a41…` (pinned).
- Every GPU step ran under one `flock` hold on `/tmp/m1-gpu.lock`
  (inode 29 before and after; nested `flock -n` refused; never
  unlinked). `63c1d3cf` not merged; no RoPE code in qmm_vec.

## Reproduce

```sh
ssh jwm1
flock /tmp/m1-gpu.lock bash /var/tmp/SwigluEpilogue/run-jwm1.sh
# build: /var/tmp/SwigluEpilogue/build-jwm1.sh
# confirmation A/B only:
#   ab_decode.py --rounds 5 --out out/ab-confirm.json (arms as in run-jwm1.sh)
```

Artifacts in `receipts/2026-09-15-swiglu-epilogue/`:
`dispatch-{base,cand,cand-off}.json`, `digest.txt`, `ab.txt`,
`ab.json`, `ab-confirm.json`, `lock.txt`, `host.txt`,
`started.txt`, `finished.txt`.

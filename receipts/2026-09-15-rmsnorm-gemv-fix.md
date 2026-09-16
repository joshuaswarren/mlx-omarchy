# 2026-09-15 RMSNorm GEMV prologue short-leg fix — five designs measured; gates jointly unreachable; scope knob shipped; NO-LAND on the ticket gates

## Verdict

**No variant satisfies all four ticket gates (short flat/up, ctx1053 up,
pins hold, dispatches ≤ 154) simultaneously.** The gates are measured to
be mutually exclusive on jw16: the ctx1053 win comes from deleting the
47 norm dispatches, and the deleted work reappears as a per-workgroup
reduction inside the consumer GEMVs whose 608-wave MLP dispatch cannot
hide it. Five reduction designs were built, bit-exactness-verified, and
raced; every alternative to the landed tree measured slower. The branch
ships the investigation, the reverts of all failed designs, and one
new measured deliverable: the `MLX_OMARCHY_FUSED_GEMV_RMSNORM_MLP=0`
scope knob (default behavior byte-identical to the landed fold).

## The measured map (short leg, clean A/B, jw16mbp1)

| configuration | dispatches | short tok/s | ctx1024 tok/s |
|---|---|---|---|
| cand (a21b3c81, SwiGLU-only) | 201 | 190.4 | 136.9 (receipt) / 130.5 (today, noisy) |
| **landed fold (94e95db3 shader)** | **154** | **176.8 (−7.3%)** | **142.1 (+3.8%)** |
| fold + 4-slot stride | 154 | 176.3 | 144.3 |
| fold + shuffleXor finish | 154 | 147.7 | — |
| fold + subgroupAdd finish | 154 | 145.8 | 131.5 |
| fold + word-aligned tree | 154 | 166.9 (+ 104 fold-test assertion failures) | 132.0 |
| fold + scalar-view loads | 154 | 169.6 | 130.3 |
| **D-const: reduction deleted** | 154 | **186.8 — the ceiling** | — |
| qkv-scope (new knob) | 178 | 185.7 (−2.5%, disjoint) | 143.6 (+10% median, wide spread) |

All runs pin short `7fd25a869ff21678` / ctx `7da83f06ec9f001d` except
the deliberate D-const diagnostic (wrong values by construction; the
A/B harness correctly refused it after the timing leg).

## Why the gates cannot coexist

1. **D-const ceiling.** With the per-workgroup reduction deleted
   entirely (rms_scale = 1.0, normalized dot intact), short = 186.8 —
   still −1.8% vs cand 190.4. The normalized dot's per-element
   `ROUND_STORAGE` discipline (load w, two multiplies, f16 round trip)
   is the pins' price and cannot be optimized away. No correct fold
   reaches cand at short.
2. **Every tree alternative loses to the plain tree.** shuffleXor
   (−29 tok/s vs tree) and subgroupAdd (−31) both die in this divergent
   prologue context on Honeykrisp; the word-aligned repack (6 barriers,
   coalesced constant-extraction loads) also lost (166.9) and broke 104
   fold-test assertions; scalar-view loads lost too (169.6). The
   landed 9-barrier shared tree is the local optimum.
3. **Per-dispatch algebra.** The fold costs ~0 on the 112/144-workgroup
   attention GEMVs (41.3 vs 42.0 µs) and ~+21 µs on the 608-workgroup
   MLP pair (63.1 → 84.2 µs): the ~0.28 µs reduction is per-wave
   serialized and only exposed at 76 waves. Four-slot slicing dilutes
   the reduction but pays it back in per-column loop overhead (89.4 µs).
4. **qkv-scope splits the trade, does not close it.** 5-round
   interleaved battery: short 185.65 [185.36–186.06] vs cand
   190.39 [190.09–190.51] (−2.5%, disjoint — not flat); ctx median
   +10.0% (143.56 vs 130.48) with heavy spread [114.98–147.48].
   Dispatches 178.

## What shipped on the branch

- `MLX_OMARCHY_FUSED_GEMV_RMSNORM_MLP=0` (fused_chain.{h,cpp}): scopes
  the prologue fold to the attention GEMV groups. Default off — the
  default dispatch behavior is byte-identical to the landed fold
  (154 dispatches re-verified on the tip wheel).
- Reverts of every failed design (shuffleXor, subgroupAdd finish,
  4-slot stride, word-aligned tree, scalar view). The tip shader equals
  94e95db3's qmm_vec.comp.
- fused_chain 34/34 (36,190 assertions) and kv producer-direct 352
  assertions on the tip wheel.

## Recommendation to the gate owner

Pick one, with numbers:
1. **Keep the landed fold** (154): ctx +3.8%, short −7.3% — status quo.
2. **Default the scope knob to qkv** (178): short −2.5%, ctx median
   +10% (needs a repeat battery to confirm against today's ctx noise) —
   requires relaxing the ≤154 gate to ≤178.
3. **Drop the prologue fold** (revert to a21b3c81): short and ctx at
   baseline; requires dropping the ctx-rises and ≤154 gates.

This run cannot choose; the gates as written are infeasible.

## Artifacts

- Branch `agent/rmsnorm-gemv-epilogue` tip `c2fc9246` (knob + reverts;
  shader identical to 94e95db3).
- jw16: `/var/tmp/SwigluRmsJw16/jw16-out-gemv-fix/` — `ab.txt`,
  `ab.json` (5-round interleaved, provenance-verified), dispatch
  counts for both knob states, test logs, lock/host/started/finished.
- jw16: `/var/tmp/SwigluRmsJw16/profile-attrib/` — per-dispatch
  profiles (diag-cand, diag-rms, diag-fix) and token tables
  (`/tmp/tok10-{cand,rms,fix}.json`).
- Wheels: rms tip wheel `+c2fc9246` (dist-rms), cand `+a21b3c81`
  (dist-cand), diag wheels `+diag.a21b3c8` / `+diag.a5756be` /
  `+diag.af332e7`; D-const wheel (uncommitted patch, since removed).
- Local sim: `/tmp/normsim*.c` — 6,000-row × 12-size × 2-contraction
  bit-exactness checks for each candidate association.

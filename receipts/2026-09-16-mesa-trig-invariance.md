# Mesa trig invariance: the hole was never trig lowering - it was one FMA contraction, and it is fixed

Date: 2026-09-16
Task: close the `wave/DecodeEpilogueFold` RoPE question - the 2026-09-14
and 2026-09-15 receipts named a Honeykrisp "trig-lowering hole" (the exp
to cos/sin to product-subtract chain lowering differently inside
`qmm_vec.comp` than inside `fast_rope.comp`) as outside GLSL source
control and unfixable without a Mesa change. This battery dumps the
actual push-constant words and the driver-exact trig bits, names the
real root cause, fixes it from shader source, and re-runs the full
jwm1 battery.

## Verdict

**The digest gate is closed. The perf gate still fails: does not land.**

- The 1053-token ctx1024 greedy digest, broken by every previous fold
  build as `31267e7ed4c6d0dc`, now holds the pin `7da83f06ec9f001d` -
  in the digest section and in all 12 interleaved A/B rounds.
- The short pin `7fd25a869ff21678` holds on both arms in all rounds.
- Dispatches drop exactly as priced: 249 -> 177 vk/token, 858
  gpu_primitive_dispatches unchanged, fold-off restores 249.
- Decode tok/s does not rise: ctx1024 median -0.445 % (97.9160 vs
  98.3539), short median -1.845 % (112.7935 vs 114.9134) against the
  <=1 % short allowance. Same perf shape as the pre-fix fold
  (2026-09-15: -1.62 % short), so the cost is the fold's epilogue
  serialization, not the contraction fix. Both perf conditions of the
  land rule fail; the branch stays on `wave/DecodeEpilogueFold` and
  nothing merges to main except this receipt.

## Root cause (measured, not inferred)

Three attribution steps, all on jwm1, all reproducible:

1. **The push-constant words are identical.** `MLX_OMARCHY_ROPE_BITS`
   (new env-gated dump at both dispatch sites, `primitives.cpp`) prints
   the words every rope dispatch actually receives. Eager FastRopeF16
   and the folded QmmVecQ4MultiSubgroupF16 take `alpha=3f800000`,
   `beta=3edd0c55` on every one of 2039 dispatches; the fold's offset
   word is the literal position, the eager arm reads the same int from
   the device buffer. The alpha-bits hypothesis is dead
   (`jwm1/evidence-bits.txt`).

2. **Sums and trig are bit-equal.** The GEMV/Add sums are bit-exact
   (only outputs 3/4 ever mismatch), and an f32 impulse rope probe
   (x = 1,0 stores cos/sin at full f32) gives driver-exact trig bits
   per mismatching (offset, i). The typed `f16(cos)` agrees between
   arms on the whole theta grid - which is why the impulse test never
   caught any of this: at x = (1,0) every rounding pattern collapses to
   the same f16 word.

3. **The divergence is one contraction.** Matching each arm against
   rounding formulas computed with the driver's own bits (21 boundary
   records, 21/21 fits):
   - eager = `f16(fl32(x1*c) - fl32(x2*s))` - the plain f32 chain, one
     final rounding. Every `rope_round` in the compiled fast_rope
     pipeline is elided; the source-level typed round trips are dead
     in both pipelines.
   - fold = `f16(fma(x1, c, -fl32(x2*s)))` - NIR contracted the
     subtract into an FMA that keeps `x1*cos` exact. Boundary products
     flip by one f16 ulp, rate ~7e-5 per rotation, which a 1053-token
     decode always hits.

So Mesa was never "non-invariant" in its trig: same words, same theta
bits, same driver sin/cos. What differs per compilation unit is FFMA
formation in the rotation epilogue - and that IS fixable from source.

## The fix and the one thing that did not work

- `precise` (NoContraction) is unusable: glslang's propagation reaches
  through the trig argument chain and flips the driver's sin/cos
  selection - the impulse sweep went to 93 TRIGDIFFs with `precise` on
  the rotation, and a bitcast fence around cos/sin does not stop the
  propagation. (Both runs archived; do not retry `precise` here.)
- The fix (`qmm_vec.comp`, commit `705271f4`): route the four rotation
  products through `rope_scratch`, a shared-float array - same-
  invocation store/load of its own slot, no barrier, runtime indices
  keep the store alive. The rotation's add/sub then consume memory
  loads, which NIR cannot contract. Trig stays float32; the dead
  `ROPE_ROUND` macro is deleted.
- Full omarchy fused-chain suite: 33/33 cases, 1,264,015 assertions,
  including the hardened sweep (offsets 0..63 and 1000..1120, three
  draws) bit-identical.

## Battery (jwm1, 2026-09-16)

| arm | vk/token | short pin | ctx1024 pin | short decode med | ctx1024 decode med |
|---|---|---|---|---|---|
| base `b79a4b68` | 249 | holds | `7da83f06ec9f001d` | 114.9134 | 98.3539 |
| cand `705271f4` (fold on) | **177** | holds | **`7da83f06ec9f001d`** | 112.7935 (-1.845 %) | 97.9160 (-0.445 %) |
| cand, `MLX_OMARCHY_FOLD_EPILOGUE=0` | 249 | - | `7da83f06ec9f001d` | - | - |

12 interleaved rounds, alternating arm order, fresh bench_decode
process per leg, provenance `verified=match` on every run. The A/B
driver aborts on the first digest mismatch; it ran to completion.

## What would make it land

The digest question is closed; what remains is launch-bound decode
throughput on M1 Max. The fold saves 72 dispatches/token yet loses
~1.8 % short / ~0.4 % ctx1024 decode: the epilogue's serialization
(exchange + rotation after the reduction) costs more than the saved
launches at 0.5B. Options worth pricing before the next attempt: fold
only the keys/queries members whose windows already exist (drop the
per-token Add merges elsewhere), or measure with the fold restricted
to ctx-length prefill-adjacent paths where the dispatch saving
dominates. Do not re-attribute the digest: it holds.

## Identity

- Base: `b79a4b68`, wheel
  `mlx_omarchy-0.32.2.dev202609151921+b79a4b68-cp314-cp314-linux_aarch64.whl`
  (the 2026-09-15 battery's base; unchanged).
- Cand: `705271f4` (`wave/DecodeEpilogueFold`), wheel
  `mlx_omarchy-0.32.2.dev202609162257+705271f4-cp314-cp314-linux_aarch64.whl`
  sha256 `8e9317cf899bf896fba534c2d6f9abd14d0518c4e68406db6c128f81f2ec2985`,
  built with the jwm1 canonical CMAKE_ARGS (`MLX_BUILD_CPU=OFF`); venv
  payload gate member == loaded.
- Host: jwm1-linux (M1 Max, Honeykrisp), Python 3.14.
- Every GPU step under one `flock` hold on `/tmp/m1-gpu.lock` (inode 35
  before and after; nested `flock -n` refused; never unlinked).
  `63c1d3cf` untouched.

## Reproduce

```sh
ssh jwm1
flock /tmp/m1-gpu.lock /var/tmp/DecodeEpilogueFold/run-jwm1-trigfix.sh
# wheel: ~/src/mlx-epifold-carm scripts-local/build-wheel-jwm1.sh
# test suite: cmake --build .work/mlx/build-tests -j8 \
#   --target omarchy_fused_chain_tests && \
#   ./build-tests/tests/omarchy/omarchy_fused_chain_tests -tc='*RoPE*'
# bit-dump: MLX_OMARCHY_ROPE_BITS=1 on any rope exercise
```

Artifacts in `jwm1/`: `dispatch-{base,cand,cand-off}.json`, `digest.txt`,
`ab.txt`, `ab.json`, `evidence-bits.txt` (ROPEBITS words, PROBE pairs),
`lock.txt`, `started.txt`, `finished.txt`, `run-jwm1-trigfix.sh`.

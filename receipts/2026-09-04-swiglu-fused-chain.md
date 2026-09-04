# FuseDecodeChains: compiled-tape fused elementwise chain (SwiGLU), default off

Date: 2026-09-04. Branch `decode-fusion/swiglu-chain`, two commits:
`ee56928` (mechanism) + `35f2dab` (FusionReview hardening), off
origin/main `999e25c`, in the dedicated worktree
`~/.config/superpowers/worktrees/mlx-omarchy/decode-fusion`. No main
push; root owns merges (including the compute.h enum-tail conflict at
the integration root, and the one residual review fix - see the
hardening section). This lands the tape-interpreter unlock the
decode-fusion-plan receipt named for the swiglu mass, GATED OFF by
default until the M1 hardware A/B below runs.

## Why the decode swiglu chain does not fuse today

`gate * sigmoid(gate) * up` is THREE separate MLX graph primitives
(Sigmoid, Multiply, Multiply). The backend sees one primitive per eval
and cannot see neighbors - precisely why fused RoPE was possible
(fast::rope is ONE primitive) and swiglu was priced as a dead mass
(72 dispatches/token on the 4-bit model, 24 layers x 3) in
`decode-fusion-plan` (`receipts/2026-09-04-decode-fusion-plan.md`).
With mlx-lm's default compile ON, the same three ops sit inside a
Compiled tape whose interpreter dispatches PER NODE
(`eval_compiled_tape`), so compilation changes attribution, not count:
the 4-bit stream is 633 dispatches/token on every leg. The unlock is
inside the tape interpreter: consecutive same-shape float unary/binary
tape nodes are a straight-line program and can run as ONE interpreted
dispatch.

## What shipped

- `overlay/mlx/backend/omarchy/fused_chain.{h,cpp}` (new):
  `FusedChain` - packs carried tape nodes into an instruction program
  (op | a | b | dst words; max 8 instructions, 3 leaf buffers) and
  dispatches one kernel. Operand forms: the previous member's register
  or a contiguous leaf with direct / scalar / mod-last / div-last
  addressing (covers swiglu's [n] gate/up leaves, [K] scales, [B,1]
  rows, and scalars). Storage offsets are carried per leaf.
- `overlay/mlx/backend/omarchy/shaders/fused_chain.comp` (new): the
  interpreter. f32 registers, 8 per invocation, grid-stride loop.
- `overlay/mlx/backend/omarchy/compiled.cpp`: the tape interpreter
  opens a chain on a fusable node, extends it only through
  tail-register consumers, and closes it at any non-join point, tape
  output, or multi-consumer boundary (consumer counts computed from
  the trace; interior members are unreachable to non-chain ops by
  construction). The census instrument, the debug switches, the
  per-node counters, and the shapeless-tape eval-time output shapes
  are all preserved; a fused dispatch feeds the same
  `compiled_tape_dispatches` accounting.
- `overlay/mlx/backend/omarchy/compute.{h,cpp}`: `FusedChainF32` /
  `FusedChainF16` kernel enum + dispatch.
- `overlay/mlx/backend/omarchy/CMakeLists.txt`: shader + source.
- `overlay/tests/omarchy/test_fused_chain.cpp`,
  `overlay/tests/omarchy/decode_dispatch_count.cpp` (new): the
  evidence suites.

Zero `primitives.cpp` hunks (coordinated with Bf16DecodePath, whose
RoPE/SDPA hunks are untouched by this). RoPE/SDPA/quantized paths
untouched.

## Semantics preserved, and how false passes are excluded

- **Dtype rounding**: the per-node path rounds EVERY node output to
  the storage dtype (each node materializes an f16/f32 buffer). The
  chain shader applies the identical rounding to each register write
  (`ROUND_INTERMEDIATE`), so a fused chain is BIT-EXACT against the
  unfused path - the f16 equivalence case compares with `==` (epsilon
  0), not a tolerance band. The prior draft of this work kept
  intermediates in f32 and needed a 2e-3 f16 band; that slack is gone.
- **Op formulas mirrored case-for-case from elementwise.comp**:
  sigmoid is the UNGUARDED `1/(1+exp(-x))` (the draft's sign-guarded
  form produces inf/inf NaN for x >~ 88 where the per-node path
  returns 1.0); Maximum/Minimum use the NaN-PROPAGATING comparator
  form (plain GLSL max/min would return the non-NaN operand,
  silently diverging from MLX semantics).
- **Special values, no finite-input false pass**: the special-values
  case feeds 0, +-1, +-20, +-88, +-89, +-100, +-inf, NaN, 65504, 5e-8
  through swiglu and max/min compositions in f32 AND f16 and compares
  WIDENED BIT PATTERNS (uint32), so a NaN/inf mismatch cannot hide
  behind `==`, a tolerance, or an abs().
- **The fused path cannot silently not-run**: both f32 and f16
  equivalence cases first assert the compiled 3-op evaluation costs
  exactly ONE dispatch (a per-node fallback would pass value checks
  while fusing nothing).
- **Offsets and broadcasting**: sliced (offset) leaves, tiled [K]
  leaves, [B,1] rows, and scalar constants each have a bit-exact case.
- **Compile-disabled behavior**: eager (`CompileMode::disabled`) is
  the reference for every equivalence case; the fused path only exists
  inside compiled tapes, so `MLX_DISABLE_COMPILE=1` streams are
  structurally untouched.
- **bf16 refusal untouched**: `unsupported_tape_bfloat16()` still
  fires before chains are ever considered (tested).
- **Loud refusals moved, not lost**: leaf span/bound checks run at
  `try_add` time - a chain that cannot dispatch is never opened, so
  `evaluate()` cannot drop carried nodes (the draft refused at
  dispatch time, which would have thrown a MISLEADING "op not
  implemented" error after nodes were skipped).

## FusionReview hardening (35f2dab)

1. **Extensions require the tail register.** A same-shape node with
   no data dependency on the open chain's tail used to join as a
   NON-tail interior member; closing the chain materializes the tail
   only, so a valid graph like `x*x + y*y` resolved nothing for the
   first mul and hit the loud tape refusal. The sibling is now
   refused as an extension (it closes the chain and opens a fresh
   one). New bit-exact test for exactly that graph.
2. **Grid stride** in `fused_chain.comp`, mirroring
   `elementwise.comp`: the dispatcher caps group count at
   `kMaxComputeGroupCountX`, and the naive tail vanished past the
   cap. Proven the napkin way (2026-08-31): a DIRECT ONE-GROUP
   dispatch of `FusedChainF32` over 600 elements, EVERY value beyond
   the first 256 compared bitwise against eager - no large
   allocations, no hazard reproduction.
3. **Leaf rollback**: leaves pushed for a REJECTED `try_add` are
   erased (erase, not resize - `array` is not default-constructible),
   so a refused add leaves no orphan slots, empty chain included.
4. **Zero off-path cost**: `MLX_OMARCHY_FUSED_CHAIN` is read ONCE per
   tape evaluation and handed to the chain; with it off the
   interpreter builds no consumer map / materialize set and runs the
   pre-chain per-node loop byte for byte.

Known residual, fixed at the INTEGRATION root by the parent (not on
this branch): the f16 capability refusal sits after leaf encoding on
35f2dab, so that one path can leave rolled-back-less leaves; the
parent moved the capability check before leaf encoding in the
integration cherry-pick.

## The gate

`MLX_OMARCHY_FUSED_CHAIN` (env_flag truthy: 1/on/true/yes). DEFAULT
OFF: the gate is read once per tape evaluation, `try_add` refuses
everything when it is off, and the tape runs the per-node path
byte-for-byte as before this change. No default-on flip until the
hardware gate below is green.

## Evidence (llvmpipe: dispatch counts and structure only; NO speed claims)

On 35f2dab: `omarchy_fused_chain_tests`: 14/14 cases, 3000 assertions
green (the twelve original cases plus the independent-sibling case
and the forced one-group stride case; includes the 3->1 dispatch
pins, f16/f32 bit-exact equivalence, bitwise special values, offset
and broadcast leaves, gate-off parity, bf16 refusal).
`omarchy_compiled_tape_tests`: 11/11, 1747 assertions.
`omarchy_runtime_tests`: 26/26, 6249 assertions.

`omarchy_decode_dispatch_count` (faithful Qwen2.5-0.5B decode token,
batch 1, fp32 structure; 24 layers), re-run on 35f2dab:

```
decode dispatches/token (layers=24): eager=482 compiled_pernode=482 compiled_fused=434
```

- compiled per-node == eager exactly: the interpreter change is
  stream-neutral with the gate off.
- compiled fused = per-node - 48 = -2/layer: exactly the swiglu
  (sigmoid + mul + mul -> 1) collapse, nothing else. Asserted by the
  test (`CHECK_EQ(compiled_fused, compiled - 2 * n_layers)`).
- The measurement is identical to the ee56928 run: the stride shader
  and gate plumbing change no dispatch counts.

On the real 4-bit model this is the -48/token predicted by the
decode-fusion-plan receipt (633 -> ~585); the receipt's numbers stand
as the model-level baseline.

## M1 hardware A/B recipe (root; M1 is exclusively root's)

1. Build a diagnostics wheel from this branch (or the integration
   root carrying it) on jwm1: `bash scripts/build-wheel.sh
   --diagnostics`, install into a fresh venv with
   `mlx-lm==0.31.3 --no-deps` + numpy (2026-09-03 provisioning
   rule), verify `libmlx.so` sha256 wheel-member == installed
   (provenance gate).
2. Correctness gate FIRST (unfence bar):
   a. `MLX_OMARCHY_FUSED_CHAIN=1` C++ battery on the Apple GPU:
      `omarchy_fused_chain_tests` must be 14/14 BIT-EXACT on
      Honeykrisp (the shader uses no uint16 block IO and no
      per-element RMW, the two documented llvmpipe/M1 hazards;
      still, Honeykrisp is the judge).
   b. Greedy identity: pinned prompt "What is the capital of
      France?", temp 0, seed 0, 32 tokens,
      `Qwen2.5-0.5B-Instruct-4bit` snapshot `a5339a41...`: token
      stream with the gate ON must equal the stream with the gate
      OFF (bit-exact shader semantics predict identity, not just
      plausibility).
3. Dispatch A/B: `MLX_OMARCHY_GPU_PROFILE=/tmp/prof-{off,on}.jsonl`
   for both legs over `scripts/profile_generate.py`, then
   `scripts/chain_census.py` against the compute.h enum: gate OFF
   must reproduce 633/token; gate ON must show ~585/token with
   ElementwiseF16 down 48 and every other kernel count unchanged.
4. Performance A/B: same pinned protocol as the release-basis table
   (median tokens/sec over >=3 runs, harness commit recorded). The
   claim to price is one fused kernel replacing three on a
   4864-wide f16 buffer 24x per token. llvmpipe numbers above are
   counts, never speed.
5. If 2a, 2b, 3, and 4 are green: flip the default by changing the
   single gate decision in `eval_compiled_tape`
   (`env_flag("MLX_OMARCHY_FUSED_CHAIN")`) to default-true (or
   inverting the gate into an opt-OUT), update
   `docs/install-omarchy.md` env docs, and record the flip in a
   receipt. Until then this stays opt-in.

## Reproduce (this host, llvmpipe)

```bash
cd ~/.config/superpowers/worktrees/mlx-omarchy/decode-fusion
git checkout 35f2dab
bash scripts/prepare-mlx.sh
cmake -B .work/build -S .work/mlx -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build .work/build -j4 --target omarchy_fused_chain_tests \
  omarchy_decode_dispatch_count
MLX_OMARCHY_ALLOW_NON_APPLE=1 .work/build/tests/omarchy/omarchy_fused_chain_tests
MLX_OMARCHY_ALLOW_NON_APPLE=1 .work/build/tests/omarchy/omarchy_decode_dispatch_count
```

Provenance: llvmpipe via `MLX_OMARCHY_ALLOW_NON_APPLE=1` (software
Vulkan, Mesa 22.3.6-1+deb12u2, LLVM 15.0.6), x86_64 Linux
6.17.2-1-pve, 16-core EPYC, shared dev host (builds capped -j4);
dispatch COUNTS and bit-level equality only - no timing or Apple-GPU
performance claim is made or derivable from this receipt.

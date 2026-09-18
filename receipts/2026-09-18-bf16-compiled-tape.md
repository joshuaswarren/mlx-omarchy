# bf16 compiled-tape refusal lift — jw16 (M1 Max T6001) — 2026-09-18

Verdict: **GREEN with one named fence.** The tape-level bfloat16 refusal
(`[omarchy] Compiled tape bfloat16 is refused`) is removed: bf16 compiled
tapes execute, and every current-gen model that previously demanded
`MLX_DISABLE_COMPILE=1` generates with compile ON, coherent, with
generated-id digests identical to eager. Root cause of the original
refusal: it fenced the stale-shape corruption (root-caused at `13d83f7`
one day after the gate was installed), not a dtype defect. One residual:
fused bf16 chains corrupt in-model on three models and are fenced to the
per-node path (bit-exact) pending a pinned mechanism
([known-defects.md](../docs/known-defects.md), Live in v0.3.5). F7GdnCorrectness's
independent DivLast leaf-matcher root cause in the same file may be the
mechanism; if their fix verifies on the corrupt matrix, the fence lifts.

## Root cause of the original refusal (receipt archaeology)

- The gate was installed 2026-09-01/02 after M1 mlx-lm bf16 garbage
  (`receipts/2026-09-02-m1-bf16-compiled-tape.md`). That receipt's own
  evidence - prefill bit-identical through all 24 layers, divergence at
  decode step 2, unique garbage per run, llvmpipe clean, every
  fixed-shape probe clean, and the "broadcast Sigmoid bf16" refusal on
  shape change - is the stale-shape defect signature. The stale-shape
  root cause landed `13d83f7` the next day; the 2026-09-03 receipt
  names the Sigmoid refusal "the stale-shape symptom" (hypothesis 8).
  The bf16 gate was never retested after the fix and shipped as "a
  separate, still-live bf16 defect". The dtype was never at fault.

## Change (branch `bf16-tape-gate-lift`, off main `bc2f1fc6`, c136912f lineage)

| commit | content |
| --- | --- |
| `cb8c0638` | tape-level refusal removed (`tape_has_bfloat16`/`unsupported_tape_bfloat16` deleted); three refusal-pinning tests flipped to bit-exact bf16 tape tests incl. a new bf16 shapeless trace-then-reuse case |
| `ef85dc72` | bf16 chains fenced from tape fusion after in-model corruption was found (see below) |
| `11f2d3b5` | fence scoped to the tape interpreter (compiled.cpp skips `try_add` for bf16 nodes); FusedChain + eager planner byte-for-byte stock - the `can_start` fence had regressed the proven eager bf16 swiglu single-dispatch path (caught by the dispatch-count pins) |
| `79b15be2` | docs: compatibility.md rewrite, known-defects live entry, matrix/install/guide rows |

Ancestry: `git merge-base --is-ancestor 63c1d3cf HEAD` exit 1 = **PASS**
(repeated at `cb8c0638` and on the pushed branch); `c136912f` is an
ancestor. Pushed: `origin/bf16-tape-gate-lift` at `11f2d3b5`.

## Verification (T6001 Honeykrisp, jw16)

Unit (C++, raw-uint16 bit compare — the bf16 bit pattern is the contract):

| battery | result |
| --- | --- |
| `omarchy_compiled_tape_tests` (incl. bf16 widened set + bf16 shapeless reuse) | 12/12 cases, 2096/2096 assertions |
| poison arm (`MLX_OMARCHY_POISON_FREED=1`) | 12/12, zero signatures |
| `omarchy_fused_chain_tests` | 33/33 (pre-fence bit-exact proof), 31/33+2 planner dispatch-count pins caught the `can_start` over-fence → `11f2d3b5` |
| `omarchy_primitive_tests` | 103/103 |

Upstream suite: `test_compile.py` **68/68** (the three named-refusal
cases — `test_compile_nonfinite_constants`,
`test_compiled_subnormal_bool_cast`, `test_inf_constant` — now pass).

Note: the first bit-compare attempt via `doctest::Approx(...).epsilon(0)`
reported false mismatches on all-zero rows on BOTH llvmpipe and T6001;
raw uint16 comparison shows the buffers are identical — the host
comparison was the broken instrument (the ledger's rule again).

Model matrix (mlx-lm 0.31.3, greedy temp 0, fixed prompt, digest =
sha256[:16] of generated ids; compiled = default compile ON, eager =
`MLX_DISABLE_COMPILE=1`):

| model | mode | reps | digest | vs eager digest | tok/s | output |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct-bf16 (the original corruption model) | compiled | 3 | `c2d5348ed63fb217` x3 | **identical** | 49.6-51.7 | coherent |
| | eager | 2 | `c2d5348ed63fb217` x2 | — | 50.5 | coherent |
| Qwen2.5-0.5B-Instruct-4bit | compiled | 1 | `b7002ae46ed3157c` | — | 173.8 | coherent |
| Ternary-Bonsai-8B-mlx-2bit | compiled | 1 | `66a0f9e93353d85e` | — | 5.8 | coherent |
| Qwen3.5-9B-MLX-4bit | compiled (fenced) | 2+1 | `910abe30d4305271` | **identical** | 10.7-12.9 | coherent `<think>` text |
| | eager | 1 | `910abe30d4305271` | — | 11.1 | coherent |
| gemma-4-31b-it-4bit | compiled (fenced) | 1 | `20b1cc9f572ef2c8` | **identical** | 2.5 | = eager (raw prompt; template artifact, not numerics) |
| | eager | 1 | `20b1cc9f572ef2c8` | — | 2.5 | |
| Ministral-3-8B-Instruct-2512-4bit | compiled (fenced) | 1 | `d4735e3a265e16ee` | **identical** | — | = eager (EOS at raw prompt; template artifact) |
| | eager | 1 | `d4735e3a265e16ee` | — | — | |

**Compile ON changes no generated id on any model in the matrix.** The
unfenced fused bf16 path corrupted Qwen3.5-9B, gemma-4-31B, and
Ministral-3-8B deterministically (`FUSED_CHAIN=0` restored the exact
eager digest on Qwen3.5) — fenced by `ef85dc72`/`11f2d3b5`. Poison
armed through a full wrong-output run: no signature, so not recycled
storage. Isolated fragments (pure-bf16, cast-mixed, shapeless
reuse, offset views, 3-D model shapes) all bit-exact — the mechanism
needs the full-model context. Compiled-vs-eager tok/s is parity within
noise on this stack (consistent with TCF-2), so the table records
correctness-at-parity, not a speedup claim.

Bonsai-2-27B: not re-run on jw16 this window (jw14m2, its qualification
host, was unreachable all session; F7GdnCorrectness's T6021 run of
their da43969e wheel - which supersedes the fence's reason - gives
96-step greedy clean, "Paris", 1.87 tok/s).

## Open items recorded honestly

1. **Parakeet E2E 104/104 + the four AC/ACO arms were NOT re-run
   against the fence wheel** (window budget; llm-inference restored
   for F7's urgent window leg). My branch does not touch the f16/f32
   chain paths, the ANE pipeline, or the runner, and the arms
   previously passed from main bytes; they must be re-certified from
   this branch's wheel before release tagging.
2. **fc battery dispatch-count pin** ("nonidentity broadcast keeps the
   per-node fallback (shapeless)", f32, EAGER leg): 2 vs expected 1 on
   my branch, 1 on stock main - isolated A/B on jw16, deterministic.
   My branch's only libmlx delta is inside `eval_compiled_tape`, which
   the eager leg does not execute; the mechanism is not understood and
   is recorded rather than explained away. F7GdnCorrectness's
   leaf-matcher work in the same file is the active investigation.
3. **Wheel provenance**: the verification wheel
   `0.32.3.dev202609182240+79b15be2` was built on jw16 from a tree
   whose files match `11f2d3b5` (source greps verified: scoped fence
   present in compiled.cpp, fused_chain.cpp stock) while the git HEAD
   stamp reads 79b15be2 (jw16 fetches via bundle; the last commit was
   scp-synced). Behavior discriminator: the fc dispatch-count pin
   above fails on 11f2d3b5 content and passes on stock.

F7GdnCorrectness's independent root cause in the same file
(`da43969e`, leaf_mode_for DivLast misindexing on broadcast leaves)
is the prime suspect for the in-model fused-bf16 corruption; if their
fix verifies on the corrupt matrix with fusion ON, the fence lifts in
its own commit with a fresh digest sweep. That sweep plus the Parakeet
re-certification are the remaining bars between this branch and
release tagging.

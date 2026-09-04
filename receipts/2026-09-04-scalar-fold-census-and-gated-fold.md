# Scalar fold: the count==1 census, a gated host fold, and its blocker

Date: 2026-09-04. Branch `scalar-folding` stacked on `fused-rope` @
`3c32cce`. Measured on llvmpipe (Mesa LLVM 15.0.6) via
`MLX_OMARCHY_ALLOW_NON_APPLE=1`, Qwen2.5-0.5B-Instruct-4bit snapshot
`a5339a41`, mlx-lm 0.31.3, greedy, prompt "What is the capital of
France?", diagnostics wheels (profiling on).

## Census first: the 290, attributed

Reproduced the baseline exactly on origin/main @ `238a977` (worktree
`/tmp/sf-main`, same protocol as the elementwise-chain census):
**1,953 dispatches/token median, 290 count==1**. Attribution by kernel +
operation code (scripts/scalar_census.py, per median token):

| kernel | op | dtype | n/token | operand residency |
|---|---|---|---|---|
| ElementwiseF32 | Add | f32 | 48 | inputs: fold-chain scalars |
| ElementwiseF32 | Multiply | f32 | 48 | positions + scale constant |
| ArangeF32 | arange(1) | f32 | 48 | no inputs (alpha constant) |
| CastI32F32 | offset cast | i32->f32 | 48 | input: host-written scalar constant |
| ReduceF32 | trig-arg gate max | f32 | 96 | input theta (32e) in-flight at gate time |
| LogSumExpF16 | sampler lse | f16 | 2 | input logits 151,936e device-resident |
| ArgReduceF16 | sampler argmax | f16 | 2 | input logits 151,936e device-resident |

Residency verdict: 192/token (the positions chain) have all-constant or
fold-chain inputs. 96/token (the sin/cos trig-argument gate's max
reduce) have an in-flight input - the gate already synchronizes and
reads on host, but the reduce input is not host-final when the gate
runs. 2/token (sampler) read a 151,936-element device buffer: folding
them is exactly the forced sync this backend cannot afford.

On the integrated tree (`fused-rope` @ `3c32cce`) the rope composition
is gone and with it the whole 192+96 mass: measured decode is **753
dispatches/token (f16) / 774 (bf16, MLX_DISABLE_COMPILE=1), count==1 =
2/token (sampler only, both dtypes)**. Evaluations per token (ctypes
trace snapshot, diag wheel): 806 evals / 751 device dispatches (f16).
The origin/main tree predates `mlx_omarchy_trace_snapshot`, so the
2,682-eval baseline could not be re-measured there with this
instrument.

## The profitability rule

> Fold iff every input's bytes are final on the host at eval time, with
> no pending GPU dispatch behind them; if any input needs a device read
> to become final, keep the dispatch.

Reviewer check: `host_final()` in scalar_fold.cpp mirrors the fold
guards exactly - constants (no primitive), this fold's own outputs
(mirror-named, size/dtype-guarded), nothing else. Event and status
shortcuts are deliberately absent: a dispatched array can report a
signaled event while its dispatch sits in an open command buffer
(measured: an exp+add repro returned 1.0 through an event-true read).

## What upstream provides

Nothing folds single-element ops host-side. Upstream gives: scalar
arrays host-written at creation (Metal shared storage; this backend's
Load idiom), `CopyType::Scalar` host reads behind a synchronize
(copy.cpp), the trig gate's existing host read, and `use_fallback` to
route fast:: primitives to composed graphs (which still dispatch per
primitive). The fold generalizes the existing scalar_read idiom into
scalar_compute; it extends rather than parallels anything upstream.

## Implementation (opt-in: MLX_OMARCHY_SCALAR_FOLD=1)

scalar_fold.cpp/h, hooked in gpu::eval() after the trace counters.
Fold set = the measured host-final subset: Arange f32 (size 1,
formula-replicating `alpha + beta*0.0f` - the -0.0 case is load-bearing),
AsType i32->f32 (size 1), Add/Multiply f32 (size 1, NaN operands
excluded: NaN payload propagation is implementation-defined and
differs between host SSE and the shader). Fresh `allocate_omarchy`
allocation per output, synchronous host write, never in-place on
inputs; evaluated state (set_data flags) matches the normal path.

Exact-match evidence: omarchy_scalar_fold_tests, 4 cases / 1,005
assertions, differential against the shader leg (size-2 element 0 =
identical per-element math, no fold) over +/-0, denormals, infs, NaN
payloads, 2^24 boundary: all bit-identical where the fold fires; the
NaN-pair cases assert the kernel's bits after the fold refuses them.
Dispatch-avoidance is asserted via vk_compute_dispatches deltas (0 for
folded size-1 evals incl. the full rope positions chain; 1 for size-2
and in-flight-input cases).

## Measured (llvmpipe)

| tree | before | after (gate on) | wall/token |
|---|---|---|---|
| origin/main (rope composition) | 1,953 disp/tok, 290 count==1 | **1,761 disp/tok, 98 count==1** | 4,253 -> **1,922 ms** (2.2x) |
| fused-rope 3c32cce (integrated, f16) | 753 disp/tok, 2 count==1 | 753 / 2 (gate fires nowhere in decode) | ~unchanged |
| fused-rope 3c32cce (bf16, no compile) | 774 disp/tok, 2 count==1 | not re-run (fold fires nowhere) | - |

Gate off: 753/2 reproduced byte-identical to the pre-fold tree.
Batteries on the gated build: scalar_fold 1,005/1,005; runtime
6,247/6,247; eq_math 116/116; compiled_tape 875/875.

## Why default-off: the unresolved ordering edge

test_runtime.cpp "small eager output stays ordered across deep submit
boundaries" fails with the fold enabled: a folded host write consumed
by a dispatch across deep submit boundaries reads a recycled page's
stale contents (thetas came back 0,0,2,6,12,20,30,42 = iter*(iter-1)).
The consumption pattern (host-written data read by a kernel submitted
after intervening heavy submissions) needs an encoder-level fix - pin
or drain at the fold's write - which belongs to the temporaries-flush
contract owner. The measured decode chains fold fully host-side or
within one batch, so the win is real; the edge blocks default-on.
RETRACTED (same night): an earlier version claimed a related pre-existing defect (`add(exp(0.5), 1.0)` garbage on 238a977 without the fold). The control run was contaminated - the worktree venv had a fold-carrying wheel reinstalled over the clean one, so the reproduction exercised this fold's since-removed is_available shortcut. Clean-tree hunts do not reproduce eager garbage on 238a977.

## M1 protocol for BenchQueueM1 (consolidated)

Model snapshots pinned above + bf16 snapshot `56d07e76`; mlx-lm 0.31.3;
greedy temp 0 seed 0; prompt "What is the capital of France?"; 32
max tokens; llvmpipe numbers are the llvmpipe receipts above. On jwm1:

1. Build this branch's wheel with scripts/build-wheel.sh --diagnostics.
2. Legs: (a) f16 4bit gate off, (b) f16 gate on
   (MLX_OMARCHY_SCALAR_FOLD=1), (c) bf16 gate off
   (MLX_DISABLE_COMPILE=1), (d) bf16 gate on. 32 tokens, median.
3. Record: decode tok/s, prefill tok/s, dispatches/token
   (MLX_OMARCHY_GPU_PROFILE + scripts/chain_census.py or
   scripts/scalar_census.py), evaluations/token (ctypes
   mlx_omarchy_trace_snapshot), plus the ordering test
   (omarchy_runtime_tests) under both gate states.
4. Keep rule: keep default-off unless (i) Honeykrisp shows no ordering
   regression on the gated legs AND (ii) the gated f16 leg beats gate
   off on tok/s at >=5% with dispatches/token not worse; if the
   ordering test is green on both states, prefer default-on only after
   the encoder-level pin/drain fix lands.

## Reproduce

```bash
# census on any tree
MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_GPU_PROFILE=/tmp/p.jsonl \
  venv/bin/python scripts/profile_generate.py --model <snapshot> \
  --prompt "What is the capital of France?" --max-tokens 32 --temp 0 \
  --seed 0 --markers /tmp/m.jsonl
python3 scripts/scalar_census.py /tmp/p.jsonl \
  --compute-h overlay/mlx/backend/omarchy/compute.h --markers /tmp/m.jsonl

# fold battery (opts itself in)
cmake --build <tests-build> --target omarchy_scalar_fold_tests
MLX_OMARCHY_ALLOW_NON_APPLE=1 ./tests/omarchy/omarchy_scalar_fold_tests
```

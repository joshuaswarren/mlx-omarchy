# Compiled-forward chain census: chains inside tapes, fusable ceiling

Date: 2026-09-04. MEASUREMENT ONLY - no fusion engine built.

Provenance (every number below comes from this environment):
`mlx-omarchy 0.32.2.dev202609040343+diag.79c2585, libmlx sha256
dbfa9f15e6b07148bec6f388d80d38d5293db5a27ea9a399795d4dc8433f8525 +
mlx.core extension a53f742538df765a0572a680a3bd5edee4d442e619752b41af1de5aecfcbc988,
wheel RECORD verified=match, mx.__version__ == dist_version,
mlx-lm 0.31.3, model mlx-community/Qwen2.5-0.5B-Instruct-4bit snapshot
a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3, llvmpipe (Mesa LLVM 15.0.6)
via MLX_OMARCHY_ALLOW_NON_APPLE=1, Linux 6.17.2-1-pve x86_64,
harness commit 79c2585`
(`scripts/mlx_provenance.py` JSON, `verified: "match"` on both binaries).

## Tree this describes

Worktree off `origin/main` (`298cd90` - main WITH the fused fast::RoPE,
WITHOUT the SDPA f16-scores branch), branch `ccc/compiled-chain-census`,
plus the two instrument commits cherry-picked from `ew-fused-elementwise`
(`414bb5c` census counters + `79c2585` probe). The instruments are
env-gated and add no dispatches when off; dispatch counts below describe
main's decode behavior. The eager-leg median (751) matches the 753
protocol number for main-with-RoPE, confirming the tree identity.

## What was built

The combination that had never been run: `MLX_OMARCHY_TAPE_CENSUS`
(tape wiring dump) against `scripts/probe_compile_forward.py` (decode
layer re-expressed functionally, whole layer wrapped in `mx.compile`),
plus a tape-internal chain analysis mode in `scripts/chain_census.py`
(`--tapes DUMP`, with `--self-test`). The dump gives, per tape
evaluation: node list, op, dtype kind, tape-wide use count, input
wiring, output flag.

Command:
```bash
MLX_OMARCHY_ALLOW_NON_APPLE=1 \
MLX_OMARCHY_GPU_PROFILE=/tmp/ccc-prof.jsonl \
MLX_OMARCHY_TAPE_CENSUS=/tmp/ccc-tapes.txt \
  ./.work/venv-run/bin/python scripts/probe_compile_forward.py \
  --model <snapshot a5339a41> --markers /tmp/ccc-markers.jsonl \
  --max-tokens 32
python3 scripts/chain_census.py /tmp/ccc-prof.jsonl --compute-h \
  .work/mlx/mlx/backend/omarchy/compute.h --markers /tmp/ccc-markers.jsonl
python3 scripts/chain_census.py --tapes /tmp/ccc-tapes.txt
```

## Result 1: dispatches per token, tape vs eager (median, 32 tokens/leg)

| leg | dispatches/token | tape-path | eager |
|---|---|---|---|
| eager (real mlx_lm loop) | 751 | 72 (9.6%) | 679 |
| compiled layers (hypothetical) | 991 | 360 (36.3%) | 631 |

One eager-leg outlier token (1,550 dispatches, 144 tape) is a cache
growth event; medians exclude nothing but are unaffected. Compile makes
the total WORSE on this backend (751 -> 991): the functional full-slot
attention (arange/compare/where over the whole cache) adds ~240 eager
dispatches, reproducing the earlier finding that compiled tapes alone
collapse nothing.

## Result 2: what the tapes actually contain

`is_fusable` upstream places only unary/binary/ternary/Broadcast
primitives in a Compiled tape. With fast::RoPE a real primitive (not a
composition), NO Qmm/RmsNorm/RoPE/SDPA node enters any tape: the
compiled layer decomposes into 8 tiny clusters per layer per token,
24 layers => 192 tape evaluations/token, 264 dispatch-recording nodes
(360 GPU dispatches on the tape path: each SelectF16 carries a
LogicalOrBool + ClearU32 rider, +96). Six unique tapes in the whole run:

| tape | nodes | dispatch nodes | evals | role |
|---|---|---|---|---|
| Sigmoid+4xBroadcast+2xMultiply | 7 | 3 | 816 | standalone shapeless swiglu (eager legs) |
| Sigmoid+2xMultiply | 3 | 3 | 768 | swiglu inlined in compiled layer |
| 3xBroadcast+LessEqual+Select | 5 | 2 | 768 | attention mask |
| 2xBroadcast+Select | 3 | 1 | 1536 | kv cache where-select |
| Broadcast+Add | 2 | 1 | 2304 | residual add (x3/layer) |
| Broadcast+Equal | 2 | 1 | 768 | positions==offset |

## Result 3: chain-length histogram INSIDE tapes (per decode token)

Chain = maximal run of fusable nodes where every INTERIOR node's
consumers lie within the run (sole-consumer rule); Broadcast views ride
free. Length counted in dispatch-recording nodes.

Eager leg (72 tape dispatches/token in the standalone swiglu):
```
len=1: 72 chains   len=0 (views only): 48
```
Compiled leg (264 tape dispatch nodes/token):
```
len=1: 144 chains   len=2: 24   len=3: 24   len=0 (views only): 72
```
The len-3 chains are the inlined swiglu (Sigmoid->Mul->Mul, uses=1
everywhere); the len-2 chains are the mask (LessEqual->Select). Each
layer contributes exactly one of each.

## Result 4: fusable ceiling, multi-consumer blocking quantified

Sole-consumer rule:
- Eager leg: **0 of 751 dispatches/token (0.0%)**. Every interior of
  the only tape has uses=2 (the two Broadcast views feed each Multiply),
  so a sole-consumer fuser fuses nothing - ElementwiseFusion's finding,
  reproduced from the dump.
- Compiled leg: **72 of 991 dispatches/token (7.3% of the token; 20.0%
  of the 360 tape-path dispatches; 27.3% of the 264 tape
  node-dispatches)**. Nothing else in any tape is fusable.

Multi-consumer blocking:
- Compiled leg: **zero chains blocked**. All 8 clusters per layer are
  sole-consumer-clean; the region rule (multi-consumer capable) saves
  exactly the same 72/token - **the harder fuser buys nothing**.
- Eager leg: the one blocked structure is the standalone swiglu region
  (interiors uses=2): 1 region blocked per tape, 24 tapes/token. A
  multi-consumer-capable fuser would save 48/751 = 6.4% there - the
  ONLY place the harder rule pays.

Mechanism finding worth keeping: the SAME swiglu is unfusable as a
standalone shapeless compile (the tracer materializes Broadcast views
with uses=2) and perfectly fusable inlined in a layer compile at exact
shape (views folded away at trace time). Whether a fuser has work
depends on compiler decisions upstream of it, not on the model.

## Comparison (one sentence)

Under the compiled forward, chains inside tapes reach length 3 with a
sole-consumer fusable ceiling of 72 of 991 (7.3%) dispatches/token,
versus effective chains of length 1 and a ceiling of 0 of 751 (0.0%)
on the eager path mlx_lm actually runs - where the only tape's interiors
all have uses=2 and only a multi-consumer-capable rule (6.4%) could
fuse anything.

## Fidelity caveat

The compiled leg is a functional re-expression of the decode layer
(explicit cache in/out, full-slot where-select, additive -inf mask,
host-side offsets), not what mlx_lm runs - and this run diverged from
eager greedy after token 1 again (parity False; compiled text coherent
but repetitive). The census therefore describes a hypothetical loop. It
is faithful in tape STRUCTURE (same ops, same shapes, decode-shape
traces after the one prefill->decode retrace), but (a) a real
compile-the-forward integration would need upstream functional-cache
changes whose op mix may differ, (b) the +240 eager full-slot dispatches
inflate the denominator, so 7.3% is the ceiling of THIS re-expression,
and (c) token positions after the divergence differ, which does not
change per-token tape structure at fixed shapes.

## Verdict

Generic tape-level fusion is not worth building.

1. In the loop that actually runs, the fusable ceiling is 0.0%
   (sole-consumer) or 6.4% (multi-consumer-capable, swiglu only).
2. Even in the maximally favorable hypothetical (every layer compiled),
   the sole-consumer ceiling is 7.3% of dispatches, and the
   multi-consumer-capable rule adds exactly zero on top - the larger,
   riskier fuser has no additional ceiling to chase.
3. Upstream's own fuser already emits the maximal fusable clusters; the
   compiled clusters ARE the chains, and they are 1-3 dispatches each.
   The remaining 90%+ of decode dispatches (Qmm, SDPA pair, RMSNorm,
   RoPE, kv copies, casts, riders) are exactly the dedicated-kernel
   targets already named in the earlier receipt.

Stop rule fires: stay on dedicated kernels.

## Reproduce

```bash
git worktree add ../mlx-ccc origin/main -b ccc/compiled-chain-census
cd ../mlx-ccc
git cherry-pick c9fa0a5 0440dd3   # census instrument + probe, from ew-fused-elementwise
bash scripts/build-wheel.sh --diagnostics
python3 -m venv .work/venv-run
./.work/venv-run/bin/pip install dist/mlx_omarchy-0.32.2.dev202609040343+diag.79c2585-*.whl "mlx-lm==0.31.3" numpy
# then the three commands under "What was built"
./.work/venv-run/bin/python scripts/chain_census.py --self-test
```

Self-test: all assertions pass. Raw artifacts: `/tmp/ccc-prof.jsonl`
(57,339 events), `/tmp/ccc-tapes.txt` (6,960 tape evaluations),
`/tmp/ccc-markers.jsonl` (66 markers).

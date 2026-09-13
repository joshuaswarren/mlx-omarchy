# Parakeet encoder op inventory vs mil-hwx-compiler H13 coverage (2026-09-13)

Plan sections 18 and 47: compare the actual public Parakeet encoder op
inventory against current compiler coverage. Host-only; no hardware.

Inputs, both pinned:

- encoder: `encoder.mlpackage` at HF revision
  `b650695c2322ee5281dff48d7345b2f3a58ff018` — 3351 ops, 29 distinct
  op types, live histogram from `coreml.mlpackage.inspect()`
- compiler: locked source at commit
  `83a4434810b0981d1764e6231f6a679df6f35241` (ane-compiler.lock,
  release `ane-parity-83a4434`), H13 lowering surface extracted from
  `plugins/H13/ANEH13Compiler.mm` and `lib/IR/ANEOperationGraph.mm`

Machine-readable result: `encoder-coverage.json`. Build log:
`build-inventory.log`.

## Headline

- **supported**: 16 op types, 2866 of 3351 ops (85.5%)
- **unsupported-with-owner**: 13 op types, 485 ops (14.5%)
- **unsupported-blocking**: none

No op in the encoder lacks an owner able to implement it under the
plan's ownership rules (§19: compiler repo owns missing-op semantics;
the mlx-omarchy Core ML frontend owns package-level weight semantics).

## Classified table

| op | count | class | owner / evidence |
|---|---:|---|---|
| const | 1783 | supported | ANEOperationGraph.mm:6 Constant kind; consumed for folding/weights/splats at ANEH13Compiler.mm:145,174,235,396,606,1518 |
| constexpr_lut_to_dense | 194 | unsupported-with-owner | mlx-omarchy Core ML frontend preferred: pre-expand palettized weights to dense fp16 consts before compiler input (compiler registry knows only `constexpr_affine_dequantize`) |
| linear | 194 | supported | ANEH13Compiler.mm:1748 (lowers to matmul `transpose_y=true` + folded bias) |
| add | 183 | supported | ANEH13Compiler.mm:64 |
| transpose | 146 | unsupported-with-owner | mil-hwx-compiler: IR-known (`ANEOperationKindLayout`) but no H13 lowering branch; fold into matmul/conv operands or lower as layout op |
| reshape | 145 | supported | ANEH13Compiler.mm:157 |
| mul | 128 | supported | ANEH13Compiler.mm:65 |
| layer_norm | 120 | supported | ANEH13Compiler.mm:91 |
| conv | 77 | supported | ANEH13Compiler.mm:1450 |
| matmul | 72 | supported | ANEH13Compiler.mm:1312/1748; attention runtime-operand form inside README envelope |
| silu | 72 | supported | ANEH13Compiler.mm:82 |
| select | 48 | unsupported-with-owner | mil-hwx-compiler: no semantic; part of the boolean attention-mask path |
| slice_by_index | 48 | unsupported-with-owner | mil-hwx-compiler: registry has `slice_by_size`, not `slice_by_index` |
| pad | 24 | unsupported-with-owner | mil-hwx-compiler: pad exists only as a conv attribute, not as an op |
| sigmoid | 24 | supported | ANEH13Compiler.mm:81 |
| softmax | 24 | supported | ANEH13Compiler.mm:90 |
| split | 24 | supported | ANEH13Compiler.mm:160/1631 |
| expand_dims | 13 | supported | ANEH13Compiler.mm:159 |
| cast | 11 | unsupported-with-owner | mil-hwx-compiler: no cast lowering (fp16 boundary conversions); frontend could also insert explicit converts |
| less | 4 | unsupported-with-owner | mil-hwx-compiler: boolean comparison; lower via fp16 arithmetic on the mask path |
| floor | 3 | unsupported-with-owner | mil-hwx-compiler: no unary lowering |
| floor_div | 3 | unsupported-with-owner | mil-hwx-compiler: no binary lowering |
| relu | 3 | supported | ANEH13Compiler.mm:1690 |
| sub | 3 | supported | ANEH13Compiler.mm:68 (x minus same-shape const tensor envelope) |
| logical_and | 1 | unsupported-with-owner | mil-hwx-compiler: boolean path |
| logical_not | 1 | unsupported-with-owner | mil-hwx-compiler: boolean path |
| reduce_min | 1 | unsupported-with-owner | mil-hwx-compiler: reduce supports sum/max/mean only |
| reduce_sum | 1 | supported | ANEH13Compiler.mm:92 |
| tile | 1 | unsupported-with-owner | mil-hwx-compiler: no tiling semantic |

## Reading the gaps

The 485 unsupported ops cluster into three work streams:

1. **Weight palettization (194 ops)** — every
   `constexpr_lut_to_dense` expands palettized weights. The frontend
   can pre-expand them to dense fp16 consts at package-conversion
   time; this needs no H13 work and unblocks 194 ops plus their
   dependent consts. Cheapest first move.
2. **Layout/data movement (146 + 48 + 24 + 1 = 219 ops)** —
   `transpose`, `slice_by_index`, `pad`, `tile`. `transpose` is
   already IR-known; the others need registry entries plus H13
   lowerings (or folds into consumer operands).
3. **Boolean/mask path (48 + 11 + 4 + 3 + 3 + 1 + 1 + 1 = 72 ops)** —
   `select`, `cast`, `less`, `floor`, `floor_div`, `logical_and`,
   `logical_not`, `reduce_min`. The H13 datapath is fp16; these need a
   concrete fp16 mask representation decision in the compiler repo
   before lowering. Highest design risk of the three streams.

Envelope caveat: "supported" means an H13 lowering branch exists.
Geometry envelopes (matmul/conv/broadcast forms) are enforced per
shape at compile time with named refusals; the encoder's concrete
shapes are only exercised when Phase 6 encoder compilation runs.

## Reproduce

```sh
PYTHONPATH=overlay/tools python3 - <<'PY'   # from the repo root
from coreml.mlpackage import inspect
inv = inspect("<cache>/b650695c…/encoder.mlpackage")
print(inv["op_histogram"])
PY
```

H13 surface: `grep -n 'isEqualToString' .work/ane-compiler/mil-hwx-compiler/plugins/H13/ANEH13Compiler.mm`
after `scripts/prepare-ane-compiler.sh` (locked source only; nothing
reads `~/src/mil-hwx-compiler`).

## Not established by this receipt

- Compilation of the encoder (any part) — inventory comparison only.
- Geometry-envelope satisfaction for the encoder's concrete shapes.
- Any ANE device execution (Phase 4+ territory).

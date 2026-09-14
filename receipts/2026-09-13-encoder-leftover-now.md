# Encoder leftover H13 ops now (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute. No jwm1.
No full-encoder ANEC is claimed.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

This updates `receipts/2026-09-13-encoder-split-compile.md` leftover table,
which predated later peels and the H13 envelope pins.

## Pins

| item | value |
| --- | --- |
| overlay | mlx-omarchy `origin/main` `b9e0aff55ff628973790aab66ed24473f8c9697a` (merge of `f86bcf1a` select runtime-a) |
| compiler source | mil-hwx-compiler `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` (`feat(h13): name concat as an ISA hole`) |
| `build/mil-hwxc` sha256 | `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71` |
| host | `omp-studio-local` Linux |
| source MIL | `receipts/2026-09-13-slice-layout-rewrite/model-fp16-noslice.mil` |
| MIL bytes / sha256 | 1361187 / `58694e04d605c0973f411a77e8b3ad3fc5426c4f15dd3ad12fbcca377c53cce1` |
| encoder | `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018` |

The on-disk MIL is peel + fold + mask_layout + heads_layout + the 749-slice
rewrite. Later modules on this commit (`attention_layout`, `conv_layout`,
`select_runtime`, last-dim `slice_layout`) were planned against that MIL.
The 99 MB last-dim / K-birth expansions were not emitted.

## Graph

6723 tensor ops (25 types). `pad` / `logical_not` / `logical_and` /
`reduce_min` / `split` remain 0.

Later-peel plans on this MIL:

| module | rewritten | refused |
| --- | ---: | ---: |
| `attention_layout` | 24 | 0 |
| `conv_layout` | 2 | 0 |
| `select_runtime` | 0 | 24 (`cond is not bool`; `var_373` is fp16 `[1,1,375,375]`) |
| `slice_layout` last-dim | 24 | 0 |

`select_runtime` does not apply: mask_lowering already rewrote the bool
cond to fp16 0/1. `conv_layout` only respells `pad_type`/`groups`; encoder
CHW still misses the table.

## Leftover (fail H13 on encoder form)

Probed with `build/mil-hwxc --target H13 --format anec`. Exit 65, empty
output dir, unless noted.

| leftover op | count | diagnostic on encoder form |
| --- | ---: | --- |
| `concat` | 96 | `h13.invalid-concat-parameters` (`x0`/`x1`… form, 72 heads `[1,8,375,128]` + 24 rel-pos `[1,8,749,375]`); `h13.unsupported-concat` (`values` tuple) |
| `select` | 24 | `h13.boolean-outside-envelope` (const `-inf` `a`, fp16 cond `[1,1,375,375]`, scores `[1,8,375,375]`). Same geometry with bool cond is still `h13.select-needs-decoded-encoder` |
| `transpose` | 73 | `h13.nonfoldable-transpose` (48 rank-3 `[0,2,1]` tail-swap `[1,375,1024]↔[1,1024,375]`; 24 rank-4 `[0,2,1,3]` `[1,8,375,128]→[1,375,8,128]`; 1 rank-4 `[1,256,375,16]→[1,375,256,16]`) |
| `slice_by_index` last-dim `[1,8,375,375]` | 24 | `h13.noncontiguous-slice` (3000 chunks of 375 spaced 749). `slice_layout` now plans these 24 rewritten; expansion not kept |
| `layer_norm` | 120 | `h13.norm-outside-envelope` (`[1,375,1024]`, gamma/beta present) |
| `silu` | 72 | `h13.unary-outside-envelope` (48 `[1,375,4096]` + 24 `[1,1024,375]`; was `h13.unsupported-program`) |
| `sigmoid` | 24 | `h13.unary-outside-envelope` at `[1,1024,375]` |
| `softmax` | 24 | `h13.norm-outside-envelope` at `[1,8,375,375]` |
| `conv` | 101 | `h13.conv-outside-envelope` on all nine encoder forms, including corpus-spelled 1×1 valid `[1,256,750,32]` |

Unit slices from the 749-rewrite (`[1,1,750,375]` × 192 and
`[1,1,749,375]` × 192) are views, not this leftover set.

## No longer leftover

`matmul` × 72. All three encoder geometries compile as one
`apple-parity-batched-matmul` program (this host, same `mil-hwxc`):

| form | n | flags | programs / TDs | rc |
| --- | ---: | --- | --- | ---: |
| scores `[1,8,375,128]×[1,8,375,128]` | 24 | `tx=0 ty=1` | 1 / 209 | 0 |
| scores rewritten `[1,8,375,128]×[1,8,128,375]` | 24 | `tx=0 ty=0` | 1 / 208 | 0 |
| rel-pos `[1,8,375,128]×[1,8,128,749]` | 24 | `tx=0 ty=0` | 1 / 208 | 0 |
| PV `[1,8,375,375]×[1,8,375,128]` | 24 | `tx=0 ty=0` | 1 / 208 | 0 |

`attention_layout` is not required to drop the 24 score leftovers: the
`ty=1` runtime-B=8 row is already in `kBatchedTasks`.

## Not established

- Full encoder compile (not attempted).
- Emission of the 99 MB last-dim or K-birth MIL.
- Hardware execution.
- Inspector acceptance of the four one-program matmul packages.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac
execution. No files under `ane-linux-experiments` were edited or
executed.

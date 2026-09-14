# Encoder leftover H13 ops (2026-09-14)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute. No jwm1.
No full-encoder ANEC is claimed.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

Re-emitted encoder MIL from mlx-omarchy `origin/main` `02715a05` with every
small peel, including select bool cond. Leftover table against mil-hwxc
`738ce0a` (binary) / `c2cf32e4` (prior concat-hole pin). The 24 selects are
**gone**.

## Pins

| item | value |
| --- | --- |
| overlay | mlx-omarchy `origin/main` `02715a055cde88d0d18b8c185fecf9d143cf266a` (`tools: convert H13 v2 packages to schema-4`; contains `92d2ab8e` select bool cond) |
| compiler source | mil-hwx-compiler `738ce0a0eeea2650a696da1af0591f973e5d3741` (`h13: name encoder envelopes and leftover holes`) |
| prior compiler pin | `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` (`feat(h13): name concat as an ISA hole`) |
| `build/mil-hwxc` sha256 | `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71` (`make build/mil-hwxc` up to date) |
| host | `omp-studio-local` Linux |
| source MIL | `receipts/2026-09-13-slice-layout-rewrite/model-fp16-noslice.mil` |
| source sha256 | `58694e04d605c0973f411a77e8b3ad3fc5426c4f15dd3ad12fbcca377c53cce1` |
| emitted MIL | `receipts/2026-09-14-encoder-leftover/model-fp16-allpeels.mil` |
| emitted bytes / sha256 | 1361130 / `a7f08439c281b8e1e1dfe7dbabbef8bf844a7094c1fe98a00688340fcb8db4dd` |
| encoder | `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018` |

The on-disk MIL is peel + fold + mask_layout + heads_layout + the 749-slice
rewrite + `conv_layout` + `select_runtime` (runtime-a + bool cond). Last-dim
slice and attention K-birth expansions were planned 24/0 and not kept
(101 MB / 76 MB). `make` did not rebuild `mil-hwxc`; the binary already
matched `738ce0a`.

## Graph

6718 tensor ops (25 types). `pad` / `logical_not` / `logical_and` /
`reduce_min` / `split` remain 0.

Peels on this MIL:

| module | rewritten | refused | emitted |
| --- | ---: | ---: | --- |
| `conv_layout` | 2 | 0 | yes |
| `select_runtime` | 24 | 0 | yes (`a = var_8_to_fp16_rt`, `cond = var_373_bool`) |
| `attention_layout` | 24 | 0 | no (K-birth 76 MB) |
| `slice_layout` last-dim | 24 | 0 | no (101 MB) |

`select_runtime` now applies: one runtime fill `[1,8,375,375]` and one
promoted bool cond `var_373_bool` `[1,1,375,375]`. Post-rewrite plan is 0/0.
`conv_layout` only respells `pad_type`/`groups`; encoder CHW still misses
the table.

## Leftover (fail H13 on encoder form)

Probed with `build/mil-hwxc --target H13 --format anec`. Exit 65, empty
output dir, unless noted. Receipts:
`receipts/2026-09-14-encoder-leftover/probe-leftover.json`,
`receipts/2026-09-14-encoder-leftover/probe-select.json`.

| leftover op | count | diagnostic on encoder form |
| --- | ---: | --- |
| `concat` | 96 | `h13.invalid-concat-parameters` (`x0`/`x1`… form, 72 heads `[1,8,375,128]` + 24 rel-pos `[1,8,749,375]`); `h13.unsupported-concat` (`values` tuple) |
| `transpose` | 73 | `h13.nonfoldable-transpose` (48 rank-3 `[0,2,1]` tail-swap `[1,375,1024]↔[1,1024,375]`; 24 rank-4 `[0,2,1,3]` `[1,8,375,128]→[1,375,8,128]`; 1 rank-4 `[1,256,375,16]→[1,375,256,16]`) |
| `slice_by_index` last-dim `[1,8,375,375]` | 24 | `h13.noncontiguous-slice` (3000 chunks of 375 spaced 749). `slice_layout` still plans these 24 rewritten; expansion not kept |
| `layer_norm` | 120 | `h13.norm-outside-envelope` (`[1,375,1024]`, gamma/beta present) |
| `silu` | 72 | `h13.unary-outside-envelope` (48 `[1,375,4096]` + 24 `[1,1024,375]`) |
| `sigmoid` | 24 | `h13.unary-outside-envelope` at `[1,1024,375]` |
| `softmax` | 24 | `h13.norm-outside-envelope` at `[1,8,375,375]` |
| `conv` | 101 | `h13.conv-outside-envelope` on encoder 1×1 valid `[1,256,750,32]` (nine encoder forms still miss the table) |

Unit slices from the 749-rewrite (`[1,1,750,375]` × 192 and
`[1,1,749,375]` × 192) are views, not this leftover set.

## No longer leftover

`select` × 24. Gone. Rewritten selects bind runtime-a `[1,8,375,375]` and
bool cond `[1,1,375,375]`. Standalone compiles as one `apple-parity-boolean`
program, 5 TDs, rc 0:

| form | rc | encoder / code |
| --- | ---: | --- |
| runtime-a / runtime-b / bool cond `[1,8,375,375]` | 0 | `apple-parity-boolean` / 5 TDs |
| encoder broadcast bool cond `[1,1,375,375]` | 0 | `apple-parity-boolean` / 5 TDs |
| fp16 cond `[1,1,375,375]` (control) | 65 | `h13.boolean-outside-envelope` |

`matmul` × 72 still compile as one `apple-parity-batched-matmul` program
(this host, same `mil-hwxc`):

| form | n | flags | programs / TDs | rc |
| --- | ---: | --- | --- | ---: |
| scores `[1,8,375,128]×[1,8,375,128]` | 24 | `tx=0 ty=1` | 1 / 209 | 0 |
| scores rewritten `[1,8,375,128]×[1,8,128,375]` | 24 | `tx=0 ty=0` | 1 / 208 | 0 |
| rel-pos `[1,8,375,128]×[1,8,128,749]` | 24 | `tx=0 ty=0` | 1 / 208 | 0 |
| PV `[1,8,375,375]×[1,8,375,128]` | 24 | `tx=0 ty=0` | 1 / 208 | 0 |

## Not established

- Full encoder compile (not attempted).
- Emission of the 101 MB last-dim or 76 MB K-birth MIL.
- Hardware execution.
- Inspector acceptance of the select or matmul packages.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac
execution. No files under `ane-linux-experiments` were edited or
executed.

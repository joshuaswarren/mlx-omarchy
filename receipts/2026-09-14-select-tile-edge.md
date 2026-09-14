# Select leftover -inf vs tile_layout 384 / 71 tiles (2026-09-14)

Host-only. Reused exec3 `host-compare/` tensors. No ANE submit, no T6001, no retry, no `ane-linux-experiments`.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Verdict

**Named miss: `not-tile_layout-384-71tile`.**

The 5412 leftover `-inf` after invert-cond are **not** host `tile_layout.h` bool packing, **not** the 9-column 384 pad, **not** 16 KiB tile edges, and **not** the 71st-tile allocation tail.

`tile_layout.h` (`column * 1`, `row_stride=384`, `plane_stride=144000`) matches header ch7. Prefix c0–c3 is dense `0x01`. That packing cannot present `false` at leftover pixels. Device `y` is still `-inf` there, so invert-cond wanted `1.0` and the kernel wrote `a`.

**Related 384-stride fact (not the host pack):** anec L2 words are `0x180=384` and `0xc00=3072=8*384`. Leftover is the first 8-row L2 tile plus H=8, identical on NCHW c0–c3, 48-period right wedge (`384/8=48`). That is a device first-L2-tile cond read, not the 71 × `0x4000` BO.

## Pins

| item | value |
| --- | --- |
| worktree | mlx-omarchy `main-land2` `59c614c2865c38bb91b75cff75217f1646e4b12b` |
| exec3 layout | `receipts/2026-09-14-jwm1-select-exec3-layout.md` |
| exec3 submit | `receipts/2026-09-14-jwm1-select-island-exec3.json` |
| bool packing | `receipts/2026-09-14-select-bool-packing.md` |
| overlay `tile_layout.h` | `c88b4d7a0c844a084a5a8ea251403daae5a6c783d2d5fa2574275f278b493f22` |
| anec | `27dc7f35fc9efc9431e4d05847798390fe0cfe79f930c3040f32dd76ef8e8e73` |
| `y.out.bin` | `4aff9aa221eb85b4be8d07f614be82dcdc4fe7399cd276cc7a23421e053df6cf` |
| invert-cond equal | 1119588 / 1125000 |
| leftover | 5412 = 4 × 1353 |

## Commands

```text
python3 - <<'PY'
# reshape [1,8,375,375]; leftover = (y != invert_cond)
# map leftover onto 384-stride, 0x4000 tiles, 8*384 L2, 48-col groups
PY
sha256sum overlay/mlx/backend/omarchy/ane/tile_layout.h \
  receipts/2026-09-14-jwm1-select-island-exec3/host-compare/* \
  receipts/2026-09-14-jwm1-select-island-exec3/attn-select/program-0.anec
```

No ANE. No formatter.

## Why tile_layout / 71 tiles are not the leftover

Header ch7: `[1,8,375,375,144000,384]`, `tiles=71`, alloc `71*0x4000=1163264`. `8*144000=1152000`, tail `11264` bytes **after** plane 7. Leftover is plane 0 start.

| hypothesis | arithmetic | leftover |
| --- | --- | --- |
| 9 pad columns of 384 | `9*384=3456` | no; leftover W starts at 32, not 375 |
| 16 KiB tile 0 edge | `16384/384=42` rows + 256 B | no; leftover is H=0..8 only, all inside tile 0 (`8*384+374=3446<16384`) |
| 71st-tile / alloc tail | bytes `1152000..1163263` | no; opposite end of the BO |
| `ane_tile` `sizeof(uint16_t)` | `R/2` width | exec3 worker uses `__ane_send` memcpy after `tile_layout.h`, not `ane_tile` |
| in-process `pack_binding` `column*2` | fp16-only | not the exec3 worker path |

c0–c3 leftover masks are identical. H≥9 leftover 0. c4–c7 leftover 0. Got at leftover is `0xfc00`; invert wanted `0x3c00`.

All-`0x01` prefix: any 1-byte 384-stride pack still stores `0x01` at leftover `(h,w)`. Host pack cannot be the false cond.

## What the 384-stride *does* match

Anec `27dc7f35` L2: `0x11f0/0x11f4/0x11f8/0x16f0/0x16f4 = 0xc00` (3072), next to `0x180` (384). First 8-row L2 tile is H=0..7. Leftover also includes H=8 (first row of the next 8-row tile) and then stops.

Group starts period 48 (`384/8`):

| H | n | first W | groups (start,end,len) |
| ---: | ---: | ---: | --- |
| 0 | 167 | 72 | (72,79,8), (120,135,16), (168,187,20), (216,243,28), (264,299,36), (312,355,44), (360,374,15) |
| 1 | 167 | 32 | (32,35,4), then +48 |
| 2 | 168 | 40 | +8 vs H1, then +48 |
| 3 | 155 | 48 | |
| 4 | 143 | 56 | |
| 5 | 139 | 64 | |
| 6 | 139 | 72 | |
| 7 | 139 | 80 | |
| 8 | 136 | 88 | last group (328,367,40); no H≥9 |

H1–H8 first W = `32+8*(H-1)`. H0 starts at 72 (same as H6). Occupancy grows toward the right of the 8×384 tile.

Named follow-up (not this host pack): first 8-row L2 bool tile, 48-period right wedge, compare-to-`0x0001` cond read. No resubmit.

## Not done

No resubmit. No T6001. No overlay edit. No compiler edit. No `ane-linux-experiments` edit or execute.

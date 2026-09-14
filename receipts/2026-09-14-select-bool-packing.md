# Select bool packing vs anec (2026-09-14)

Host-only. No ANE execute, no jwm1, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Verdict

**Named hole: `cond-header-u8-384` vs `cond-dma-fp16-768`.**

The 5-TD `apple-parity-boolean` anec *header* wants bool cond as 1-byte elements, row 384, plane 144000, channel 7. The CLI worker packed that: 1-byte `0x01`/`0x00` on sequential src2 → ch7.

The *task stream* that reads ch7 uses fp16 DMA (row 768, plane 288000). Adjacent `0x01,0x01` become fp16 `0x0101`. That is the first u16 of device `y` (`receipts/2026-09-14-jwm1-select-mismatch.md`).

No host adapter can close this. Ch7 is allocated 1 163 264 bytes (71 tiles). An fp16 `[1,8,375,375]` surface is 2 310 144 bytes (141 tiles). You cannot pack 768-stride fp16 0/1 into the bool allocation.

## Pins

| item | value |
| --- | --- |
| worktree | mlx-omarchy `main-land2` `c87b67c52feed6c6098d25e0c2190797883cdfbd` |
| overlay (exec2 worker) | `15ecc733ee91327dbf7887eac21ef14bebe824ce` |
| bool ABI | `389fb4de82f3b338319cf974f29446bd1eed3b04` |
| compiler (anec emit) | `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` |
| inspect_anec tree | mil-hwx-compiler `738ce0a0eeea2650a696da1af0591f973e5d3741` |
| anec | `receipts/2026-09-13-attn-select-island/attn-select/program-0.anec` |
| anec sha256 | `860de06c53e3fdfed9452859042df59902fb733bef0a364b134dd9d9d6374cdd` |
| anec bytes | 9216 |
| encoder | `apple-parity-boolean` |
| operation | `select` |
| taskDescriptors | 5 |

## Commands

```text
python3 - <<'PY'
# sha256, HEADER unpack, inspect_anec.load_package, task DMA
# package: receipts/2026-09-13-attn-select-island/attn-select
# inspect_anec: ~/src/mil-hwx-compiler/research/inspect_anec.py
PY
```

No ANE. No formatter.

## ANEC header vs overlay ABI

`HEADER = struct.Struct('<QIIQQII32I192Q')`. Measured:

| ch | tiles | alloc | nchw | role |
| ---: | ---: | ---: | --- | --- |
| 3 | 141 | 2310144 | zeros | scratch |
| 4 | 141 | 2310144 | `[1,8,375,375,288000,768]` | `y` fp16 |
| 5 | 141 | 2310144 | `[1,8,375,375,288000,768]` | `ninf_rt` fp16 |
| 6 | 141 | 2310144 | `[1,8,375,375,288000,768]` | `b` fp16 |
| 7 | 71 | 1163264 | `[1,8,375,375,144000,384]` | `cond` bool |

`surface_layout([1,8,375,375], bool_elements=True)` = `[1,8,375,375,144000,384]`. Header ch7 matches.

Compiler `booleanTensor(7, shape, true)` at `c2cf32e4` uses element 1, row `(375+63)/64*64 = 384`. Same.

Overlay CLI pack (`overlay/mlx/backend/omarchy/ane/tile_layout.h` + `worker_libane.cpp`): `ane_element_size(bool)=1`, `plane*plane_stride + row*row_stride + column*1`. Matches header ch7. Exec2 worker `762dd1de` used this path (`mlx-omarchy-ane-worker` → `LibaneDevice::pack`). Dense fill was 1 125 000 bytes, first 562 500 `0x01`, rest `0x00`.

## Task DMA (not the header)

Raw word8 at `0x1000+32` = `0x04823025`. Five linked tasks:

| task | src1 | src2 | dst | src1 row/plane | src2 row/plane |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | **5** | 0 | 3 | **384 / 144000** | disabled |
| 1 | 3 | **7** | 3 | 768 / 288000 | **768 / 288000** |
| 2 | 0 | 0 | 3 | (constants) | |
| 3 | 3 | 6 | 3 | 768 / 288000 | 768 / 288000 |
| 4 | 3 | 3 | 4 | scratch | write y 768 / 288000 |

Bool-stride DMA reads **ch5** (`ninf_rt` fp16 header). Ch7 (`cond` bool header) is read as **fp16 768**.

`encodeANEC` does call `bindTasks`. Select `taskSurfaceChannels = {7,6,4,5}` is in `c2cf32e4`. The emitted stream still has 384-DMA on ch5 and 768-DMA on ch7.

## Why `y` starts `0x0101, 0x3c00`

Host replay of worker 1-byte pack, then fp16 768-stride read of that tile (task 1 src2 on ch7): first 16 u16 are `0x0101`. First 8000: `0x0101` 7895, `0x0000` 84, `0x0001` 21.

Sibling device `y.out`: first u16 `0x0101, 0x3c00` repeating; `0x0101` only in NCHW ch0–ch1 (linear 0..281249). One fp16 plane is 288000 bytes = two bool planes of 144000. Cond-true is bool planes 0–3, so fp16 ch0–ch1 are the bitcast-true region. `0x3c00` is `b`. Not a slot swap.

Constant section (2048 bytes, 46 nonzero) prefix `0x8001, 0x0001, 0x0000, 0x0000, 0x3c00…` matches the `select_rrb_1x8x375x375` oracle. Kernel compare-to-`0x0001` cannot see 1-byte `0x01` once DMA pairs it to `0x0101`.

## inspect_anec

`load_package(attn-select)` → `ValueError: incorrect physical layout`.

`check_layout` requires `nchw == surface_layout(nchw[:4])` with default `bool_elements=False` (fp16 768). Ch7 384 fails. `surface_layout` already knows 1-byte bool; `check_layout` does not pass it.

`convert_tensor` always uses `logical // 2` and `column * 2`. It cannot pack this cond.

## Overlay split (not the exec2 path)

`runtime_detail.h pack_binding` still uses `logical / 2` and `column * 2` for every dtype. In-process runtime would pack cond as fp16 into the 384-stride allocation. Exec2 did not use that function.

Smallest overlay adapter for that split: `pack_binding` / `unpack_binding` should call `ane_element_size` like `tile_layout.h`. That makes the in-process path match the **header**. It does not move 384-DMA onto ch7, and it does not change exec2.

## Hole

Do not change worker 0x01/0x00. That packing matches header ch7 and compiler `booleanTensor(..., true)`.

Do not pack fp16 0/1 onto ch7: allocation is half-size.

Close is a new anec whose bool-stride DMA selects ch7 (header cond), or a proven `bindTasks` fix. Not a host pack tweak.

## Not done

No ANE. No jwm1. No overlay edit. No compiler edit. No `ane-linux-experiments` edit or execute.

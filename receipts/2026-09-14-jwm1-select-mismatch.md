# jwm1 select-island y.out mismatch (2026-09-14)

Host-only. Copied `/var/tmp/jwm1-select-island-exec2/y.out.bin` after exec2. No ANE submit, no T6001, no retry.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Verdict

**named miss.** Device `y` is not `cond.bin`, not `cond` viewed as fp16, not `ninf_rt.bin`, not `b.bin`, and not a slot-swap identity of any of those.

The first u16 `0x0101` is the cond fill `0x01,0x01` as little-endian fp16. That is only the first value. The whole buffer does not match cond.

## Pins

| item | value |
| --- | --- |
| overlay (exec2 worker) | `15ecc733ee91327dbf7887eac21ef14bebe824ce` |
| bool ABI | `389fb4de82f3b338319cf974f29446bd1eed3b04` |
| worker sha256 | `762dd1de868b52e4d6ffa30e18d842d9341db463b06aef2355f18fb1eb248929` |
| anec sha256 | `860de06c53e3fdfed9452859042df59902fb733bef0a364b134dd9d9d6374cdd` |
| graph | `998f81dee926a012be067467aee2b4bf138146f9944fb36ae40a732f40c89463` |
| host | `jwm1-linux` |
| prior exec | `receipts/2026-09-14-jwm1-select-island-exec2.json` |

MIL: `select(a=ninf_rt, b=b, cond=cond)`. Shape `[1,8,375,375]`.

## Copied hashes

| file | bytes | sha256 | first u16 |
| --- | ---: | --- | --- |
| `y.out.bin` | 2250000 | `a5a22246dc963d0d949016c2ea96fd7f2b3663535b1b4767f8d44a9733a3d7ac` | `0x0101, 0x3c00, 0x0101, 0x3c00, …` |
| `cond.bin` | 1125000 | `658def707323f286aa8acbf7765984e75f7f3167b686f61b9d729d372d2da08e` | `0x0101` repeating (bytes `0x01`) |
| `ninf_rt.bin` | 2250000 | `98cabc7db74c4a173e1bc2a1e22afc5c3595a5a5f91a53a19c2e89e903726d3b` | `0xfc00` (`-inf`) |
| `b.bin` | 2250000 | `fe0afb9bd9567bca8c5ab2c23d39f023c1e335e0940679cd55ecad9a163bc017` | `0x3c00` (fp16 `1.0`) |
| `y.bin` expect | 2250000 | `dd8dbb3a55336d4597c637553165f028a5304065f68c6ce8bfa915f301b2364c` | `0xfc00` |

`y.out.bin` sha256 matches exec2. Cond fill is first 562500 bytes `0x01`, rest `0x00`.

## Identity

u16 counts over 1125000 fp16 elements unless noted.

| candidate | equal | identity |
| --- | ---: | --- |
| `ninf_rt.bin` | 0 / 1125000 | miss |
| `b.bin` | 555149 / 1125000 | miss |
| expect `y.bin` | 277575 / 1125000 | miss (same as exec2) |
| `cond.bin` bytes vs `y[:1125000]` | 704337 / 1125000 | miss (also 2× size) |
| `cond` pairwise as fp16 vs `y[:562500]` | 282775 / 562500 | miss |

Slot swap would require `y` byte-identical to one input. It is not.

## What the first u16 is

`y[0] = 0x0101`. Cond starts `0x01,0x01`. Viewing those two bool bytes as fp16 is `0x0101` = `1.531839e-05`. That hypothesis holds for element 0 only.

`ninf` (`0xfc00`) never appears in `y` (count 0). So device `a` was not the ninf buffer.

## Pattern (not a match)

`y` u16 histogram: `0x3c00` 555149, `0` 429454, `0x0101` 140312, 38 other values totaling 85. Unique = 41.

`0x0101` lives only in NCHW channels 0 and 1 (linear 0..281249). Channel counts: `[70498, 69814, 0, 0, 0, 0, 0, 0]`. Cond-true is channels 0–3.

Channel 0 row 0 starts `0x0101, 0x3c00` repeating: cond-as-fp16 interleaved with `b`, not a copy of either buffer.

## Not done

No resubmit. No T6001. No `ane-linux-experiments` edit or execute.

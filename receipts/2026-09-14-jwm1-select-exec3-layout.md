# jwm1 select exec3 y vs expect layout (2026-09-14)

Host-only. Copied existing `/var/tmp/jwm1-select-island-exec3/y.out.bin` and the exec1 input tensors. No ANE submit, no T6001, no retry.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Verdict

**Named polarity: invert-cond.** Device `y` is `select(b, a, cond)` / `select(a, b, ~cond)`, not MIL `select(a=ninf_rt, b=b, cond)`.

That polarity is identical to **suffix-true** under the MIL contract (last 4 NCHW channels true). Prefix-true cond + invert-cond === suffix-true cond + MIL select.

**Transpose HW:** not a distinct layout. Invert/expect are constant per NCHW channel, so `transpose(H,W)` does not change the score.

**NCHW vs NHWC:** NCHW. Output is two 4-channel blocks. NHWC reinterpret is ~50% and rejected.

**Not closed:** 5412 leftover `-inf` in the invert-true region (NCHW c0–c3, H=0..8, identical mask). Not invert, transpose, NHWC, or prefix/suffix.

## Pins

| item | value |
| --- | --- |
| worktree | mlx-omarchy `main-land2` `f040880838582347dd452292e0c4627c6759d1ab` |
| overlay (exec3 worker) | `15ecc733ee91327dbf7887eac21ef14bebe824ce` |
| compiler | mil-hwxc `8f97f504edb4ed525767fad6d90f814ff1c13241` |
| anec | `27dc7f35fc9efc9431e4d05847798390fe0cfe79f930c3040f32dd76ef8e8e73` |
| worker | `762dd1de868b52e4d6ffa30e18d842d9341db463b06aef2355f18fb1eb248929` |
| graph | `998f81dee926a012be067467aee2b4bf138146f9944fb36ae40a732f40c89463` |
| prior exec | `receipts/2026-09-14-jwm1-select-island-exec3.json` |
| MIL | `select(a = ninf_rt, b = b, cond = cond)` shape `[1,8,375,375]` |

## Copied hashes

| file | bytes | sha256 | first u16 |
| --- | ---: | --- | --- |
| `y.out.bin` | 2250000 | `4aff9aa221eb85b4be8d07f614be82dcdc4fe7399cd276cc7a23421e053df6cf` | `0x3c00` |
| expect `y.bin` | 2250000 | `dd8dbb3a55336d4597c637553165f028a5304065f68c6ce8bfa915f301b2364c` | `0xfc00` |
| `cond.bin` | 1125000 | `658def707323f286aa8acbf7765984e75f7f3167b686f61b9d729d372d2da08e` | prefix `0x01` |
| `ninf_rt.bin` | 2250000 | `98cabc7db74c4a173e1bc2a1e22afc5c3595a5a5f91a53a19c2e89e903726d3b` | `0xfc00` |
| `b.bin` | 2250000 | `fe0afb9bd9567bca8c5ab2c23d39f023c1e335e0940679cd55ecad9a163bc017` | `0x3c00` |

`y.out.bin` sha256 matches exec3. Unique device values: `{0x3c00: 557088, 0xfc00: 567912}`.

## Commands

```text
scp jwm1:/var/tmp/jwm1-select-island-exec3/y.out.bin \
    jwm1:/var/tmp/jwm1-select-island-exec/{y,cond,ninf_rt,b}.bin \
    receipts/2026-09-14-jwm1-select-island-exec3/host-compare/
sha256sum receipts/2026-09-14-jwm1-select-island-exec3/host-compare/*
python3 - <<'PY'
# reshape [1,8,375,375]; score invert-cond, ~cond, transpose HW,
# NHWC/NHCW, prefix vs suffix channel polarity; locate leftover infs
PY
```

No ANE. No formatter.

## Scores (exact u16 / 1125000)

| hypothesis | equal |
| --- | ---: |
| MIL `select(a,b,cond)` vs got | 5412 |
| invert-cond `select(b,a,cond)` vs got | **1119588** |
| MIL `select(a,b,~cond)` vs got | **1119588** |
| suffix-true MIL `select(a,b,cond)` vs got | **1119588** |
| invert-cond identical to suffix-true MIL | yes |
| got.transpose(H,W) vs invert | 1119588 (not distinct) |
| got as NHWC→NCHW vs invert | 562466 |
| got as NHCW→NCHW vs invert | 564198 |
| cond packed as NHWC, invert vs got | 557092 |

Raw vs expect is the same 5412 as exec3 (`first_got=0x3c00`, `first_expect=0xfc00`). That first-element pair is invert-cond at index 0.

## Prefix / suffix

Cond fill is prefix-true: first 562500 bytes `0x01` = NCHW c0–c3 all true, c4–c7 all false.

| region | expect | got |
| --- | --- | --- |
| prefix 562500 (c0–c3) | all `-inf` | 557088 `1.0` + 5412 `-inf` |
| suffix 562500 (c4–c7) | all `1.0` | all `-inf` |

Prefix-true + invert-cond is the same tensor as suffix-true + MIL select. Channel-reverse of got scores 1119588 vs expect for the same reason.

## NCHW

| ch | cond | got inf | got 1.0 |
| ---: | --- | ---: | ---: |
| 0–3 | true | 1353 each | 139272 each |
| 4–7 | false | 140625 | 0 |

c0–c3 leftover inf masks are identical. Rows H≥9 of those channels are all `1.0`. c4–c7 are all `-inf`. That is NCHW channel blocks, not NHWC.

## Leftover 5412 (not a named layout)

All 5412 invert-mismatches are `-inf` where invert wants `1.0`, in c0–c3, H=0..8 only (9 rows). Row inf counts `{136,139,143,155,167,168}`; columns start at W=32. Not 9 pad columns of the 384-wide surface (`9*384=3456`). Not closed by invert, transpose, NCHW/NHWC, or prefix/suffix.

## Not done

No resubmit. No T6001. No `ane-linux-experiments` edit or execute.

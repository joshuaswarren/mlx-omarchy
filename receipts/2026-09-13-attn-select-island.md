# Attn score + masked select island (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute. No jwm1.
No full-encoder ANEC is claimed.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Pins

| item | value |
| --- | --- |
| overlay | mlx-omarchy `origin/main` `0cd6fa59e8db1dc76051871497186d84e4ffb3b6` |
| compiler source | mil-hwx-compiler `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` (`feat(h13): name concat as an ISA hole`) |
| `build/mil-hwxc` sha256 | `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71` |
| host | `omp-studio-local` Linux |
| schema | `mil-hwxc.h13-anec-package.v2` |

Artifacts: `receipts/2026-09-13-attn-select-island/`.

## Standalone

Both compile, one program each.

### Matmul `[1,8,375,128]×[1,8,128,375]`, `tx=0`, `ty=0`

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-attn-select-island/attn-matmul.mil \
  --model-root receipts/2026-09-13-attn-select-island \
  --target H13 --format anec \
  --output receipts/2026-09-13-attn-select-island/attn-matmul
compiled target=H13 artifacts=1 format=anec output=.../attn-matmul
```

| field | value |
| --- | --- |
| exit | 0 |
| programs | **1** |
| encoder | `apple-parity-batched-matmul` |
| taskDescriptors | 208 |
| `program-0.anec` sha256 | `a3aa2fe1333a48eba3e7fcae5b848563e2bfa747a232dadf9ed2d568821c5627` |
| bytes | 182400 |

Inputs `x` fp16 `[1,8,375,128]`, `w` fp16 `[1,8,128,375]`. Output `product` fp16 `[1,8,375,375]`.

### Select runtime-a, bool cond `[1,8,375,375]`

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-attn-select-island/attn-select.mil \
  --model-root receipts/2026-09-13-attn-select-island \
  --target H13 --format anec \
  --output receipts/2026-09-13-attn-select-island/attn-select
compiled target=H13 artifacts=1 format=anec output=.../attn-select
```

| field | value |
| --- | --- |
| exit | 0 |
| programs | **1** |
| encoder | `apple-parity-boolean` |
| taskDescriptors | 5 |
| `program-0.anec` sha256 | `860de06c53e3fdfed9452859042df59902fb733bef0a364b134dd9d9d6374cdd` |
| bytes | 9216 |

Inputs `ninf_rt` fp16, `b` fp16, `cond` bool, all `[1,8,375,375]`. Output `y` fp16 `[1,8,375,375]`.

## Combined island

Same two ops in one MIL: matmul then `select(a=ninf_rt, b=product, cond)`. Default `--schedule per-op`.

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil receipts/2026-09-13-attn-select-island/attn-select-island.mil \
  --model-root receipts/2026-09-13-attn-select-island \
  --target H13 --format anec \
  --output receipts/2026-09-13-attn-select-island/attn-select-island
compiled target=H13 artifacts=2 format=anec output=.../attn-select-island
```

| field | value |
| --- | --- |
| exit | 0 |
| programs | **2** (not fused) |
| dispatchPlan | `[0, 1]` |
| intermediate | `product` |

| program | op | encoder | TDs | file | sha256 | bytes |
| --- | --- | --- | ---: | --- | --- | ---: |
| 0 | matmul | `apple-parity-batched-matmul` | 208 | `program-0.anec` | `a3aa2fe1333a48eba3e7fcae5b848563e2bfa747a232dadf9ed2d568821c5627` | 182400 |
| 1 | select | `apple-parity-boolean` | 5 | `program-1.anec` | `860de06c53e3fdfed9452859042df59902fb733bef0a364b134dd9d9d6374cdd` | 9216 |

The two anec hashes are the standalone programs. Inputs: `x`, `w`, `ninf_rt`, `cond`. Output `y` fp16 `[1,8,375,375]`.

`--schedule chain` is `h13.chain-outside-envelope` (`H13 composed scheduling needs at least two operations and at most two boundary inputs`). Four boundary inputs. rc=65, no package.

## Not established

- Full encoder compile (not attempted).
- Hardware execution.
- Fusion of this island.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac
execution. No files under `ane-linux-experiments` were edited or
executed.

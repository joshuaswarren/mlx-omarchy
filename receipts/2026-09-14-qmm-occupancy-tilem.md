# Occupancy 16-row coopmat prefill tile

Date: 2026-09-14

## Verdict

Land the 16-row coopmat twin, selected when the shape-derived grid is below 6 workgroups per GPU core. Delete the 8-row variant. Digests stay on the pinned Q4 IDs on both hosts. The default 32-row SPIR-V is byte-identical to the shipped shader.

jw16 M1 Max short-prompt prefill median rises 26.74% (360.2682 → 456.6207 tok/s). The 1053-token prefill is +1.23% and is not the reason to land. jwm1 base M1 does not fire at 1053 (no timing regression: 1112.6267 → 1112.1047 tok/s, −0.05%). It does fire on the 30-token prompt (+13.67% prefill: 332.0043 → 377.3840 tok/s). Decode is unchanged on both hosts.

Code commit: `f45f76a7b948512fc876b00a9d5d6059301883be` on `agent/qmm-occupancy-tilem`. This receipt is the landing record; do not merge `63c1d3cf`.

## Discriminator

The pick is occupancy, not a device-name tile constant. Floor is 6 workgroups per GPU core, the midpoint between the measured crossover on jw16 at m=1053, k=896:

| n | workgroups | wg/core | 32-row median | 16-row median | 16 vs 32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 132 | 4.1 | 213.6 µs | 165.7 µs | −22.4% |
| 256 | 264 | 8.2 | 225.7 µs | 300.7 µs | +33.3% |
| 384 | 396 | 12.4 | 249.7 µs | 434.8 µs | +74.1% |
| 896 | 924 | 28.9 | 509.2 µs | 759.5 µs | +49.2% |

16 rows win only at 4.1 wg/core and lose from 8.2 up. 6 × cores = 192 workgroups on the 32-core Max (strictly between 132 and 264) and 48 on the 8-core M1.

Vulkan, `Device::capabilities()`, and `mx.device_info()` expose no GPU core count. Cores come from a measured name table (`Apple M1 (G13G B1)` → 8, `Apple M1 Max (G13C C0)` → 32); unknown names return 0 and keep the shipped 32-row tile. Occupancy math is then `cores * 6`. `MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE=0` restores the shipped pick.

An 8-row 32-lane twin was built and measured and lost even at 4.1 wg/core (about 58% slower than 32 rows). It is not in CMake. Only `qmm_coopmat_f16` (default 32) and `qmm_coopmat_m16_f16` (`-DTILE_ROWS=16`) remain.

## Identity

- Branch HEAD (code): `f45f76a7b948512fc876b00a9d5d6059301883be` (`omarchy: dispatch a 16-row coopmat prefill tile on starved grids`), parent `d9272a80`.
- Wheel used on both hosts: `mlx_omarchy-0.32.2.dev202609141611+23102851-cp314-cp314-linux_aarch64.whl`, SHA-256 `9ed0cbb5fecde8899dfb47ee53bde28df03d04fb05fc8ef84b8a522a7751e40a`. Built on `jw16mbp1-linux`, installed on jwm1 with `scripts/mlx_provenance.py` `verified=match`.
- Installed bits: `core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea`, `libmlx.so=sha256:b59c8b757da69da6`.
- mlx-lm: `0.31.3`. Python 3.14.7. Kernel `7.1.6-1-1-ARCH`. Mesa Honeykrisp `26.3.0.devel.hk6f6afc8-1`.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
- `resolved_model`: `xai-oauth/grok-4.6`. `fallback`: true (session is not an OpenAI model).
- Protocol: `MLX_DISABLE_COMPILE=1`, greedy temp 0, seed 0, EOS suppressed, 32 generated tokens, `bench_matrix.prompt_text` for `short` (30 tokens) and `ctx1024` (1053 tokens). Five interleaved off/on rounds after warmup. Off arm sets `MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE=0`.

## SPIR-V (non-firing default)

glslc 2026.3. Default 32-row binary matches the shipped shader on both hosts. The 16-row twin is a distinct binary.

```text
2581623778f0bed9951be3e46175f3da3d8aa85dd61a7125a416052759838e83  shipped / new-default
8a9692c77f57ea6b242f9963b08af14c996ba92f5cce638035a3e397f90b518f  new-m16
```

## jw16 (Apple M1 Max, G13C C0, 32 cores)

Lock inode 12. Nested `flock -n` exit 1 during the hold. Released, not unlinked. `llm-inference.service` inactive.

| Leg | off median | on median | on vs off | digest |
| --- | ---: | ---: | ---: | --- |
| short prefill, 30 tokens | 360.2682 tok/s | 456.6207 tok/s | +26.74% | `7fd25a869ff21678` |
| short decode, 30/32 | 169.7826 tok/s | 169.4604 tok/s | −0.19% | `7fd25a869ff21678` |
| ctx1024 prefill, 1053 tokens | 3570.9544 tok/s | 3615.0316 tok/s | +1.23% | `7da83f06ec9f001d` |
| ctx1024 decode, 1053/32 | 144.1140 tok/s | 141.8734 tok/s | −1.55% | `7da83f06ec9f001d` |

Both arms: short first=`9707,0,2585` last=`646,387,7881`; ctx1024 first=`13060,498,369` last=`3897,553,279`. Five of five rounds.

## jwm1 (Apple M1, G13G B1, 8 cores)

Same wheel, `verified=match`. Lock inode 29. Nested `flock -n` exit 1 during the hold. Released, not unlinked, inode 29 after. Boot id `95872130-fd28-4d67-8247-03a0e8b61202` unchanged. `llm-inference.service` inactive. jw16 GPU was not used (held by DecodeEpilogueFold).

At m=1053 every measured shape predicts 32 rows (132, 924, 5016 workgroups, all ≥ 48). At m=30 the 16-row twin fires for n=128 and n=896 (8 and 56 workgroups) and stays 32-row for n=4864 (152 workgroups).

Shape A/B kernel medians, compiled-in floor vs `WG_PER_CORE=0`:

| m | n | k | off (32) µs | on µs | rows on | on vs off |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 128 | 896 | 247.951 | 211.173 | 16 | −14.83% |
| 30 | 896 | 896 | 258.097 | 211.887 | 16 | −17.90% |
| 30 | 896 | 4864 | 1011.265 | 749.520 | 16 | −25.88% |
| 30 | 4864 | 896 | 432.747 | 431.602 | 32 | −0.26% |
| 1053 | 128 | 896 | 465.156 | 464.322 | 32 | −0.18% |
| 1053 | 896 | 896 | 1769.200 | 1765.408 | 32 | −0.21% |
| 1053 | 896 | 4864 | 8968.649 | 8968.313 | 32 | −0.00% |
| 1053 | 4864 | 896 | 8687.192 | 8698.730 | 32 | +0.13% |

End-to-end legs, five interleaved rounds:

| Leg | off median | on median | on vs off | digest |
| --- | ---: | ---: | ---: | ---: |
| short prefill, 30 tokens | 332.0043 tok/s | 377.3840 tok/s | +13.67% | `7fd25a869ff21678` |
| short decode, 30/32 | 111.4325 tok/s | 111.7721 tok/s | +0.30% | `7fd25a869ff21678` |
| ctx1024 prefill, 1053 tokens | 1112.6267 tok/s | 1112.1047 tok/s | −0.05% | `7da83f06ec9f001d` |
| ctx1024 decode, 1053/32 | 96.3968 tok/s | 96.1332 tok/s | −0.27% | `7da83f06ec9f001d` |

Same first/last IDs as jw16. Five of five rounds. Non-firing 1053 path is digest-identical and not slower. Short-prompt path fires and is faster.

## Artifacts

Under `receipts/2026-09-14-qmm-occupancy-tilem/`:

- `jw16/final-legs.jsonl`, `jw16/spirv-gate.txt`, `jw16/provenance-final.txt`, `jw16/lock-final.txt`
- `jwm1/legs.jsonl`, `jwm1/shapes.jsonl`, `jwm1/spirv-gate.txt`, `jwm1/provenance-legs.txt`, `jwm1/lock-legs.txt`

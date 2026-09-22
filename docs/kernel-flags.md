# Kernel feature and serve-patch flags (Qwen3.8-2B integration, 2026-09-22)

Runtime switches for the integrated kernel set (fused GDN decode/prefill,
bf16 coopmat qmm prefill, q4 gemv xpack) and the vendored mlx-lm serve
patches. All default to the shipped configuration; the flags exist for
A/B gating and qualification, not for normal operation.

## GPU kernel flags (backend, wheel)

| Flag | Default | Effect |
|---|---|---|
| `MLX_OMARCHY_KV_DIRECT` | on (`1`) | Direct decode KV window storage in fused chains. `0` composes ops; output-neutral. |
| `MLX_OMARCHY_NO_COOPMAT` | unset (route on) | `1` disables the bf16 coopmat qmm-prefill route; output-neutral. |
| `MLX_OMARCHY_QMM_VEC_Q4_WORD` | on | Packed q4 word path in qmm gemv (xpack). `0` reverts to scalar; may change which ulp-level near-ties flip. |
| `MLX_OMARCHY_QMM_TILE`, `_QMM_TILE_RB`, `_QMM_COOPMAT_WG_PER_CORE` | tuned defaults | Tile/workgroup sizing overrides for qualification. |
| `GDN_FALLBACK_DEBUG` | unset | Prints fused-eligibility inputs for the GDN prefill route when set. |
| `MLX_OMARCHY_FUSED_AB` | — | ANE-side experimental flag; **ignore on the GPU path** — it has no effect on the Vulkan backend. |

## mlx-lm serve patches (vendored in `patches/`, applied by
`scripts/apply-mlx-lm-patches.sh`, wired into `install.sh`)

- `mlx-lm-gated-delta-fast-route.patch` — **default ON.** Routes
  `mx.fast.gated_delta_update` in mlx-lm 0.31.3 to the fused kernel in the
  mlx-omarchy wheel. Required for the measured serve numbers.
- `mlx-lm-convring.patch` — **default OFF.** Rolling conv-state ring; enable
  per venv with `MLX_OMARCHY_CONV_RING=1 scripts/apply-mlx-lm-patches.sh
  /path/to/venv`.
- Both patches are idempotent and fail loudly on mlx-lm version drift.

## Acceptance bar

Cross-host digest equality is not a gate (see README performance section
and `ane-linux-experiments` receipts 2026-09-22-qwen38-integration /
2026-09-22-qwen38-correctness): per-host determinism plus logit-level
equivalence (≤ 2 bf16 quanta) is.

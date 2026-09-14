# Decode epilogue fold: 72 fewer dispatches, 1053 digest does not hold

Date: 2026-09-14
Task: fold FastRopeF16 and SwigluF16 into the fused Q4 GEMV store
(priced by `receipts/2026-09-14-decode-gap-attribution.md`).

## Verdict

**The fold is implemented and dispatch-correct. It does not land.**

On jw16 Honeykrisp, one decode token drops from 249 compute dispatches
to **177** (exactly the priced −72) and from 2 submissions to 1.
`gpu_primitive_dispatches` stays 858, so the fold did not reintroduce
host-side fixups. `MLX_OMARCHY_FOLD_EPILOGUE=0` restores 249 / 2 on the
same cand wheel.

The pinned short digest holds on both arms:
`7fd25a869ff21678`. The pinned 1053 digest does **not**: cand produces
`31267e7ed4c6d0dc` against pinned `7da83f06ec9f001d`. First three ids
match (`13060,498,369`); later ids diverge. The same cand wheel with
`MLX_OMARCHY_FOLD_EPILOGUE=0` restores `7da83f06ec9f001d`, so the
divergence is the fold, not the rest of the shader layout.

C++ bit-exact tests of the isolated RoPE fold (offsets 0, 7, 1052) and
the isolated SwiGLU fold passed on Honeykrisp after a rounding fix
(packHalf2x16 → `float(float16_t(x))`, commit `50870b69`). They did not
catch the full-model 1053 greedy divergence. jwm1 was not measured:
Main queued this slice behind AneResidentCache / QmmOccupancyTileM /
MelFrontendPerf, and the 1053 digest already fails the hard gate.

A digest change is a failed fold, not a new baseline. Default remains
the unfolded graph. Branch `wave/DecodeEpilogueFold` keeps the code and
the `MLX_OMARCHY_FOLD_EPILOGUE=0` bisection knob.

## Identity

- Branch: `wave/DecodeEpilogueFold` at `50870b69` (fold) on parent
  `b79a4b68` (`wave/DecodeEpilogueFold-base`).
- Stale DecodeGemv worktree (`stale/2026-09-02-decodegemv-worktree`)
  was read and not merged: it is a dense/qmm `m==1` kernel split,
  superseded by `QmmVecQ4Multi` on this lineage.
- Wheels, both built on jw16 with that host's glslc  (Arch openblas
  include + FetchContent seed, private `.work` per side):
  - base `mlx_omarchy-0.32.2.dev202609141639+b79a4b68-cp314-cp314-linux_aarch64.whl`
    sha256 `beeeaee99834a24db788ffa8ea809689667343327a271bafd8576a2779c719b7`
    libmlx.so `fdfc6a6db21e4f23…` verified=match
  - cand `mlx_omarchy-0.32.2.dev202609141654+50870b69-cp314-cp314-linux_aarch64.whl`
    sha256 `b0683fcafab1eb05af8303eb7a9e96e1deffe31280a6c21066fc7bfef45aa46f`
    libmlx.so `7defa9b4bf0f8f7e…` verified=match
- Host: `jw16mbp1-linux`, Apple M1 Max, lock inode 12 held with
  `flock -w 60` (nested `flock -n` returned 1), never stolen, released
  (`flock -n` = 0 afterwards). No ANE, reboot, 1x896, SET write.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` snapshot
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
- mlx-lm 0.31.3. `MLX_DISABLE_COMPILE=1` `HF_HUB_OFFLINE=1`.
- Agent model: `anthropic/claude-fable-5-1`.

## Dispatch counts (jw16, 8 greedy tokens, prompt 30)

| arm | vk_compute_dispatches / token | vk_submissions | gpu_primitive_dispatches |
| --- | ---: | ---: | ---: |
| base `+b79a4b68` | 249 | 2 | 858 |
| cand fold on `+50870b69` | **177** | 1 | 858 |
| cand `FOLD_EPILOGUE=0` | 249 | 2 | 858 |

Raw JSON: `receipts/2026-09-14-decode-epilogue-fold/jw16/dispatch-*.json`.

## Digests (jw16)

| arm | short | 1053 |
| --- | --- | --- |
| pinned | `7fd25a869ff21678` | `7da83f06ec9f001d` |
| base | `7fd25a869ff21678` | `7da83f06ec9f001d` |
| cand fold on | `7fd25a869ff21678` | **`31267e7ed4c6d0dc`** |
| cand fold off | (not re-run short) | `7da83f06ec9f001d` |

cand 1053 ids: first=`13060,498,369` last=`315,279,28431`
pinned 1053 ids: first=`13060,498,369` last=`3897,553,279`

## Rates (incomplete: one interleaved round, aborted on digest)

jw16 round 0 only, not a median, not a claim:

- short: base 169.71 tok/s, cand 186.76 tok/s
- ctx1024: base 132.12 tok/s, cand not reported (digest fail)
- isolation ctx1024 wall: base 7.2 ms/token, cand-on 6.5, cand-off 7.6

Priced gain was 143.19 → 159.05 tok/s at 1053 on jw16. Not measured.

jwm1: not run.

## Tests

llvmpipe (this workstation, `MLX_OMARCHY_ALLOW_NON_APPLE=1`):
`omarchy_fused_chain_tests` 32/32, `omarchy_kv_ops_tests` 16/16.

jw16 Honeykrisp, under the lock, after `50870b69`:
`omarchy_fused_chain_tests -tc=*gemv*` 6/6, 51269 assertions;
`omarchy_kv_ops_tests -tc=producer-direct*` 1/1, 352 assertions.
The RoPE case had failed 534 assertions under packHalf rounding
(`54559007`) at offsets 7 and 1052, output 3 (q_rot); offset 0 and all
GEMV/Add outputs were already bit-exact. `float(float16_t(x))` cleared
it.

## Why it did not land

Hard gate: digests identical on both hosts, including 1053. cand fold-on
fails that gate on jw16. The isolated C++ folds are bit-exact, the
full greedy 1053 graph is not. Next step, not done here: dump per-layer
q_rot / k_rot / swiglu against the per-node path at context 1053 and
find the first differing tensor. Until that is bit-exact, leave the
fold behind `MLX_OMARCHY_FOLD_EPILOGUE` (defaults on in this branch;
do not merge).

## Diff

`50870b69` on `wave/DecodeEpilogueFold`. Files:

- `overlay/mlx/backend/omarchy/shaders/qmm_vec.comp`
- `overlay/mlx/backend/omarchy/fused_chain.{h,cpp}`
- `overlay/mlx/backend/omarchy/primitives.cpp`
- `overlay/mlx/backend/omarchy/compute.h` (binding budget 19 → 22)
- `overlay/tests/omarchy/test_fused_chain.cpp`
- `docs/install-omarchy.md`, `docs/chip-portability.md`

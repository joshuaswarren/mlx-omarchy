# bf16 decode dispatch census: gates off vs on (RoPE direct + SDPA fast)

Date: 2026-09-04. Branch `bf16-decode-path` @ `a3d05a4` (base `999e25c`).
Author: Bf16DecodePath worker. llvmpipe dispatch COUNTS and structure
only - no Apple-GPU speed claim, and no timing claim of any kind (the
box ran at load ~62 with seven workers; inter-token times are
meaningless and not reported as evidence).

## Result

| leg | dispatches/token (median) | greedy |
|---|---|---|
| bf16 eager, gates OFF (default) | **774** | (see caveat) |
| bf16 eager, both gates ON | **606** | "The capital of France is Paris." |

Delta: **-168/token**, measured, matching the fusion-plan designs 1+2
phase-A prediction (-120 SDPA, -48 RoPE). 4-bit/f16 paths untouched
(FastRopeF16/MatmulF16 legs never selected by these gates; gate-off
code paths are byte-identical to base).

## Kernel composition, median decode token

Gates OFF (774): MatmulBF16 169, ElementwiseBF16 121, CastBF16F32 120,
CopyGeneralBF16 96, CastF32BF16 72, FastRmsNormBF16 49, FastRopeF32 48,
MatmulF32 48, ElementwiseF32 24, SoftmaxF32 24, GatherBF16 1,
LogSumExpBF16 1, ArgReduceBF16 1 - byte-for-byte the pinned main
composition (fusion-plan receipt, Result 1).

Gates ON (606): MatmulBF16 217, ElementwiseBF16 121, CopyGeneralBF16
96, FastRmsNormBF16 49, FastRopeBF16 48, CastF32BF16 48, SoftmaxBF16
24, GatherBF16 1, LogSumExpBF16 1, ArgReduceBF16 1.

Shifts: CastBF16F32 120 -> 0 (72 SDPA q/k/v + 48 rope up-casts
deleted); CastF32BF16 72 -> 48 (SDPA output downcast deleted, rope's
proven tail stays); FastRopeF32 -> FastRopeBF16 (packed-word leg
selected); MatmulF32 48 and SoftmaxF32 24 and ElementwiseF32 24 deleted
into MatmulBF16/SoftmaxBF16 (+48 matmul, +24 softmax).

## Provenance and procedure

- Wheel: `mlx_omarchy-0.32.2.dev202609041412+999e25c` built from the
  branch working tree (diagnostics, MLX_OMARCHY_GPU_PROFILING=ON),
  wheel sha256 `94d80d31f7be36a7e7589e23f74ba688c4939d145b96d304ebc6be864594f857`;
  installed `libmlx.so` sha256 `3cb0e3c44eb9ca8050943df3eff65007d0ef4cfe06c3708e821b186e5f8b3cd3`
  == wheel member (verified=match).
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-bf16` snapshot
  `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`; mlx-lm 0.31.3; prompt
  "What is the capital of France?", temp 0, seed 0;
  `MLX_DISABLE_COMPILE=1` (bf16 compiled tapes are refused by design).
- Gates-off leg: `MLX_OMARCHY_ROPE_BF16_DIRECT` and
  `MLX_OMARCHY_SDPA_BF16_FAST` unset. Gates-on leg: both `=1`.
- Census: `scripts/chain_census.py <profile>.jsonl --compute-h
  .work/mlx/mlx/backend/omarchy/compute.h --markers <markers>.jsonl`;
  one torn tail line dropped from the killed gates-off profile.

## Caveats, plainly

- Gates-off leg: PARTIAL - the harness was killed at the 1800 s
  command timeout after 29 decode tokens (28 of 29 at exactly 774,
  first token 748); its greedy text corrupted mid-run (see the
  offset-scalar finding below). Dispatch structure is value-independent
  (a wrong offset changes position math, not the kernel stream), so the
  774 count is valid; the corruption is why the leg is labeled partial.
- Gates-on leg: 8 decode tokens to EOS, clean text, min=median=max=606.
- Both legs were measured under host contention (load ~62, seven
  workers); llvmpipe dispatch counts are timing-independent, and no
  performance number is claimed from this box.

## Offset-scalar defect: two independent live events on 999e25c+gates-off

1. First census attempt: `Vulkan timeline counter failed to advance for
   10000 ms` watchdog kill (fresh process, retry).
2. Second attempt, fresh process: named refusal during prefill -
   `RoPE rotational argument magnitude 1025376256.000000 exceeds the
   built-in accuracy limit 100000.000000` - the rope gate read garbage
   from the offset scalar and refused by name (correct contract
   behavior). Third attempt survived prefill and then produced
   corrupted greedy text without a refusal (offset values wrong but
   under 1e5). All three legs ran with the gates OFF, i.e. the RoPE/SDPA
   code path is byte-identical to base `999e25c`; the a108741 bf16
   drain does not prevent the corruption, consistent with the
   hypothesis that a buffer on the bf16 path is released before its
   writer retires and the page is recycled (Bf16ReadinessAudit owns the
   lifetime audit; no cause claim here).

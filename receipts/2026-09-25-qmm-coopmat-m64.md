# 2026-09-25 — qmm coopmat TILE_ROWS=64 for big-M prefill (jwm1/T8103)

Branch `agent/jwm1-parity5-coopmat-m64` (9b250d7 + ba38c1c on top of
`ecda0fa33`). Lane: Jwm1Parity5; receipts and the final per-metric matrix
live in ane-linux-experiments
`receipts/2026-09-25-jwm1-parity5-gpu-parity-m64/`.

## Why

M1 prefill-512 is compute-bound on the qmm coopmat pipe: 235 tok/s vs the
macOS Metal 343.73 bar (0.685x). The fork's coopmat cell measured 1034.6
GFLOP/s at the dominant gate_up shape (`2026-09-12-prefill-fma-qualify`)
while Metal reaches ~1.41 TFLOP/s on the same shapes. In
`shaders/qmm_coopmat.comp` the packed weight word is re-staged and
re-dequanted (8 fma/lane) once per (m-tile, n-tile, k-step): at M=512,
TILE_M=32 runs 16 m-tiles, so every weight word is loaded and dequanted
16x per output pass.

## Change

- `TILE_ROWS=64` (`SUBGROUP_ROW_BLOCKS=4`); the drain loop and row guards
  were already generic in `SUBGROUP_ROW_BLOCKS`. Weight staging cost per
  FLOP halves; the MMA per output element is unchanged.
- `qmm_coopmat_m64_{f16,bf16,x32}` shader variants (CMake), append-only
  compute-kernel ids (GPU-profile ids stable).
- `coopmat_tile_rows` now returns the widest tile (64 -> 32 -> 16) whose
  grid still fills cores x workgroups_per_core, restricted to M > 32:
  below one m32 tile the clamped-row MMA waste outweighs the staging
  saving (TTFT-path small-M prefill keeps the 16/32 heuristic).

Bit-exactness: per-output fp32 k chain unchanged (CHUNK_K=64, STEP_K=16,
same coopMatMulAdd sequence); only the m/n work partition moves. Rows
past matrix_m are clamped on load and guarded on store as before.

## Gates

PENDING (this file is completed at landing): 10x interleaved r1
contracts ctl-vs-cand with paired t-CI on prefill/TTFT/decode/e2e, 0
token flips per pass, r1 digest `486872c410629f1d` per pass, and one
10-pass cand contract whose digest must equal the dbf704971617fdfc pin.

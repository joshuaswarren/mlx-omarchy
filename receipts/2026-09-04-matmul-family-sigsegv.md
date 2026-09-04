# omarchy_matmul_family_tests SIGSEGV: pre-existing on origin/main 999e25c

Date: 2026-09-04. Evidence preserved at Main's request; no reruns beyond
what is recorded here. Author: Bf16DecodePath worker.

## Claim

`omarchy_matmul_family_tests` crashes with SIGSEGV in full-binary runs
on the dev box on PRISTINE origin/main commit `999e25c` (no working-tree
changes). The crash position wanders between cases and the executed
case count varies run to run. The failure is therefore NOT attributable
to the bf16-decode-path diff (RoPE/SDPA gates), which was absent from
the pristine build below.

## Build provenance (pristine side)

- Worktree: `/home/joshuawarren/.config/superpowers/worktrees/mlx-omarchy/bf16-wt`
  at `999e25c`, working tree stashed (`git stash`), then
  `./scripts/prepare-mlx.sh` re-staged (overlay = 999e25c only).
- Configure: `cmake -S .work/mlx -B build-bf16 -DCMAKE_BUILD_TYPE=Release
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF
  -DMLX_BUILD_BENCHMARKS=OFF`
- Target: `cmake --build build-bf16 --target omarchy_matmul_family_tests
  -j$(nproc)` - zero errors.
- Run command: `MLX_OMARCHY_ALLOW_NON_APPLE=1
  ./build-bf16/tests/omarchy/omarchy_matmul_family_tests`
- Box state: shared dev host, load average ~62 on 16 cores at the time
  of the later runs; llvmpipe (Mesa software Vulkan) via
  MLX_OMARCHY_ALLOW_NON_APPLE=1.

## Recorded runs (pristine 999e25c build)

Three consecutive full-binary runs, 2026-09-04:

1. `test_matmul_family.cpp:682: FATAL ERROR: test case CRASHED: SIGSEGV`
   - "test cases: 3 | 2 passed | 1 failed | 4 skipped"
2. `test_matmul_family.cpp:1096: FATAL ERROR: test case CRASHED: SIGSEGV`
   - "test cases: 5 | 4 passed | 1 failed | 2 skipped"
3. `test_matmul_family.cpp:682: FATAL ERROR: test case CRASHED: SIGSEGV`
   - "test cases: 3 | 2 passed | 1 failed | 4 skipped"

Assertions completed before each crash all passed (12,345 in a
representative earlier run on a mixed build).

Isolation check (same day, same box, MY branch build before rebase):
`--test-case="segmented mm writes per-segment contractions"` passes 3/3
in isolation; the crash only appears in full-binary runs, and moves to
a different case (`gather qqmm dequants with scales only`, line 1096)
between runs - consistent with cross-case state or an allocator/lifetime
sensitivity, not a single bad case.

Earlier same-day observation on a mixed build (shared checkout,
94d967c + qmm-prefill-tile's 93ba83d + bf16 work in the tree): same
SIGSEGV at the same line 682 case in full runs, 1 failed of 8 cases,
5 of 5 full runs failing while isolation passes - before the pristine
builds above proved main-alone is red.

## What this does and does not establish

- Established: current origin/main is red for this battery on the dev
  box; the red is independent of the bf16-decode-path diff (pristine
  side built and crashed without it).
- Not established: root cause (no debugger run, no poison detector -
  per Main, no crash-reproduction workflows from this worker).
- The batteries that cover the bf16-decode-path change
  (omarchy_fast_ops_tests 21/21, omarchy_runtime_tests 26/26) are green
  on the same builds; sdpa_equivalence.py ALL PASS in all three gate
  states.

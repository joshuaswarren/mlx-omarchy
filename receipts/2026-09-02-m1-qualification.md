# 2026-09-02 — M1 qualification of feat/vulkan-primitives @ 5f8ba16

Purpose: qualify today's coverage waves + defect fixes on the real target.
Everything landed 2026-09-02 was verified only on llvmpipe until this run.

**Measurement condition (added 2026-09-03):** all jwm1-linux timings in this document (build wall times, prompt/gen tok/s) were taken with only 1 of 8 CPU cores online (GRUB entry `Omarchy ANE test`, which supplies a static device tree and left cores 1-7 offline). The machine now boots all 8 cores. GPU-bound figures were materially unaffected (matmul TFLOP/s median 0.1556 -> 0.1576); these host-bound figures may improve on re-measurement.

## Devices

| Role | Machine | Device string |
|---|---|---|
| Target | jwm1 (192.168.3.66), Asahi Linux, Apple M1 (T8103, 8 GPU cores, 16 GB) | pending on-device capture |
| Reference | dev box, x86_64, Mesa lavapipe / llvmpipe (software), `MLX_OMARCHY_ALLOW_NON_APPLE=1` | llvmpipe |

**Hardware-state caveat, prominent:** the M1 ran SINGLE-CORE for every number
below. `nproc`=1; dmesg shows CPU1-7 failing to come online at boot
("failed in unknown state : 0x0"), a cold reboot did NOT fix it, root cause is
an m1n1 1.5.2 spin-table/enable-method mismatch with kernel
7.1.6.asahi1-1 (escalated to Joshua; m1n1 flash needs hands-on DFU recovery).
Earlier M1 receipts (2026-08-31, 2026-09-01) were measured on the 8-core
configuration. Comparisons against them carry that confound. Build wall times
in this receipt are single-core numbers.

## Commit under test

`5f8ba16cf9080ca3eb8a5ad59e3377df28450650` (feat/vulkan-primitives), the
dev box's committed HEAD at run time; origin already matched. The dev box
working tree additionally holds UNCOMMITTED sibling edits (primitives.cpp,
linalg_lu.comp, linalg/matmul/primitives tests, docs) — none of that is in
this qualification. Both machines built from clean detached worktrees at
exactly this commit. (While this qualification ran, Main committed
`09d9e15` and `b188d13` — docs-only receipts/disclosures. They do not touch
the tested C++ tree; 5f8ba16 remains the code this receipt describes.)

## LUF status at the tested commit (scope finding)

At 5f8ba16, LUF is STILL GATED:
`overlay/mlx/backend/omarchy/primitives.cpp:3218` refuses every call with
`[LUF] gated pending numeric verification`, and the committed suite carries
three gate cases ("luf stays gated pending numeric verification", "luf
batched stays gated", "luf gate fires before the singular check"). The
pivot-search sentinel fix, the ungating, and the 300x300 case exist ONLY as
uncommitted sibling edits and are NOT part of this commit. What this run
therefore verifies on-device: the gate fires by name on Honeykrisp. The
ungated 300x300 LUF run cannot exist at this commit on any machine.

## llvmpipe reference (dev box, same commit, 2026-09-02)

Fresh worktree `mlx-omarchy-qual-llvmpipe` at 5f8ba16,
`./scripts/prepare-mlx.sh` + CMake `-DMLX_BUILD_OMARCHY=ON
-DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF
-DMLX_BUILD_TESTS=ON` (python bindings/examples/bench off), ninja `-j24`,
build + 2 battery passes in 50 s total. Every binary rc=0 in both passes;
pass-1 vs pass-2 logs byte-identical (`diff -r` clean).

| binary | cases | assertions |
|---|---|---|
| omarchy_ane_bundle_tests | 12 | 131 |
| omarchy_compiled_tape_tests | 8 | 343 |
| omarchy_copy_offset_tests | 7 | 68 |
| omarchy_distributed_tests | 7 | 23 |
| omarchy_error_contract_tests | 3 | 14 |
| omarchy_fast_ops_tests | 11 | 375 |
| omarchy_fft_ops_tests | 17 | 1376 |
| omarchy_indexing_ops_tests | 36 | 1723 |
| omarchy_kv_ops_tests | 14 | 407 |
| omarchy_linalg_ops_tests | 27 | 497 |
| omarchy_matmul_family_tests | 6 | 13164 |
| omarchy_primitive_tests | 86 | 602555 |
| omarchy_reduce_ops_tests | 14 | 1934 |
| omarchy_runtime_tests | 22 | 6188 |
| omarchy_shape_ops_tests | 24 | 266 |
| **total** | **294** | **629,064** |

Note: the session's quoted baseline (~298 cases / ~813,750 assertions,
matmul_family red) corresponds to the tree WITH uncommitted sibling test
edits, not to 5f8ba16. At the committed HEAD the whole battery is green on
llvmpipe, including omarchy_matmul_family_tests (6/6). The same-commit
pairing above is the correct llvmpipe-vs-M1 comparator and is what this
receipt uses throughout.

## M1 C++ battery — REAL HARDWARE RESULTS

Device evidence captured 2026-09-02 10:17:57 CDT by `mlx-omarchy-info` on
jwm1: device `Apple M1 (G13G B1)`, driver `Mesa Honeykrisp`, Vulkan API
`1.4.354`, shader float16/int16/16-bit storage all yes, unified memory yes,
`total memory: 7738 MiB`. `MLX_OMARCHY_ALLOW_NON_APPLE` NOT set anywhere.
Build: fresh worktree `~/src/mlx-omarchy-qual` at 5f8ba16, `nproc`=1
confirmed in the build log, `./scripts/prepare-mlx.sh` + cmake + ninja
`-j1`: **338 s wall** (10:07:33 -> 10:13:11 CDT).

Battery: 15 binaries, `ulimit -c 0`, run from
`/home/joshuawarren/src/mlx-omarchy-qual/.work/build`. **Four full passes**
(2 script invocations x 2 passes; passes 1+2 10:17:57-10:18:41 CDT, 44 s;
passes 3+4 44.9 s).

### Per-binary counts (identical in all 4 passes)

| binary | cases | asserts | failed | verdict |
|---|---|---|---|---|
| omarchy_ane_bundle_tests | 12 | 131 | 0 | PASS |
| omarchy_compiled_tape_tests | 8 | 343 | 0 | PASS |
| omarchy_copy_offset_tests | 7 | 68 | 0 | PASS |
| omarchy_distributed_tests | 7 | 23 | 0 | PASS |
| omarchy_error_contract_tests | 3 | 14 | 0 | PASS |
| omarchy_fast_ops_tests | 11 | 375 | 0 | PASS |
| omarchy_fft_ops_tests | 17 | 1376 | 0 | PASS |
| omarchy_indexing_ops_tests | 36 | 1723 | 599 | **FAIL (2 cases)** |
| omarchy_kv_ops_tests | 14 | 407 | 0 | PASS |
| omarchy_linalg_ops_tests | 27 | 497 | 0 | PASS |
| omarchy_matmul_family_tests | 6 | 13164 | 0 | PASS |
| omarchy_primitive_tests | 86 | 602555 | 30 | **FAIL (2 cases)** |
| omarchy_reduce_ops_tests | 14 | 1934 | 0 | PASS |
| omarchy_runtime_tests | 22 | 6189 | 0 | PASS |
| omarchy_shape_ops_tests | 24 | 266 | 0 | PASS |
| **total** | **294** | **629,065** | **629** | **FAIL — 4 cases** |

285 of 294 cases pass; 629 of 629,065 assertions fail (4 cases, 2 binaries).

### M1 vs llvmpipe — every named difference

1. **omarchy_primitive_tests::`LogicalAnd and LogicalNot match host
   references`** (test_primitives.cpp:5449). 28 failed assertions: 12x
   `CHECK_EQ(landed.data<bool>()[i], and_expected[i])` -> `false, {?}` at
   line 5468 (packed-bool read-back returns bytes that are not valid 0/1
   bools) and 16x at line 5476 for the scalar-condition broadcast leg. The
   test's own comment names the trigger: 33 elements cross the 32-bit word
   packing boundary. Deterministic in all 4 passes. llvmpipe: green.
2. **omarchy_primitive_tests::`integer Remainder, DivMod, Power, Sign, and
   Abs match host references`**. 2 failed assertions at line 68
   (check_int32_values helper): both `CHECK_EQ( -2, -3 )` — device produced
   -2 where the host reference expects -3. Silent wrong integer values,
   deterministic. llvmpipe: green.
3. **omarchy_indexing_ops_tests::`masked_scatter fills true positions from
   the source in order`** (test_indexing_ops.cpp:602). 1 failed assertion:
   `CHECK( 5 == Approx( -3 ) )` — a scattered value landed as 5 where -3
   was written. llvmpipe: green.
4. **omarchy_indexing_ops_tests::`masked_scatter carries the scan across
   chunks and rows`** (test_indexing_ops.cpp:656). 598 failed assertions:
   `0 == Approx( -2 )`, `0 == Approx( -3 )`, ... — the scan across chunks/
   rows lands zeros instead of the scattered negatives. llvmpipe: green.
5. **omarchy_runtime_tests assert count**: 6189 on M1 vs 6188 on llvmpipe
   (+1 platform-conditional assertion; both passes green, all pass).
   Everything else: case and assert counts are IDENTICAL to llvmpipe.

### Convergence gates (the headline question) — NO divergence

**omarchy_linalg_ops_tests: 27/27 cases, 497/497 assertions, rc=0, on the
real M1.** Eigh and SVD Jacobi sweep caps did NOT trip on Honeykrisp for the
same inputs that converge on llvmpipe; the refusal-by-name paths are
exercised by their own cases and also pass. No case that converges on one
device refuses on the other. FFT likewise green: 17/17 cases, 1376/1376.

### Nondeterminism — verdict over 4 passes

`diff -r` across passes: every log byte-identical across all 4 passes with
ONE exception — omarchy_runtime_tests.log line 8, a `[receipt]` TIMING line
printed by the binary's own scheduling probe:
`serialized_median=0.4347s/0.4336s/0.3973s... concurrent_median=... 
ratio=1.27865/1.20835/1.12741`. This is wall-clock measurement jitter with a
consistent outcome (concurrent faster than serialized every pass), not a
result variation. **No test case's PASS/FAIL or values varied between runs.
Zero result nondeterminism in 4 passes x 629,065 assertions.** (Context:
M1Bf16Tape separately reproduced bf16 COMPILED-TAPE corruption on this box
at nproc 1; the compiled_tape suite itself is green in all 4 passes here.)

## Build-integrity check (Main's staging hazard)

Main warned that a build consumes the STAGED `.work/mlx` copy, refreshed only
by `prepare-mlx.sh`. This run is clean by construction — a fresh isolated
worktree, zero source edits, one `prepare-mlx.sh` before the single configure/
build — and it was verified after the fact:
`diff -r overlay/mlx/backend/omarchy .work/mlx/mlx/backend/omarchy` in the
qual worktree returns clean (STAGING_MATCHES_OVERLAY) and `git status
--porcelain` in the worktree is empty. Every number in this receipt comes
from code byte-identical to commit 5f8ba16.

## Model qualification — mlx-lm on the M1 (real device)

Wheel: `mlx_omarchy-0.32.2.dev20260902+5f8ba16-cp314-cp314-linux_aarch64.whl`,
2,799,362 bytes, sha256 `2a52423699c61517ffba9cc84eec7598fb573cef61bbf3ff38a239d8cbe9f9de`
(`scripts/build-wheel.sh`, 679 s single-core). Installed into a FRESH venv
(`~/src/mlx-omarchy-qual/.work/venv-qual`) with mlx-lm 0.31.3 (upstream mlx
correctly skipped via its Darwin marker). Verified before runs:
`version 0.32.2.dev20260902+5f8ba16`, `device Device(gpu, 0)`,
`device_name Apple M1 (G13G B1)`, `architecture honeykrisp`.
`MLX_DISABLE_COMPILE=1` on every leg (bf16 compiled tape is gated by design);
`MLX_OMARCHY_ALLOW_NON_APPLE` never set. Models from the pinned local
snapshots used by every prior receipt.

Six legs, `--max-tokens 32` (all ended on natural EOS):

| leg | prompt | text (verbatim) | prompt tok/s | gen tok/s | peak GB |
|---|---|---|---|---|---|
| bf16 greedy | "Hi" | `Hello! How can I assist you today?` | 21.452 | 2.810 (10 tok) | 0.993 |
| bf16 greedy | France | `Paris` | 30.364 | 5.208 (2 tok) | 0.993 |
| bf16 temp 0.9 seed 0 | "Hi" | `Hello! How can I assist you today?` | 20.427 | 2.616 (10 tok) | 0.995 |
| q4 greedy | "Hi" | `Hello! How can I assist you today?` | 17.971 | 2.883 (10 tok) | 0.292 |
| q4 greedy | France | `Paris` | 21.192 | 5.057 (2 tok) | 0.292 |
| q4 temp 0.9 seed 0 | "Hi" | `Hello! How can I assist you today?` | 16.962 | 3.005 (10 tok) | 0.292 |

### Text vs recorded receipts — NO DEGRADATION

- `Paris` for the France prompt, both bf16 and 4-bit — identical to the
  2026-09-01 receipts (6049968 release verification and b9745b2 qualification).
- Coherent greeting for "Hi", both dtypes — identical to the receipts' text
  (`Hello! How can I assist you today?`, 10 tokens, natural EOS).

### Sampling is genuinely sampling (control legs)

The seeded temp-0.9 legs landed on the modal sequence, so two UNSEEDED
temp-0.9 legs were run as controls. Both diverged from greedy and stayed
coherent:

- bf16 unseeded temp 0.9: `Hello! How may I assist you today? My name is
  Qwen, and I can be really helpful with a wide variety of topics. Please
  feel free to`
- 4-bit unseeded temp 0.9: `Hello! How may I assist you today? I'm here to
  help with any questions you might have about AI and other technological
  topics. Please feel free to ask`

### Run-to-run variation (model level)

All six legs were re-run end to end into a second output set. **Generated
text: byte-identical in 6/6 legs.** Only tok/s jittered (bf16 gen 2.810 ->
2.913; q4 gen 5.057 -> 5.005 tok/s) and peak memory was stable (bf16
0.993-0.995 GB both passes; q4 0.292 GB both passes).

### Numbers vs the 2026-09-01 receipts

| metric | 09-01 receipts | today (single core) | delta |
|---|---|---|---|
| bf16 "Hi" gen tok/s | 2.044-2.520 | 2.616-2.913 | faster |
| bf16 "Hi" prompt tok/s | 16.139-18.584 | 20.427-21.452 | faster |
| bf16 peak GB | 0.993-1.025 | 0.993-0.995 | same |
| 4-bit France gen tok/s | 4.223-4.269 | 5.005-5.208 | faster |
| 4-bit France prompt tok/s | 19.156-19.197 | 21.192 | faster |
| 4-bit peak GB | 0.292-0.320 | 0.292 | same |

Caveat: the 09-01 numbers were measured on the 8-core configuration; today's
on one core. Direction of the comparison is still favorable and memory is
bit-for-bit the q4 number from the release receipt.

## Verdict

**Today's work HOLDS UP on real hardware, with two named exceptions to fix.**

1. The two suites this qualification targeted as unknowns are CLEAN on the
   M1: linear algebra 27/27 cases, 497/497 assertions green — Eigh/SVD
   Jacobi convergence behaves IDENTICALLY on Honeykrisp and llvmpipe (no
   sweep-cap trip on either); FFT 17/17, 1376/1376 green. LUF's committed
   contract (refuse by name) also verifies on-device; note the ungating +
   300x300 case are NOT in commit 5f8ba16 (uncommitted sibling work), so
   "LUF verified on-device" is true only for its gate at this commit.
2. **Four silent wrong-value cases found — they do not raise, they return
   wrong numbers:** packed-bool LogicalAnd transport (28 asserts, invalid
   bool bytes), integer Remainder/DivMod/Power/Sign/Abs leg (-2 where -3 is
   correct, 2 asserts), and masked_scatter (599 asserts, scattered values
   lost/zeroed across chunks and rows). All deterministic across 4 passes,
   all green on llvmpipe at the same commit — device/kernel interactions,
   NOT Honeykrisp nondeterminism.
3. **Zero run-to-run nondeterminism** in 4 battery passes x 629,065
   assertions (single timing-receipt line in omarchy_runtime_tests jitters;
   results never do) and in 2 model passes (text byte-identical).
4. End-to-end model behavior matches the receipts verbatim in both dtypes,
   greedy and sampled, with faster tok/s and identical peak memory.
   The README's end-to-end claim stands at 5f8ba16.

Bottom line for the fix owners: the M1-only failures are confined to
omarchy_primitive_tests (2 cases) and omarchy_indexing_ops_tests (2 cases),
named above with exact assert lines. Everything else that landed today —
including the two suites that had never touched real hardware — is qualified
on the real device.

## Artifacts and reproduction

Raw evidence (all four battery passes, both model passes, build logs,
llvmpipe reference passes): `receipts/m1-qualification-2026-09-02/`.

Left on jwm1: worktree `~/src/mlx-omarchy-qual` (detached at 5f8ba16, clean),
venv `~/src/mlx-omarchy-qual/.work/venv-qual`,
wheel `~/src/mlx-omarchy-qual/dist/`, logs `/tmp/qual-*`. The main checkout
`~/src/mlx-omarchy` was NOT touched by this run.

Reproduction (jwm1, single-core timing):

```sh
cd ~/src/mlx-omarchy
git worktree add --detach ~/src/mlx-omarchy-qual 5f8ba16
cd ~/src/mlx-omarchy-qual
./scripts/prepare-mlx.sh
cmake -S .work/mlx -B .work/build -G Ninja -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build .work/build -j1 --target omarchy_runtime_tests \
  omarchy_copy_offset_tests omarchy_primitive_tests omarchy_ane_bundle_tests \
  omarchy_kv_ops_tests omarchy_error_contract_tests omarchy_shape_ops_tests \
  omarchy_reduce_ops_tests omarchy_indexing_ops_tests omarchy_matmul_family_tests \
  omarchy_distributed_tests omarchy_compiled_tape_tests omarchy_fft_ops_tests \
  omarchy_fast_ops_tests omarchy_linalg_ops_tests mlx-omarchy-info
ulimit -c 0
./.work/build/tests/omarchy/omarchy_linalg_ops_tests   # 27/27 PASS
./.work/build/tests/omarchy/omarchy_primitive_tests    # 84/86, see findings
MLX_DISABLE_COMPILE=1 .work/venv-qual/bin/python -m mlx_lm generate \
  --model /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx \
  --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 32 --temp 0 --seed 0                    # Paris
```

No commits on either machine. Receipt only.



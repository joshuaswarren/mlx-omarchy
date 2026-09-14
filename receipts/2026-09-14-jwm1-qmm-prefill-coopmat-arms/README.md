# jwm1 QmmPrefillCoopmatF16: six bit-preserving arms and the kernel's measured ceiling

Date: 2026-09-14

## Verdict

The shipped `qmm_coopmat.comp` is at a local optimum on the installed
Honeykrisp fork. Six bit-preserving arms were built and screened against
it on the four real Qwen2.5-0.5B Q4 prefill cells; at the dominant cell
none is faster, five are 13 to 26 percent slower, and one is neutral at
+0.1 percent. Every arm produced
the same f16 output digest as the shipped kernel on all eight cells in
all three interleaved rounds, so the deltas are pure codegen, not
arithmetic.

Nothing is landed. `overlay/` is unchanged by this receipt.

The dominant prefill cell runs at **1035 GFLOP/s** (1053x896x9728,
17.736 ms median). The matrix pipe's own measured rate is 1459 GFLOP/s
inside the real dispatch geometry and 2148 GFLOP/s flat
(`receipts/2026-09-11-fma-ceiling` via
`receipts/2026-09-12-agx-qmm-codegen`), so this kernel is 1.41x from its
in-geometry ceiling - and this session shows that headroom is not
reachable from the shader source.

Two previously open questions are now answered with measurements:

- **The drain-remedy theory is falsified at ISA level.** The 2026-09-12
  receipt attributed the primary stall to five serialized global-load
  waits per 16-k step caused by exec-mask block exits, and named "dump
  the de-diverged shader and diff wait placement" as the open next step.
  That dump exists now (`dumps/`): removing all five guards leaves the
  wait count at **23 in both kernels** and the load-to-wait spacing at
  one to two instructions in both. The guards are not what the waits are
  attached to.
- **Cutting the waits does not help either.** The `uvec4` arm reduces
  per-step global loads from five to two by staging the x tile as one
  16-byte load per lane. It is 13.4 percent *slower* at the dominant
  cell. Exposed staging-load latency is therefore not the dominant term
  of the 1035-vs-1459 gap.

What the arms do show is that this driver is acutely sensitive to the
shared-memory access pattern: padding the shared row pitch by two or
four f32 costs 19 to 20 percent, and the shipped staging pattern is the
one that works.

## Identity

- Host: `jwm1-linux`, `aarch64`, `nproc=8`, Apple M1 T8103.
- Kernel: `7.1.6-1-1-ARCH` (asahi).
- Mesa: `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`;
  `driverInfo = Mesa 26.3.0-devel (git-6f6afc8968)`,
  `driverID = DRIVER_ID_MESA_HONEYKRISP`, `conformanceVersion 1.4.0.0`.
- Vulkan device: `Apple M1 (G13G B1)`, integrated, `api 1.4.359`,
  `cooperative_matrix_f32_8 = 1`, `max_compute_shared_memory_size =
  32768` (`device-info.json`).
- Source commit: `b41e2b74c330f910b24cab0e7516e306527858f0`, clean.
- Base build: `/var/tmp/mlx-omarchy-profile-enabled-b41e2b74`, the
  2026-09-13 profile-enabled tree
  (`-DMLX_OMARCHY_GPU_PROFILING=ON`); its installed `libmlx.so` is
  `d41f311ec19e23634213e420ec3f131ddb83509a87a16aee264b323d2fcc7051`,
  identical to `receipts/2026-09-13-jwm1-phase-profile-enabled`, so the
  base arm is that receipt's binary, not a rebuild.
- Arm builds: same tree, same CMake flags, one wheel and one venv per
  arm, local versions `diag.pad2`, `diag.pad4`, `diag.mat16`,
  `diag.swz8`, `diag.clamp`, `diag.uvec4`. Wheel and per-arm
  `libmlx.so` SHA-256 are in `identity.txt`.
- Model: `Qwen2.5-0.5B-Instruct-4bit` at
  `/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx`, config
  SHA-256 `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`,
  index SHA-256 `54001cb4c11197119c206dde28e7be08e5872aab6c6d271aed339ec77e84f870`,
  affine 4-bit group 64.
- Prompt for the end-to-end runs: the 2026-09-13 `ctx1024` prompt,
  4759 UTF-8 bytes, SHA-256
  `40e058593532cb7e1d79dd8c8274894a4d0ce02915ca365a78d14c7d051a91fb`,
  1053 chat-template tokens.
- Probe: `benchq/qmm-coop-bench/qmm_coop_bench_probe.py` SHA-256
  `e8774456d8c7a637cc7ff52121ce4b236f28316d9e39b5646b68ad7473858322`
  (unmodified, copied to `scripts/probe.py`).
- Agent model: `openai-codex/gpt-5.6-sol`; no routing fallback.
- AC connected, battery `Full` throughout (`identity.txt`).

## The arms

Each arm changes `overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp`
only, except `uvec4`, which also tightens one alignment gate in
`primitives.cpp` (`arms/primitives.uvec4.patch`). Sources in `arms/`.

| arm | change | shared bytes |
| --- | --- | ---: |
| `base` | shipped kernel, 32x32 tile, 64 lanes, 8x8x8 f32 coopmat | 4096 |
| `pad2` | shared row pitches padded 2 f32 (x 16->18, w 32->34) | 4480 |
| `pad4` | shared row pitches padded 4 f32 (x 16->20, w 32->36), matching upstream Metal's 16-byte `BK_padded` | 4864 |
| `mat16` | driver's 16x16x16 f32 coopmat shape instead of 8x8x8: one A and two B fragments per step, no `ks` loop | 4096 |
| `swz8` | swizzled tile order so 8 consecutive workgroup ids share a column tile and differ in row tile | 4096 |
| `clamp` | the five guarded staging loads de-diverged by clamping the row/column index and masking the word; addressing left inline | 4096 |
| `uvec4` | x tile staged as one 16-byte load per lane instead of four 32-bit loads | 4096 |

`pad2`/`pad4` test H3 from the 2026-09-12 handoff ("step-level 32->34 was
never isolated"). `mat16` is new: `hk_physical_device.c` advertises
8x8x8 and 16x16x16 for f32 and f16 A/B off the same hardware matrix MAC,
and `agx_nir_lower_simdmat.c` expands 16x16x16 into the 2x2x2 tiling, so
the arm asks the driver for the same 16 hardware ops per step with a
third of the fragment loads. `clamp` isolates the de-divergence in
`8f6de34c` from the array hoisting that commit also introduced.

## Kernel-isolated results

Protocol: `qmm_coop_bench_probe.py`, `mx.quantized_matmul` at the real
prefill shapes, 4 warmups plus 30 timed reps per cell, per-round median;
three rounds, arms interleaved inside one GPU lock hold (`ab_all.sh`);
table is the median of the three per-round medians (`ab-summary.md`,
raw rows in `ab-all.jsonl`).

| cell | base | pad2 | pad4 | mat16 | swz8 | clamp | uvec4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262x896x128 | 0.568 ms / 106 | 0.550 ms / +3.3% | 0.547 ms / +3.7% | 0.591 ms / -3.9% | 0.672 ms / -15.4% | 0.520 ms / +9.3% | 0.509 ms / +11.6% |
| 1053x896x128 | 0.646 ms / 374 | 0.579 ms / +11.6% | 0.583 ms / +10.8% | 0.652 ms / -1.0% | 0.607 ms / +6.4% | 0.635 ms / +1.7% | 0.601 ms / +7.4% |
| 262x896x896 | 0.806 ms / 522 | 0.870 ms / -7.4% | 0.864 ms / -6.8% | 0.883 ms / -8.8% | 0.955 ms / -15.6% | 0.920 ms / -12.4% | 0.814 ms / -1.0% |
| 1053x896x896 | 1.979 ms / 854 | 2.364 ms / -16.3% | 2.333 ms / -15.2% | 2.501 ms / -20.9% | 1.965 ms / +0.7% | 2.311 ms / -14.3% | 2.179 ms / -9.1% |
| 262x4864x896 | 2.709 ms / 843 | 3.920 ms / -30.9% | 3.929 ms / -31.0% | 3.981 ms / -31.9% | 4.256 ms / -36.3% | 3.761 ms / -28.0% | 3.472 ms / -22.0% |
| 1053x4864x896 | 9.406 ms / 976 | 11.825 ms / -20.5% | 11.665 ms / -19.4% | 12.462 ms / -24.5% | 9.456 ms / -0.5% | 11.254 ms / -16.4% | 10.441 ms / -9.9% |
| 262x896x9728 | 5.087 ms / 898 | 6.302 ms / -19.3% | 6.226 ms / -18.3% | 6.725 ms / -24.4% | 5.073 ms / +0.3% | 6.038 ms / -15.7% | 5.818 ms / -12.6% |
| 1053x896x9728 | 17.736 ms / 1035 | 22.289 ms / -20.4% | 22.001 ms / -19.4% | 23.861 ms / -25.7% | 17.714 ms / +0.1% | 21.323 ms / -16.8% | 20.484 ms / -13.4% |

Base column is median ms and GFLOP/s; arm columns are median ms and
speed delta against base, positive meaning faster.

The base arm reproduces the pinned number: 17.736 ms / 1035 GFLOP/s with
digest `5179630cd4a7c3f9`, against 17.5371 ms / 1046.7 on 2026-09-12 and
17.5996 ms / 1043.0 on 2026-09-11, same digest.

The two `x896x128` cells are the k and v projections: 132 or 36
workgroups of 64 lanes, work-starved, and the arms move them by up to
+11.6 percent. That is the same non-generalizing micro-win the
2026-09-10 screen rejected; those two cells are 20.3 ms of the 702 ms
QMM prefill budget (2.9 percent), and every arm that wins there loses
five to ten times as much on the cells that matter.

Digest equality across every arm, cell and round: **PASS**, one digest
per cell (`ab-summary.md`).

## End-to-end prefill

Protocol: `run_profile.sh`, one locked run per arm of
`scripts/profile_generate.py` at 1053 chat-template tokens,
`--max-tokens 2 --temp 0 --seed 0`, `MLX_DISABLE_COMPILE=1`,
`MLX_OMARCHY_GPU_PROFILE` on, then `scripts/profile_analyze.py`. Host
prefill rate is 1053 tokens over the `prefill_start`/`prefill_done`
marker interval.

| metric | base | clamp | delta |
| --- | ---: | ---: | ---: |
| host prefill | 1109.827 ms | 1267.190 ms | +14.18% |
| host prefill rate | **948.80 tok/s** | **830.97 tok/s** | **-12.42%** |
| `QmmPrefillCoopmatF16` | 163 / 704.425 ms | 163 / 842.031 ms | +19.53% |
| QMM share of prefill-window GPU busy | 76.59% | 79.33% | |
| prefill-window GPU busy | 919.732 ms | 1061.460 ms | |
| whole-run GPU busy fraction | 92.48% | 90.26% | |
| prefill dispatches | 1063 | 1063 | |
| `MatmulRbF16` | 46 / 105.690 ms | 46 / 104.964 ms | -0.69% |

This is the transfer function between the two instruments: `clamp`'s
-16.8 percent at the dominant kernel cell landed as -12.4 percent prefill
tok/s, i.e. prefill tok/s moves about 0.74x the kernel delta on this
model. `MatmulRbF16` is unchanged, as expected for a qmm-only edit.

Note the anchor: 948.80 tok/s is the profile-enabled build with the GPU
profiler compiled in and recording. The shipped wheel measures 1111.58
tok/s on the same prompt and protocol
(`receipts/2026-09-14-jwm1-gpu-parity-rerun.md`), 60.38 percent of the
pinned native base-M1 1840.90 tok/s.

## ISA attribution

Captured with `AGX_SIMDMAT=1 MESA_SHADER_CACHE_DISABLE=true
AGX_MESA_DEBUG=shaders` on the installed driver - no ICD override -
while the single-shape dominant-cell probe ran (`dump.sh`, dumps in
`dumps/`, counters in `scripts/isa_count.py` and `scripts/isa_waits.py`).

| final packed ISA, whole kernel | base | clamp |
| --- | ---: | ---: |
| instructions | 699 | 721 |
| `wait` | 23 | 23 |
| global `load` | 7 | 7 |
| `lload` / `lstore` | 40 / 48 | 40 / 48 |
| `barrier` | 10 | 10 |
| `if` / `pop_exec` | 34 / 35 | 29 / 30 |
| `mov` | 19 | 54 |
| `csel` | 0 | 11 |
| load to next wait, instruction gaps | 4, 1, 1, 1, 1, 1, 1 | 4, 1, 1, 1, 1, 1, 2 |
| waits within 2 instructions before a barrier | 0 of 23 | 0 of 23 |

The base count of 699 static instructions matches the 2026-09-12
receipt exactly, so this dump pipeline and that one describe the same
code.

Reading: `clamp` removed exactly five guard triplets (`if` 34->29,
`pop_exec` 35->30, one per de-diverged load) and the scheduler still
emitted 23 waits with the same load-to-wait spacing. The waits are a
property of how the AGX scheduler pairs each global load with its own
wait, not of the exec-mask block exits - so the source-level remedy the
drain theory implied cannot work, and what the arm actually bought was
22 more instructions and 35 more `mov` in the hot loop, for -16.8
percent. H2's alternative ("waits sink to the barrier") is also refused:
zero of 23 waits sit within two instructions of a barrier in either
kernel.

`uvec4` confirms it from the other side. Its dump shows the compiler
unrolled the four-step chunk loop (64 matrix ops, 4 steps x 16) and the
per-step order is `load wait lstore x8 load wait lstore x8 barrier` -
two global loads per step instead of five, exactly as designed - and it
is still 13.4 percent slower. Removing three exposed memory latencies
per step did not pay for a worse shared-store pattern.

`mat16` yields one reusable fact beyond its timing: its digests are
identical to the 8x8x8 kernel on all eight cells, so the driver's
2x2x2 lowering of a 16x16x16 multiply-add preserves per-output
ascending-k accumulation order.

## The named ceiling

- Measured now: **1035 GFLOP/s** at the dominant prefill cell.
- Matrix pipe inside the real dispatch geometry: **1459 GFLOP/s**
  (zero-traffic coopmat ladder, `receipts/2026-09-11-fma-ceiling`, cited
  through `receipts/2026-09-12-agx-qmm-codegen`).
- Matrix pipe flat: **2148 GFLOP/s**, same source.
- Kernel headroom to the in-geometry ceiling: **1.41x**.

Arithmetic, not a measurement: giving the QMM term all of that headroom
takes 704.425 ms to 499.7 ms, prefill-window GPU busy to 715.0 ms and
host prefill to about 905 ms, i.e. 948.80 -> about 1163 tok/s on this
profile-enabled anchor (+22.6 percent). Scaled onto the shipped wheel's
1111.58 tok/s with today's measured decomposition, the same headroom is
worth about 1394 tok/s, 75.7 percent of the pinned native 1840.90.
So even a perfect QMM kernel does not reach native prefill parity: the
remaining budget is `MatmulRbF16` at 105.7 ms, `SoftmaxF16`, the 7.5
percent GPU idle inside the prefill window, and the host time outside
the GPU span.

Every source-level route into that 1.41x is now measured and rejected:
operand transport and staging chunk size (2026-09-11 qmm-coop-datapath),
bit-preserving ILP schedules (2026-09-11 coopmat-ilp-chains), f16
operands (2026-09-11 qmm-f16-operand: +3.5 to +4.3 percent paired
prefill but digests move away from native, not landable), de-diverged
staging with hoisting (8f6de34c), and this receipt's six arms. The
remaining lever is the driver: the 2026-09-12 receipt's
`hk/agx-wait-batching` patch is committed but inert on the target
(frame detection never fires; root cause and a named ~30-line fix are
recorded there), and the flat-versus-in-geometry gap, 2148 versus 1459,
is a second driver-side item nobody has attacked.

## Suggested next step

Not "another shader arm". Two ranked items:

1. Build the named ~30-line fix to `agx_insert_waits.c` from
   `receipts/2026-09-12-agx-qmm-codegen` (defer merge resolution to
   else-arrival) and A/B it by ICD file against the base. This receipt
   raises its prior: the waits are demonstrably scheduler-placed rather
   than CFG-placed, so batching them is the only route that touches
   them. A dump after the patch should show fewer than 23 waits;
   if it does not, the pass is still inert.
2. H1 from the 2026-09-12 handoff, still unrun: re-run the zero-traffic
   coopmat ladder with a 4 KiB-shared skeleton, 128-lane workgroups and
   a flat per-tile dispatch, to find whether the 2148-to-1459 loss is
   shared footprint, subgroup residency or the tile loop. That decides
   whether the kernel's ceiling is 1459 or higher.

## Lock and hardware safety

- `/tmp/m1-gpu.lock` inode 29, always taken with `flock -w 60`, never
  stolen, never unlinked; nested `flock -n` returned 1 inside every
  profile run (`profiles/*/lock.json`). Held only for the duration of
  each screen or profile run and released immediately: the seven-arm
  three-round screen held it for 45 s, each profile run for about 3 s.
- Lock free after the work, `fuser` empty (`identity.txt`).
- Coordinated with `DecoderLstmSemantics` (no jwm1 GPU need),
  `Ane1x896Execute` (ANE only, no GPU lock) and `Jw16MaxAttribution`
  (jw16 only) over IRC before and during the holds.
- `ane.ko` stayed loaded and untouched: no ANE command, no unload, no
  reboot, no 1x896 bundle, no SET write.
- No driver install, no ICD override, no `VK_DRIVER_FILES`: every
  measurement ran on the system fork driver.
- The grouped-base checkout's `dist/` and `.work/venv-run` were not
  modified. Arm wheels and venvs live under `/var/tmp/qmm-coop-arms`.
- Arms were built in place in the profile-enabled tree, and its two
  touched sources were restored to the pristine overlay afterwards,
  verified by hash: `qmm_coopmat.comp`
  `68cc00fff6ea13a125b59ff2256f0aef224edfc2bf66675bc0a5e44f95d5c1b4`
  and `primitives.cpp`
  `1e3a0a24b3ea301715aee932a0642bbd461481d68cb94ea679a04e146e5c65ff`,
  both equal to `overlay/` (`identity.txt`). The base venv's
  `libmlx.so` is unchanged from 2026-09-13.
- AC connected, battery `Full`, before and after.

## Artifacts

- `ab-all.jsonl`, `ab-summary.md`: raw and summarized seven-arm screen.
- `profiles/base/`, `profiles/clamp/`: `analysis.txt`, `markers.jsonl`,
  `profile.jsonl.gz`, `run.log`, `lock.json`.
- `dumps/dump-{base,clamp,uvec4}.log.gz`: AGX shader dumps.
- `arms/qmm_coopmat.*.comp`, `arms/primitives.uvec4.patch`: every arm
  source plus the shipped kernel for diffing.
- `identity.txt`, `device-info.json`: host, driver, model, prompt,
  wheel, library and restore hashes; lock state.
- `scripts/`: every script used, including the arm generators
  (`make_clamp.py`, `make_swz.py`, `make_uvec4.py`), the build and
  install drivers, the ISA counters and the profile runner.

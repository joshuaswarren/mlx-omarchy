# jw16 M1 Max GPU attribution: phase-isolated 1053-token Q4 profile

Date: 2026-09-14
Task: with the Mesa driver now equal on both hosts
(`receipts/2026-09-14-jw16-honeykrisp-mesa-parity/`), take a phase-isolated
`MLX_OMARCHY_GPU_PROFILE` capture of the 1053-token Q4 leg on
`jw16mbp1-linux`, attribute the residual M1-Max-specific deficit, and name
whether it is occupancy, tiling constants sized for 8 cores, submission
overhead, or memory bandwidth. Attribution only: no kernel was changed.

## Verdict

**The deficit is occupancy, and its cause is tiling constants sized for an
8-core part.** It is not submission overhead and it is not memory bandwidth;
both are excluded quantitatively below.

The 1053-token prefill deficit decomposes exactly into three multiplicative
factors (native M1 Max Metal scaling over native base-M1 Metal is the 4.3720x
yardstick; Linux realizes 3.2172x):

| factor | measured | share of native scaling captured |
| --- | ---: | ---: |
| `QmmPrefillCoopmatF16`, 163 identical dispatches | 3.5933x | 0.8219 |
| all other prefill GPU work | 3.3734x cumulative | 0.9388 |
| everything outside GPU-busy time | 3.2172x cumulative | 0.9537 |
| product | | **0.7359** |

Observed end-to-end: 44.43% / 60.38% = 0.7358. The decomposition closes to
0.01%. **82% of the shortfall lives inside the one hot kernel**; the rest of
the GPU work costs 6%, and every host-side, submission and gap cost together
costs 4.6%.

Inside that kernel the loss is occupancy, measured on jw16 itself from the
four real prefill shapes. The dispatch grid is purely shape-derived
(`n_groups = ceil(n/32)`, `m_groups = ceil(m/32)`, cap 65535); nothing in the
dispatch path has a device-width term. The resulting workgroup counts put
every shape above the saturation knee on an 8-core part and put two of four
below it on a 32-core part:

| n | k | workgroups | wg/core on jw16 (32) | wg/core on jwm1 (8) | jw16 TFLOP/s | rel |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4864 | 896 | 5016 | 156.8 | 627.0 | 3.97 | 1.00 |
| 896 | 4864 | 924 | 28.9 | 115.5 | 3.77 | 0.95 |
| 896 | 896 | 924 | 28.9 | 115.5 | 3.47 | 0.87 |
| 128 | 896 | 132 | **4.1** | 16.5 | **1.09** | **0.28** |

Highest-value single change: **make the coopmat prefill kernel's M-tile
device-width aware** — add a finer-M specialization of
`shaders/qmm_coopmat.comp` and select it when
`m_groups * n_groups` falls below the measured knee for the part in hand.
Evidence and expected gain in "Proposal".

## Identity

- Receipt checkout (local `mlx-omarchy`): `b28deb1cbf7899cab71df08d432dd47cb31862ce`
  (`main`; working tree dirty from unrelated sibling work, not used for the
  measurement).
- Measured source: `git worktree add --detach /var/tmp/mlx-omarchy-prof-b41e2b74 b41e2b74`,
  commit `b41e2b74c330f910b24cab0e7516e306527858f0`, clean. jw16's own checkout
  sits at `3db3cb9a`; `git diff --stat b41e2b74 3db3cb9a` is one file,
  `receipts/…/2026-09-13-jw16-m1max-gpu-performance.json`, 108 insertions — no
  code difference, so the diagnostics build is code-identical to the wheel the
  parity receipts measured.
- Host: `jw16mbp1-linux`, `aarch64`, Apple M1 Max T6001, GPU `Apple M1 Max (G13C C0)`,
  `nproc=10`, kernel `7.1.6-1-1-ARCH`, `boot_id f6ad865b-b85e-499e-8308-28b57b24f0c8`
  (unchanged; no reboot).
- Vulkan: `driverName = Honeykrisp`, `driverInfo = Mesa 26.3.0-devel (git-6f6afc8968)`,
  `apiVersion 1.4.359`, ICD `asahi_icd.aarch64.json`. Recorded at the top of the
  run (`driver.txt`) because `driverName` alone cannot distinguish the fork from
  stock. `mx.device_info()` reports `cooperative_matrix_f32_8: 1`,
  `max_compute_shared_memory_size: 32768`, `timestamp_period_ns: 1`
  (`device.txt`). No `AGX_*`/`MESA_*`/`VK_*` override other than the documented
  `MESA_SHADER_CACHE_DISABLE=true`.
- Diagnostics wheel: `mlx-omarchy==0.32.2.dev202609141516+b41e2b74`, file
  `dist-diag/mlx_omarchy-0.32.2.dev202609141516+b41e2b74-cp314-cp314-linux_aarch64.whl`,
  SHA-256 `6c05aebc29bdfc6a45a29dc3268deab0b8b89356e7a8aece453c1caedec17782`,
  built on jw16 with `-DMLX_OMARCHY_GPU_PROFILING=ON`
  (`==> Created wheel … Successfully built mlx-omarchy`, wall 2m3s, 10 cores).
- Profiler gate, positive and negative control in the same run
  (`profiler-gate.txt`): the diagnostics `lib/libmlx.so`
  (SHA-256 `d92f6e7cb408fbbb6f024be25ce4eb9e19471e29c57ea2b90337bee0e3f30887`)
  contains the `MLX_OMARCHY_GPU_PROFILE` literal 3 times; the wheel the parity
  receipts measured (`libmlx.so` SHA-256 `f2d45e601f05dd22…`) contains it 0
  times, which is why no phase profile existed for jw16 before today. The
  Python bindings are byte-identical in both
  (`core.cpython-314-aarch64-linux-gnu.so` SHA-256 `4ad850da16c300ea9f60aa7247f034f358aa16d7ba2fd146979a01c867e3cfa5`).
- Interpreter: `/var/tmp/mlx-omarchy-prof-b41e2b74/venv-diag/bin/python`
  (Python 3.14.7) — a `cp -a` copy of `~/.local/share/mlx-omarchy-test-venv`
  with the diagnostics wheel force-reinstalled `--no-deps --no-index`.
  `mlx_lm 0.31.3`. The measured test venv, `~/src/mlx-omarchy/dist/`, and
  `~/src/mlx-omarchy/.work/` were **not** modified (the private build used
  `MLX_OMARCHY_WORK_DIR` and a private `dist-diag/`).
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, local snapshot
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, `HF_HUB_OFFLINE=1`.
  `num_hidden_layers 24`, `hidden_size 896`, `intermediate_size 4864`,
  `num_attention_heads 14`, `num_key_value_heads 2` (kv width 128).
- Prompt: `scripts/bench_matrix.py:prompt_text(manifest, "ctx1024")`, 4759
  UTF-8 bytes — byte-identical to the expansion the parity receipts used.
- Power during the run: `macsmc-ac/online=1`,
  `tps6598x-source-psy-0-003a/online=1`, `macsmc-battery/status=Full`.
- Agent model: `anthropic/claude-opus-5`; routing fallback: false.

### Build-environment deviations, and why they cannot touch the result

`scripts/build-wheel.sh` does not build on jw16 as shipped. Two additions were
needed, both outside the Vulkan backend:

1. `-DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas`.
   Upstream `find_path(BLAS_INCLUDE_DIRS cblas.h /usr/include …)` fails on this
   host because Arch's `blas-openblas 0.3.34-1` installs `cblas.h` under
   `/usr/include/openblas`, not `/usr/include`; the configure step aborts with
   `BLAS_INCLUDE_DIRS … set to NOTFOUND` (`build-diag.log`). CPU BLAS include
   path only.
2. `-DFETCHCONTENT_SOURCE_DIR_{JSON,GGUFLIB,FMT,NANOBIND}` pointing at copies of
   the already-downloaded sources from jw16's existing
   `.work/mlx/build/*/mlx.core/_deps`. `prepare-mlx.sh` recreates `.work/mlx`
   from scratch on every invocation, so every build re-runs `FetchContent`, and
   the `nlohmann/json` release download failed against github
   (`status_code: 22`). Third-party source provisioning only; same sources, no
   version change.

Neither flag reaches `overlay/mlx/backend/omarchy/`. The profiler gate above
proves the only backend-relevant difference from the measured wheel is the
profiling define.

## Protocol

One outer `flock -w 60 /tmp/m1-gpu.lock` (inode 12) held across both captures,
never stolen, never unlinked; `flock -n` returned 1 while held and 0 after
release (`lock.txt`). Each capture is a fresh process with a scrubbed
environment (`env -i` plus `HOME`, `PATH`):

```text
MLX_DISABLE_COMPILE=1
HF_HUB_OFFLINE=1
MESA_SHADER_CACHE_DISABLE=true
MLX_OMARCHY_GPU_PROFILE=<out>/profile-<tag>.jsonl
MLX_OMARCHY_GPU_PROFILE_LABEL=jw16-max-attrib-<tag>
scripts/profile_generate.py --model <snapshot> --prompt <ctx1024>
  --max-tokens {2,32} --temp 0 --seed 0 --markers <out>/markers-<tag>.jsonl
```

`--max-tokens 2` reproduces the jwm1 protocol exactly
(`receipts/2026-09-14-jwm1-gpu-parity-refresh.md`, `…-phase-profile-enabled/`)
so the kernel tables are comparable; `--max-tokens 32` was added to get 31
decode intervals for the decode leg. Both runs emitted the pinned greedy text
(`Thank you` at `--max-tokens 2`, the same continuation jwm1 produced).
Generation rc=0 for both; profiles 328373 and 2180253 bytes.

No driver change, no ANE command, no reboot, no 1x896, no SET write, no
`bench_decode` re-measurement (every clean-run rate quoted here comes from an
existing receipt).

Scheduling: `IslandsExecJw16` asked for a quiet box for its ANE encoder-island
submits, the build was held until it reported its last island landed and the
device-live check recorded, and the box was then idle
(`load average: 0.35` after a 100 s settle) before the profile ran. No ANE lane
was in flight during either capture.

### Instrumented rates are not parity rates

Instrumented host phases (`markers-m1053-t2.jsonl`): load 375.641 ms, prefill
562.353 ms, 2-token decode 21.478 ms, first inter-token 16.573 ms. That prefill
is 1872.3 tok/s against a clean-run 3576.23 tok/s, because
`MESA_SHADER_CACHE_DISABLE=true` forces pipeline compilation inside the
measured window (the same ~260 ms absolute penalty jwm1's refresh receipt
carries: 1204.479 ms instrumented against 947.297 ms clean). **Only GPU-busy
sums and per-dispatch grids are used for attribution; every rate is the
existing uninstrumented median.**

## What is bit-identical across the two machines

Whole-run dispatch counts and per-kernel counts match jwm1's refresh capture
exactly, so the graph, the kernel routing and the dispatch grids are identical
and the only variable is the hardware:

| | jwm1 (refresh receipt) | jw16 (this capture) |
| --- | ---: | ---: |
| whole-run dispatches | 1312 | 1312 |
| `QmmPrefillCoopmatF16` | 163 | 163 |
| `MatmulRbF16` | 46 | 46 |
| `QmmVecQ4MultiSubgroupF16` | 288 | 288 |
| `SoftmaxF16` | 23 | 23 |
| `SwigluF16` | 95 | 95 |
| analyzer "prefill window" dispatches | 1063 | 1063 |

The one structural difference is submission batching: jwm1 used 12 submissions,
jw16 used 9. The capture resolves jw16's 1312 dispatches into one true prefill
pass of 565 dispatches in 3 submissions plus 3 decode steps of 249 dispatches
in 2 submissions each (565 + 3×249 = 1312). Since every per-kernel count is
identical, the same 565/249 partition must hold on jwm1, which puts jwm1's
prefill in 6 submissions against jw16's 3 — **fewer host round trips on jw16,
i.e. in jw16's favour**, so it cannot be part of the deficit. (Inference from
the identical counts; jwm1's per-submission list is not in its receipt and
jwm1 was not touched for this task.)

Note also that the analyzer's "prefill window" is prefill **plus the first two
decode steps** (1063 = 565 + 2×249) on both hosts; the label is a
join-attribution artifact the analyzer itself warns about. It stays usable as a
cross-machine ratio because the composition is identical.

## Prefill results

True prefill pass (submissions 1-3, 565 dispatches):

| quantity | value |
| --- | ---: |
| GPU busy | 254.138 ms |
| intra-submission gaps | 13.416 ms (562 pairs, mean 23.9 us, p50 28.2 us) |
| inter-submission gaps | 0.685 ms (2 gaps) |
| GPU span | 268.238 ms |
| **GPU busy fraction** | **94.74%** |
| clean-run prefill wall (existing median, 3576.2252 tok/s) | 294.445 ms |
| GPU busy as a share of the clean wall | 86.3% |

Whole-run kernel table (`analysis-m1053-t2.txt`), against jwm1's refresh table:

| Kernel | n | jwm1 gpu ms | jw16 gpu ms | jwm1/jw16 | jw16 share of busy |
| --- | ---: | ---: | ---: | ---: | ---: |
| QmmPrefillCoopmatF16 | 163 | 702.034 | 195.371 | **3.593x** | 68.5% |
| MatmulRbF16 | 46 | 106.812 | 33.203 | 3.217x | 11.6% |
| QmmVecQ4MultiSubgroupF16 | 288 | 28.244 | 12.161 | 2.322x | 4.3% |
| SoftmaxF16 | 23 | 27.473 | 7.882 | 3.485x | 2.8% |
| SwigluF16 | 95 | 22.099 | 6.089 | 3.629x | 2.1% |
| whole-run busy / span | | 942.136 / 1084.801 | 285.222 / 377.060 | 3.303x busy | 75.64% |

`QmmPrefillCoopmatF16` is 195.371 / 274.938 = **71.06%** of the 1063-window GPU
busy (jwm1: 702.034 / 927.471 = 75.69%), and 76.9% of the true 565-dispatch
prefill pass. It is the kernel to fix on both machines; it is *more* dominant on
jwm1 only because everything else on jwm1 is slow too.

Do **not** read the whole-run busy fraction (86.85% jwm1 vs 75.64% jw16) as
jw16 idling more. Both figures are contaminated by the same machine-invariant
absolute pipeline-compilation stall (`MESA_SHADER_CACHE_DISABLE=true`;
jw16's first decode submission shows a 28.505 ms gap before it, the second
26.676 ms, and the third only 5.063 ms), and an absolute stall weighs more on a
3.6x faster GPU. The honest idle number is the true-prefill 94.74% above.

### Per-shape occupancy census (`shape-census.txt`, `shape_census.py`)

Exact shapes are recovered from the recorded dispatch, not guessed: the
dispatch path sets `gx = ceil(n/32)` and `gy = ceil(m/32)`, and the `InputW`
binding range is exactly `n*k/2` bytes for affine 4-bit weights, so
`k = 2*range/n`. The four recovered shapes are exactly the Qwen2.5-0.5B
projection set at m=1053: `n=4864,k=896` (gate/up), `n=896,k=4864` (down),
`n=896,k=896` (q/o), `n=128,k=896` (k/v).

| n | k | wgroups | wg/core (32) | wg/core (8) | cnt | tot ms | mean us | GFLOP | TFLOP/s | rel |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4864 | 896 | 5016 | 156.8 | 627.0 | 46 | 106.349 | 2311.9 | 9.178 | 3.97 | 1.00 |
| 896 | 4864 | 924 | 28.9 | 115.5 | 23 | 56.023 | 2435.8 | 9.178 | 3.77 | 0.95 |
| 896 | 896 | 924 | 28.9 | 115.5 | 46 | 22.394 | 486.8 | 1.691 | 3.47 | 0.87 |
| 128 | 896 | 132 | 4.1 | 16.5 | 48 | 10.605 | 220.9 | 0.242 | 1.09 | 0.28 |

Total 722.67 GFLOP in 195.371 ms = **3.699 TFLOP/s** on jw16; the identical
722.67 GFLOP in jwm1's 702.034 ms = **1.029 TFLOP/s**. Against each part's
fp32 FMA peak (cores x 128 lanes x 2 x 1.296 GHz nominal: 2.654 TFLOPS for 8,
10.617 for 32) that is **38.8% of peak on jwm1 and 34.8% on jw16** — the wider
part loses 10.1% of pure per-core efficiency.

The achieved rate is monotone in workgroups per core and has not flattened
until ~157/core. The same grids are 16.5-627 wg/core on 8 cores — every shape
above the knee — which is exactly why `TILE_M = TILE_N = 32` with a 64-lane
workgroup is a good choice on the part it was tuned on and a lossy one on a
32-core part.

Priced: if every shape ran at the best observed rate (3.97 TFLOP/s), the
kernel would take **182.034 ms** instead of 195.371 ms. That is −13.337 ms
(−6.83% of the kernel, −5.25% of prefill GPU busy, −4.53% of the clean prefill
wall), which would move the kernel's scaling from 3.5933x to 3.8566x
(0.8219 → 0.8821 of native scaling, closing **33.8%** of the kernel shortfall)
and the leg from 3576.23 to ≈3745.9 tok/s, **44.43% → 46.54% of native**. It
also restores 96.4% of jwm1's per-core efficiency (182.034/195.371 = 0.932
against the 0.899 measured ratio).

## Exclusions

**Memory bandwidth is excluded, two-sidedly.** Summing the recorded binding
ranges, the 163 coopmat dispatches move 1.308 GB face value, which is
6.69 GB/s on jw16 — 1.67% of the M1 Max's 400 GB/s. Even under the worst case
that the `x` operand gets zero cache reuse across a dispatch's `gx` column
tiles, traffic is 23.397 GB: 119.76 GB/s = **29.9% of peak on jw16** against
33.33 GB/s = **48.8% of peak on jwm1**. Bandwidth pressure is *lower* on the
Max, so it cannot be the Max-specific loss.

**Submission and host overhead is excluded as the primary term.** Everything
outside GPU-busy time accounts for the 0.9537 factor — 4.6% of the shortfall.
The measurable piece is the intra-submission gap: 13.416 ms over 562 pairs,
mean 23.9 us per dispatch, an absolute driver/barrier cost that does not shrink
with a wider GPU (4.6% of jw16's clean prefill wall against 1.4% of jwm1's).
That 23.9 us also *includes* the profiler's own two `vkCmdWriteTimestamp` plus
the per-command-buffer `vkCmdResetQueryPool`, so 13.416 ms is an upper bound,
and jw16 already issues *half* as many prefill submissions as jwm1 (the
inference above, from the identical per-kernel counts). Driving it
to zero would buy at most the same ~4.5% the occupancy fix buys, with no
evidence any of it is actually reducible.

**Tiling constants and occupancy are the same finding here, not alternatives.**
The only device-width lever the dispatch path has is the workgroup count, and
the workgroup count is fixed by `TILE_M`/`TILE_N`. Verified by source read at
`b41e2b74`: `overlay/mlx/backend/omarchy/primitives.cpp:6960-6974` computes
`m_groups = (matrix_m + 31)/32`, `n_groups = (matrix_n + 31)/32`, clamps both
to `kMaxComputeGroupCountX = 65535` (`compute.h:19`, never binding here) and
dispatches; `shaders/qmm_coopmat.comp` hard-codes `TILE_M = TILE_N = 32`,
`STEP_K = 16`, `layout(local_size_x = 64)`. There is no core count, no
occupancy target, and no device query anywhere in that path, and
`mx.device_info()` exposes no core count to build one from.

## Decode leg (secondary; occupancy consistent, not priced)

Clean-run 1053/32 decode scales 1.5067x on Linux against Metal's 2.0216x, a
0.7453 deficit ratio — the same magnitude as prefill's 0.7359 (and exactly the
50.45/67.70 = 0.7452 of the parity receipts). The decode gemv grid is
`min(ceil(n/8), 65535)` workgroups (`primitives.cpp:6893-6899`), and the
recorded grids put three of four dispatch groups far below the prefill knee on
the Max:

| workgroups | wg/core (32) | wg/core (8) | dispatches | mean us | tot ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1216 | 38.0 | 152.0 | 792 | 64.2 | 50.813 |
| 144 | 4.5 | 18.0 | 792 | 33.7 | 26.653 |
| 112 | 3.5 | 14.0 | 792 | 50.2 | 39.731 |
| 112 | 3.5 | 14.0 | 792 | 28.8 | 22.826 |

Decode work is 249 dispatches and 2 submissions per token on jw16; the
identical per-kernel counts imply the same partition on jwm1, whose
per-submission breakdown is not in its receipt.

**This leg cannot be priced with this instrumentation, and the numbers above
must not be converted into a saving.** Summed bracketed kernel durations in the
decode window are 339.800 ms over 31 intervals = 10.961 ms per token, which
*exceeds* the clean-run 6.984 ms per token. For ~30 us kernels the
`BOTTOM_OF_PIPE` timestamp pair forces each dispatch to drain before `t1`,
removing inter-dispatch overlap that a clean run keeps; the shape census's
`n`/`k` recovery is also invalid for the gemv path, whose group width and
fused-group routing differ from the `TILE_N=32` assumption (the "n=9728,k=448"
row is the gate/up weight buffer misread by that assumption, not a real shape).
A decode-side price needs a coarser timing scheme (timestamp per submission,
not per dispatch), a wall-clock A/B of a candidate change, or the
amortized-grid method `receipts/2026-09-14-decoder-projector` used for the same
problem: dispatch many calls' worth of work in one dispatch and subtract a
same-shape control, which lifts the signal orders of magnitude above the noise
floor and, as a side effect, separates arithmetic cost from the
latency-bound-at-one-call regime that this occupancy finding is about.

## Proposal: device-width-aware M-tiling for `QmmPrefillCoopmatF16`

One change, attacking the 82% factor with the measured occupancy curve behind
it.

**What.** Add a finer-M specialization of `shaders/qmm_coopmat.comp`
(`TILE_M = 16`: one 8-row `coopmat` block per subgroup, `acc[1][4]`, 64 lanes,
`TILE_N` and the k loop untouched; optionally `TILE_M = 8` with a 32-lane
workgroup). In `QuantizedMatmul::eval_gpu`, after `m_groups`/`n_groups` are
computed, step down to the finer specialization while
`m_groups * n_groups < occupancy_target(device)` and a floor is not hit.
Calibrate `occupancy_target` from this curve: the knee is not reached below
~157 workgroups per core, and the cliff is at 4.1.

**Why this and not the alternatives.**

- Coarser M (`TILE_M = 64`) is contraindicated by the same curve: it would
  halve 924 workgroups to 462 (14.4/core) and drop the q/o and down shapes
  toward the 4.1-wg/core cliff.
- Split-K would fill the machine but changes each output's f32 accumulation
  order, which breaks the pinned generated-ID digests this project gates on.
- Batching submissions attacks a 4.6% factor with a ≤4.5% ceiling that is
  partly the profiler's own cost, on a host that already issues half of jwm1's
  prefill submissions.
- Bandwidth work attacks a term at 1.67% (29.9% worst-case) of peak that is
  *less* pressured on the Max than on the base M1.

**Expected gain, and its bound.** The ideal-occupancy floor is the ceiling:
−13.337 ms of 195.371 ms, 3576.23 → ≈3745.9 tok/s, 44.43% → 46.54% of native.
Realization will be partial, and the receipt should say so up front: the staged
16x32 weight tile is reused across `TILE_M` rows, so halving `TILE_M` doubles
shared-memory staging traffic per output. The trade is strongly favourable only
where the occupancy deficit is large — the `n=128` k/v projections at
rel 0.28 (132 → 528 workgroups at `TILE_M=8`, 16.5/core; ceiling −7.68 ms,
−3.9% of the kernel) are the first and possibly only shapes worth enabling.
The two 924-workgroup shapes at rel 0.87-0.95 are a second, more marginal step
(ceiling −5.64 ms combined) where the reuse loss may eat the gain outright;
they need an A/B, not an assumption.

**Prerequisite, named because it does not exist yet.** Selection needs a GPU
core count, and neither `Device::capabilities()` nor `mx.device_info()` has
one; Vulkan does not report it. It must come from an architecture table keyed
on the reported device name (`Apple M1 (G13G B1)` = 8, `Apple M1 Max (G13C C0)`
= 32) or from an Asahi-specific property, with a conservative default that
keeps `TILE_M = 32` for unknown parts — which preserves today's behaviour
everywhere including jwm1.

**Correctness gate.** A finer M tile leaves each output's ascending-k f32
accumulation chain and the two 8-wide `coopMatMulAdd` updates per output
untouched, so it is digest-preserving by construction — but it must still be
verified per leg against the pinned digests (`7fd25a869ff21678` short,
`7da83f06ec9f001d` at 1053), on jw16 **and** on jwm1, plus a jwm1 no-regression
run to confirm jwm1's already-saturated grids stay on `TILE_M = 32`.

## Lock and hardware safety

- `/tmp/m1-gpu.lock` inode 12 taken with `flock -w 60`, held exclusively across
  both captures; nested `flock -n /tmp/m1-gpu.lock -c true` returned 1 while
  held. Not stolen, not unlinked; `flock -n` free after release (`lock.txt`).
- No ANE command, no `accel0` access, no module load/unload: `ane 65536 0`
  still live at refcount 0, `boot_id` unchanged, no reboot, no 1x896, no SET
  write.
- No driver change. jw16 remains on `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`,
  the parity state the project expects.
- `llm-inference.service` `inactive` throughout.
- `~/src/mlx-omarchy/dist/`, `~/src/mlx-omarchy/.work/` and
  `~/.local/share/mlx-omarchy-test-venv` were not modified; the diagnostics
  build, its venv and its outputs live under `/var/tmp/mlx-omarchy-prof-b41e2b74/`
  and `/var/tmp/jw16-max-attrib-20260914/`. The `b41e2b74` worktree is still
  registered (`git worktree list`) and can be removed with
  `git worktree remove /var/tmp/mlx-omarchy-prof-b41e2b74`.
- Nothing was run on jwm1. Every jwm1 number here is quoted from
  `receipts/2026-09-14-jwm1-gpu-parity-rerun.md`,
  `receipts/2026-09-14-jwm1-gpu-parity-refresh.md` or
  `receipts/2026-09-13-jwm1-phase-profile-enabled/`, and every jw16 clean-run
  rate from `receipts/2026-09-14-jw16-honeykrisp-mesa-parity/`.

## Artifacts

`receipts/2026-09-14-jw16-max-gpu-attribution/`, hashes in `SHA256SUMS`:

| file | what |
| --- | --- |
| `profile-m1053-t2.jsonl.gz`, `markers-m1053-t2.jsonl` | raw capture, jwm1-comparable 2-token protocol |
| `profile-m1053-t32.jsonl.gz`, `markers-m1053-t32.jsonl` | raw capture, 32-token protocol for the decode window |
| `analysis-m1053-t2.txt`, `analysis-m1053-t32.txt` | `profile_analyze.py` output |
| `shape-census.txt`, `shape_census.py` | per-shape grid/occupancy/TFLOP-per-shape census |
| `compute-b41e2b74.h`, `profile_analyze-b41e2b74.py` | the `ComputeKernel` enum the capture's kernel ids index into, and the analyzer that produced the two `analysis-*.txt` files, both copied from the measured worktree (SHA-256 verified equal to it) |
| `driver.txt`, `device.txt`, `provenance.txt`, `profiler-gate.txt`, `lock.txt` | identity, profiler gate, lock evidence |
| `generate-m1053-t2.log`, `generate-m1053-t32.log` | generation output |
| `runner-jw16-max-attrib-run.sh` | the locked runner, as executed |
| `build-diag-wheel.sh`, `build-diag.log` | diagnostics wheel build, as executed |

The set is self-contained and re-derivable:

```sh
sha256sum --check SHA256SUMS
zcat profile-m1053-t2.jsonl.gz > /tmp/p.jsonl
./shape_census.py /tmp/p.jsonl --compute-h compute-b41e2b74.h --cores 32
./profile_analyze-b41e2b74.py /tmp/p.jsonl \
  --markers markers-m1053-t2.jsonl --compute-h compute-b41e2b74.h
```

Both are archived rather than referenced because the local `main` tree's
`scripts/profile_analyze.py` cannot read this capture: its enum header lacks
`QmmPrefillCoopmatF16`, and it raises
`TypeError: '<' not supported between instances of 'str' and 'NoneType'` on the
per-phase sort. The archived analyzer reproduces `analysis-m1053-t2.txt`
byte-for-byte apart from the echoed input path.

`/var/tmp/jw16-max-attrib-20260914/profile-m1053-t32.jsonl` on jw16 is the
uncompressed original of the 32-token capture.

# 2026-09-02 — GPU profile of a decode step on jwm1 (Honeykrisp) + reusable profiling harness

Purpose: measure where wall time actually goes during a decode step — GPU kernel
execution versus idle between dispatches versus host round trips — and rank
optimization opportunities from measurements, not intuition. Measurement only;
nothing optimized, nothing committed.

**Measurement condition (added 2026-09-03):** all jwm1-linux timings in this document were taken with only 1 of 8 CPU cores online (GRUB entry `Omarchy ANE test`, which supplies a static device tree and left cores 1-7 offline). The machine now boots all 8 cores. GPU-bound figures (kernel medians, GPU-busy shares) were materially unaffected (matmul TFLOP/s median 0.1556 -> 0.1576); the host-bound figures (tok/s, wall shares) may improve on re-measurement.

## Headline

**The GPU is busy 5.6-6.9% of the wall time.** Everything else is host-side
submission structure: one dispatch per submission, a full host join between
recordings, and 150-260 us of host descriptor-setup cost per dispatch. The
pure GPU execution time of a 41-token bf16 prefill (113-119 ms) is within
10% of what macOS MLX spends for the same work (377.9 tok/s = 108 ms) — the
20-37x end-to-end gap is structure, not kernel arithmetic. The suspected
barrier pair is real but second-order: blanket pre+post dispatch barriers
cost ~35 us per intra-submission gap; the host join structure costs ~212-225
us per dispatch and there are ~1,500 dispatches per decode token.
Overall kernel duration across ALL dispatches (observed from the event
streams): q4 n=21,177, median 17.4 us, mean 22.5 us, p90 30.8 us; bf16
n=20,278, median 17.0 us, mean 22.0 us, p90 30.9 us.


## Environment

- Host: jwm1 (192.168.3.66), Apple M1 (T8103, 8 GPU cores, 16 GB), Omarchy
  Linux, kernel 7.1.6-1-1-ARCH, single core online (`nproc=1`; m1n1 1.5.2 /
  spin-table mismatch, unchanged since the 2026-09-02 qualification
  receipt). Cold reboot does not fix it; box untouched this session.
- Vulkan device: `Apple M1 (G13G B1)`, driver `Mesa Honeykrisp`, API
  1.4.354, `timestampPeriod = 1 ns`, `timestampValidBits = 64` (captured by
  the harness meta line in every profile file listed under Artifacts).
- Base commit: `008e86bfc26b4f0fc7e31ee08aa2c5b145c84b1e` (origin/main).
  jwm1: fresh detached worktree `~/src/mlx-omarchy-profile` at exactly this
  commit plus the uncommitted harness diff below. Dev box: the same commit
  with the same uncommitted diff.
- Models (pinned snapshots used by every prior receipt):
  `/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx`,
  `/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx`.
- Python: venv `~/src/mlx-omarchy-profile/.work/venv-profile`, mlx-lm
  0.31.3, wheel `mlx_omarchy-0.32.2.dev20260902-cp314-cp314-linux_aarch64.whl`
  built from this worktree (`DEV_RELEASE=1 scripts/build-wheel.sh`), sha256
  `deadcc560416da0ae25a9f1a820144f96b78139b033138196ebc22914a0e9f92`.
- `MLX_DISABLE_COMPILE=1` on every model leg; `MLX_OMARCHY_ALLOW_NON_APPLE`
  never set on jwm1; `ulimit -c 0` everywhere; build exit status 0
  (single-core `ninja -j1`, test binaries + wheel).

## The harness (left in the tree, env-gated)

New file `overlay/mlx/backend/omarchy/gpu_profiler.h` plus a 42-line diff
across `encoder.cpp`, `vulkan.h`, `device.h`, `device.cpp` (uncommitted;
`git diff` in either worktree shows it; Main commits). Analysis tooling:
`scripts/profile_analyze.py` (parses the event stream, maps kernel enums to
names by parsing `compute.h`), `scripts/profile_generate.py` (receipt-identical
mlx-lm run that writes CLOCK_MONOTONIC phase markers aligned with the
harness host timestamps).

Usage:

```sh
# profile any workload (C++ side)
MLX_OMARCHY_GPU_PROFILE=/tmp/p.jsonl \
MLX_OMARCHY_GPU_PROFILE_LABEL=my-label \
  <workload command>

# receipt-identical generation with phase markers
MLX_DISABLE_COMPILE=1 MLX_OMARCHY_GPU_PROFILE=/tmp/p.jsonl \
  python3 scripts/profile_generate.py --model <path> --prompt "Hi" \
  --max-tokens 32 --temp 0 --seed 0 --markers /tmp/m.jsonl

# report
python3 scripts/profile_analyze.py /tmp/p.jsonl \
  --compute-h overlay/mlx/backend/omarchy/compute.h --markers /tmp/m.jsonl
```

What it records (NDJSON, one file): a `meta` line (device, timestamp period
and valid bits, pool capacity, label); one `d` line per compute dispatch
(kernel enum, groups, params.count, host record cost in ns, GPU ticks t0/t1,
binding list); one `s` line per submission (host cost of submit); one `j`
line per join (host cost of the completion-timeline wait and of the
noncoherent invalidate); one `end` line with totals. GPU ticks convert with
`meta.period_ns`.

Mechanism: two `vkCmdWriteTimestamp` per dispatch (BOTTOM_OF_PIPE; t0 after
the pre-dispatch barrier, t1 after the dispatch) plus one
`vkCmdResetQueryPool` per command buffer; results read back during
`join_last_completion`, when the device has already guaranteed
availability. Query pools are per-encoder (16,536-slot budget, overflow
drops timestamps and counts them; none dropped in any run below). No new
synchronization; nothing blocks the queue; the four new device-table entries
(CreateQueryPool, GetQueryPoolResults, CmdResetQueryPool,
CmdWriteTimestamp) are all core Vulkan 1.0.

### Instrument inertness (required before trusting any profile)

- Env unset: every hook is one predictable branch; the only unconditional
  changes are the four device-table loads and one capability field
  (`queue_timestamp_valid_bits`).
- Env set, llvmpipe dev box: omarchy_runtime_tests 22/22 cases, 6188/6188
  assertions, rc=0.
- Env set vs unset on the M1, failing suites re-run both ways: identical
  failed-case counts (eq_math 1, primitive 1, scatter_determinism 15,
  select_layout 3). Profiling does not change results.
- Model-level: instrumented CLI output text byte-matched the receipts
  (`Hello! How can I assist you today?`; `Paris`).

### Instrumentation overhead (measured, jwm1, real device)

mlx-lm generate, same command, warm, env OFF then ON:

| leg | prompt tok/s OFF | ON | delta | gen tok/s OFF | ON | delta |
|---|---|---|---|---|---|---|
| q4, France prompt, 41-prompt-tok | 21.034 | 17.405 | -17.2% | 4.873 (2 tok) | 3.940 | -19.1% |
| bf16, "Hi", 30-prompt-tok | 21.628 | 16.249 | -24.9% | 2.816 (10 tok) | 2.091 | -25.7% |

Uninstrumented parity with prior receipts (same machine, same day): q4
France 21.034/4.873 vs receipt 21.192/5.057; bf16 "Hi" 21.628/2.816 vs
21.452/2.810 (qualification) and 20.848/2.862 (compiled-tape receipt).

The overhead is real (per-dispatch fprintf emission + hash lookup +
timestamp bookkeeping on one core) and its direction is conservative for
every headline claim: kernel durations (t1-t0) are pure GPU time and
unaffected; host-idle gaps are INFLATED by instrumentation, so the measured
GPU busy fraction is a lower bound on the true one. If the instrumented
profile says 93% idle, the uninstrumented truth is at least that idle.
Harness improvement path (not done, to keep the instrument frozen during
measurement): single vfprintf per event, 1 MiB setvbuf, optional binding
capture.

## Battery gate (env OFF, jwm1, this worktree)

24 binaries, `ulimit -c 0`, sequential: 20 green, 4 red.

Green (cases): ane_bundle 12, compiled_tape 8, complex_ops 16, conv 7,
copy_offset 7, distributed 7, eig_ops 9, error_contract 3, fast_ops 11,
fast_regression 2, fft_general 14, fft_ops 17, indexing_ops 43, kv_ops 14,
linalg_ops 30, matmul_family 6, reduce_ops 21, runtime 22, shape_ops 24,
take_fill 8.

Red (owned elsewhere; correctness agent dispatched by Main):
- omarchy_eq_math_tests: 1 failed case (sin/cos tolerance, test_eq_math.cpp:230-231)
- omarchy_primitive_tests: 1 failed case (CHECK_EQ, test_primitives.cpp:5652)
- omarchy_scatter_determinism_tests: 15 of 21 cases; per Main's later
  correction, 12 are NAMED REFUSALS (VK_EXT_shader_atomic_float absent on
  the M1 Honeykrisp device; the capability gate refusing by name is the
  contract working), the rest under investigation
- omarchy_select_layout_tests: 3 failed cases (diff<=tol, test_select_ops.cpp:75)

Inertness of this harness w.r.t. the gate: identical counts with profiling
ON (section above). These suites are newer than the 5f8ba16 M1
qualification; pre-existence at clean 008e86b is being established by
M1RedInvestigation with a clean-HEAD build, not claimed here.

## Profile results (jwm1, real Honeykrisp; full analyze output in Artifacts)

Workloads: (a) `profile_generate.py` q4 "Hi" 10 generated tokens + 41-token
prefill; (b) same bf16; (c) instrumented CLI legs for receipt parity.
Model load records zero dispatches (mlx-lm quantizes lazily on first
forward, so prefill-phase counts include weight preparation).

| metric | q4 driver | bf16 driver | bf16 CLI | q4 CLI (France) |
|---|---|---|---|---|
| GPU busy fraction | **6.91%** | **6.74%** | **6.70%** | **5.60%** |
| dispatches | 21,177 | 20,278 | 20,278 | 7,089 |
| submissions | 18,631 | 17,732 | 17,732 | 6,263 |
| dispatches per submission | 1.14 | 1.14 | 1.14 | 1.13 |
| span (first..last GPU tick) | 6,891.9 ms | 6,618.3 ms | 6,624.1 ms | 2,857.9 ms |
| GPU busy | 476.3 ms | 446.3 ms | 443.6 ms | 160.0 ms |
| inter-submission gaps: n / total | 18,569 / 6,290.2 ms | 17,670 / 6,049.5 ms | 17,670 / 6,058.8 ms | 6,209 / 2,656.1 ms |
| inter gap p50 / p90 / p99 / max | 225.5 us / 487 us / 2.49 ms / 14.68 ms | 212.3 us / 422.8 us / 2.57 ms / 30.5 ms | 212.9 us / 424.8 us / 2.58 ms / 29.8 ms | 222.1 us / 470.8 us / 6.41 ms / 15.2 ms |
| intra-submission gaps (barrier pair): n / total | 2,607 / 125.4 ms | 2,607 / 122.5 ms | 2,607 / 121.6 ms | 879 / 41.8 ms |
| intra gap p50 / p90 / max | 35.6 us / 87.1 us / 209.5 us | 35.1 us / 86.7 us / 223.4 us | 35.1 us / 86.4 us / 147.9 us | 35.0 us / 86.5 us / 175.0 us |
| host dispatch-record cost p50 / mean | 166.3 us / 234.7 us | 160.4 us / 260.8 us | — | — |
| host join wait p50 / total | 180.5 us / 5,380.1 ms | 159.3 us / 5,199.2 ms | — | — |
| host submit() p50 / total | 54.3 us / 1,103.5 ms | 52.4 us / 1,033.0 ms | — | — |

Decode-step attribution (from phase markers, q4 / bf16):

- dispatches per decode token: **1,584.9 / 1,517.4**
- GPU busy per decode token: **35.7 ms / 33.3 ms** (357.3 ms / 332.8 ms over
  10 tokens)
- decode joins per token: 1,390.5 / 1,323.0 — i.e. essentially one
  submission+join per dispatch
- join wait per decode token: **371.8 ms / 378.5 ms** (isolated
  join_last_completion cost; single-core upper bound)
- intra-submission (barrier) gap cost per decode token: ~9-12 ms

Top kernels by cumulative GPU time (q4 driver; bf16 analogous, bf16/f16
swapped, MatmulBF16 70.0 ms in place of QmmF16):

| kernel | n | total | share | mean | p50 |
|---|---|---|---|---|---|
| ElementwiseF16 | 5,759 | 137.5 ms | 28.9% | 23.9 us | 29.4 us |
| ElementwiseF32 | 4,335 | 102.5 ms | 21.5% | 23.6 us | 22.8 us |
| CopyGeneralF16 | 3,521 | 66.2 ms | 13.9% | 18.8 us | 16.6 us |
| QmmF16 | 2,022 | 62.0 ms | 13.0% | 30.7 us | 30.6 us |
| CastF32F16 | 1,437 | 26.4 ms | 5.5% | 18.3 us | 16.5 us |
| CastF16F32 | 861 | 22.4 ms | 4.7% | 26.0 us | 23.7 us |
| ArangeF32 | 1,150 | 21.0 ms | 4.4% | 18.3 us | 16.6 us |
| FastRmsNormF16 | 586 | 14.1 ms | 3.0% | 24.1 us | 29.5 us |
| CastI32F32 | 575 | 10.8 ms | 2.3% | 18.8 us | 16.7 us |
| MatmulF32 | 574 | 7.6 ms | 1.6% | 13.2 us | 12.8 us |

Every kernel type, including `ArangeF32` (an iota) and `MatmulF32`, runs in
12-35 us: the per-dispatch GPU floor dominates everything; no kernel is
arithmetically large. The ranking by total time is a ranking by dispatch
COUNT, not by per-kernel expense.

Dependency proxy (consecutive dispatch pairs sharing no buffer = pairs a
blanket barrier provably does not need to order): q4 **74.8%** disjoint
(15,846/21,176), bf16 **71.1%** (14,408/20,277). So at most ~25-29% of
consecutive pairs can be genuinely data-dependent; the other ~3/4 are
ordered only by the blanket barrier.

### What the profile says, plainly

1. The dominant cost is the submission/join structure, not the barriers and
   not the kernels: inter-submission gaps are 91.3% (q4) / 91.4% (bf16) of
   the GPU-timeline span, at ~1.14 dispatches per submission.
2. `join_last_completion` alone costs ~372-378 ms per decode token on this
   single-core host (~1,390 joins/token at p50 ~160-180 us), before counting
   the ~160-260 us host record cost of each of the ~1,500 dispatches.
3. The blanket pre+post barrier pair is second-order: ~35 us per
   intra-submission gap, ~9-12 ms per decode token, with a measured
   71-75% barrier-free ceiling for a dependency-gated replacement.
4. The kernels themselves are floor-bound: 33-36 ms of GPU busy per decode
   token against macOS MLX's 3.4 ms/token (q4) and 16.3 ms/token (bf16)
   end-to-end. q4 decode has ~10x GPU-side headroom (QmmF16 at the ~30 us
   floor, 194 qmm dispatches/token), bf16 ~2.6x.
5. Prefill has the same disease: 41-token bf16 prefill = 113-119 ms GPU
   busy vs macOS 108 ms end-to-end at 377.9 tok/s; our 21 tok/s is host
   structure, all of it.

## Ranked opportunities (estimates from these measurements; not implemented)

1. **Batch submissions; remove the per-recording host join.**
   Change: keep command buffers open across evals; submit on a budget
   (nodes or bytes) like llama.cpp `max_nodes_per_submit` (default 100);
   join only when a host read demands it; use the fence/almost-ready
   pattern so the next encode starts while the GPU runs.
   Evidence: 1.14 dispatches/submission; inter-submission gaps 6.05-6.29 s
   of a 6.6-6.9 s span (91%); join wait 5.2-5.4 s per 10-token run;
   91-93% GPU idle.
   Win: removes up to ~5-6 s of the ~6.9 s profiled span; decode bound
   moves from ~2.8-5 tok/s to the host-record cost (~370 ms/token stays
   until item 2 lands, so items 1+2 together are what unlock ~20 tok/s
   decode; item 1 alone still yields a multiple on prefill tok/s because
   prefill amortizes joins across 41 tokens already).
   Risk: MEDIUM. Lifetimes and ordering were audited correct-by-construction
   under one-dispatch-per-submission (Wave11); batching changes which
   submission owns temporaries and semaphore ordering. Gate: full battery
   on the real device must stay at the pre-change baseline; any red is a
   stop.

2. **Stop creating a descriptor pool + set per dispatch.**
   Change: cache descriptor sets per pipeline (or pool reuse with reset at
   safe points); pre-allocate per binding count.
   Evidence: dispatch_compute today runs CreateDescriptorPool +
   AllocateDescriptorSets + UpdateDescriptorSets per dispatch; host record
   cost p50 160.4 us / mean 234.7-260.8 us per dispatch (q4/bf16), ~1,500
   dispatches/token = ~370-390 ms/token of host time; llvmpipe smoke of the
   same hook shows 18-64 us, so this is M1 driver-call cost, not
   fundamentals.
   Win: ~370 ms/token -> ~15-30 ms/token expected; with item 1, this is
   what carries decode to the ~20 tok/s GPU-side ceiling.
   Risk: LOW-MEDIUM. Descriptor lifetime must respect in-flight
   submissions; the existing keepalive mechanism (temporaries moved into
   the dispatcher entry) is the pattern to reuse.

3. **Replace blanket per-dispatch barriers with dependency-gated ones.**
   Change: track unsynced buffer ranges; emit one barrier only when a
   dispatch's bindings overlap an unsynced written range (llama.cpp model).
   Evidence: intra-submission gaps p50 35 us, ~122-125 ms per run, ~9-12
   ms per decode token; 71-75% of consecutive pairs provably disjoint.
   Win: ~70-75% of barrier gaps (~7-9 ms/token GPU) plus reduced driver
   barrier cost; small against items 1-2, but it compounds with them and it
   also shrinks the per-kernel floor each kernel pays waiting on a full
   flush.
   Risk: MEDIUM-HIGH. A missed dependency is a silent wrong value — the
   exact class that produced twelve defects. The range tracker must be
   conservative (over-approximate overlaps), gated by name where uncertain,
   and validated by the full device battery plus bit-comparison fixtures.

4. **Cut dispatch count: eliminate cast/copy/arange churn.**
   Change: native-dtype elementwise paths (stop upcasting f16/bf16 to f32
   and back per op), fuse elementwise chains, avoid materialized index
   arrays (ArangeF32 1,150 dispatches/run).
   Evidence: Cast* + CopyGeneral* + Arange = 7,574 of 21,177 q4 dispatches
   (35.8%) and 24-26% of GPU busy; 1,517-1,585 dispatches per decode token
   where the model math needs a few hundred; Main's corroborating external
   datapoint: the same fusion measured +53% tok/s on a Vulkan LLM path and
   nothing on CUDA.
   Win: multiplies items 1-3 (fewer dispatches = proportionally less host
   record, fewer joins, fewer barriers); ~25% direct GPU-busy cut.
   Risk: LOW-MEDIUM numerically (bf16 native rounding differs from
   upcast-f32-round; needs the pinned-fixture comparison the test rules
   already require), with zero ordering risk.

5. **Kernel arithmetic (GEMV/quantized-matmul tiling).**
   Change: register-tiled matmul/GEMV, no BK-loop unroll (llama.cpp
   disables unrolling on this exact driver; AGX is register-bound; no
   cooperative matrices on Honeykrisp; float atomics ABSENT on this device
   — no kernel may plan around them).
   Evidence: QmmF16 30.7 us / MatmulBF16 34.6 us mean at the dispatch
   floor; q4 GPU busy/token 35.7 ms vs macOS 3.4 ms/token end-to-end —
   ~10x q4 GPU-side headroom, ~2.6x bf16.
   Win: bounded by items 1-2 landing first; on today's wall it is worth
   ~0.3% (7 ms of 535 ms/token), which is exactly why it ranks last.
   Risk: kernel correctness; DecodeGemvPath owns the first slice (m==1
   vec path) with before/after receipt discipline.

## Single-core caveats

- jwm1 has one CPU online. Every host-side number here (dispatch record
  cost, join wait, submit cost, tok/s) is an upper bound: a healthy 8-core
  host would cut them, plausibly several-fold for the record cost (driver
  calls contend with the completion thread today).
- GPU-side numbers (kernel t0/t1 durations, busy fraction, gap structure
  between kernels) are device timestamps and unaffected by host contention.
- The busy-fraction CONCLUSION survives the caveat directionally: a faster
  host would shrink the measured idle, but the idle is ~91% of the span; to
  make the GPU 80% busy, host costs would need to fall ~25x, beyond any
  multi-core effect.
- Instrumented runs overstate idle by the instrumentation overhead measured
  above (17-26% tok/s), which makes the busy fraction a lower bound.
- Thermal procedure: sequential runs, one session, box otherwise idle
  (peers held off the single core during measurement legs); no cooling or
  clock instrumentation applied beyond prior receipts' practice.

## Artifacts (jwm1)

- Profiles: `/tmp/prof-q4-hi-driver.jsonl`, `/tmp/mark-q4.jsonl`,
  `/tmp/prof-bf16-hi-driver.jsonl`, `/tmp/mark-bf16.jsonl`,
  `/tmp/prof-q4-france-cli.jsonl`, `/tmp/prof-bf16-hi-cli.jsonl` (NDJSON;
  analyze command above). Battery logs `/tmp/battery-*.log`; profiler-ON
  re-runs `/tmp/on-*.log` + `/tmp/on-*.jsonl`.
- Build: `/tmp/profile-build.log` (ninja rc=0), `/tmp/profile-wheel.log`
  (wheel receipt lines, sha256 above).
- Dev box: harness diff uncommitted on the same commit;
  `receipts/2026-09-02-gpu-profile-decode.md` (this file) on the dev box.
- mlx-omarchy-info on jwm1 this session: device `Apple M1 (G13G B1)`,
  driver `Mesa Honeykrisp`, API 1.4.354, unified memory, timeline
  semaphore, total memory 7738 MiB.

No commits made. Main commits.
> **Annotation 2026-09-03:** the decode tok/s figures in this document are EOS-truncated short-burst rates, not steady-state decode (generation stopped after 2-10 tokens under `--max-tokens 32`). They are not comparable across machines or wheels. See `receipts/2026-09-03-decode-metric-fix.md`; replacement protocol: `scripts/bench_decode.py` (pinned length, token-count assertion).


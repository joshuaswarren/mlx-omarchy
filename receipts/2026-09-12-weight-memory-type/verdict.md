# Weight memory type: runtime facts, microbenchmark, and paired model A/B — memory type is not the deficit, and no device-private memory exists to route to

- schema: mlx-omarchy/weight-memory-type/1 (CLOSED 2026-09-12)
- agent: WeightMemoryType; branch `weight-memory-type` (bench arms 66fc5e7f,
  allocator env-gate 278bf468; never merged to main absent a repeating gain)
- host: placeholder jwm1 — Apple M1 (G13G B1), aarch64, Omarchy Linux kernel
  `7.1.6-1-1-ARCH`, fork driver `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`
- wheel: `mlx_omarchy-0.32.2.dev202609120900+d389c24-cp314-cp314-linux_aarch64.whl`
  sha256 `dc9f418338ebe9c86b76a8d0ab73005444ef32702c2dd2cf4e690b9d7a0516a4`,
  source commit `d389c24` (stamped in wheel version and asserted by
  bench_decode's provenance gate in every run)
- microbench binary sha256 `47cdad1bd9385dec9d319bf73bb72e9ad330e4013f4dae8c7a17019b2a7b2b29`,
  production shader frozen copy `qmm_vec_base.comp` sha256 `56aff1d3…`
  (diff-gated against `overlay/mlx/backend/omarchy/shaders/qmm_vec.comp` in-window)
- windows: microbench = ONE flock 2026-09-12T08:43Z (7 interleaved legs,
  ~350 s); model A/B = ONE flock 09:05–09:16Z (644 s, 20 runs); probes
  09:1xZ (two short locked runs). Loadavg sampled every 10 s in both
  windows (files attached).
- timing: wall-anchored only (host CLOCK_MONOTONIC around whole submits),
  same instrument and method as receipts/2026-09-11-q4-memory-roof. Timings
  read from JSON result lines only.

## 1. The facts, dumped at runtime: honeykrisp has ONE heap and NO device-private memory

From `vulkaninfo` and from inside the instrument
(`--dumpmem`, querying `vkGetPhysicalDeviceMemoryProperties` on the live
device — both agree):

```
memoryHeaps: count = 1
  memoryHeaps[0]: size = 8113881088 (7.56 GiB)  flags = DEVICE_LOCAL
memoryTypes: count = 2
  memoryTypes[0]: heap 0, propertyFlags 0x000f
      DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
  memoryTypes[1]: heap 0, propertyFlags 0x0007
      DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
allocator_pick (production selection rule over the live table): type 0
```

There is **no non-host-visible DEVICE_LOCAL type on this driver**. "Stage
weights into device-private memory" has no destination on Honeykrisp: both
types are the same unified heap, differing only in HOST_CACHED. This kills
the entire family of "where do the weights live" hypotheses at the root:
weights already live in device-local memory, mapped and coherent.

Allocation classes: every class (weights, KV cache, activations, scalar
fills, logits) routes through the single `VulkanAllocator::malloc`; there is
no separate staging class (the mapped coherent memory IS the stage), and the
selection rule returns type 0 for every allocation on the stock path. The
runtime per-class dump with the experiment gate armed
(`probe-alloc-dump.stderr.log`, BF16 model load + 32-token decode):

| class (by size) | n | type |
|---|---|---|
| >=256 MB (embedding/lm_head) | 1 | 1 (0x0007) |
| 1–32 MB (120 per-layer dense weights = 4 matrices x 24 layers + KV) | 120 | 1 (0x0007) |
| <1 MiB (decode activations, scalar fills, logits) | 1000+ | 0 (0x000f) |

Reach proof for the gate: first-call smoke plus this dump — the conditional
path is demonstrably executed, not just compiled.

## 2. Microbenchmark: same read-heavy kernel over the same bytes, both types

One locked window, interleaved a/b legs, 3 rep pairs, plus a default leg
proving the allocator's own pick == type 0. Judged on wall GB/s (the
device-timestamp `med_gpu_ns` peak fields remain the known invalid-dispatch
readings documented in the roof receipt; they are not the judged number).

| arm | type 0 (HOST_CACHED) GB/s | type 1 (uncached) GB/s | paired delta (geomean) |
|---|---|---|---|
| `pat_w1_r8` — the exact production Q4 GEMV byte pattern, 266 MB | 58.36 / 58.22 / 58.34 | 58.13 / 58.92 / 58.88 | **−0.14% = same** (+0.40/−1.19/−0.91%) |
| `peak_read` — flat 256 MB linear stream | 59.43 / 59.58 / 59.19 | 63.29 / 62.60 / 63.03 | **+6.0%** (+6.49/+5.07/+6.49%) |
| `peak_copy` — 256 MB read+write | 57.98 / 57.81 / 56.77 | 58.38 / 57.82 / 58.07 | +1.0% (noise) |

- The production Q4 pattern measures the SAME from both types: within-type
  spreads are 0.25%/1.4%, the three pairs straddle zero. **The Q4
  memory-type hypothesis is dead.**
- The +6.0% `peak_read` effect is real (3/3 pairs agree, tight spreads, and
  drift direction argues against an ordering artifact: type 1 legs ran
  later, thermal drift would slow them) — but it is a FLAT-STREAM effect
  that does not reach the Q4 GEMV pattern arm.
- The tiny `q4wall` shape timings in this instrument are L2-cache games
  (the `down` dispatch reads a hot 2.4 MB weight 21 times back-to-back:
  21 → 33 GB/s apparent), not token-representative; not used for decisions.

## 3. Model A/B: paired, interleaved, one window — nothing converts

Change under test: `MLX_OMARCHY_BIG_UNCACHED=1` routes allocations
>= 1 MiB to type 1 (env-gated; default path byte-identical). Weights are
written once at load and read every token; the floor keeps readback-scale
buffers (token logits, scalar fills, decode activations) host-cached.

Digests first: env ON, both drivers, 3 reps after a discarded warmup per
driver, every repetition asserted against the canonical pins —
**ALL_DIGESTS_HELD** (fork Q4 `7fd25a…`/`4cc089…`/`7da83f…`, fork BF16
`f26175…`/`8690dc…`/`ff5029…`, stock BF16 `7fc0f9…`/`46108a…`/`ff5029…`),
and every perf run's six legs asserted too. Moving allocations between the
two types moves no arithmetic, and the digests confirm zero value movement.

Paired decode tok/s (A = flag off, B = flag on, alternating in one window):

| leg | clean pairs p1–p4 (loadavg <= 0.92) | geomean |
|---|---|---|
| Q4 short | +2.47 / +0.72 / −0.01 / +0.92% | +1.02% |
| Q4 long | +0.06 / −0.31 / +0.29 / +0.80% | +0.21% |
| Q4 1K ctx | −2.64 / +0.05 / +0.48 / −4.66% | −1.71% (sign-inconsistent) |
| BF16 short | +0.19 / −0.13 / +0.22 / +0.44% | +0.18% |
| BF16 long | +0.43 / −0.26 / +0.00 / +0.33% | +0.12% |
| BF16 1K ctx | +0.08 / +0.36 / +2.47 / +0.73% | +0.91% |

Contamination disclosure: pairs p5–p6 ran while a co-tenant build drove
1-min loadavg from 1.1 to 7.07 (attached `memtype-loadavg`/window log);
their swings (−39% then +63% on Q4 long) are the contamination, not the
change, and they are discarded. p1–p4 are the decision set.

**The +6.0% flat-stream probe win converts to ~0.1–0.9% at the token on
BF16 decode — inside the 5% session noise floor, with no repeating
signature — and to nothing on Q4.** A third clean negative in a row
(after chunked softmax −2.5%, packed pair loads +1.1%): the remaining
decode deficit is not the memory system.

## 4. Verdict

1. Memory type is not the difference. On Honeykrisp there is no
   device-private memory to route weights to, and the two types that exist
   stream identically for the production GEMV pattern and near-identically
   for whole tokens.
2. The allocator gate stays on the branch, default-off (a no-op by
   construction), with the digest matrix green under it. Do not merge
   absent a repeating gain; it is kept so the question can be re-asked in
   one window, not re-derived.
3. Bookkeeping for the Q4 decode wall: bytes moved — at roof; arithmetic
   order — pinned bit-for-bit; access pattern — measured-unreachable
   (roof receipt); memory type — this receipt. The 41 → 60 GB/s GEMV gap
   now has no remaining hypothesis inside the memory system.

## Files

- `run-window.sh` — the locked window driver (digest matrix + paired A/B)
- `memtype-{def,a1..a3,b1..b3}-20260912T084300Z.ndjson` + `-session.txt` +
  `-loadavg.log` — microbenchmark raw artifacts (binary + shader sha256s in header)
- `fork-warmup|stock-warmup|fork-d1..d3|stock-d1..d3.{json,log}` — digest runs
- `base-p1..p6|bigunc-p1..p6.{json,log}` — paired perf runs
- `perf-runs.json` — extracted decode rates (A/B) + digest summary
- `probe-alloc-dump.{log,stderr.log}` — per-class allocation dump at real model load
- `verdict.json` — machine-readable summary

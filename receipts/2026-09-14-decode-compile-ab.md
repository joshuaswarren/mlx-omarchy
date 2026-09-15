# 2026-09-14 Decode A/B: MLX compile ON vs OFF on jw16 and jwm1 — NAMED RESULT, NO DEFAULT CHANGE

Date: 2026-09-14 ticket; runs executed 2026-09-15T05:26:08Z–05:27:16Z UTC.

## Verdict

**Compile ON holds both pins on both laptops (40/40 runs) but does not
raise decode tok/s toward native Metal beyond noise: named result, no
default change.** `MLX_DISABLE_COMPILE=1` stays the receipt protocol.
Nothing was merged (`63c1d3cf` still untouched; no repo state changed
beyond this receipt).

Two findings the ticket was after:

1. **The compiled path is digest-safe.** With `MLX_DISABLE_COMPILE`
   unset (libmlx default, i.e. what "native Metal compiles the graph"
   means on this stack), every run on both hosts reproduced
   short `7fd25a869ff21678` and ctx1053 `7da83f06ec9f001d`. The
   Honeykrisp bit-exactness breakers named in the ticket — the
   GEMV-epilogue RoPE and RMSNorm folds into `qmm_vec` producing
   `31267e7ed4c6d0dc` — are properties of the **unmerged fold
   branches** (`receipts/2026-09-14-gemv-rmsnorm.md`: fold arm
   `DIGEST MISMATCH 31267e7ed4c6d0dc`), not of `main`'s compiled-tape
   path. Enabling compile does not smuggle the fold digests into the
   default configuration.
2. **Compile ON cannot move this decode gap.** Per-token dispatch
   probes: `vk_compute_dispatches` **249** and `vk_submissions` **2**
   with compile ON and OFF, on both hosts; only
   `gpu_primitive_dispatches` drops **858 → 810** (−48 eval_gpu calls
   per token merged into tape evals). The GPU command stream is
   byte-identical, which is exactly why digests hold and tok/s is
   flat — and it is consistent with the attribution receipt's thesis
   (`receipts/2026-09-14-decode-gap-attribution.md`): the deficit is
   per-dispatch serial cost × 249, so only a dispatch-count reduction
   moves it. The fold work reduces the count but breaks the pins; the
   named hole stands.

## Result medians (5 interleaved rounds × 2 legs × 2 arms per host)

Same wheel, same interpreter, both arms per host; off = `MLX_DISABLE_COMPILE=1`,
on = variable removed (never `=0`). Native divisors: base-M1 150.57 /
140.38 tok/s, M1 Max 286.96 / 283.79 tok/s (pinned
`native-2026-09-06-summary.json` per the 2026-09-13 parity receipt).

| host | leg | off median | on median | raw Δ | paired Δ (median of per-round) | % of native off → on |
|---|---|---:|---:|---:|---:|---|
| jwm1 (M1) | short 30/32 | 111.4156 | 111.7582 | +0.31% | **+0.28%** | 73.99% → 74.22% |
| jwm1 (M1) | ctx1053 1053/32 | 96.4552 | 96.6449 | +0.20% | **+0.20%** | 68.71% → 68.85% |
| jw16 (M1 Max) | short 30/32 | 168.5784 | 166.6779 | −1.13% | **−0.99%** | 58.75% → 58.08% |
| jw16 (M1 Max) | ctx1053 1053/32 | 134.5339 | 143.1468 | +6.40% | **+0.51%** | 47.41% → 50.44% |

Prefill medians (not part of the land rule): jw16 short 364.48 → 373.81,
ctx1053 3840.61 → 3766.23; jwm1 short 329.03 → 334.51, ctx1053
1112.14 → 1123.90 tok/s.

**The one eye-catching cell deflates under pairing.** jw16 ctx1053's
+6.40% raw median gap is driven by a single round (r4 off dipped to
131.5677 while on held 143.1468); per-round paired deltas are
−0.47%, +0.51%, −1.69%, +0.58%, +8.80% — median **+0.51%**. All four
cells are tok/s-neutral within run noise.

All 40 runs digest-pinned:

- jw16: 20/20 `7fd25a869ff21678` (short) and `7da83f06ec9f001d`
  (ctx1053), both arms.
- jwm1: 20/20, same pair, both arms.
- `prompt_tokens` 30/1053 asserted on every leg; provenance
  `verified=match` asserted on every leg.

## Per-run decode tok/s (round order as run; arm order alternates)

| host | leg | off | on |
|---|---|---|---|
| jw16 | short | 171.2547, 167.8313, 169.0972, 168.3519, 168.5784 | 166.4856, 167.8202, 167.6283, 166.6779, 165.1470 |
| jw16 | ctx1053 | 144.6486, 144.9594, 134.5339, 133.2006, 131.5677 | 143.9733, 145.7012, 132.2551, 133.9678, 143.1468 |
| jwm1 | short | 106.6442, 111.4496, 111.4156, 111.4643, 111.3485 | 112.3081, 111.7582, 111.4807, 111.8359, 110.4409 |
| jwm1 | ctx1053 | 96.6035, 96.5114, 96.4552, 95.7787, 95.2021 | 96.1254, 96.9235, 96.6449, 96.6807, 96.0973 |

## Dispatch probes (tokens 8, median per token; 3 runs per cell, all identical)

| config | vk_compute_dispatches | gpu_primitive_dispatches | vk_submissions |
|---|---:|---:|---:|
| compile OFF, jw16 | 249 | 858 | 2 |
| compile ON, jw16 | 249 | **810** | 2 |
| compile OFF, jwm1 | 249 | 858 | 2 |
| compile ON, jwm1 | 249 | **810** | 2 |

Compile engages (−48 host-side primitive evals per token, stable across
all 6 on-probes) but emits the identical per-token GPU command stream.

## Identity

- Wheel (both hosts, bit-identical):
  `mlx_omarchy-0.32.2.dev202609122355+b41e2b74-cp314-cp314-linux_aarch64.whl`,
  SHA-256 `81743cd1a631f6c5d8aa7ff9538d1ec4ddcb2a57c184da7c55345e48d349a240`
  (same build the 2026-09-14 jwm1 parity rerun and both attribution
  captures used). jw16's copy shipped from jwm1's `dist/` via the
  workstation this session; `pip install --force-reinstall --no-deps`
  into a fresh clone of jw16's `venv-base`.
- Loaded binaries identical on both hosts:
  `core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea`,
  `libmlx.so=sha256:03b3f4b9024927f9`; provenance `verified=match`
  every leg (jw16 prints `harness=b41e2b74-dirty` from its checkout
  dirt; jwm1 `harness=b41e2b74` — harness files hashed equal, below).
- Python 3.14.7, mlx-lm 0.31.3, Vulkan Honeykrisp Mesa 26.3.0-devel
  (git-6f6afc8968), `Device(gpu, 0)`.
- Models: jw16 HF snapshot `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`;
  jwm1 `/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx`
  (both `mlx-community/Qwen2.5-0.5B-Instruct-4bit`).
- Harness SHA-256 (identical on both hosts):
  `ab_compile.py` `5617dffcaf8740cf73b028090a72ca098b81dd280fbbf8724b32c187b2393776`,
  `dispatch_count.py` `1767156509f2d6754ef1036e5a8b859d2e2d1e3da8843ecf4fd8a21757edcc44`;
  `bench_decode.py` `f5062d88f34b0845c1e59b0b35d2e33ae02180f636a0f65c543a4366ec2bef7f`,
  `bench_matrix.json` `df8eb9f3ed84182604379b7cde070ade706bfe590fbf83a35b1877cfaf43f258`
  (both hosts, equal to the jwm1 parity-rerun pins).
- Hosts: `jw16mbp1-linux` (aarch64, 7.1.6-1-1-ARCH, battery Full, AC
  online) and `jwm1-linux` (aarch64, 8 cores).
- Receipt checkout: local `mlx-omarchy` `64e52943`.

## Protocol

One `flock` hold per host on `/tmp/m1-gpu.lock` (jw16 inode 12, jwm1
inode 29, same before and after; nested `flock -n` refused while held;
never unlinked; never stolen — `flock -w 600`). Under the hold: 6
dispatch probes (off ×3, on ×3), then the interleaved A/B: 5 rounds ×
legs {short 30/32, ctx1024 1053/32} × arms {off, on}, arm order
alternated per round, every leg a fresh `bench_decode.py` process with
`HF_HUB_OFFLINE=1 --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4`.
Unlike the fold A/Bs, a digest mismatch does NOT abort here: the
compile-on digest is a result. A 1-round smoke on jw16 validated the
harness before the full runs (smoke artifact removed after).

## Decision

Acceptance branch: "compile ON matches pins AND tok/s rises toward
Metal → document as the path; breaks pins or does not rise → named
result, no default change." Pins held; tok/s did not rise (paired
medians −0.99% to +0.51%). **Named result: `MLX_DISABLE_COMPILE=1`
remains the protocol and default; the compiled path is proven
digest-safe and provably inert for this gap (249 dispatches
unchanged); the only lever that moves decode remains dispatch-count
reduction, which stays blocked at the ctx1024 pin by the `qmm_vec`
epilogue folds (`31267e7ed4c6d0dc`).** No default was flipped, no
branch merged, `63c1d3cf` not merged.

## Artifacts

- jw16: `/var/tmp/DecodeCompileAB/jw16-out/` — `ab.txt`, `ab.json`
  (medians + per-run + provenance), `dispatch-{off,on}-{1,2,3}.txt`,
  `started.txt`, `finished.txt`, `host.txt`, `lock.txt`.
- jwm1: `/var/tmp/DecodeCompileAB/jwm1-out/` — same set.
- Harness: `/var/tmp/DecodeCompileAB/{ab_compile.py,dispatch_count.py,mlx_provenance.py,run-jw16.sh,run-jwm1.sh}`;
  jw16 venv `venv-ab`; jw16 wheel copy
  `/var/tmp/mlx_omarchy-0.32.2.dev202609122355+b41e2b74-cp314-cp314-linux_aarch64.whl`.
- Runner log (jwm1): recovered from background job `bg_4`; identical
  content in `jwm1-out/*.txt`.

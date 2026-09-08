# Parity-baseline audit — 2026-09-08 — ParityBaseline

Scope: audit of the measurement harness and historical native receipts
before the first c2548675 hardware window; native oracle provenance; the
historical long-prompt digest divergence.

## Harness audit (all at c2548675 unless noted)

`scripts/bench_matrix.py` (last touched by `a34cdc5a`, lineage
`871ff4ea`/`d8a8e429`):

- Models resolve from the local HF cache only; offline hosts never
  fetch. A pinned revision that does not match the local snapshot skips
  the leg; `--expect-pins` refuses the whole run (exit 4) when a model
  resolves to a different revision, so cross-machine comparisons cannot
  silently compare different weights.
- Each leg runs `scripts/bench_decode.py` as a subprocess with the
  manifest env (`MLX_DISABLE_COMPILE=1`), parses decode/prefill rates,
  the provenance line, the `generated_ids sha256:<16hex> n=N` line, and
  the bench-reported prompt-token count. Leg agreement is enforced:
  requested == leg tokens, decode_tokens == tokens-1, digest present,
  digest length 16 hex, n == tokens. A violated agreement is
  `failed`, never a number.
- `prompt_tokens` is trusted only from bench_decode's measured
  count (a transformers 5.x `BatchEncoding` bug poisoned derived prefill
  rates; the independent tokenizer probe was removed deliberately).
  `prefill_tok_s = prompt_tokens / prefill_s`.
- Contention: `clean_check` scans process names (not args) for
  model-serving processes; contended runs are labeled, and this
  baseline's runner asserts clean. Power state is recorded from
  pmset/`/sys/class/power_supply`, never inferred. Host facts include
  vulkaninfo `driverName`/`deviceName` on Linux and `system_profiler`
  chipset/GPU-core facts on macOS. Paths are sanitized ($HOME → `~`)
  before they reach reports.

`scripts/bench_matrix.json` (schema bench-matrix/1):

- Generation: temp 0.0, seed 0, warmup 4 tokens, engine bench_decode,
  env `MLX_DISABLE_COMPILE=1` (eager policy on BOTH platforms).
- Models: `qwen25-0.5b-4bit` pinned
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`; `qwen25-0.5b-bf16` pinned
  `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`; 7B/14B optional
  resolve-or-skip (absent on this M1 → skipped).
- Prompts: short = `Hi` (30 chat-template tokens); long = embedded
  deterministic text, 1208 chars (262 tokens); ctx1024 = numbered
  template, 20 entries, 4759 chars (1053 tokens); ctx4096 =
  explicit-selection only.
- Workloads: `short-decode-32` (30/32), `long-decode-128` (262/128),
  `longctx-1024-decode-32` (1053/32). Cross-platform identity key is
  the generated-IDs digest, never decoded text.

`scripts/bench_decode.py`:

- EOS suppressed (`tokenizer.eos_token_ids = set()`), so exactly
  `--tokens` tokens are produced and asserted against the request;
  load excluded; prefill timed to the first token; 4 untimed warmup
  tokens; decode rate spans tokens-1 inter-token gaps.
- Provenance gate runs BEFORE any mlx import: `mlx_provenance`
  compares the loaded `libmlx.so`/core extension hashes against the
  `--wheel` file; mismatch refuses to emit numbers (exit 3).
- Exact IDs travel with every run (digest line + JSON line).
- `ids_digest` = sha256 over the comma-joined ASCII token IDs, first
  16 hex chars.

`scripts/profile_generate.py` / `scripts/profile_analyze.py`:

- profile_generate reuses the mlx-lm load + stream_generate path and
  writes CLOCK_MONOTONIC markers (load/prefill/decode/tok) on the same
  clock as the C++ harness host timestamps. Requires
  `MLX_OMARCHY_GPU_PROFILE=<path>` and a wheel built with
  `-DMLX_OMARCHY_GPU_PROFILING=ON` (release wheels compile the harness
  out; the env var is then inert — verified in
  `overlay/mlx/backend/omarchy/CMakeLists.txt` and `gpu_profiler.h`).
- profile_analyze: kernel names come from the ComputeKernel enum
  declaration order in `overlay/mlx/backend/omarchy/compute.h`; device
  ticks unwrap via `meta.valid_bits` and convert with
  `meta.period_ns`; outputs GPU busy fraction, intra/inter-submission
  gap split, per-kernel totals/mean/median/share, dispatches per
  submission, host join/submit/record costs, and a clearly-labeled
  dependency proxy. Dispatch records are written delayed, so phase
  attribution follows the SUBMIT record's marker window; the two
  unknown categories are kept distinct rather than folded into phases.

### Compile-policy matching (explicit)

`bench_matrix.execute_leg` applies `env.update(manifest.generation.env)`
to every leg on BOTH platforms, and `bench_matrix.json` sets
`generation.env.MLX_DISABLE_COMPILE=1`. The native "default" bundle
therefore already ran eager under the same manifest as the Linux runs —
it is the matched-policy oracle. The "nocompile" bundle re-exported the
same variable in the outer shell (belt and braces); identical digests
across all 5+5 native reps confirm the policies coincide. The default
bundle's medians are the authoritative target set; the nocompile bundle
is the policy robustness confirmation, not a different policy.

Audit verdict: the harness enforces pinned weights, fixed lengths,
greedy policy, eager compile policy, clean contention, AC power, binary
provenance, and exact-ID capture. No gaps that would invalidate a
baseline. One explicit provenance gap exists on the native side (below).

## Native oracle provenance (receipts/native-baseline-2026-09-06)

Both bundles (default `native-2026-09-06.json` + rep1-5, and
nocompile `native-2026-09-06-nc{1..5}.json` + summary) ran on the
actual target machine, booted macOS:

- Host: Apple M1, 8 cores, 17179869184 bytes RAM, macOS 14.8.9
  (23J631), GPU chipset Apple M1, 8 GPU cores, Metal 3. AC power,
  100%, charged. clean_check clean (283 / 281 processes scanned).
- Stack: python 3.11.16, upstream MLX 0.32.2 from PyPI (extension
  sha256 `32a9f068…` matches the installed wheel RECORD — verified),
  mlx-lm 0.31.3, transformers 5.16.1, numpy 2.4.6.
- Same manifest, same pins, same prompts, same eager policy.
- 5 isolated reps each; every leg digest stable across reps AND
  across both compile policies; medians agree within ~0.3%.

Native oracle (target M1, medians of 5, DEFAULT bundle = authoritative):

| leg | decode tok/s | prefill tok/s | ids digest |
|---|---:|---:|---|
| 4bit short 30/32 | 150.57 | 294.1 | `7fd25a869ff21678` |
| 4bit long 262/128 | 146.77 | 1213.0 | `254d73fd93164b98` |
| 4bit ctx1024 1053/32 | 140.38 | 1840.9 | `7da83f06ec9f001d` |
| bf16 short 30/32 | 56.24 | 232.6 | `7fc0f968789b1882` |
| bf16 long 262/128 | 55.62 | 1007.7 | `407b7624ed1b3b29` |
| bf16 ctx1024 1053/32 | 54.25 | 1653.1 | `ff502900d2a179a5` |

(bf16 row quoted from the nocompile summary per the coordinator's
request; the default-policy bf16 medians differ by ≤0.4% with identical
digests. Q4 rows are from the default summary.)

Explicit provenance gap: the native receipts record
`binary_provenance.omarchy.verified = no-metadata` and
`source.harness_commit = unknown` — correct behavior for a PyPI
upstream install (there is no mlx-omarchy dist to check); the binary is
pinned instead by the wheel-RECORD hash match. The native side is
therefore an upstream-MLX 0.32.2 oracle, not an mlx-omarchy oracle; the
Linux side is the fork. This is the correct comparison for the parity
goal and is recorded as-is, not papered over.

## Historical evidence contamination (resolved by the 09-06 receipts)

`receipts/2026-09-04-native-output-comparison.json` compared Linux
against a machine recorded in its own `stacks` block as
**Apple M1 Max, 32 GPU cores, macOS 26.6.2, upstream PyPI wheel,
contended window** — NOT the target M1. It is a different chip and an
uncontrolled window; it makes no timing claim, and its digests are
corroborative only. The target-M1 09-06 receipts independently
reproduce its long128 digest (`254d73fd93164b98`) on the right
hardware, so the historical comparison's identity conclusions happen to
stand, but its native side is superseded by the on-target receipts.
## Long-prompt digest divergence (facts and partial mechanism)

Deterministic per-backend (5/5 Linux reps at c2548675; 5/5 native
reps, both compile policies on the target M1). 4bit short and ctx1024
digests MATCH cross-OS exactly; only long128 differs. First difference
at zero-based index 20 (20-token identical prefix); streams re-align
at indices 22-23 and diverge again from index 45; 86 of 108 tail
positions differ; both continuations coherent, no garbage (2026-09-04
comparison receipt; native IDs in that receipt are from an M1 Max,
but the long128 digest and divergence pattern reproduce on the target
M1 — corroborative, not on-target identity proof).

### Plausible mechanism, NOT closure

Linux margin probe (margins-long128.json, digest independently
reproduces the exact Linux stream `4cc08910089477fd`): the long128
fork sits on an EXACT float32 argmax tie at generated index 20 — top1
and top2 logits both 20.953125 (` review`=3395 vs ` carefully`=15516,
margin 0.0000).

This is a plausible mechanism, not a closure of the numerical gate:

- We have only Linux-side logit evidence. The native-side logit pair at
  index 20 has not been captured on the target M1 (would require a
  coordinated macOS reboot + a probe-side harness analogous to
  margin-probe.py; not in this window). Linux showing an exact tie
  does not prove native shows an exact tie — native could legitimately
  have a strict ordering on the same pair and pick 15516 deterministically
  while Linux (under different ULP rounding) hits the tie.
- Standard MLX argmax tie-breaking is lowest-index-first. Token 3395
  has the lower index. Linux picked 3395 (consistent with lowest-index
  tie-break on a tied pair). Native picked 15516, which is INCONSISTENT
  with a naive lowest-index tie-break on a tied pair — strongly
  suggesting either (a) native logits at this position are NOT actually
  tied (some ULP difference picks 15516 outright), or (b) the two
  backends use different argmax tie-break conventions. Both branches
  are unresolved without paired native logits.
- 2/3 workloads matching cross-OS outright (short + ctx1024) still
  suggests that the fork's reduction-order noise is bounded, but it
  does not prove bit-identical numerics — only that no generated
  position in those workloads hit a disagreement.

Conclusion: the Linux exact-tie at index 20 is necessary evidence
that a backend ULP difference COULD flip this argmax; the native side
is the missing half of the proof. The numerical gate is UNRESOLVED
without a paired native-margin probe or an existing native-golden
evidence at the same position. No tolerance was relaxed and no
model, prompt, or length changed (the gate explicitly forbids that),
but the question remains open.

bf16 observations (recorded, not blockers): bf16 long128 Linux digest
`ad964232ee67fecd` differs from native `407b7624ed1b3b29` — same
tie-cascade signature expected, not margin-probed; bf16 short32
`f26175202f3dabe9` is the known post-`2f54fcb` stream whose divergence
from native starts after the EOS position in the EOS-suppressed
continuation (docs/known-defects.md); bf16 ctx1024 matches native
exactly.
- CPU-oracle attempt: the fork's CPU backend cannot run the pinned
  4-bit model through mlx-lm's cached decode — `RuntimeError: NYI`
  while evaluating the prompt-cache state
  (`oracle-cpu-long.out`/`window2-cpu-nyi.log`). Recorded as evidence
  of a CPU-path gap; the Vulkan-vs-reference question is carried by
  the margin probe instead.

## Ranked costs (GPU-timestamp profile, diag wheel) — INSTRUMENTED ONLY


The profiling harness is intrusive: on long-128 the instrumented span

is 6.718 s versus ~2.27 s decode + ~0.64 s prefill on the unprofiled
release wheel — i.e. the profile roughly doubles wall time. ALL
GPU-busy/gap fractions below are therefore instrumented-build metrics,
NOT unprofiled GPU utilization; the kernel-share ranking (which kernel
consumes the most GPU-busy time) is the robust part, while absolute
busy/gap splits must be re-derived from an unprofiled scaling benchmark
before any "dispatch-bound" claim is treated as native behavior. The
coordinator owns that cross-check (assigned to DecodeParity).
prefill + decode; phases separated by markers in the .analysis.txt
files):

| leg | dispatches | GPU busy | intra-submission gap p50 | share of span idle |
|---|---:|---:|---:|---:|
| short-32 | 7,045 | 19.8% | 42.0 µs | ~69% intra gaps |
| long-128 | 76,075 | 22.6% | 37.5 µs | ~74% intra gaps |
| ctx1024-32 | 19,915 | 9.1% | 34.9 µs (p90 183 µs, p99 5.1 ms) | ~86% intra gaps |

The dominant cost is dispatch overhead, not kernel time: at ~35-42 µs
median gap per dispatch × 70-76k dispatches, the GPU idles most of the
span. Kernel busy-time ranking (share of GPU-busy, consistent across
legs):

1. ElementwiseF16 31-35%
2. QmmVecQ4WordSubgroupF16 21-23%
3. CopyGeneralF16 10.7-11.7%
4. FastRopeF16 9.7-11.3%
5. FastRmsNormF16 7.0-10.7%
6. MatmulF16 5.5-7.9%
7. SoftmaxF16 ~3%

Implication for the waves: kernel-side wins are bounded by the busy
fraction (~10-23% of wall); the larger lever is per-dispatch
submission/floor cost and dispatch count, consistent with the decode
A/B floors (~54 µs fixed + ~23 µs/dispatch host submit + ~21 µs GPU
floor) DecodeParity measured. Raw evidence:
`profile/*.analysis.txt`, `profile/*.profile.jsonl`,
`profile/*.markers.jsonl`.

## Window record

- window.log (attempt 4): metadata + vulkaninfo receipt, baseline
  5x3x2 complete, profiling legs complete. Attempts 1-3 retained as
  `window-attempt{1,2,3}-*.log` with their failure causes (venv
  interpreter symlink resolution, engine hook path, manifest path) —
  process evidence, not measurement evidence; no numbers were emitted
  by any failed attempt (bench_matrix refuses to emit without legs).
- window2: CPU oracle legs → long leg NYI (above); log retained.
- window3: margin probe attempt, failed on a probe-side token-feed
  shape bug (margins-long128.err); no numbers claimed.
- window4: margin probe on GPU — exact tie found at index 20
  (margins-long128.json); log `window4.log`. Lock verified free after
  release (fuser empty) before sibling windows opened.
- Profile artifacts: `profile/{short-32,long-128,ctx1024-32}.{
  analysis.txt,profile.jsonl,markers.jsonl}` + `profile-summary.json`
  on the -diag wheel (`5e201380…`, `MLX_OMARCHY_GPU_PROFILING=ON`,
  verified: 3 harness symbols present in wheel `libmlx.so`).
- Release wheel: `255c2f93…7e02`
  (`mlx_omarchy-0.32.2.dev202609081618+c254867`), provenance
  verified=match on every measured leg.

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
  vulkaninfo `driverName`/`deviceName` on Linux and
  `system_profiler` chipset/GPU-core facts on macOS. Paths are
  sanitized ($HOME → `~`) before they reach reports.

`scripts/bench_matrix.json` (schema bench-matrix/1):

- Generation: temp 0.0, seed 0, warmup 4 tokens, engine bench_decode,
  env `MLX_DISABLE_COMPILE=1` (eager policy on BOTH platforms).
- Models: `qwen25-0.5b-4bit` pinned `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`;
  `qwen25-0.5b-bf16` pinned `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`;
  7B/14B optional resolve-or-skip (absent on this M1 → skipped).
- Prompts: short = `Hi` (30 chat-template tokens); long = embedded
  deterministic text (262 tokens); ctx1024 = numbered template, 20
  entries (1053 tokens); ctx4096 = explicit-selection only.
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

Native oracle (target M1, medians of 5):

| leg | decode tok/s | prefill tok/s | ids digest |
|---|---:|---:|---|
| 4bit short 30/32 | 150.57 | 294.1 | `7fd25a869ff21678` |
| 4bit long 262/128 | 146.77 | 1213.0 | `254d73fd93164b98` |
| 4bit ctx1024 1053/32 | 140.38 | 1840.9 | `7da83f06ec9f001d` |
| bf16 short 30/32 | 56.24 | 232.6 | `7fc0f968789b1882` |
| bf16 long 262/128 | 55.62 | 1007.7 | `407b7624ed1b3b29` |
| bf16 ctx1024 1053/32 | 54.25 | 1653.1 | `ff502900d2a179a5` |

(bf16 row from the nocompile summary; the default-policy bf16 medians
differ by ≤0.4% with identical digests.)

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
No M1 Pro/Max receipt is used as a baseline number anywhere in this
assignment.

## Long-prompt digest divergence (facts)

- Linux `4cc08910089477fd` is deterministic across wheels
  (4f27136, a34cdc5, 348919c, 417c06e), across days (09-03 → 09-07),
  and across every 5-rep batch; native `254d73fd93164b98` is
  deterministic across 5 reps, both compile policies, and the M1 Max.
- 4bit short and ctx1024 digests MATCH cross-OS exactly; only long128
  differs. First difference at zero-based index 20 (20-token identical
  prefix); streams re-align for indices 24-44 and diverge again from
  index 45; 86 of 108 tail positions differ; both continuations are
  coherent, no garbage (2026-09-04 comparison receipt).
- bf16: the RoPE dense-promotion stride defect (fixed at `2f54fcb`)
  made Linux bf16 diverge from the first token; after the fix the
  eager stream matches native through EOS+4 tokens and diverges in the
  EOS-suppressed continuation (first difference index 14, EOS at 9).

Working explanation (to be closed by in-window evidence, no tolerance
relaxation): identical digests on two of three workloads plus a
deterministic low-index fork on the third is the signature of
backend-level floating-point reduction-order differences that only
become visible where an argmax sits on a near tie. The in-window
experiments quantify this: (1) `cpu-oracle.py` runs the same engine on
the stock CPU backend of the SAME Linux wheel — if CPU yields the
Linux digest, the fork's tokenizer/template/sampling path is
consistent and the divergence lives in GPU kernel numerics; (2)
`margin-probe.py` measures the per-step top-2 logit margin on the GPU
stream — a near-zero margin at the fork index proves a tie-break
sensitive to ULP-level differences, which cross-backend comparisons
routinely produce and which token-identity checks cannot "pass"
without making the kernels bit-identical (out of scope for a
performance-parity effort and not required by any gate).

## Window plan (single flock, /tmp/m1-gpu.lock)

1. Claim lock; print hostname; verify load quiet.
2. Metadata receipt (bench_matrix --mode metadata) + vulkaninfo
   driverInfo snapshot (Honeykrisp Mesa 26.3.0-devel git-6f6afc8968,
   Apple M1 (G13G B1)) — verified before the window.
3. `run-baseline.py` — 5 reps × 6 legs (2 models × 3 workloads), full
   IDs per rep via MLX_PAIR_IDS (unique file per rep), provenance
   asserts per rep.
4. `run-profile.py` on the -diag wheel — profile_generate + analyze
   for short-32, long-128, ctx1024-32.
5. `cpu-oracle.py` legs (long128 + short control + ctx1024 control).
6. `margin-probe.py` long128.
7. Release lock; notify DecodeParity / PrefillParity / ANEParity /
   Main with paths.

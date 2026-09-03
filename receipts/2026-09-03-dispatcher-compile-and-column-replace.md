# 2026-09-03 — poll-interval and ordering bakeoff, compiled-path gates, README column re-measure (jwm1, 8 cores)

Second follow-up to `receipts/2026-09-03-decode-ab-and-affinity-jwm1.md`,
after that receipt's provisioning bug was found and fixed. Read the
provisioning section first: it changes the meaning of every number taken
before it.

## Provisioning bug and redo discipline (important)

The shared support-requirements file used to provision every fresh venv
contained a direct-URL pin,
`mlx-omarchy @ file:///…/mlx_omarchy-0.32.2.dev20260903-…whl#sha256=6e54ab2b…`,
emitted by `pip freeze` for a wheel-installed distribution. The exclusion
filter only removed `==`-form lines, so this line survived, and pip
silently reinstalled the stale 06:06 wheel (5,300,664-byte `libmlx.so`,
`bb90d191…`) AFTER the wheel under test in every fresh venv. Three venvs
used earlier in this session were poisoned this way; measurements taken in
them measured the stale binary while claiming to test other commits.

Redo discipline for every number below: fresh venv, wheel installed first,
then `mlx-lm==0.31.3 --no-deps`, then a requirements file with both
`mlx-omarchy` lines removed, then a provenance gate — the sha256 of the
installed `mlx/lib/libmlx.so` must equal the sha256 of that member inside
the wheel under test. All failures loud.

## Bakeoff redo (wheels unchanged; provisioning fixed)

Same protocol as before: prompt "What is the capital of France?" for every
leg, `MLX_DISABLE_COMPILE=1`, `--max-tokens 32 --temp 0 --seed 0`, 5 runs,
median. These are EOS-truncated short-burst rates and are comparable only
within the table.

| leg | loaded libmlx.so (prefix) | bf16 prefill | bf16 decode | 4-bit prefill | 4-bit decode |
|---|---|---|---|---|---|
| base `3fdcc20` | (gate green, wheel `5d2e185e…`) | 46.298 | 5.417 | 28.788 | 5.605 |
| fix `4ea2f47` (1 ms in-flight poll) | (gate green, wheel `061dfd36…`) | 46.114 | 5.280 | 29.063 | 5.622 |
| ordering `ff4b05a` (single commit: each submission waits for the stream's previous one) | (gate green, wheel `f8e35d7b…`) | 34.468 | 3.831 | 22.503 | 3.417 |

- Leg 1 verdict (unchanged in direction, now measured honestly): the 1 ms
  in-flight poll does not move decode (bf16 -2.5%, 4-bit +0.3% — noise).
  The dispatcher-wakeup hypothesis stays dead.
- Leg 2 verdict (REVERSED by the redo): the ordering fix is **not free**
  on hardware. It costs **-25.3% bf16 prefill, -22.6% 4-bit prefill,
  -27.5% bf16 decode, -39.2% 4-bit decode** versus its parent `4ea2f47`.
  The contaminated leg-2 run had measured the stale binary on both sides,
  which is exactly the failure mode the provenance gate now prevents.
  This prices the correctness fix: it is a real regression that ships to
  every eager user, on the table alongside its correctness benefit.

## Compiled path: two gates, two generations

- Clean `ff4b05a` venv, compile genuinely ON (`env -u
  MLX_DISABLE_COMPILE`), 4-bit snapshot
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, France prompt: **aborts
  loudly at run 1** with
  `RuntimeError: [omarchy] Cos argument magnitude 808400896.000000
  exceeds the built-in accuracy limit 100000.000000` raised from
  `mx.async_eval` inside `mlx_lm`'s generate step. The earlier
  silent-corruption run is now attributable: it happened on the stale
  generation (which lacked this gate's behaviour on this input), not on
  `ff4b05a`.
- Current main (`d1a6bfd`, provenance-gated wheel `064a58ee…`,
  libmlx.so `a84080c4…`): the differential harness
  (`scripts/differential_compile.py --mode realpath`) refuses before
  executing with `[omarchy] broadcast Sigmoid is not implemented …
  shape=[1,36,…]` at a 36-token prompt AND `shape=[1,30,4864]` at a
  2-token prompt; llvmpipe shows the same class of refusal
  (trace-history-dependent, shapeless-fragment retrace).
- Isolated fragment pin on hardware (`qwen2.swiglu`, f16, shape
  (1,7,4864)): compiled fragment PASSES, eager PASSES — the gap needs the
  model-graph tape context, matching the bisection on llvmpipe.
- Both observations stand WITHOUT a shared-cause claim: stale generation
  silently corrupted; `ff4b05a` aborts at the Cos accuracy gate; current
  main refuses at the broadcast-Sigmoid gate. The compiled path is fenced
  on current main, not proven fixed.
- C++ batteries on hardware at the `ff4b05a` tree (built from source,
  `-DMLX_BUILD_TESTS=ON -DMLX_BUILD_OMARCHY=ON`; venv contamination does
  not apply to these binaries): `omarchy_compiled_tape_tests` 8/8 (343
  assertions), `omarchy_eq_math_tests` 7/7 (116), `omarchy_runtime_tests`
  22/23 — the one failure is the stream-overlap timing assertion
  (`ratio 1.20867 < 1.10`, serialized 0.380567 s vs concurrent 0.314865 s,
  400 iters), fixed on main by `81613a1`, which replaces the upper timing
  bound with byte-for-byte output verification of both copies and keeps
  the anti-lock lower bound.

## README 8-core column: clean re-measure

Wheel from main `e7a6542` (6797038 bytes, `ba0afdb788e1e023c449f286b4d6f7831c1f4ca1d3a238008a03725875f6cf55`),
provenance gate green, `scripts/bench_decode.py`, 36-token prompt,
`--tokens 64`, EOS suppressed, load excluded:

| row | value |
|---|---|
| bf16 prefill | 28.6 tok/s (1.259 s for the 36-token prompt) |
| bf16 decode | 3.29 tok/s over 63 tokens (mean 304.0 ms/token) |
| 4-bit prefill | 21.2 tok/s (1.695 s) |
| 4-bit decode | 2.96 tok/s over 63 tokens (mean 337.8 ms/token) |

These supersede both the morning column (stale wheel, short bursts) and
the intermediate re-measure taken in the contaminated venv (which was
this stale wheel's speed, not current main's). Note current main contains
the `ff4b05a` ordering wait, so this column already carries its cost.

## Submissions per decode token (profiling wheel, clean venv)

Built the `--diagnostics` wheel from main (6803461 bytes,
`e073c10b7044f0cf13487e9c6862baeb376bd2783f146568f5ba7bd72e246653`),
provenance-gated, prompt "Count from 1 to 100." (the profiler path has no
EOS suppression, so a long greedy generation is required), 96-token cap,
`MLX_DISABLE_COMPILE=1`:

| metric | bf16 (96 tokens) | 4-bit (54 tokens) |
|---|---|---|
| dispatches per decode token | **95.0** | **94.2** |
| submissions per decode token | **1740.7** | **1800.0** |
| GPU busy per decode token | 1.55 ms | ~1.4 ms |
| join wait per decode token | 240 us | 232 us |

Whole-stream context (bf16): 184086 dispatches, 172456 submissions, GPU
busy 8.57% of wall; inter-submission host round-trip gaps total 37.4 s of
the 42.0 s span (p50 146.9 us); host `submit()` p50 66.5 us over 172456
calls.

**Verdict: the per-submission-overhead theory is SUPPORTED.** A decode
token is ~95 dispatches wrapped in ~1800 submissions, does ~1.5 ms of GPU
work, and takes ~500 ms of wall: ~99.7% of decode wall is host-side, and
inter-submission host round trips alone account for ~89% of the span.
Committing less often in eager mode is the validated lever and ships
without the compiled path.

The published v0.3.3-diag.1 asset was not used for this measurement: at
the time the shared freeze was still contaminated, so its payload looked
broken in my venv. Direct wheel inspection shows the asset itself is
correctly built (3 profiler literals in libmlx.so).

## Compiled-path gates on hardware (7cf5e9f wheel, `1176dad7…`, gate green)

- Sync-val (Khronos layer, settings-file mechanism): **no hazard named.**
  The compiled run aborts at the Cos accuracy gate before any GPU-side
  hazard appears. Baseline and sync-val abort identically, with
  nondeterministic Cos magnitudes across runs (808400896, 899036608,
  953366592, 959657664), and generated text degrades ("Parisse",
  "udes!") before the abort.
- GPU-assisted validation: **the same compiled run produced the correct
  "Paris"**, clean exit. GPU-AV's serialization removes the wrong-value
  behaviour - behavioural confirmation that the defect is an
  asynchronous execution race, not a resource hazard.
- Fail-closed gate (`7cf5e9f`): eager unaffected (clean "Paris");
  override permits compiled execution (tiny broadcast-sigmoid tape runs
  with correct values). **The default refusal does not fire on the M1:**
  `compiled_tapes_refused()` returns false on the real Honeykrisp device,
  so `CHECK_EQ(refused_default, !dev.non_apple_dev())` fails
  (test_compiled_tape.cpp:628); without the override the battery is 2/10
  (compiled tapes still execute and return wrong values) and with the
  override it is 9/10 for the same policy check. Reported to
  CompiledFailClosed - the gate is inert on the exact device it targets.

## Differential harness on hardware (d1a6bfd wheel, `064a58ee…`, gate green)

`probe_tape_eager.py` rc=0: all variants bitwise-clean over 32 iterations
on the real device. `differential_compile.py --mode realpath` refuses
before executing at both prompt lengths with `[omarchy] broadcast Sigmoid
is not implemented for the Omarchy Vulkan backend` (`shape=[1,36,…]` and
`shape=[1,30,4864]`). The isolated `qwen2.swiglu` fragment call passes
compiled and eager. On current main the compiled path is fenced by the
loud refusal; the silent corruption is only observed on the stale
generation (see the bakeoff section above).

## Provisioning rule going forward

Never provision a measurement venv from a freeze that contains `@`
direct-URL pins for the package under test; exclude by name prefix, and
gate every run on installed-`libmlx.so` == wheel-member hashes. A
harness-level gate in `scripts/bench_decode.py` is expected from
ReleaseAssetGate; adopt it when it lands.

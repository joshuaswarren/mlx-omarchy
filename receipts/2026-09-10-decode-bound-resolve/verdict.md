# Decode-bound resolution: direct discrimination of GPU-bound vs host-paced, plus the first compiled-vs-eager Q4 measurement

- schema: mlx-omarchy/decode-bound-resolve/1
- date_utc: 2026-09-11
- host: jwm1-linux (Apple M1 T8103, 8 GPU cores, Honeykrisp, linux-asahi 7.1.6), coordinated GPU lock `/tmp/m1-gpu.lock`, quiet-gated, per-leg loadavg recorded (max 1.01 in any counted leg)
- baseline: release wheel `mlx_omarchy-0.32.2.dev202609101916+b6d662a` (libmlx sha256 68e477ed... family, provenance `verified=match` on every leg via `bench_decode --wheel`), ablation wheel `0f79507+debug dev202609110036` (the DecodeAttribution pair), host-trace wheel `12012beb dev202609110320` (HostPathOverhead). mlx-lm pinned 0.31.3.
- receipt-only: nothing merged; no product source changed. Instruments live in this directory and in `~/src/mlx-DecodeBoundResolve/` on jwm1.
- canonical digests asserted: short `7fd25a869ff21678`, long `4cc08910089477fd`, 1K `7da83f06ec9f001d` — 100/100 measured legs that were required to hold them did (see tables; ablation legs flip by design).

## The contradiction being resolved

- `receipts/2026-09-10-decode-attribution-finish` reads its ablation table as GPU-bound: ablating the quantized GEMV shaders removes 5.68 ms of an 8.87 ms token.
- `receipts/2026-09-10-hostpath` reads its host-phase trace as host-paced: "6.4 ms Python + 2.6 ms C++ against 8.95 ms wall; the host thread is busy essentially the whole token wall; decode on this stack is HOST-paced, not GPU-paced."

Both are raw measurements of the same 8.9 ms token. They cannot both be limiter claims.

## Part 1 — the discriminating experiment

Design (no inference from region occupancy; every knob moves exactly one resource):

1. **Hold GPU fixed, scale host only.** `ATTR_SPIN_US`: a pure-Python busy spin per token in the consumer loop (1 ms / 4 ms). No mlx call, no allocation, digest untouched.
2. **Hold host fixed, scale GPU only.** `ATTR_GUP_K`: a K-deep data-dependent chain of large f16 matmuls (`n=2048`, unit calibrated in-leg at 7.9-8.1 ms; also one 8192×8192 macro point at 254 ms), one `mx.eval` per token outside the token graph. Same Python graph, digest untouched.
3. **Hold host fixed, scale GPU down.** The ablation wheel (`MLX_OMARCHY_ABLATE=gemv`, `=all`), the DecodeAttribution instrument, re-run under the new instruments.
4. **CPU-vs-wall at token granularity.** `time.thread_time_ns` and `time.process_time_ns` deltas per token, per-token wall gaps, all per-token pairs stored (`tok_cpu_vs_wall_ms`). A thread that EXECUTES accumulates CPU time; a thread blocked at the token sync does not. This splits "host working" from "host waiting" inside the very region the hostpath receipt bucketed as Python.
5. **Submitted-vs-completed.** In mlx-lm 0.31.3 the decode loop is pipelined (`generate_step`: builds token n+1 and `mx.async_eval`s it BEFORE `y.item()` syncs token n — `mlx_lm/generate.py:450-464` in the pinned venv). The submit boundary is the async_eval; the completion boundary is the `y.item()` return at the yield. The per-token wall/CPU pairs measure exactly that interval; the monkeypatched `mx.eval` saw **0 eval calls per token** in steady state (only n=0), confirming submissions ride `async_eval` and completions ride `item()`.

All arms ran eager (`MLX_DISABLE_COMPILE=1`), canonical prompts from the committed `bench_matrix.json`, warmup 4, temp 0, seed 0, EOS suppressed, pinned 32 tokens, digest-gated per leg.

### Results — short-decode-32 (medians of 3 reps; full data in `attr/legs.ndjson`, `attr2/legs.ndjson`)

| arm | tok/s | wall ms/tok | thread CPU ms/tok | Δwall vs base |
|---|---|---|---|---|
| base (release wheel) | 113.1 | 8.84 | 5.7-6.3 | — |
| spin 1 ms | 112.8 | 8.86 | +0.1-0.3 CPU | **+0.02** |
| spin 4 ms | 111.2 | 8.99 | +0.8-0.9 CPU | **+0.15** |
| gemv ablated | 311.7 | 3.21 | 2.4-2.5 | **−5.63** |
| all ablated | 422.7 | 2.37 | 2.26 | **−6.47** |
| gup2 (~16 ms GPU chain) | 27.0 | 37.04 | 15.4 | **+28.2** |
| gup6 (~48 ms GPU chain) | 14.6 | 68.54 | 17.9 | **+59.7** |
| gup macro (254 ms 8192 matmul) | 3.7 | 273.84 | 20.4 | **+265.0** |

(base/spin/gup rows from `attr2`, ablation rows from `attr` — same session, same driver, same wheel pair. The old-worker macro arm used one 8192x8192 matmul per token instead of a K-chain.)

Slopes:

- **Host-scale slope: +0.005-0.04 ms wall per ms of added host work.** Four milliseconds of pure host work per token vanishes into the token-boundary sync.
- **GPU-scale slope: 1.04-1.21 ms wall per ms of added GPU work** across +23 to +254 ms (superlinear at the smallest scale from L2/register interference between the dummy matmul chain and the model kernels). Every added GPU ms lands on the wall.
- **Ablation (GPU down):** removing the GEMV payload removes 5.6 ms of wall, the whole payload 6.5. The skeleton (all-ablated) is 2.37 ms wall at 2.26 ms thread CPU — 95% CPU-bound: **that skeleton IS the true host cost of a decode token.**

### Verdict

**The token is GPU-paced.** The decode wall is set by the GPU payload exposed at the per-token sync, not by host throughput:

- host-add slope ≈ 0.03; GPU-add slope ≈ 1.0-1.2; GPU-remove −5.6 ms for the GEMV class alone.
- true host cost = skeleton = **2.3-2.4 ms/token** (2.26 ms of it CPU), not 8.95 ms.
- exposed GPU work = wall − skeleton ≈ **6.5 ms/token** at short-32.

**Which receipt dies and why.** `receipts/2026-09-10-hostpath`'s headline — "host-paced, not GPU-paced; the dominant host term is Python eager graph construction (6.4 of 8.95 ms); remove the Python term first" — **does not survive**. Its instrument bucketed `wall − timed mx.eval calls` as "Python", but in the pipelined mlx-lm loop the per-token completion sync (`y.item()`) sits in exactly that bucket, so ~3-4 ms of GPU-wait (and its busy-poll CPU) was booked as Python graph construction. Direct refutation: host-add slope ≈ 0.03; true host cost 2.3-2.4 ms (ablation skeleton, same graph); GPU-add slope ≈ 1.0-1.2; and Part 2's compiled mode (the Python term measurably shrunk) moving decode by −2.0% to +0.6%. A correction block is appended to that receipt's `verdict.json`.

**What stands.** The hostpath receipt's C++ counters (record/descriptor ≤ 0.76 ms/token, 2.2 submissions/token), its replay prototype (bit-identical, 2.6x slower), and the DecodeAttribution receipt's GPU-bound reading and marginals table, which this work re-measures and confirms (gemv 313.4 -> 311.7 tok/s; ctx-gemv 151.4 -> 151.5; all-ablated 395.9 -> 422.7, session variance on the same side of a 4.7x gap). One refinement to DecodeAttribution's text: its "skeleton minus cadence = host pacing + tail kernels" residual is now measured directly — the skeleton is ~95% real host CPU at 2.26 ms/token.

**Implication (measurement-backed):** host-side work is worth at most ~2.4 ms of the 8.9 ms token; the 6.5 ms GPU payload (quantized GEMV above all) is the gap to native. Whole-graph capture/replay or C++ decode loops address the small term only.

## Part 2 — compiled vs eager Q4 (first measurement on this backend)

Protocol: canonical engine `scripts/bench_decode.py` from the committed b6d662a tree, paired alternating legs (eager = `MLX_DISABLE_COMPILE=1`, compiled = unset), 3 reps per workload per driver, prompt token counts from the manifest, provenance-gated, digest-checked per leg. Drivers: installed fork `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8` and stock `mesa 26.1.7-1` (swapped via pacman inside the window, driverInfo verified both ways, fork restored after).

### Decode tok/s (median of 3, 30/32, 262/128, 1053/32)

| leg | fork eager | fork compiled | Δ | stock eager | stock compiled | Δ |
|---|---|---|---|---|---|---|
| short-decode-32 | 111.12 | 108.89 | −2.0% | 103.05 | 101.61 | −1.4% |
| long-decode-128 | 109.01 | 109.69 | +0.6% | 81.96 | 81.70 | −0.3% |
| longctx-1024 | 97.05 | 97.17 | +0.1% | 52.06 | 52.00 | −0.1% |

### Prefill tok/s (median of 3)

| leg | fork eager | fork compiled | Δ | stock eager | stock compiled | Δ |
|---|---|---|---|---|---|---|
| short | 333.3 | 337.1 | +1.1% | 169.5 | 172.4 | +1.7% |
| long | 952.7 | 985.0 | +3.4% | 309.3 | 312.3 | +1.0% |
| longctx-1024 | 1110.8 | 1128.6 | +1.6% | 361.6 | 364.2 | +0.7% |

### Digests

**36/36 legs reproduce the canonical generated-id digests in BOTH modes on BOTH drivers** (eager and compiled, 3 reps × 3 workloads × 2 drivers). Compiled tapes change no generated id. This closes the compile-identity question the differential-compile defect (ff4b05a) had left open for Q4 decode at b6d662a.

### Compiled mode is genuinely engaged

Host-trace counters on the 12012beb wheel, one leg each (same digest `7fd25a869ff21678` both modes):

- eval nodes: 915.2 → 864.3 per token (−5.6%); allocations: 646.2 → 595.2 per token (−7.9%) — the graph is measurably reshaped when compilation is enabled.
- GPU dispatches: 272.9 per token in BOTH modes; submissions 69 per window (2.16/token) in both; no replay-counter hits in either.
- tok/s 111.94 (eager) vs 112.32 (compiled) on that wheel.

So compiled mode removes host nodes/allocations but **neither the dispatch count nor the GPU payload** — and because Part 1 shows the token is GPU-paced, the removed host work was already hidden. Measured consequence: compiled mode is speed-neutral (−2.0% to +0.6% decode; +0.7% to +3.4% prefill).

### Fraction of the committed base-M1 native baseline

Committed denominator: `receipts/2026-09-10-native-macos-metal-baseline/committed-base-m1/native-baseline-baseM1.json` — decode 150.57 / 146.77 / 140.38 tok/s and prefill 294.1 / 1213.0 / 1840.9 tok/s (short/long/1K).

| leg | fork compiled decode | fraction | fork compiled prefill | fraction | stock compiled decode | fraction |
|---|---|---|---|---|---|---|
| short | 108.89 | **72.3%** | 337.1 | **114.6%** | 101.61 | 67.5% |
| long | 109.69 | **74.8%** | 985.0 | **81.2%** | 81.70 | 55.7% |
| 1K ctx | 97.17 | **69.2%** | 1128.6 | **61.3%** | 52.00 | 37.0% |

(Eager fork fractions are 73.8/74.3/69.1% decode — compiled mode moves none of them materially.)

### Denominator caveat and the exact follow-up it implies

**The native denominator was measured eager.** The committed macOS protocol ran native MLX 0.32.2 through mlx-lm with no compile path enabled (mlx-lm calls no `mx.compile` in generate; the committed receipt records no compile configuration). On the Linux side, compiled mode does not win, so no parity claim changes hands today. **But if compiled mode ever wins on this backend, an apples-to-apples parity claim requires a compiled native baseline too**, namely:

1. on the same base-M1 Mac, same macOS/MLX 0.32.2 build and pinned model snapshot `a5339a41`, run the identical committed bench protocol (same prompts, warmup 4, pinned lengths, EOS suppressed, 5 reps, median);
2. with the decode step compiled the way the Linux tape path compiles it — wrap the per-token `TransformerBlock` call in `mx.compile` (or enable the native equivalent of the tape path), so the comparison is compiled-vs-compiled;
3. prefill compiled separately (compile the prompt-processing call), since tape effects differ by phase;
4. digest-checked against the native eager digests (`7fd25a869ff21678` / `254d73fd93164b98` / `7da83f06ec9f001d`) so identity is proven on the native side as well;
5. commit it as a sibling of `committed-base-m1/native-baseline-baseM1.json` and only then re-derive the fractions above.

## Policy compliance

- never merged: all instruments live on jwm1 (`~/src/mlx-DecodeBoundResolve/`) and in this receipt directory; no product source, shader, or protocol file changed.
- digest gates: canonical pins asserted on every non-ablation leg (100/100 hold); ablation legs flip by design and are recorded; part-2 gates both modes.
- windows: four announced top-level flocks (R1, R2, R1b, R1c), one per window, quiet-gated except R1b (post-R2 immediate start; per-leg loadavg max 1.01, no contamination); two pacman driver swaps inside R2 with driverInfo verification and fork restoration confirmed before release.
- raw data: `attr/`, `attr2/`, `compile-fork/`, `compile-stock/`, `tape/` (copies of the jwm1 results directories), scripts: `attr_worker.py`, `attr_driver.py`, `compile_ab.py`, `analyze_attr.py`, wrappers `R1.sh`, `R2.sh`, `R1c.sh`.

# Q4 decode gap attribution on jwm1: it is dispatch count, not the GPU

Date: 2026-09-14
Task: attribute the 1053-token Q4 **decode** deficit on `jwm1-linux`
(67.70% of native base-M1 Metal) from existing captures, decide whether
decode is dispatch-bound, launch-bound, bandwidth-bound or kernel-bound, and
name the single highest-value change. Attribution only: no kernel was
changed, nothing was run on jwm1, no lock was taken.

## Verdict

**Decode is dispatch-bound, and the GPU is almost idle while it happens.**
Per token the graph issues 249 dispatches and 498 compute-to-compute
pipeline barriers, two per dispatch; 151 of those 249 dispatches (60.6%)
launch 14 workgroups or fewer and 79 launch exactly one. At jwm1's clean
10.522 ms/token the part
is running at **3.9% of its fp32 arithmetic peak and 40.5% of its memory
bandwidth**. Neither arithmetic, nor bandwidth, nor GEMV occupancy, nor
host-side command recording can be the deficit; each is excluded
quantitatively below.

The decisive measurement needs no per-dispatch timing at all. The decode
deficit is the **same absolute milliseconds per token on an 8-core part and
on a 32-core part**:

| | jwm1 (8 cores, 68.25 GB/s) | jw16 (32 cores, 400 GB/s) | ratio |
| --- | ---: | ---: | ---: |
| Linux 1053/32 decode | 10.52242 ms/token | 6.98394 ms/token | 1.507x |
| native Metal, same silicon | 7.12352 ms/token | 3.52373 ms/token | 2.022x |
| **deficit** | **3.39890 ms/token** | **3.46021 ms/token** | **0.9823** |

The GPU between those two columns is 3.30x faster on the same code (measured
prefill GPU-busy ratio, `receipts/2026-09-14-jw16-max-gpu-attribution.md`).
A GPU-resident cost would have shrunk by roughly that factor. It changed by
1.8%. Whatever the deficit is, it does not scale with the GPU — so it is a
per-token serial cost, and with 249 dispatches per token it is 13.65 us per
dispatch on jwm1 (13.90 on jw16).

Splitting by context length separates that cost from the KV stream, because
**the dispatch count per token does not depend on context** (verified from
the grid census: every recorded decode grid derives from `hidden_size`,
`intermediate_size`, head count or vocabulary; none derives from sequence
length, and the KV write has no dispatch of its own):

| term | jwm1 | jw16 | what it is |
| --- | ---: | ---: | --- |
| A, context-independent (deficit at 30-token context) | 2.6795 ms (78.8%) | 2.4087 ms (69.6%) | 249 dispatches x 10.76 us (jwm1) / 9.67 us (jw16) |
| B, KV-length-dependent (growth 30 -> 1053) | 0.7194 ms (21.2%) | 1.0515 ms (30.4%) | the KV stream in `SdpaDecodeNativeF16` |
| total at 1053 | 3.3989 ms | 3.4602 ms | |

Term A is 1.11x apart across a 3.30x GPU. Term B is a 12.57 MB incremental
KV read per token which Linux serves at **10.46 GB/s on jwm1 and 11.53 GB/s
on jw16** — 1.10x apart across a 5.86x bandwidth difference — against
native's 26.08 and 322.94 GB/s. That last figure is also the byte model's
own check: 322.94 GB/s is 80.7% of the M1 Max's 400 GB/s specification,
which is what a well-formed pure stream should reach, so the 12.57 MB
increment is the right quantity rather than a coincidence of subtraction.
Both terms are device-invariant. Both are therefore serialization, not
capability.

**Highest-value single change: extend the existing GEMV store-epilogue
fusion to RoPE and SwiGLU**, removing 72 of the 249 dispatches per token
(28.9%). Priced at term A's own rate: −0.775 ms/token, 95.04 -> 102.59
tok/s, **67.70% -> 73.08% of native**, digest-preserving by construction.
Evidence, bound and alternatives in "Proposal".

## Identity

- Receipt checkout (local `mlx-omarchy`): `b28deb1cbf7899cab71df08d432dd47cb31862ce`
  (`git describe --always --dirty`: `v0.3.2-4-gb28deb1c-dirty`; the working
  tree is dirty from unrelated sibling work and was not used — every number
  here comes from archived captures and from `git show b41e2b74:…`).
- Analyzed source commit: `b41e2b74c330f910b24cab0e7516e306527858f0`, the
  commit both parity legs and both profiles were measured at.
- Captures analyzed, both already in this repository, neither re-measured:
  - `receipts/2026-09-13-jwm1-phase-profile-enabled/profile.jsonl`,
    SHA-256 `1ad4c795e78ec3630c9935e1599c0b91ce465babdf18f4bb8d49d8d8be49ff60`
    — jwm1, `Apple M1 (G13G B1)`, 1312 dispatches, 12 submissions, three
    complete decode steps (submission pairs 7/8, 9/10, 11/12).
  - `receipts/2026-09-14-jw16-max-gpu-attribution/profile-m1053-t32.jsonl.gz`,
    SHA-256 `ab4228cac173eba4329a380992783489fb81a4d6e065d327c9cbdcb1c5930582`
    — jw16, `Apple M1 Max (G13C C0)`, 8782 dispatches, 69 submissions, 33
    complete decode steps.
  - Kernel enum header `receipts/2026-09-14-jw16-max-gpu-attribution/compute-b41e2b74.h`,
    SHA-256 `5b1b4135c9a87fee92958bd9ece698fd1aee5d661e93988bd33b81ce100318f5`.
- Clean-run rates, all quoted, none re-measured: jwm1 from
  `receipts/2026-09-14-jwm1-gpu-parity-rerun.md` (107.285 and 95.0352 tok/s);
  jw16 Honeykrisp-fork from
  `receipts/2026-09-14-jw16-honeykrisp-mesa-parity/README.md` (169.6769 and
  143.1857 tok/s). Native divisors are the pinned
  `native-2026-09-06-summary.json` medians selected by the 2026-09-13 parity
  receipt: base-M1 150.57 / 140.38 tok/s, M1 Max 286.96 / 283.79 tok/s.
- Source read at `b41e2b74`: `overlay/mlx/backend/omarchy/encoder.cpp`
  (barrier and dispatch path, lines 32-103 and 440-565),
  `overlay/mlx/backend/omarchy/gpu_profiler.h` (record contract, lines 1-52),
  `docs/install-omarchy.md` (fused-GEMV and gated-barrier semantics, lines
  44-62). The superseded plan is `ef188d58:docs/plans/2026-09-06-decode-gap-plan.md`.
- Model, identical in both captures and both parity legs:
  `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, affine 4-bit group 64, tied
  embeddings, `hidden_size 896`, `num_hidden_layers 24`,
  `intermediate_size 4864`, 14 heads / 2 KV heads (kv width 128), vocabulary
  151936. Both captures and both parity legs ran `MLX_DISABLE_COMPILE=1`, so
  the graph the profile records is the graph the parity rates measured.
- Agent model: `anthropic/claude-opus-5`; routing fallback: false.
- Vendor specification figures, labelled as such and not measured here:
  68.25 / 400 GB/s memory bandwidth; fp32 FMA peak 2.654 / 10.617 TFLOPS by
  the convention of `receipts/2026-09-14-jw16-max-gpu-attribution.md`
  (cores x 128 lanes x 2 x 1.296 GHz).

## Protocol

Host-only, on the workstation, from files already in the repository. **No
command ran on jwm1 or jw16, `/tmp/m1-gpu.lock` was never taken, no wheel
was built, no kernel or shader was edited, no driver was touched, no ANE
command was issued.** jwm1 was contended for the whole task
(`ExportGeometryFix`, `ParakeetE2E`, then `IslandsExecJwm1` and
`MelFrontendPerf`); nothing here needed it.

Re-derivable:

```sh
cd receipts/2026-09-14-decode-gap-attribution
./decode_census.py ../2026-09-13-jwm1-phase-profile-enabled/profile.jsonl \
  --compute-h ../2026-09-14-jw16-max-gpu-attribution/compute-b41e2b74.h \
  --decode-subs 7,8,9,10,11,12            # -> census-jwm1.txt
./decode_census.py ../2026-09-14-jw16-max-gpu-attribution/profile-m1053-t32.jsonl.gz \
  --compute-h ../2026-09-14-jw16-max-gpu-attribution/compute-b41e2b74.h \
  --decode-subs 66,67,68,69               # -> census-jw16.txt
./decode_census.py --deficits-only       # -> deficits.txt
```

## Per-token dispatch census (jwm1, last decode step, submissions 11+12)

Counts and grids are recorded, not timed, so they are exact. The bracketed
column is shown only to locate the work, never as a price (see
"Exclusions"):

| kernel | n | per layer | grids (workgroups x count) | bracketed ms | share |
| --- | ---: | ---: | --- | ---: | ---: |
| `QmmVecQ4MultiSubgroupF16` | 96 | 4.000 | 112x48, 144x24, 1216x24 | 9.990 | 57.5% |
| `QmmVecQ4WordSubgroupF16` | 1 | — | 18992x1 | 2.127 | 12.2% |
| `SdpaDecodeNativeF16` | 24 | 1.000 | 14x24 | 1.517 | 8.7% |
| `FastRmsNormF16` | 49 | 2.042 | **1x49** | 1.314 | 7.6% |
| `FastRopeF16` | 48 | 2.000 | **1x24, 2x24** | 1.154 | 6.6% |
| `SwigluF16` | 24 | 1.000 | **5x24** | 0.459 | 2.6% |
| `LogSumExpF16`, `ElementwiseF16`, `ArgReduceF16`, `TakeF16` x2, `TakeU32`, `DequantF16` | 7 | — | 1x6, 594x1 | 0.536 | 3.1% |
| **total** | **249** | **10 per layer + 9 outside** | | 17.379 | |

The four GEMVs per layer are the already-fused set: 144 workgroups for the
fused q/k/v (1152 outputs), 1216 for the fused gate/up (9728), and two of
112 for `o` and `down` (896 each). `MLX_OMARCHY_FUSED_GEMV` and
`MLX_OMARCHY_FUSED_CHAIN` have already taken a Qwen2 decode layer from 22
dispatches to the documented 14 (`docs/install-omarchy.md`); the measured
layer is 10 — four GEMVs, two RMS norms, two RoPEs, one SDPA, one SwiGLU —
better than the documented figure.

The jw16 capture's last decode step is **structurally identical**: 249
dispatches, the same 151 at <=14 workgroups, the same 79 at exactly one, the
same 235.59 MB distinct binding footprint over the same 829
`(buffer, offset, range)` triples. The two hosts run the same graph with the
same grids, which is what makes the cross-host deficit comparison a clean
natural experiment.

Every one of those 249 dispatches carries **two** pipeline barriers in the
default configuration, verified by source read: a pre-dispatch barrier
(`HOST_WRITE|TRANSFER_WRITE|SHADER_WRITE -> SHADER_READ|SHADER_WRITE`, stages
`HOST|TRANSFER|COMPUTE_SHADER -> COMPUTE_SHADER`) at `encoder.cpp:486` and a
post-dispatch barrier (`SHADER_WRITE -> SHADER_READ|TRANSFER_READ|HOST_READ`,
stages `COMPUTE_SHADER -> COMPUTE_SHADER|TRANSFER|HOST`) at `:546`, both
recorded unconditionally because `MLX_OMARCHY_GATED_BARRIERS` defaults off.
(The heavier `ALL_COMMANDS` / `MEMORY_READ|WRITE` form in
`record_dependency_barrier()` at `:75` is used only by the gated path and by
the `MLX_OMARCHY_TAPE_FULL_BARRIERS` diagnostic, so it is not what decode
pays today.) The capture's own barrier accounting agrees:
`emitted=1312 skipped=0` for dispatch barriers and `emitted=2624` including
transfer and fill nodes, i.e. 498 barriers per decode token.

## Exclusions

**Arithmetic is excluded, decisively, on both parts.** A decode token is
1.078 GFLOP (2 x 494.0M weight MACs plus the 1053-position attention). At
jwm1's 10.522 ms that is 102.5 GFLOP/s = **3.86% of the 2.654 TFLOPS fp32
peak**; on jw16, 154.4 GFLOP/s = 1.45% of 10.617 TFLOPS. Native Metal on the
same silicon reaches 5.70% and 2.88%. No plausible kernel-quality argument
lives in a 25x headroom.

**Memory bandwidth is excluded.** Required traffic per token is 290.88 MB
(201.28 MB of layer weights at 0.5625 B/element, 76.58 MB of tied lm_head,
12.94 MB of KV at context 1053, 0.09 MB of norms) — consistent with the
235.59 MB distinct footprint the profile records, which is a lower bound
because the recorded binding list does not resolve every weight of a fused
multi-weight GEMV. That is 27.64 GB/s on jwm1 Linux = **40.5% of the 68.25
GB/s specification**, against native's 40.83 GB/s (59.8%) — so the target
rate is demonstrated achievable on the same memory system. On jw16 it is
41.65 GB/s = **10.4% of 400 GB/s**, with native itself only at 20.6%.

**GEMV occupancy is excluded as term A's content.** The decode grids are
1216, 144, 112 and 112 workgroups — 152.0, 18.0, 14.0 and 14.0 per core on
jwm1's 8 cores, and 38.0, 4.5, 3.5 and 3.5 on jw16's 32. By the occupancy
curve measured in `receipts/2026-09-14-jw16-max-gpu-attribution.md` those
jw16 numbers sit at or below the 4.1-wg/core cliff while jwm1's sit above
it — yet the two hosts' term A differs by only 1.11x. A cost that occupancy
explained could not be that flat across that difference. (Occupancy remains
the correct finding for jw16 *prefill*; it is not this one.)

**Host-side command recording is excluded.** The per-dispatch record cost
`h` sums to 0.250 ms per token on jwm1 (p50 0.83 us) and 0.176 ms on jw16
(p50 0.71 us) — at most 9% of term A, and that is the instrumented figure.

**A per-submission model is excluded.** Decode is 2 submissions per token on
both hosts. Fitting the recorded `vkQueueSubmit` durations against dispatch
count gives a per-submission intercept of −285 us (jwm1, median pair over
all 12 submissions) and −335 us (jw16, over 69), i.e. indistinguishable from
zero or negative; 2.4-2.7 ms per token cannot be produced by two
submissions each costing nothing fixed. The cost tracks dispatches.

## What is not resolved, and why it does not change the answer

**Term A's host/GPU split is open.** The recorded `vkQueueSubmit` duration
is per-dispatch — 10.01 us/dispatch on jwm1's warmest decode step (2142.1 us
for 214 dispatches against 306.6 us for 35, slope 10.25 us, intercept −52 us)
— and that figure is the same order as term A's 10.76 us/dispatch. It is
tempting and it must be resisted: the same measurement on jw16 is **25.83
us/dispatch, stable across all 33 decode steps** with no warming trend. The
faster host pays 2.5x more inside `vkQueueSubmit` than the slower one, which
means that duration contains something other than command translation —
most plausibly absorbed back-pressure from the submission ring or the
scheduler throttle. It cannot be converted into a price, and this receipt
does not.

**Bracketed per-dispatch GPU durations remain unusable**, exactly as
`receipts/2026-09-14-jw16-max-gpu-attribution.md` warned. On jwm1 the last
decode step's bracketed busy is 17.379 ms and its span 23.183 ms against a
10.522 ms clean wall (1.65x and 2.20x); on jw16, 10.718 and 16.690 ms
against 6.984 ms (1.53x and 2.39x). The profiler adds two timestamps and an
isolating execution barrier per dispatch and writes `t0`/`t1` at
`BOTTOM_OF_PIPE` (`gpu_profiler.h:22` and `:41-42`, emitted at `:177` and
`:237`), which drains each dispatch and deletes the overlap a clean run
keeps. Every intra-dispatch gap figure
inherits the same defect (jwm1 p50 14.1 us, jw16 13.2 us over 247 pairs) and
is reported here only as corroboration of shape, not as a quantity.

**Why the answer survives anyway.** Term A is device-invariant and scales
with dispatch count, and that is true whether its content is host time
inside `vkQueueSubmit`, GPU drain from the 498 compute-to-compute barriers,
or per-launch
fixed cost in the driver's command stream. All three divide by the same
number. The recommended change reduces that number, so it pays off under
every candidate mechanism — which is precisely why it is the right pick
while the mechanism is unresolved. The cheap experiment that *would* resolve
it is named below and costs no code.

**Also not done here, deliberately:** no 30-token-context profile exists on
either host, so the context-independence of the dispatch count is
established structurally (from the recorded grids) rather than by a second
capture; no clean-run decode A/B of any candidate change was measured,
because that needs jwm1 and jwm1 was contended.

## Supersedes

`ef188d58:docs/plans/2026-09-06-decode-gap-plan.md` priced this gap when
jwm1 decode ran at 23.5 tok/s at 1053 context, four times slower than
today's 95.0. Its headline conclusion — "kernel work + full barriers, not
one-dispatch-per-submit", with 14-20 ms of 32 ms attributed to kernel
arithmetic — no longer holds: kernel arithmetic now runs at 3.86% of peak,
and the deficit that remains does not move when the GPU gets 3.30x wider.
Its TOP-1 item, the dependency-gated-barrier A/B, has still never been run:
no receipt in this repository sets `MLX_OMARCHY_GATED_BARRIERS`, and
`docs/install-omarchy.md` still records the mode as "defaults off pending
the M1 A/B".

## Proposal: fold RoPE and SwiGLU into the GEMV store

One change, to one existing mechanism, attacking term A's multiplier.

**What.** The fused GEMV already folds an epilogue into its store: "the bias
or residual `Add` that is a projection's only consumer is folded into that
GEMV's store" (`docs/install-omarchy.md` at `b41e2b74`). Extend that same
epilogue to the two per-layer elementwise consumers that currently get their
own dispatches:

- `FastRopeF16` after the fused q/k/v GEMV — 48 dispatches per token, 2 per
  layer, launching **1 and 2 workgroups**;
- `SwigluF16` after the fused gate/up GEMV — 24 dispatches per token, 1 per
  layer, launching **5 workgroups**.

72 of 249 dispatches per token, 28.9%, and 144 of the 498 barriers.

**Why these two and not the others.** Both are pure per-element maps on
values the GEMV has just produced in registers. Folding them leaves each
output's f32 accumulation chain untouched, so the stored value is
bit-identical by construction and the pinned generated-ID digests are
preserved — the same argument the existing `Add` fold already relies on.
`FastRmsNormF16` (49 dispatches, every one launching a single workgroup) is
the next 49 dispatches and the cheapest-looking remainder, but it is a
*prologue*, not an epilogue: a reduction over the GEMV's input. Folding it
makes every one of 112-1216 workgroups recompute the 896-element reduction,
and bit-identity then
requires reproducing `FastRmsNormF16`'s exact reduction tree in a different
workgroup geometry. That is step two, gated on a digest A/B, not step one.

**Expected gain, and its bound.** Priced at term A's own measured rate
(10.76 us/dispatch on jwm1, from the 30-token-context deficit divided by
249):

| change | dispatches/token | jwm1 ms/token | tok/s | % of native |
| --- | ---: | ---: | ---: | ---: |
| today | 249 | 10.5224 | 95.04 | 67.70% |
| + RoPE and SwiGLU folded | 177 | 9.7476 | 102.59 | **73.08%** |
| + RMS norm folded (step two) | 128 | 9.2203 | 108.46 | 77.26% |
| every <=14-workgroup dispatch gone (ceiling) | 98 | 8.8975 | 112.39 | 80.06% |

The same change on jw16 prices at −0.697 ms/token, 143.19 -> 159.05 tok/s,
50.45% -> 56.04% of native. These are *conservative* in one direction and
optimistic in another, and the receipt should not hide either: conservative
because removing a dispatch removes Linux's whole per-dispatch cost, not
only the part that exceeds native, so the wall should fall by at least this
much; optimistic because term A's per-dispatch rate is an average over a mix
of 1-workgroup and 1216-workgroup dispatches, and if the fixed cost is
sublinear in command count the small ones may be cheaper than average. The
80.06% row is a ceiling, not a forecast.

**Alternatives, and why they rank lower.**

- *Dependency-gated barriers* (`MLX_OMARCHY_GATED_BARRIERS=1`) attack the
  barrier half of term A only, and only partly: gated mode drops the
  post-dispatch barrier unconditionally (249 of 498 per token) and skips
  pre-barriers on non-overlap, which the capture's binding-disjointness proxy
  puts at at most 38.7% of consecutive decode pairs. It needs no build and no
  kernel, so **run it first as the discriminating experiment**: one locked
  jwm1 A/B on the pinned 1053/32 leg with digest verification resolves term
  A's host/GPU split and, if it moves decode, is itself a zero-code win. It
  is the cheaper experiment; it is not the larger change.
- *Splitting `SdpaDecodeNativeF16`'s KV length across workgroups* attacks
  term B, the smaller term (0.72 ms on jwm1), and it is the right eventual
  fix: 14 workgroups — one per query head, independent of KV length — stream
  the KV at 10.46 GB/s on jwm1 and 11.53 on jw16 while native reaches 26.08
  and 322.94. But a length-split needs flash-style partial-softmax
  combination, which reorders the softmax accumulation and therefore breaks
  the pinned digests unless the combination order is made to reproduce the
  current result exactly. Higher risk, smaller term, second.
- *Fewer submissions* is already exhausted: 2 per decode token on both
  hosts, with a fitted per-submission constant indistinguishable from zero.
- *GEMV kernel quality or occupancy work* attacks a term running at 3.86% of
  arithmetic peak and 40.5% of bandwidth, on a machine where the deficit does
  not shrink when the GPU gets 3.30x wider.

**Correctness gate.** Per-leg digest verification on both hosts, against the
pinned `7fd25a869ff21678` (short) and `7da83f06ec9f001d` (1053) on jwm1 and
jw16's own pinned pair, plus the existing runtime test suite for the fused
GEMV path. A fold that changes any digest is a failed fold, not a new
baseline.

## Artifacts

`receipts/2026-09-14-decode-gap-attribution/`:

| file | what |
| --- | --- |
| `decode_census.py` | the census, decomposition and pricing, as run |
| `census-jwm1.txt` | its output on the jwm1 capture |
| `census-jw16.txt` | its output on the jw16 capture |
| `deficits.txt` | `--deficits-only`: every rate-derived number in this receipt, including the proposal table |
| `SHA256SUMS` | hashes of the four files above |

The script reads the two archived captures and the archived enum header in
place; it writes nothing and touches no device. `./decode_census.py
--deficits-only` needs no capture at all and reproduces the deficit
decomposition, the traffic and arithmetic exclusions, the KV-stream figures
and the proposal's pricing table from the quoted clean-run rates alone.

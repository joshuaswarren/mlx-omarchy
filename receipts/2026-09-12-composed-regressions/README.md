# Composed regressions — 2026-09-12

Three items from the composed-tree qualification at ab08be8b: the Q4
short-prefill regression, the affine storage-offset wrong value, and the
copy-offset submission-count failure. Fix branch:
`composed-regressions-20260912`, fix commit `69a01db3`.

## Item 1 — Q4 short-prefill regression: REAL, cause named, fixed

**Breaker: `de780aa2` "restore the scalar-fill host drain the batching
commit removed".**

`de780aa2` put an unconditional `encoder.synchronize()` (commit + host
join) at the top of `copy_gpu_inplace`'s `CopyType::Scalar` branch. Every
GPU scalar fill now flushes the open batch (splitting the submission) and
blocks the host until the device drains. The Q4 short-prefill path
performs scalar fills; the per-fill join destroys host/device pipelining:
+51 ms per prefill call. Decode is unaffected (few fills per token),
matching the wrong-value-sweep receipts' decode-only timing tables.

### Bisect (one session, fork driver, corrected instrument, warmup
discarded, digests held `7fd25a869ff21678` on every row)

| commit | prefill_s (med of 2-3) | tok/s | state |
|---|---|---|---|
| 63c9a8d8 wheel (Bf16PrefillBase artifact, sha16 2a32a2611ce8fee2) | 0.089306 | 335.9 | anchor start |
| bddc061f | 0.090693 | 330.8 | fast |
| 5aab281f | 0.089693 | 334.5 | fast |
| **de780aa2** | **0.139922** | **214.4** | **-36% step (breaking commit)** |
| 193e1aba | 0.137369 | 218.4 | slow |
| 87b4bed4 | 0.138876 | 216.0 | slow |
| e1896643 | 0.139317 | 215.3 | slow |
| 485404cf | 0.141312 | 212.3 | slow |
| ed3c60f4 | 0.140692 | 213.2 | slow |
| 944a49ba | 0.137248 | 218.6 | slow |
| 2b5ba672 | 0.136473 | 219.8 | slow |
| ab08be8b wheel (sha16 9d5745b7a3b4f066) | 0.140323 | 213.8 | anchor end |
| 63c9a8d8 anchor re-run | 0.090932 | 329.9 | drift -1.8% |

Same-session drift on identical wheels was under 2%; the step at
`de780aa2` is 25x that. The 256-key perf gate (`2b5ba672`) is exonerated:
every commit before it in the wave already measures slow. The qmm layout
merge (`5aab281f`) is exonerated for prefill: it measures fast.

### Why 485404cf did not recover batching

485404cf's recycled-gated drain sits inside `fill_pattern`, which is
called only from the Scalar branch AFTER de780aa2's unconditional
synchronize. The narrow gate never executes first; it is dead code on
every real path. Copy-offset isolation runs agree: FAIL at de780aa2 x3
and at 485404cf x3.

### The fix (69a01db3)

- `copy_gpu_inplace` Scalar: drop the unconditional host drain.
- `fill_pattern` recycled-storage fill: order through a timeline wait on
  the device completion semaphore at the newest reserved value
  (`encoder.wait_outstanding_submissions()`). The wait rides the fill's
  own submission: no host stall, no submission split, and cross-stream
  in-flight writes stay ordered exactly as the drain ordered them.
- `fill_pattern` edge bytes (host memset of lead/tail) and the
  dtype-converting scalar scratch (host memcpy into a possibly recycled
  block) keep a real host drain when their target is recycled: a GPU
  semaphore cannot order host writes to mapped memory. Fresh allocations
  skip the stall entirely.

This restores e5072671's contract — back-to-back scalar fills stay
ordered in one submission — without re-introducing the batching commit's
wrong values: the conv/pool garbage came from fills losing to in-flight
writes of a recycled block's previous occupant, and the timeline wait
covers that ordering device-side.

## Item 2 — affine storage-offset wrong value: PRE-EXISTING, filed

`omarchy_primitive_tests` "quantized matmul binds affine streams at
storage offsets", m=1, one output element off at eps 4e-3.

- Fails on M1 hardware at EVERY checkpoint: bddc061f (x3,
  case-name-verified), 5aab281f, 8fd294ed, de780aa2, 485404cf, f425f46f,
  ab08be8b. Confirmed independently in two agents' windows.
- The test case text is byte-identical bddc061f..ab08be8b, and so are the
  qmm shaders (`qmm_vec.comp`, `qmm.comp`) and the affine dispatch block;
  tonight's qmm branch (cddac6bc) changes only rank-1 acceptance and
  guards, which the rank-2 m=1 route never reaches.
- Therefore the wrong value is OLDER than tonight's stack. QmmLayoutParity
  was right that it is pre-existing and wrong that it was "green on M1".
- The suspect introduction is `1f6a7bf8` (2026-09-06, "Bind affine QMM
  scales and biases directly; drop the packing copies"): the vec route's
  aux streams carry storage offsets via push constants (`aux_offset` =
  scale base item, `aux_size` = bias base item) while the buffers are
  bound whole (`binding()` uses offset 0), so the m=1 qmm_vec route
  (QmmVecQ4Word*F16) mis-composes the view offset for the f16 affine
  streams; the general m>1 route reads them correctly.
- Impact is synthetic-only today: real models pass standalone scale/bias
  arrays at offset 0 (every canonical digest holds). Known defect, not
  fixed tonight; route left emitting values, because the failing case is
  a non-zero-offset view that real models do not produce and a refusal
  would break the passing m=7 general route.

## Item 3 — submission-count test: contract decided, code fixed

`omarchy_copy_offset_tests` "back-to-back scalar fills stay ordered in
one submission" (e5072671, 2026-09-10): two scalar fills plus dependent
ops must land in exactly one `vk_submissions` increment with correct
values. Both halves are the contract: ordering (values 22/121) and one
submission (the measured batching cost — always-drain timing cost
1.3-5.7% decode in the sweep, and the tonight bisect shows the join costs
36% of prefill).

Evidence chain (isolated build+run per checkpoint): PASS at bddc061f x5,
5aab281f, 8fd294ed; FAIL from de780aa2 on (x3 at de780aa2, x3 at
485404cf, x3 cold at ab08be8b), signature 7-vs-6 (delta 2: the first
fill's drain splits the submission). The test needed no change — the fix
removes the split while keeping ordering, so the test passes as written.

## Instrument finding (side product, fixed first)

`bench_decode` times prefill with `time.monotonic_ns()` but prints only
`prefill %.3fs`; `bench_matrix` regex-parsed that print
(`PREFILL_RE`), quantizing every published prefill_s to whole
milliseconds. The 12-matrix Q4 short value 0.090/333.333 was exactly
reproducible because the true span is one deterministic forward pass,
not because the clock froze. Quantization bound: +-0.5 ms per leg
(0.556% at the 0.090 s span; proportionally less elsewhere) — too small
to explain the 90-to-140 ms gap, so the regression was real. Fixed in
`162ec5c3`: parse the result_line JSON's 6-decimal `prefill_s`, keep the
print regex as fallback. RULE: the JSON field is the instrument; never
re-derive timings from printed output.

## Verification (window3, this session)

See `window3-results.md` — filled from the verification run on the fix
wheel (built from 69a01db3, farm4).

## Raw data

- jwm1:/home/joshuawarren/src/bisreg-window/ — window1/2/3 logs and
  per-run JSONs (D*, B-*, A2-*, T-*, G-*).
- jwm1:/home/joshuawarren/src/bisreg-farm*.log, bisreg-phase2.log,
  bisreg-venvfix2.log — build and venv-staging receipts.
- Sibling receipts: receipts/2026-09-12-composed-main-qualification/
  (bisect-observations.md, corrections/README.md) @ composed-main-qual.
- Wheels: 63c9a8d8 sha16 2a32a2611ce8fee2 (mlx-Bf16PrefillBase/dist),
  ab08be8b sha16 9d5745b7a3b4f066, fix 69a01db3 (see window3 results).

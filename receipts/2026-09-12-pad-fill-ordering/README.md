# Pad fill ordering — 2026-09-12

Pad and concat filled whole regions with recycled garbage after
`3f7e549a` ("omarchy: order recycled scalar fills through the completion
timeline"). First failing commit named by run, not assumed. Fix:
`pad-fill-ordering` commit `182a02bb` — one gated drain in
`copy_gpu_inplace`'s Scalar branch; the prefill restoration stands.

## Three commits, one path, two shipped wrong values

Read this before editing copy.cpp's Scalar branch:

1. `de780aa2` added an unconditional host drain before every scalar fill
   to fix a LOST FILL WRITE (fills into freshly recycled storage lost to
   the previous occupant's in-flight writes — convolve boundary garbage,
   MaxPool1d wrong values). The drain also made the branch's host reads
   of the fill value safe. Cost: +51 ms per prefill call.
2. `3f7e549a` correctly removed the unconditional drain (it was the
   measured prefill breaker) and replaced it with a timeline wait —
   which orders GPU work only. The removal exposed the second defect:
   the host read of a GPU-produced fill value now raced its producer,
   and pad/concat filled regions with recycled garbage.
3. `182a02bb` (this fix) narrows the drain to what actually needs it:
   GPU-produced scalars drain; host-written scalars skip. Both wrong
   values die with it.

The pattern to fear: both failures surfaced as garbage in an output
region, neither looked like a synchronization bug, and in both cases the
symptom appeared only when a drain that had silently been load-bearing
was removed or added.

## First failing commit: 3f7e549a

Verified on llvmpipe (`MLX_OMARCHY_ALLOW_NON_APPLE=1`, x86_64 dev box),
not inferred from the stretch:

| commit | omarchy_shape_ops_tests | omarchy_complex_ops_tests |
|---|---|---|
| 0043ef2d (parent) | 24/24, 466/466 assertions | 34/34, 1715/1715 |
| **3f7e549a** | **23/24, 5 assertions fail** | **33/34, 6 assertions fail** |
| d389c24f (main) | 23/24, same case | 33/34, same case |

The stretch 0043ef2d..d389c24f contains seven commits; only 3f7e549a
touches backend code (receipt/README/docs commits touch no overlay file;
1873305e touches zero overlay files). Failing cases:
- `Pad fills multidimensional boundaries with exact values`
  (test_shape_ops.cpp:392) — padded region read 0.25f and 60.0f where 0
  expected.
- `complex pad and concatenate transport` (test_complex_ops.cpp) — every
  padded index read the identical pair (-0.402478, -0.0108201) where
  (0, 0) expected; identical values at all wrong indexes is a fill-with-
  constant signature, not partial writes.

## Mechanism

`3f7e549a` removed the unconditional `encoder.synchronize()` at the top
of `copy_gpu_inplace`'s Scalar branch and replaced it with
`wait_outstanding_submissions()` in `fill_pattern`. The wait orders only
GPU-vs-GPU work: it attaches a timeline wait to the fill's submission so
the submission starts after every already-RESERVED (committed)
submission completes.

The Scalar branch, however, reads the fill value's bytes ON THE HOST
before anything is enqueued:

- `scalar_is_zero(in, i_offset)` — byte compare to pick the zero path,
- `scalar_fill_value(in, i_offset)` / the raw byte, halfword, word and
  64-bit memcpys that become the fill constant,
- the dtype-converting scratch memcpy source.

MLX's pad op represents `pad_value` as `astype(array(0), input.dtype())`
(mlx/ops.cpp pad()), so every constant pad's value is GPU-produced: an
int32→float32 cast scheduled just before the Pad primitive evaluates.
When the Scalar branch reads the value, the cast's output bytes may not
be published: the cast sits either in an open batch (buffer stamped
`kPendingCompletion` = UINT64_MAX, never committed — invisible to any
timeline wait by construction, because reserve() happens at commit) or
in a committed-but-incomplete submission (invisible to a HOST read — a
semaphore wait publishes bytes to the device's next submission, never to
a mapped host pointer). The host read returned the recycled block's
stale occupant bytes, `scalar_is_zero` saw non-zero, and the non-zero
float fill path (FillF32 / FillComplex64) wrote the garbage as the fill
value across the whole padded region.

Trace evidence (env-gated instrument, removed before landing): the
scalar's buffer at `comp=18446744073709551615` (kPendingCompletion),
recycled, with host-read bytes `00 00 80 3e` = 0.25f and `00 00 70 42` =
60.0f — the exact values the suite reported. The -7.0f pad in the same
test case passed because `array(-7.0f)` needs no cast, so its bytes were
host-written and current; a correct control in the wild.

This is Main's case 2: the wait was the wrong instrument. The receipt
for 3f7e549a claimed full ordering; corrected in
`receipts/2026-09-12-composed-regressions/README.md` (CORRECTION
section).

## Fix (182a02bb)

At the top of the Scalar branch, exactly where the unconditional drain
sat:

```cpp
const auto* in_buffer =
    static_cast<const omarchy::VulkanBuffer*>(in.buffer().ptr());
if (in_buffer->completion != 0 &&
    in_buffer->completion >
        encoder.device().completions().drained_value()) {
  encoder.synchronize();
}
```

- `completion == 0`: the scalar's bytes are host-written (decode and
  prefill hot path: RoPE offsets, kv offsets, host constants) — no
  drain, nothing to wait for. This is why the prefill restoration
  stands: de780aa2's cost was draining unconditionally for exactly these
  host-written fills.
- `completion > drained_value` (committed, not yet drained) or
  `kPendingCompletion` (recorded in an open batch): the bytes are
  GPU-produced and possibly unpublished — `synchronize()` commits the
  open batch and joins, making the bytes visible before the host reads.
- `fill_pattern`'s recycled-target timeline wait is unchanged: it still
  orders the fill against committed in-flight writes to the recycled
  target.

`has_primitive()` is NOT a visibility test and was not used: detach()
clears the primitive the moment an array is evaluated, and evaluated
does not mean complete — the pad value cast is detached while its bytes
are still unpublished. The completion stamp is the actual visibility
state.

## Regression test

`overlay/tests/omarchy/test_shape_ops.cpp`,
`Pad constant survives recycled storage with a pending value cast` —
the convolve churn shape (2026-09-11-wrong-value-sweep): each iteration
floods one 4096-byte recycled block with a distinct sentinel (full +
eval + free), then pads with `array(3)` through the int32→float32 cast
without evaluating in between, and checks exact values. The stale-read
content does not matter: whatever the host read instead of 3.0f becomes
the fill value or collapses the fill to zeros, and the check fails.

- Pre-fix (3f7e549a copy.cpp, test added): FAILS — 304/432 assertions,
  2/2 runs.
- Post-fix (182a02bb): PASSES — 432/432, 2/2 runs.

## llvmpipe verification at the fix commit

Clean `prepare-mlx.sh` + rebuild at 182a02bb (x86_64, llvmpipe,
`MLX_OMARCHY_ALLOW_NON_APPLE=1`):

| suite | result |
|---|---|
| omarchy_shape_ops_tests (incl. new test) | 898/898 assertions, 25/25 cases |
| omarchy_complex_ops_tests | 1715/1715 assertions, 34/34 cases |
| omarchy_copy_offset_tests | 629/629 assertions, 26/26 cases — as written, including the `vk_submissions == +1` single-submission assertion (test_copy_offsets.cpp:181) |

copy_offset passes as written because the test's fills (values 22/121)
are host-written (`completion == 0`): the gated drain never fires, so
back-to-back fills stay in one submission. The batching contract and
pad correctness do not tension under this fix — the drain splits
submissions only for GPU-produced values, which the contract test does
not exercise and real decode/prefill paths do not produce.

## M1 window (jwm1, fork honeykrisp-26.3.0-devel, /tmp/m1-gpu.lock)

Wheels (built on jwm1, same session, same build host):
- baseline d389c24f: `mlx_omarchy-0.32.2.dev202609120916+d389c24f`
- fix 182a02bb: `mlx_omarchy-0.32.2.dev202609120913+182a02bb`

(FILLED AFTER WINDOW: interleaved A/B prefill table, digest gates, C++
suites on M1, churn probes.)

## Host-read audit (the bug class: host dereferences mapped memory whose
bytes a pending submission produces)

Every site found, with verdict. "Safe (drained)" = an explicit
synchronize/join orders the read; "safe (host-written)" = the bytes were
written by the host before any submission could touch them.

1. `copy.cpp` Scalar branch value reads — `scalar_is_zero` (37),
   `scalar_fill_value` (50-84), the byte/halfword/word/64-bit memcpys
   (523/532/561/593/622), the dtype-converting scratch source (483).
   **THE BUG** — read raced the pad-value cast's unpublished bytes.
   Fixed by the gated drain; one guard covers every entry above it.
2. `slicing.cpp:74` `compute_dynamic_offset` — host read of
   `live_indices.data<int32_t>()`. Safe (drained): `eval()` +
   `encoder.synchronize()` immediately before, per its own comment.
   Caveat, not a defect: the join covers this stream; cross-stream
   producers rely on scheduler-inserted stream ordering — same
   assumption every backend read makes.
3. `primitives.cpp:3914` `linalg_check_status` — reads kernel-pinned
   status words. Safe (drained): `encoder.synchronize()` on the line
   above.
4. `primitives.cpp:4214` trig argument gate — `magnitude.item<float>()`.
   Safe (drained): eval + `synchronize("trig_argument_gate")`; the
   comment records the hardware failure (recycled-page ~1e9) that
   proved the need.
5. `primitives.cpp:10006/10012/10019` `rope_trig_gate` — offset/freqs
   `item` reads. Safe (drained): the scalar-offset path drains unless
   `status()==available && !has_primitive()` (host constant from
   construction — the comment warns is_available() alone promotes
   in-flight arrays); the vector paths drain unconditionally. Cost
   receipt: 2026-09-04-rope-gate-drain.md.
6. `primitives.cpp:10059` RoPE fallback — `result[0].eval()` +
   `synchronize("rope_fallback")`, then buffer share, no host read.
   Safe (drained).
7. `Load::eval_gpu` (`primitives.cpp:6034`) — writes host-read bytes
   INTO the output buffer (reader_ source is host I/O, not device
   memory). Safe (host-written); recycled target is provably drained
   under the allocator quarantine invariant below.
8. Host writes into mapped allocations — `make_copy_axis_metadata`
   (copy.cpp:153), fill edge bytes (copy.cpp:191), scalar scratch
   (copy.cpp:483), `custom_kernel.cpp:798`, `fused_chain.cpp:524`,
   metadata buffers (primitives.cpp 767/2249/5416/8135/8900), qmm
   scale (10665), causal mask (10714), dynamic-slice offset
   (slicing.cpp:81). Safe from THIS class (writes, not reads); safe
   from lost-write against in-flight old occupants only via the
   allocator invariant: free() quarantines any buffer stamped
   kPendingCompletion or with an undrained completion, so
   `reuse_from_cache` hands out blocks whose previous occupant is
   provably drained or host-written. Non-coherent hosts publish at
   submit (`flush_noncoherent` in submit()).
9. Completion bookkeeping (`CompletionDispatcher`, device.cpp) — pure
   host-side state under mutex; no mapped reads. Safe.

Audit verdict: after 182a02bb there is no remaining host read of
device-produced mapped memory that lacks a drain. The structural
lesson, recorded for the next person: `has_primitive()` is NOT a
visibility test — detach() clears the primitive when an array is
evaluated, and evaluated does not mean complete; the buffer completion
stamp is the visibility state. Anyone reaching for has_primitive() as
an in-flight test will re-create this bug. Any new host read of tensor
bytes must sit behind a synchronize or gate on
`completion > drained_value` the way scalar fills now do.

Safe-only-by-accident (latent, named per request — safety held by an
invariant elsewhere rather than a check at the site, the exact shape
that broke scalar_is_zero):

- Site 8 (every host write into a mapped allocation): safe only
  because free() quarantines stamps it cannot prove drained. The
  single point of failure is batch registration: a future dispatch
  path that binds a buffer without `note_batch_buffer`/
  `add_temporary` hands a block back while its producer is still
  queued, and both the lost-write class and the stale-read class
  return at once. A debug assertion at `reuse_from_cache` (block
  drained or never stamped) would make this by construction; not
  added in this fix.
- Site 1's old guard `if (in.has_primitive()) unsupported(...)` reads
  like a visibility check but only catches never-evaluated arrays;
  without the new completion gate directly below it, it kept
  "passing" while the read raced — the same trap scalar_is_zero sat
  in for months. Do not collapse the completion gate into it.
- Sites 2-6 are safe by construction: the synchronize is the
  immediately preceding statement in the same function, not an
  accident of call order.

# The RoPE gate drained the queue 48 times per token: 4-bit decode +76%

2026-09-04, jwm1 (Apple M1 G13G B1, Honeykrisp 1.4.354). Before = main
94d967c, after = a108741. Both wheels built on the M1 from detached
checkouts, fresh venv per side, provenance verified=match every run.
Qwen2.5-0.5B, pinned 64 tokens, EOS suppressed, decode over the last 63;
bf16 eager.

## What the profile showed

`MLX_OMARCHY_GPU_PROFILE` on main, 4-bit, 64 tokens:

- GPU busy 12.07% of wall
- 604 dispatches per decode token in **48 submissions**
- join wait 42.5 ms per decode token in the instrumented run

Profiling perturbs execution. These times must not be divided by the
uninstrumented benchmark latency to estimate its host/GPU split.

48 submissions for 24 decoder layers is two per layer, and the split was
mechanical: `sub N` = 5 dispatches (rope + 4 cache copies), `sub N+1` =
21 dispatches (the rest of the layer), repeating exactly. Each boundary
is a `vkQueueSubmit` plus a host join.

## The cause

`rope_trig_gate` (`primitives.cpp`) checks the rotational argument
magnitude before dispatching the fused RoPE kernel, and to read the
offset on the host it called `encoder.synchronize()` - commit the open
command buffer, then block until the device drains. That is correct for
an offset produced by GPU work. The measured mlx_lm path supplies a
Python integer, represented by an available array without a primitive.
The implementation uses those properties to select the direct read for
non-bf16 output. Whether these properties prove readiness for every
other caller requires a separate ownership/readiness audit.

The bf16 path retains synchronization.

## Numbers (M1, five runs each, same session)

| dtype | before | median | after | median | delta |
|---|---|---|---|---|---|
| 4-bit | 17.44 17.17 17.52 17.57 17.46 | 17.46 | 30.82 30.72 30.70 30.76 30.74 | 30.74 | **+76%** |
| bf16 | 8.70 8.75 8.76 8.79 8.76 | 8.76 | 8.76 8.74 8.73 8.75 8.74 | 8.74 | unchanged (fenced) |

Submissions per decode token: 48 -> 11 (`scripts/fragmentation_probe.py`,
same box). `omarchy_fast_ops_tests` 24,084 assertions green on the Apple
GPU. The earlier greedy check compared only the first 96 IDs after
decoding and re-encoding generated text, with a 128-token generation cap.
It did not establish equality of 128 actual generated token IDs.

## What removing the drain exposed

bf16 decode began refusing at about token 33 with a RoPE argument
magnitude of ~1e9. Dumping the bytes around the offset scalar:

```
floats: -0.192383 0.902344 1.57031 -1.11719 1.4375 1.52344 -2.6875
u32:    be450000 3f670000 3fc90000 bf8f0000 3fb80000 3fc30000 c02c0000
```

These are float32 values with zero low 16 bits, exactly representable in
bf16. The bit pattern does not identify the writer or establish why the
offset storage changed. A lifetime error, aliasing, or an ordering error
remains a hypothesis, not a diagnosed cause.

The commit also pins the dense temporary used by `copy_gpu` for strided
dtype casts. That fix did not resolve the bf16 observation without the
drain. bf16 therefore retains synchronization while ownership and
producer readiness are investigated. See `docs/known-defects.md`.

That is why 4-bit gets 76% and bf16 gets nothing today. The gate refused
rather than returning a wrong number, which is the contract working.

## Note on the instrument

Three theories were tested and killed against the profile before the
right one: cross-stream fences (a trace showed zero fence crossings
during generation), the 100-node batch budget (633 dispatches per token
would give 6 submissions, not 48), and descriptor-pool exhaustion (2048
sets per pool). The trace that settled it printed the submission
boundaries with their kernel names.

Logs: jwm1:`~/benchq/logs/profile-94d967c/` (profile),
`~/benchq/logs/ropegate3/` (A/B).

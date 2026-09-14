# jwm1: the AGX wait-insertion patch fires, and it does not matter

2026-09-14, jwm1 (Apple M1, G13G B1). Driver-side half of the question that
`receipts/2026-09-14-jwm1-qmm-prefill-coopmat-arms` left open from the shader
side.

## Verdict

The `agx_insert_waits` patch is **live, not inert**. The 2026-09-12 named fix
works: the region pass now accepts 75 divergent if/else frames per dispatch
where `84fcd220de1` accepted zero. Whole-kernel scoreboard waits in the
shipped `qmm_coopmat` kernel drop **23 to 19**, so the acceptance signal set
by the coopmat-arms receipt ("a dump after the patch should show fewer than
23 waits; if it does not, the pass is still inert") is met.

It buys nothing, and the ISA says exactly why. **All four removed waits are in
the epilogue. The steady-state k-loop body has 5 waits in both arms,
unchanged.** The epilogue runs once per workgroup; the loop body runs per
k-step. The removal is on the cold path by construction.

Measured, paired, same mlx build, driver as the only variable:

| metric | base | patched | delta |
|---|---:|---:|---:|
| dominant cell 1053x896x9728 | 17.7602 ms | 17.6801 ms | +0.45 % |
| `QmmPrefillCoopmatF16` GPU time, 163 dispatches | 708.191 ms | 708.265 ms | -0.01 % |
| prefill-window GPU busy | 942.533 ms | 943.473 ms | -0.10 % |

Every one of those deltas is smaller than the within-arm spread. All eight
cells are bit-identical across both arms and five rounds. Nothing landed as a
perf change, and nothing should.

**This is a verdict, not a failed arm.** It is the clean, unconfounded
isolation of wait placement that no previous experiment in this line could
give, because the kernel source is byte-identical across the two arms.

## What was under test

No driver was installed. Both arms are locally built mesa trees selected
per-command by `VK_DRIVER_FILES` on the python invocation only. The system
driver was never touched: `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`,
`/usr/lib/libvulkan_asahi.so` sha256 `abcce1bb…`, unchanged before and after
(`rollback.txt`, `identity.txt`).

| arm | tree | HEAD | `libvulkan_asahi.so` sha256 |
|---|---|---|---|
| base | `~/src/mesa-wt-waitbatch-base` | `6f6afc89684` (= the packaged base) | `c3c2a4ec…` |
| patched | `~/src/mesa-wt-waitbatch` | `ed1895cb714` | `485c6274…` |

Both `debugoptimized`, `optimization=2`, `debug=True`, `b_ndebug=if-release`.
Same venv for both arms: `mlx-omarchy 0.32.2.dev202609122355+diag.b41e2b74`,
`libmlx.so` `d41f311e…`.

### Provenance: the timed driver is not `14eb6778a69`

The 2026-09-12 receipt records the patch inert at `84fcd220de1` and names an
unbuilt fix at `8f3169a4342`. That receipt was written at 07:26; the same
agent kept committing until 08:39. **Seven further commits already existed and
had never been dumped or measured**, including `c3cebc011bd` ("fix
wait-insertion frame detection, restore hazard scoping, add firing proof
logs") and `30ffef135e0`, which reverts the 4-slot change back to 2 with a
static assert — `agx_instr::scoreboard` is a 1-bit field, so the original
"slots 2 to 4" was silently truncating every slot assignment above 1.

On top of that tip I added one commit, `ed1895cb714`, gating the pass's
tracing behind `AGXWAITS_DEBUG`: the committed code wrote several lines per
shader compile to stderr unconditionally, which would pollute timed runs. The
timed binary is therefore `14eb6778a69` **plus** `ed1895cb714`
(`debug-gating.patch`), and the firing proof below was captured from that same
`485c6274…` binary in a separate run. Full pass diff against the packaged
base: `agx_insert_waits.full.patch` (387 lines changed in
`agx_insert_waits.c`, plus a 243-line host test).

## The pass fires

`AGXWAITS_DEBUG=1` on the timed binary, one dominant-cell dispatch
(`dumps/patched.firing.log`, 482 trace lines):

| event | count |
|---|---:|
| frame ACCEPT | 75 |
| else-arrival (merge seeded with union of both arms) | 75 |
| frame close | 75 |
| carry across exec-mask fallthrough | 215 |
| conservative per-block drain | 16 |
| reject: `term->target == NULL` | 13 |
| reject: shape mismatch | 0 |
| overflow-drain at merge | 4 |

Against `84fcd220de1`, which the 2026-09-12 receipt measured at **0 pushes
across 263 blocks and 88 mask branches**. Zero shape rejections means the
lazy else-arrival merge resolution handles every CFG shape this dispatch
produces. `dumps/base.firing.log` is empty of trace lines, as it must be: the
base tree has no instrumentation.

The pass's own hazard tests pass (`agx_tests --gtest_filter='*Wait*'`):
`RawHazardWaitsAtUse`, `SlotCapacityWaitsBeforeIssue`,
`DivergentWawWaitsInElseArm`, `MergeReaderSeesCarriedPending`. All five asahi
meson suites OK. Running them did not relink the measured `.so` (hash
re-checked after: still `485c6274…`).

## Wait counts, before and after

Final packed ISA of the `qmm_coopmat` kernel, captured with
`AGX_SIMDMAT=1 MESA_SHADER_CACHE_DISABLE=true AGX_MESA_DEBUG=shaders` under
the GPU lock, one dump per arm (`dumps/`, counters `isa_count.txt`,
`isa_waits.txt`, `isa_diff.txt`).

| whole kernel | base | patched |
|---|---:|---:|
| packed instructions | 699 | 699 |
| **`wait`** | **23** | **19** |
| global `load` | 7 | 7 |
| `lload` / `lstore` | 40 / 48 | 40 / 48 |
| `barrier` | 10 | 10 |
| `if` / `pop_exec` | 34 / 35 | 34 / 35 |
| `else` | 6 | 10 |
| `simd_matrix_fmadd32` | 16 | 16 |

The base column reproduces the 2026-09-12 accounting and the coopmat-arms
dump exactly (699 instructions, 23 waits, 7 loads), so this dump pipeline
describes the same code as both.

## Where the four waits actually went

Splitting the kernel at the k-loop boundary is the whole finding. The loop
span `[65..293]` is 229 instructions with 5 waits and 5 global loads, which
reproduces the independent 2026-09-12 steady-state accounting exactly — so the
split is trustworthy, not fitted.

| region | base waits | patched waits |
|---|---:|---:|
| before loop head | 2 | 2 |
| **steady-state k-loop body (229 instr)** | **5** | **5** |
| epilogue, after loop tail | 16 | 12 |

Inside the loop the patch removed nothing. It only reordered, at exactly five
sites — one per staging load:

```
base:     … load   iadd   wait   else   …
patched:  … load   iadd   else   wait   …
```

Load-to-next-wait gaps go `4,1,1,1,1,1,1` to `5,2,2,2,2,2,2`: the extra
instruction is the `else` the wait now sits behind. Same number of exposed
memory latencies, one instruction later. The epilogue is where the count
changes, and it also picks up the 4 extra `else` and the "waits within 2
instructions before a barrier" shift from 0 of 23 to 8 of 19.

### Mechanism

This is inference from the two dumps, not a direct measurement, but it is
tightly constrained. The patched pass carries pending scoreboard state across
exec-mask fallthrough edges and inserts waits at **first use**; only real
branches, back-edges and barriers drain. If the loop's 5 waits had been
conservative block-exit drains, this pass would have removed them. It did not.

Therefore the loop's 5 waits are **true first-use RAW waits**: the scheduler
placed each global load immediately adjacent to the unpack that consumes it,
and when first use is one instruction after the load, "wait at first use" and
"wait at block exit" emit the same instruction in the same place. There is
nothing for a batching pass to batch.

That is the same fact `receipts/2026-09-14-jwm1-qmm-prefill-coopmat-arms`
found from the source side — de-diverging the staging loads removed all five
exec-mask guards and left the wait count at 23 — seen now from inside the
driver. Two independent observations, one cause.

### What this supersedes

`receipts/2026-09-12-agx-qmm-codegen`, field `primary_cause_driver`, is
**superseded by name**. Its claim is:

> The GLSL loads all four x words in one loop and unpacks them in a second
> loop — the source already batches. The driver re-serialized them.

The driver did not re-serialize them. The scheduler sank each load next to its
consumer, and the four exposed latencies per step were never a wait-insertion
artifact. The two remedies that claim implied are both now measured and dead:
the source-level one (8f6de34c de-diverged staging, -13 to -26 percent) and
the driver-level one (this receipt, flat). That field is the first thing a
reader of the 2026-09-12 receipt encounters, which is why it is called out
here rather than left implied.

## Per-cell A/B and the digest gate

Interleaved inside one lock hold, 5 rounds, arms alternating within each
round, 30 timed reps per cell per run, same venv and byte-identical kernel
source (`ab/ab.jsonl`, `ab/ab-summary.md`).

| cell | base ms | patched ms | delta | base GF | patched GF | digest |
|---|---:|---:|---:|---:|---:|---|
| 262x896x128 | 0.7210 | 0.7214 | -0.06 % | 83.4 | 83.3 | `6a31dc466db80a40` |
| 1053x896x128 | 0.5912 | 0.6108 | -3.32 % | 408.5 | 395.4 | `e9218e07c9c13e5e` |
| 262x896x896 | 0.7644 | 0.7957 | -4.09 % | 550.3 | 528.7 | `5c42552b5320472e` |
| 1053x896x896 | 1.9912 | 1.9873 | +0.20 % | 849.1 | 850.8 | `2df9480de6e511ec` |
| 262x4864x896 | 2.7290 | 2.7070 | +0.81 % | 836.8 | 843.6 | `ecd08c8841ecd105` |
| 1053x4864x896 | 9.4059 | 9.4160 | -0.11 % | 975.8 | 974.8 | `3e79733f76ed9d37` |
| 262x896x9728 | 5.0859 | 5.1027 | -0.33 % | 898.0 | 895.1 | `4e29e2dbdf33b57b` |
| **1053x896x9728** | **17.7602** | **17.6801** | **+0.45 %** | 1033.6 | 1038.3 | `5179630cd4a7c3f9` |

**Digest gate: PASS.** One digest per cell, identical across both arms and all
five rounds. The dominant cell's `5179630cd4a7c3f9` matches the pinned value
from 2026-09-11, 2026-09-12 and 2026-09-14.

Per-round dominant-cell medians: base `17.7602, 17.7990, 17.7597, 17.8221,
17.6532`; patched `17.6801, 17.6466, 17.6603, 17.8088, 17.8037`. The +0.45 %
median separation sits inside both arms' round-to-round range, so it is not a
speedup.

Verdict set is the 1053-row cells (+0.20, -0.11, +0.45). The x896x128 cells
are work-starved at 132 and 36 workgroups and swing several percent across
unrelated arms; the -3.32 % there is noise of the kind the 2026-09-10 screen
already rejected, and the -4.09 % on 262x896x896 is the same.

### The digest gate is a hazard test, and it is not a proof

An under-inserted wait is a race: it can pass once and fail under different
timing or concurrent load. Forty matching per-cell digests bound the risk and
do not eliminate it. Combined with the four passing host hazard tests, that is
enough to call the patch **safe as a diagnostic**. It is not enough to ship,
and since it buys nothing there is no reason to try. Anyone who later revives
this patch for a variant that does pay should run the six-leg generated-id
screen (`~/benchq/qmm-coop-bench/qmm_digest_screen.sh`), not the kernel probe
alone.

## End-to-end 1053-token prefill

Four interleaved rounds per arm, arms alternating, `run_profile.sh` protocol
and prompt from `receipts/2026-09-13-jwm1-phase-profile-enabled`
(`profiles/`, `profiles/prefill-summary.md`).

| metric | base median | patched median | delta | within-arm base spread |
|---|---:|---:|---:|---:|
| `QmmPrefillCoopmatF16` GPU time | 708.191 ms | 708.265 ms | -0.01 % | 0.8 % |
| prefill-window GPU busy | 942.533 ms | 943.473 ms | -0.10 % | 1.8 % |
| host wall-clock | 703.8 tok/s | 684.7 tok/s | -2.72 % | **38.1 %** |

Every delta is smaller than the within-arm spread: no resolvable difference on
any metric. `QmmPrefillCoopmatF16` ran 163 dispatches and the prefill window
1063 dispatches in every one of the eight runs.

The comparison metric is **cumulative total over all 163 dispatches, not mean
or p50**, and that matters. Those 163 dispatches are four distinct shapes with
roughly a 10x spread in per-dispatch cost, so mean and p50 are shape-mix
statistics rather than kernel statistics — the distance between the 4.3 ms
mean and the 1.68 ms p50 here is the mix, not variance
(`Jw16MaxAttribution`, `receipts/2026-09-14-jw16-max-gpu-attribution`). The
total is mix-invariant only because the dispatch set is bit-identical between
arms, which the 163/1063 counts above confirm for every run. Anyone needing a
per-shape A/B out of these captures should use that receipt's
`shape_census.py` rather than the mean.

The tok/s row should be read as noise, not as a regression. Host wall clock
swung 546 to 815 tok/s **within the base arm alone** while prefill-window GPU
busy held inside 1.8 percent, so the variance is host scheduling on a shared
box, not the driver. Kernel ms and prefill-window GPU busy are the metrics
that survive here — and the next section shows the tok/s column is also
depressed wholesale by a flag, in both arms equally.

### The tok/s gap is a cold cache times a `debugoptimized` build, measured 2x2

These arms run about 700 tok/s against the 948.80 tok/s anchor in the
coopmat-arms receipt, while `QmmPrefillCoopmatF16` lands at 708 ms against
that receipt's 704.425 ms. The GPU kernel time is the same, so the gap is
outside the kernel.

This section was wrong twice before it was right, and both wrong versions
are worth recording because each was a plausible single-cause story:

1. Draft 1: "the entire gap is `debugoptimized` mesa with assertions."
2. Draft 2, after `Jw16MaxAttribution` flagged shader-cache contamination and
   a flag-only A/B on this build measured 385 ms: "the entire gap is
   `MESA_SHADER_CACHE_DISABLE`; the build type contributes essentially
   nothing."

Draft 2 was also wrong, and `Jw16MaxAttribution` caught why: **both arms of
that A/B carried the `debugoptimized` ICD, so it measures the flag's cost
*on that build* and structurally cannot separate the two factors.** The fix
is the full 2x2 — packaged versus `debugoptimized` ICD crossed with the cache
flag, same harness, same profiler state, same wheel, same host, three runs
per cell, nothing crossing receipts (`profiles/factorial/`, `profiles/cached/`).

| ICD build | cache | `QmmPrefillCoopmatF16` | prefill-window busy | host prefill | tok/s |
|---|---|---:|---:|---:|---:|
| packaged | on | 701.500 ms | 919.403 ms | 1106.3 ms | 951.8 |
| packaged | off | 701.816 ms | 920.748 ms | 1216.7 ms | 865.5 |
| `debugoptimized` | on | 701.740 ms | 919.178 ms | 1111.9 ms | 947.0 |
| `debugoptimized` | off | 708.198 ms | 943.208 ms | 1530.8 ms | 687.9 |

| effect | magnitude |
|---|---:|
| cache flag, on the packaged build | 110.4 ms |
| cache flag, on the `debugoptimized` build | 418.9 ms |
| build type, with the cache **warm** | **5.6 ms (0.5 %)** |
| build type, with the cache **cold** | 314.1 ms |

**It is an interaction, not a main effect of either factor.** The assertions
live in the compiler path, so they cost nothing until shaders actually have
to be compiled — and then they roughly quadruple the compile penalty.

The reusable fact, stated correctly this time: **a `debugoptimized`
ICD-override build is comparable to the packaged build only once the shader
cache is warm** — 5.6 ms and 0.5 percent on host prefill, +0.240 ms and
0.03 percent on kernel GPU time. With a cold cache it is not comparable at
all. The caution belongs on the *combination*, not on the flag alone and not
on the build type alone.

Two consequences worth carrying:

- The anchor's 948.80 tok/s implies 1109.8 ms of host prefill; the warm-cache
  packaged cell here is 1106.3 ms and the warm-cache `debugoptimized` cell is
  1111.9 ms. Both land within 0.3 percent of the anchor, which is what
  licenses comparing this receipt's kernel numbers to it at all.
- The flag's apparent cost to kernel GPU time is itself build-dependent:
  +6.45 ms on `debugoptimized` but only **+0.316 ms** on the packaged build.
  Anyone correcting a packaged-build capture for cache contamination should
  use ~0.3 ms, not the 6.45 ms this receipt reported before the 2x2 existed.

None of this touches the verdict. The flag and the build are held identical
across both arms of the A/B throughout, so they are symmetric and cancel in
the pairing. They only ever affected comparison against the external anchor.

## Rollback and safety

- **No driver was installed.** The rollback was staged and byte-verified
  before any measurement rather than exercised. `pacman -U` of
  `~/src/mesa-pkg-20260908/mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz`
  (sha256 `cef58afe…`) restores the running driver; the `libvulkan_asahi.so`
  inside that package was extracted and `cmp`-verified byte-identical to the
  installed `/usr/lib/libvulkan_asahi.so` (`abcce1bb…`). A staged rollback
  nobody checked is not a rollback.
- System ICD untouched: `/usr/share/vulkan/icd.d/` files still dated Sep 8,
  `asahi_icd.aarch64.json` still pointing at `/usr/lib/libvulkan_asahi.so`.
  Arm selection is per-command `VK_DRIVER_FILES` on the python process only —
  never exported to a shared shell, never a systemd drop-in.
- `/tmp/m1-gpu.lock` inode 29, always `flock -w 60`, never stolen, never
  unlinked. Nested `flock -n` confirmed the hold from inside each profile run
  (`profiles/*/lock.json`) and inside the A/B screen. Free after the work.
- No ANE, no reboot, no 1x896, no SET writes.
- Coordinated over IRC: stood down for `DecoderProjectorKernel`'s lock window
  and kept jwm1 CPU quiet for its timing pass; sequenced the six prefill runs
  into a gap `EncoderParityAne` held open, after establishing that its 10.0 s
  encoder pass landing between two arms would have destroyed the pairing.
  `IslandsExecJwm1`, `AneChannelPolarity` and `IslandsExecJw16` are ANE-only,
  no GPU conflict.

## Where this leaves the line

The kernel is at 1035 GFLOP/s at the dominant prefill cell against a
1459 GFLOP/s in-geometry coopmat ceiling: 1.41x of headroom, worth roughly
+22 percent prefill if all of it were recovered. What is now closed:

- Six bit-preserving shader arms, measured and rejected
  (`receipts/2026-09-14-jwm1-qmm-prefill-coopmat-arms`).
- Operand transport, staging chunk size, ILP schedules, f16 operands,
  de-diverged staging (2026-09-11 receipts, 8f6de34c).
- **Wait placement, this receipt.** Retired as the dominant term. Not "we
  could not fix it" — fixed, fires, and the loop's waits turn out to be real
  data dependencies with nothing to batch.

The remaining lever is not `agx_insert_waits`. It is whatever would hoist the
staging loads away from their consumers — instruction scheduling or software
pipelining — which is a different pass entirely.

Ranked handoff, both from `QmmPrefillOpt` and endorsed here:

1. **Shared-memory throughput, not DRAM latency, is now the leading
   candidate.** If the 5 loop waits are true RAW first-use waits, and the
   `uvec4` arm cut them to 2 and lost 13.4 percent, and this patch leaves the
   loop untouched and lands flat, then exposed staging-load latency is cheap
   and presumably covered by multi-workgroup concurrency. That contradicts the
   2026-09-12 stall model, which assigned ~250 of the implied 494 clk/step to
   memory latency and the barrier path. Every negative arm on record perturbed
   shared access patterns: pitch padding -19 to -20 percent, 16x16x16 fragment
   loads -26 percent, `uvec4`'s store pattern -13 percent. Decide it with two
   diagnostic-only arms via `~/benchq/qmmpad/arm_shader.sh`, digests
   deliberately not preserved, run as ceilings: (a) staging-only, keep every
   load, unpack, dequant, shared store and both barriers, delete the
   `coopMatMulAdd`s; (b) math-only, stage one tile before the k loop. If
   staging-only plus math-only is about the shipped 17.7 ms, the phases are
   serialized and shared traffic is the binding term.
2. **H1, still unrun:** re-run the zero-traffic coopmat ladder with a 4 KiB
   shared skeleton, 128-lane workgroups and a flat per-tile dispatch, to find
   whether the 2148-to-1459 loss is shared footprint, subgroup residency or
   the tile loop. That decides whether the kernel's ceiling is 1459 or higher.

## Artifacts

- `identity.txt`, `rollback.txt` — host, driver hashes, build configs, branch
  history, lock state, rollback verification.
- `agx_insert_waits.full.patch` — the whole pass diff vs the packaged base.
- `debug-gating.patch` — `ed1895cb714`, the only change made in this session.
- `dumps/{base,patched}.dump.log.gz` — full AGX shader dumps.
- `dumps/{base,patched}.firing.log` — `AGXWAITS_DEBUG=1` traces.
- `dumps/{base,patched}.driver.json` — `/proc/self/maps` proof that each arm
  mapped exactly one, and the intended, `libvulkan_asahi.so`.
- `dumps/isa_count.txt`, `isa_waits.txt`, `isa_diff.txt` — the instruction
  tables and the aligned per-position opcode diff behind the loop/epilogue
  split.
- `ab/ab.jsonl`, `ab/ab-summary.md` — 5-round interleaved 8-cell screen.
- `profiles/prof/`, `profiles/prof2/` — the 8 A/B prefill runs, per-run
  `analysis.txt`, `markers.jsonl`, `lock.json`, plus `prefill-summary.md`.
- `profiles/cached/`, `profiles/factorial/` — the 12 runs of the 2x2 that
  separated the shader-cache flag from the `debugoptimized` build type
  (3 per cell: packaged and `debugoptimized` ICD, cache on and off).
- `scripts/` — everything used, including `wp-isadiff.py` (the loop/epilogue
  split), `whichdriver.py` (the binding proof), and the four arm wrappers
  `py-base-cached`, `py-system-cached`, `py-system-nocache` plus the two
  generated by `wp-setup.sh`.

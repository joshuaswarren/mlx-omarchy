# 2026-09-11 — Investigation of 8 alleged f16 SDPA throws in `omarchy_fast_ops_tests` on the M1

## Question

The README on the bf16-alpha-fix strand (`receipts/2026-09-11-bf16-alpha-fix/README.md`) reported:

> `omarchy_fast_ops_tests` on the M1 throws in 8 sdpa cases —
> `[omarchy] ScaledDotProductAttention dtype is not implemented
> (dtype=float16, rank-5 mask/GQA shapes)`.

…and named it as a "pre-existing defect, not this branch's", distinct from the
bf16 alpha fix. The f16 path was claimed byte-identical between main, bf16-alpha-fix,
and the strand-2 candidate.

## Investigation

### Reproduce the throw class locally (no M1 needed)

Built the current main tree (b4271903) on this dev box (x86_64, llvmpipe):

* `MLX_OMARCHY_ALLOW_NON_APPLE=1 LP_NUM_THREADS=4` — real llvmpipe caps
  (subgroup_size 4, no coopmat): **33/33 cases pass**, 1,115,786 assertions.
  Receipt: `/tmp/llvmpipe-fastops-baseline.log`.
* `MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_CAPS_SIM=m1-stock-no-coopmat`
  (simulated M1 stock Mesa: subgroup 32, no coopmat): **33/33 cases pass**,
  1,115,786 assertions. Receipt: `/tmp/sim-fastops-m1-stock-no-coopmat.log`.
* `MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_CAPS_SIM=m1-honeykrisp-fork`
  (simulated M1 Honeykrisp fork: subgroup 32, ARITH|SHUFFLE|SHUFFLE_RELATIVE,
  coopmat f32 8x8x8, 32768 shared bytes): **9 THROWS** —
  test_fast_ops.cpp lines 759, 976, 1036, 1115, 1136, 1209, 1304, 1328, 1349.
  Receipt: `/tmp/sim-fastops-m1-honeykrisp-fork.log`.

### The throw site

All 9 throws are from `omarchy::capsim::require_backed(...)`:
- Line 585: `MatmulF32Coopmat/MatmulBF16Coopmat`, requires `cooperative_matrix_fp32_8x8x8`
- Line 10388: `SdpaDecodeNativeF16`, requires the coopmat + workgroup + shared memory axes

The eight f16 SDPA cases (skipping line 759's SdpaDecodeNativeF16, which is a
different path) all reach the f16 attention score matmul whose dispatch selects
MatmulF32Coopmat when `cooperative_matrix_f32_8 && subgroup_size==32 &&
matrix_m > 1 && matrix_k % 8 == 0 && alpha == 1.0f && !use_c` holds
(`overlay/mlx/backend/omarchy/primitives.cpp:561-580`).

### Why the throw does NOT fire on real M1 hardware

`omarchy::capsim::require_backed` (capability_sim.cpp:199-218) is gated on
`caps.simulated` — the `simulated` flag is only set by `apply()` in
capability_sim.cpp:193, and `apply()` is only called when
`MLX_OMARCHY_CAPS_SIM` is set (the device constructor chooses between the
hardware report and the simulated overlay). On real hardware
(`caps.simulated == false`) `require_backed` returns silently, regardless of
whether the axis is actually backed.

The simulation refusal is a **strict-capability harness**: it throws when
a profile claims a kernel that the simulation's hardware (always llvmpipe)
does not back. It is the wrong tool for predicting what real M1 silicon does.
On the real M1 Honeykrisp fork driver, `hardware_capabilities().cooperative_matrix_f32_8`
is true (the profile was derived from that driver), so the axis is provided and
the coopmat dispatch executes normally.

The bf16-alpha-fix README's "8 throws" claim conflates the simulation's
strict-refusal pattern with a real M1 defect that never actually fired —
no M1 run produced the throw; only a `MLX_OMARCHY_CAPS_SIM=m1-honeykrisp-fork`
run on llvmpipe produces it.

### Why llvmpipe "passes all 34" while simulated M1 "throws 9"

The task brief says "34 cases". The current suite has **33 cases** (the
README's 34 included the alpha-branch coopmat regression test). The
8-cases-vs-9-throws gap maps cleanly:
- llvmpipe (real caps, no coopmat, no sim): 33/33 pass — uses non-coopmat
  kernel paths; coopmat gates false, no refuse.
- `m1-stock-no-coopmat` sim: same — coopmat gate false under simulation,
  no refuse.
- `m1-honeykrisp-fork` sim (claims coopmat): coopmat gate true under sim,
  llvmpipe doesn't back coopmat → strict refusal throws on every case
  whose dispatch path picks coopmat.
- Real M1 Honeykrisp fork: coopmat gate true, hardware backs it → no
  refusal, dispatch executes.

### Why the README message text didn't match

The README paraphrased the throw as `[omarchy] ScaledDotProductAttention
dtype is not implemented (dtype=float16, rank-5 mask/GQA shapes)`. The
actual refusal (verbatim from this run) is the
`require_backed` message naming `MatmulF32Coopmat/MatmulBF16Coopmat`
and the `cooperative_matrix_fp32_8x8x8` axis. The README's text was the
prior agent's loose summary, not a verbatim quote — there is no
`ScaledDotProductAttention dtype is not implemented` site in
primitives.cpp that could fire for rank-5 mask/GQA f16 inputs (the only
dtype gate at the SDPA entry, `require_float_dtype` with message `... dtype`,
requires `input.dtype() != out.dtype()`, but every suite case constructs
matching dtypes).

## Classification

* **Reachability:** Every one of the 8 f16 SDPA cases is reachable by
  real-model workloads (Qwen-style decode + prefill uses GQA with
  rank-5 broadcast views through this exact path on every step).
* **Outcome class:** On real M1 hardware: **silent correct execution
  through the coopmat route**, NOT a refusal. The throw class only
  manifests in `MLX_OMARCHY_CAPS_SIM=m1-honeykrisp-fork` simulation,
  where it is the documented strict-capability refusal
  (`docs/new-chip-bringup.md` §4: "a refusal is a routing fact, not
  a bug"). A refusal is permitted by the contract.
* **Real defect:** None. The prior agent's README conflated a
  simulation-only artifact with a real-M1 defect.

## Decision

No source change. The M1 battery needs `omarchy_fast_ops_tests` so that
the suite is exercised on every M1 qualification wave (otherwise the gap
between llvmpipe's "passes everything" and the M1's "untested" stays
invisible — which is the original reason this confusion was possible).
The "M1 battery never ran the suite" is the real defect; the fix is
the recording, not the code.

## Verification on the real M1

Captured 2026-09-11 13:19:09-05:00 on jwm1 (100.84.184.102), driver
`mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1` (the canonical Honeykrisp
fork build, pinned). Lock window: top-level flock `/tmp/m1-gpu.lock`,
cap 7200s, single 43-second hold (capture script `/tmp/f16sdpa_capture.sh`,
read-only w.r.t. `/tmp/fast-alpha`). Binary: `/tmp/fast-alpha`
(`61058d55a45ae631...`, bf16-alpha-fix tree at 4376ffeb, f16 path
byte-identical to main b4271903).

Result: **34/34 cases pass, 1,104,353 assertions, zero failed, zero
throws, Status: SUCCESS**. Log: `m1-fastops.log`. Capture log:
`m1-capture.log`.

Confirms the prediction: real M1 Honeykrisp fork driver backs
`cooperative_matrix_fp32_8x8x8` (the axis the eight f16 SDPA dispatch
paths select), so `omarchy::capsim::require_backed` short-circuits on
`caps.simulated == false` and the coopmat dispatch runs normally. The
"8 sdpa throws" claim is wrong — there is no real M1 defect.

## Battery addition

The recorded M1 battery previously only included
`omarchy_matmul_family_tests` and `omarchy_runtime_tests`
(`receipts/2026-09-10-bf16-decode-gemv-requal/m1_suites.sh`). The
AGENTS.md "Test rules" section now lists `omarchy_fast_ops_tests` (and
the other suites the prior M1 runs had no central record of) as the
standing M1 battery so the suite ships on every qualification window —
closing the "llvmpipe passes 34, M1 untested" gap that produced this
confusion.

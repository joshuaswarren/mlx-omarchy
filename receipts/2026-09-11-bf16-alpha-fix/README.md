# BF16 coopmat alpha fix — strand 1 of the bf16-prefill-attention split (2026-09-11)

Agent: `Bf16AlphaFixAndVerdict`. Branch: `bf16-alpha-fix`
(EE8D26FB = origin/main at split time + one fix commit, no performance
flip). Carrier commit: `c6c44674`. The performance-flip strand and its
verdict live in `receipts/2026-09-11-bf16-prefill-attention` on branch
`bf16-prefill-attn`; the two strands share no commit.

## The defect

`shaders/matmul_coopmat_bf16.comp` declared `alpha` in its params and
never read it, while `dispatch_matmul` required `alpha == 1.0f` for
every coopmat dispatch. Any alpha != 1 bf16 coopmat dispatch would have
silently computed unscaled results.

## Latent trap, not a live defect

No shipped code path could reach the broken case before the fix; the
gate happened to exclude it. Evidence, from the ee8d26fb tree:

- `MatmulBF16Coopmat` is selected in exactly one place
  (`dispatch_matmul`; the only other reference is the pipeline-creation
  case label in `compute.cpp`), and `coopmat_base` required
  `alpha == 1.0f && !use_c`.
- Every `dispatch_matmul` call site passes either alpha == 1.0f exactly
  (Matmul::eval_gpu, the probs matmuls) or is excluded from coopmat
  regardless (AddMM via `use_c=true`; the bf16 sdpa scores matmul via
  alpha = 1/sqrt(head_dim) != 1). The float-compare admits exactly
  1.0f: -0.0f, NaN, and every other value fell back to the non-coopmat
  kernel, which applies alpha correctly.
- So the kernel only ever ran with alpha == 1, where ignoring alpha is
  correct, and IEEE multiplication by 1.0f (the fix) is exact for every
  finite value, infinity, and NaN payload.

The trap would have sprung the moment someone relaxed the gate - which
is exactly what the bf16 fast-path attention scores wanted
(alpha = 1/sqrt(head_dim)). The fix lands the shader scale and the gate
relaxation as one pairing, and the regression test pins the pairing.

## What the commit contains

1. Shader: the f32 accumulator is scaled by alpha at the single bf16
   store - the same scale point as the f32 sibling kernel's 16x16 tile.
2. `dispatch_matmul`: `coopmat_alpha` admits any alpha for
   `MatmulBF16Coopmat` only; `MatmulF32Coopmat` stays alpha==1-gated
   (`matmul_coopmat.comp` still never reads alpha). `use_c` stays
   required false.
3. Regression test "scaled_dot_product_attention bf16 fast scores scale
   through MatmulBF16Coopmat" (`omarchy_fast_ops`): coopmat-gated shape
   (qL=32, k=D=8, scale=0.25) against the host reference. It pins
   `MLX_OMARCHY_SDPA_BF16_FAST=1` for its scope, so it works under both
   defaults; on a coopmat device the scores matmul routes through
   MatmulBF16Coopmat, elsewhere it exercises the non-coopmat fallback
   (llvmpipe). A missing scale fails by orders of magnitude, not a
   rounding margin.
4. `docs/compatibility.md`: MatmulBF16Coopmat alpha semantics.

No default flip, no docs flag-status change, nothing sdpa-default
related - those belong to strand 2.

## Proof 1: alpha == 1 traffic is bit-identical

Two legs:

- IEEE: alpha==1 multiplies the f32 accumulator exactly (no rounding,
  no subnormal disturbance for a finite multiplier of 1.0f), so the
  drain value is bit-identical to the stored accumulator pre-fix.
- Empirical (M1, fork + stock, 3 reps, quiet machine, driver pinned
  mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 in-window): the
  six canonical Q4 digests and all BF16 pins must hold with the
  fix wheel. The projections, residual adds, and norms - every
  alpha==1 MatmulBF16Coopmat consumer in the model - ride those
  digests. RESULTS: see `matrix/` + `digest-gates.json` (populated by
  the S1 window; verifier: `verify_digest_gates.py`).

## Proof 2: the scaled bf16 matmul computes correct values (f64 RNE)

`matmul_alpha_f64_probe.cpp` (receipt tooling, not shipped): at a
coopmat-gated sdpa shape (qL=64, k=D=8, scale=0.25), GPU output vs a
float64 round-to-nearest attention truth computed from bf16-lifted
inputs, per-route ULP statistics:

- bf16 fast route (fixed coopmat kernel, drain scale)
- f32 composition route

Run: `m1-logs/probe.log` (populated by the S1 window). Pre-fix, the
fast route would show every score displaced by the full alpha factor;
post-fix both routes sit within a few bf16 ULP of truth.

## Proof 3: the regression test fails pre-fix and passes post-fix

Two binaries from the SAME alpha tree build, differing only in the
shader file:

- `/tmp/fast-alpha`: fix as committed -> the test must PASS.
- `/tmp/fast-trap`: `ee8d26fb`'s shader (no drain scale) with the
  relaxed gate -> the test must FAIL by orders of magnitude.

Run: `m1-logs/fastops-trap-alpha-test.log` and
`m1-logs/fastops-alpha-alpha-test.log` (S1 window). The pair proves the
test guards exactly the shader change, not the gate or the test
harness. (Without the gate relaxation the test passes trivially: the
alpha!=1 scores never reach the coopmat kernel.)

## Suites

- llvmpipe (dev box, `VK_DRIVER_FILES=lvp_icd.x86_64.json`,
  `MLX_OMARCHY_ALLOW_NON_APPLE=1`), fix tree at c6c44674:
  - `omarchy_matmul_family_tests`: PASS (log: `llvmpipe-family.log`)
  - `omarchy_runtime_tests`: PASS (log: `llvmpipe-runtime.log`)
  - `omarchy_fast_ops_tests`: 34/34 cases, 1,116,299 assertions
    (log: `llvmpipe-fastops.log`; the alpha test exercises the
    non-coopmat route here - llvmpipe reports
    cooperative_matrix_f32_8=0)
- M1 (fork driver, S1 window): same three suites from the alpha tree;
  logs `m1-fork-*.log`.

## Discovered pre-existing defect (not this branch's)

Mechanism note (added after a peer suggested the throws might be
capability-simulation refusals): they are not. The window-2 phase-A
environment carries no MLX_OMARCHY_CAPS_SIM (no shell/env.d hits, clean
process environ), require_backed returns silently whenever
caps.simulated is false (capability_sim.cpp:206), and its refusal text
("[omarchy] capability simulation '...' dispatches ...") differs from
the logged text ("[omarchy] ScaledDotProductAttention dtype is not
implemented ..."), which is require_float_dtype's unsupported() contract
- i.e. on the real M1 fork driver the f16 sdpa path reaches a dtype
guard with an out whose dtype does not match q. A caps-sim run on
llvmpipe reproduces the same test set through a DIFFERENT mechanism
(plus the decode-native case 759, which passes on real hardware);
the real-hardware mechanism remains to be root-caused in the S1
control run.

`omarchy_fast_ops_tests` on the M1 throws in 8 sdpa cases -
`[omarchy] ScaledDotProductAttention dtype is not implemented
(dtype=float16, rank-5 mask/GQA shapes)`. The f16 path is byte-identical
between main, this fix, and the strand-2 candidate (the env flip only
reaches bf16), and the suite never ran on the M1 before (prior
batteries ran fam + runtime only; llvmpipe passes 34/34). See
`m1-fork-fastops-alpha.log` and the strand-2 receipt for the candidate
tree's identical failure. Owner decision required; not fixed here.

## Files

- `matmul_alpha_f64_probe.cpp` - f64 RNE probe (standalone)
- `verify_digest_gates.py` - post-hoc canonical digest gate
- `m1_window_s1.sh` - the GPU window runner (copied to /tmp on jwm1)
- `m1-logs/`, `matrix/` - S1 outputs
- `llvmpipe-*.log` - dev-box suite logs

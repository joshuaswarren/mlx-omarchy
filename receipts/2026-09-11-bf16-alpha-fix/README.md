# BF16 coopmat alpha fix — strand 1 of the bf16-prefill-attention split (2026-09-11)

Agent: `Bf16AlphaFixAndVerdict`, qualified by `Bf16AlphaQualify`. Branch:
`bf16-alpha-fix` (origin/main at split time + one fix commit, no
performance flip). Carrier commit: `c6c44674` at split time; the branch
was rebased onto origin/main `b4271903` before the M1 window - rebased
carrier `08cdfc49`, receipt `d0507d56`, probe fix `3da2b3ee`. The rebase
delta touches only `receipts/2026-09-11-bf16-prefill-attention` files,
so the library source the llvmpipe proofs ran (at `c6c44674`) is
byte-identical to the rebased tree. All M1 artifacts were rebuilt from
the rebased tip `3da2b3ee` before the window (wheel
`0.32.2.dev202609111816+3da2b3e`, sha256 `b5ed4a4f...` in
`m1-logs/s1-build/`). The performance-flip strand and its verdict live
in `receipts/2026-09-11-bf16-prefill-attention` on branch
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
  digests. RESULT (S1 window): HELD. 36/36 measured cells across
  fork+stock x r1-r3 match the canonical pins - the three Q4 digests
  `7fd25a869ff21678` / `4cc08910089477fd` / `7da83f06ec9f001d`
  (driver-independent) and the BF16 pins (short fork `f26175202f3dabe9`
  / stock `7fc0f968789b1882`, long fork `8690dc83246b39f8` / stock
  `46108ad71157cb4d`, longctx `ff502900d2a179a5`). Verifier
  `verify_digest_gates.py`: all_held=true, rc=0
  (`digest-gates.json`; runs in `matrix/r{1,2,3}-{fork,stock}-alpha/`,
  warmup run discarded). The verifier as staged counted the
  out-of-scope skipped 7b/14b legs as violations; it was fixed to skip
  them and to fail on any missing canonical leg instead.
  Wall-anchored: matrix runs 19:05:28Z-19:08:14Z+run on 2026-09-11,
  machine quiet behind the pinned lock, driver verified in-window.

## Proof 2: the scaled bf16 matmul computes correct values (f64 RNE)

`matmul_alpha_f64_probe.cpp` (receipt tooling, not shipped): at a
coopmat-gated sdpa shape (qL=64, k=D=8, scale=0.25), GPU output vs a
float64 round-to-nearest attention truth computed from bf16-lifted
inputs, per-route ULP statistics for the bf16 fast route (fixed coopmat
kernel, drain scale) and the f32 composition route. Pre-registered
bounds: fast mean_ulp <= 4.0 and fast max_ulp <= 32.

RESULT (S1 window, valid rerun): **GATE FAILED AS PRE-REGISTERED**.
`fast={"exact_frac":0.408203,"mean_ulp":2.6123,"max_ulp":242}`,
`f32comp={"exact_frac":1.000000,"mean_ulp":0.0000,"max_ulp":0}`
(`m1-logs/probe.log`, run 2026-09-11 ~14:14-05:00 under its own lock,
driver verified). mean_ulp 2.6123 passes the <=4.0 bound;
max_ulp 242 fails the <=32 bound, so the probe's CHECKs exit nonzero.

Attribution (recorded evidence, not a re-bound): the outliers are the
fast route's documented bf16 storage of scores and probs
("Scores store bf16 (2^-8 relative rounding per stored score)..." in
the sdpa primitive) amplified by sharp softmax at kL=128 - not the
alpha mechanism. A float64 simulation of the probe's exact seeded
inputs with bf16-rounded scores and probs reproduces the measurement:
sim mean 2.6953 / max 242 / exact_frac 0.358 vs measured 2.6123 / 242 /
0.408 (f64-vs-f32 softmax arithmetic accounts for the residual). The
storage cost exists identically with or without this fix; the fix's
own contribution is clean (mean 2.6 ULP; the trap-shader control at
the same shape measured mean 9668 - see the incident note).
Re-bounding the probe and re-qualifying is an owner decision; per the
standing instruction the failed gate stops the landing.

Incident (recorded): the first S1 probe run used an INVALID binary -
the rebuild script compiled the probe against `libmlx.a` while the
trap build still had the PRE-FIX shader embedded (restore-without-
rebuild ordering bug). It measured the trap at this shape
(`mean_ulp 9668, max_ulp 34046`, `m1-logs/probe-invalid-trapshader.log`
- which doubles as trap evidence at (m=128, n=128, k=8)) and was
disclosed in the window report. The rebuild script was fixed to rebuild
after the shader restore (`scripts-s1/m1_rebuild_rebased.sh`), libmlx.a
was rebuilt with the fixed shader, and the probe was relinked as
`/tmp/probe-alpha-fixed` (sha256 `30e2db7b...`, distinct from the
invalid `64bf23af...`; both in `m1-logs/s1-build/`) before the valid
rerun above.

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

RESULT (S1 window): PASS. trap FAILED the case (rc=1, CHECK
tolerance errors at test_fast_ops.cpp:77) and fast-alpha PASSED it
(rc=0) on the M1 coopmat route - the trap failing also proves the
case really routes through MatmulBF16Coopmat on the M1, so the pass
is not a fallback artifact.

## Suites

- llvmpipe (dev box, `VK_DRIVER_FILES=lvp_icd.x86_64.json`,
  `MLX_OMARCHY_ALLOW_NON_APPLE=1`), fix tree at c6c44674:
  - `omarchy_matmul_family_tests`: PASS (log: `llvmpipe-family.log`)
  - `omarchy_runtime_tests`: PASS (log: `llvmpipe-runtime.log`)
  - `omarchy_fast_ops_tests`: 34/34 cases, 1,116,299 assertions
    (log: `llvmpipe-fastops.log`; the alpha test exercises the
    non-coopmat route here - llvmpipe reports
    cooperative_matrix_f32_8=0)
  RESULT (S1 window, rebased tree, rebuilt binaries):
  - `omarchy_matmul_family_tests`: PASS 21/21 cases, 82,940,459
    assertions (`m1-fork-family-alpha.log`)
  - `omarchy_runtime_tests`: PASS 41/41 cases, 22,681 assertions
    (`m1-fork-runtime-alpha.log`)
  - `omarchy_fast_ops_tests`: PASS 34/34 cases, 1,104,353 assertions,
    0 failed / 0 skipped (`m1-fork-fastops-alpha.log`)
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

UPDATE (S1 window, 2026-09-11): the throws DID NOT REPRODUCE. The full
`omarchy_fast_ops_tests` passed 34/34 with 0 failed / 0 skipped on the
M1 with the rebased tree's rebuilt binaries, and the pre-rebuild
binaries passed 34/34 as well when run the same evening. Whatever
produced the 8 throws in the original staging run (stale build state
or environment contamination - the prepare-mlx mtime lesson is a known
stale-build mechanism), it is not present in a clean rebuild on the
pinned driver. No owner decision is pending on a live defect; the
observation stands as not-reproduced.

## Files

- `matmul_alpha_f64_probe.cpp` - f64 RNE probe (standalone)
- `verify_digest_gates.py` - post-hoc canonical digest gate
- `m1_window_s1.sh` - the GPU window runner (copied to /tmp on jwm1)
- `m1-logs/`, `matrix/` - S1 outputs (`window_s1.log`,
  `digest-gates.json`, suite/probe/fail-proof logs, per-run matrix.json
  + logs; `m1-logs/s1-build/` - rebuilt-artifact receipts)
- `scripts-s1/m1_rebuild_rebased.sh`, `scripts-s1/m1_window_s1.sh`,
  `scripts-s1/m1_probe_rerun.sh` - the exact rebuild, window, and
  probe-rerun procedures (the staged window runner and its corrected
  rebuild script)
- `llvmpipe-*.log` - dev-box suite logs

## S1 window verdict (2026-09-11)

Window: one flock acquisition on /tmp/m1-gpu.lock (13:59:25-05:00),
driver `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1` verified
in-window; probe, fail-proof pair, three suites, then digest matrix
with a discarded fork warmup and fork+stock x r1-r3 (matrix runs
19:05:28Z-19:08:14Z, each 30-60 s, machine otherwise quiet). The probe
rerun took a second short flock at 14:14:41-05:00.

| Gate | Result |
|---|---|
| Fail-proof pair (trap fails, fixed passes) | PASS |
| f64 RNE probe, fast mean <= 4.0 | PASS (2.6123) |
| f64 RNE probe, fast max <= 32 | **FAIL (242)** |
| matmul-family suite (M1) | PASS 21/21 |
| fast-ops suite (M1) | PASS 34/34, 0 skipped |
| runtime suite (M1) | PASS 41/41 |
| Six canonical Q4 digests + BF16 pins, fork+stock x 3 reps | PASS 36/36, verifier rc=0 |

**Verdict: not landable under the pre-registered gates as written** -
the probe's fast max_ulp bound failed. The recorded attribution
(Proof 2) shows the outliers come from the fast route's bf16
score/prob storage rounding, not from the alpha mechanism; the fix
itself measures mean 2.6 ULP against the trap's 9668. Re-bounding the
probe (receipt tooling, not shipped code) and re-qualifying is an
owner decision; nothing in the branch was changed to chase the gate.

Incidents, disclosed (also messaged to the peers at the time): two
full fast-ops suite runs (~45 s each) executed on the M1 GPU outside a
window during the CPU rebuild - the rebuild script's
`--list-test-matching` smoke check executes tests on this doctest
build instead of listing them - once overlapping the peer's arms
window (~13:45); digest arithmetic is deterministic so no gate was
affected, but wall-clock arm metrics in those seconds may be
perturbed. And the first probe run used the invalid trap-shader
binary (Proof 2 incident note); the invalid log is preserved.
# BF16 native-chain per-op mismatch: root cause (receipt-only, no source change)

Date: 2026-09-10
Investigator: Bf16QmmRootCause (omp subagent)
Base: b6d662a8 (`record M1 ANE crossover rejection`), branch `wave/Bf16RootCause`
M1 Linux host: joshuawarren@100.84.184.102 (Apple M1, G13G B1, Asahi + honeykrisp/coopmat)
Native capture host: macOS 26.6.2, M1 Max, stock MLX 0.32.2 (Metal)

## Question

The fixed-input, native-injected M1 receipt
(`receipts/2026-09-10-bf16-native-chain/final-fixed-{fork,stock}`) shows the
BF16 262-token leg reproducing the same per-op mismatches on both the coopmat
fork and stock:

- prefill q_proj 34/234752 (max raw delta 1), v_proj 3/33536 (max 2)
- decode q_proj 106/896 (max 20), k_proj 13/128 (max 7), v_proj 46/128 (max 14328)

while every surrounding op and all downstream ops are bit-exact. Assignment:
trace the first mismatch through QMM/dequant/accumulation and decide whether a
minimal portable kernel fix can make the leg native-exact without regressing
Q4 or stock.

## Answer (root cause)

**The mismatch is not a Linux/omarchy kernel bug, and no kernel fix exists that
would be correct. Given the captured inputs, the omarchy Linux GPU result is
bit-exact against round-to-nearest-even(float64) ground truth on every decode
element, and the macOS Metal capture is the side that deviates. The residual
Metal-vs-exact error is amplified into large bit distances by QKV-bias
cancellation. Matching it bit-for-bit from Vulkan would mean reproducing
another backend's rounding error — a strict regression of exactness, and not
portable across Metal versions. No source change was made.**

### Evidence chain

All steps ran on the M1 with the capture at
`/tmp/bf16chain3-native-fixed/import-2/capture` (72 tensors,
`capture.json` sha256 `2defcc6a…`), model
`~/models/Qwen2.5-0.5B-Instruct-bf16-mlx` (`config.json` sha256
`bc8d587c…`, equal to the capture provenance), CPU-only unless marked GPU.
Full logs: `cpu-evidence.log` in this directory; scripts attached.

1. **QKV projections carry biases.** `mlx_lm` qwen2 builds
   `q_proj/k_proj/v_proj = nn.Linear(..., bias=True)`
   (site-packages `mlx_lm/models/qwen2.py:44-46`); o/mlp projections are
   bias-free. Any reference that omits the bias is wrong for q/k/v. This
   explains why naive x@W.T checks "blow up" exactly and only on q/k/v.

2. **Bias-corrected three-way comparison** (`bias_check.py`), bf16 bit
   distances against RNE(f64) of `x@W.T + b`:

   ```
   op                |Metal-RNE(f64)|  |CPU-RNE(f64)|
   decode.q_proj     max=20  n>1=13   max=0  n>1=0   (Metal exact on 790/896, CPU on 896/896)
   decode.k_proj     max=7   n>1=2    max=0  n>1=0   (115/128 vs 128/128)
   decode.v_proj     max=14328 n>1=6  max=0  n>1=0   (82/128 vs 128/128)
   decode.o_proj     max=0            max=0
   prefill.q_proj    max=1            max=1     (Metal exact 234745/234752, CPU 234721)
   prefill.k_proj    max=0            max=0
   prefill.v_proj    max=4  n>1=1     max=2 n>1=1
   prefill.o_proj    max=13 n>1=3     max=27 n>1=2
   ```

   The CPU-backend bf16 Linear reproduces RNE(f64) **bit-for-bit on every
   decode q/k/v/o element**. Metal deviates on exactly 106/13/46 elements —
   the receipt's mismatch counts and max deltas, element for element.

3. **Mechanism, worst element** (`worst_element.py`): decode.v_proj[50]:
   `acc = -0.0127268`, `bias = +0.0127563` → exact output `+2.956e-05`
   (correct bf16 `0x37f8`). Metal produced `0x0000` (+0.0): its accumulation
   order lands near zero under cancellation and rounds to zero; the omarchy
   sequential-f32 order lands on the correctly rounded value. 4/4 elements
   with |truth| < 1e-3 mismatch; the remaining Metal mismatches are ≤20-ULP
   RNE tie flips of the same kind.

4. **Metal-side error scale** (`deviation_scale.py`): Metal's absolute
   deviation on q/k/v reaches ~1e-3 … 1e-2 against a partial-sum scale of
   ~0.3-11, i.e. ~1e-4..1e-3 relative — orders of magnitude beyond f32
   accumulation-order noise (~1e-7 relative). This bounds Metal's bf16 GEMV
   as accumulating at reduced effective precision (product rounding), not
   merely a different f32 summation order. [INFERENCE on the internal
   mechanism; the deviation measurements themselves are exact.]

5. **Dispatch path at b6d662a8** (`overlay/mlx/backend/omarchy/primitives.cpp`,
   `dispatch_matmul`): decode q/k/v (M=1, N<4096) → `MatmulBF16`
   (`shaders/matmul.comp`: per-output sequential f32 accumulation, one final
   RNE store); decode gate/up/lm_head (M=1, N>=4096) → `MatmulVecBF16`
   (`shaders/matmul_vec.comp`); prefill (M=262, aligned) →
   `MatmulBF16Coopmat`. **QMM/dequant kernels are not in the BF16 path at
   all** (quantized weights only), so "trace through QMM/dequant" terminates
   at: plain bf16 matmul, no dequant stage, single f32 accumulator.

6. **Why q/k/v and not o/gate/up/down**: only the QKV projections have
   biases, and biases put many outputs near cancellation (acc ≈ -b), where
   accumulation-order/precision differences are amplified by ~10^3. The
   bias-free projections have no cancellation structure and are bit-exact
   everywhere.

7. **Prefill (34/3-element q/v legs)**: both Metal and Linux coopmat land
   within <=1-2 bf16 bits of RNE(f64); the receipt mismatches are RNE ties
   flipped by accumulation order. 853f1cd ("Match native BF16 split reduction
   order") already aligned the coopmat SPLIT_K store order; the residue is
   tie noise, not a defect.

8. **fork == stock decode numbers**: identical (106/13/46 with equal max
   deltas) because the decode legs run the shared `matmul.comp`; the fork's
   changes touch only prefill coopmat. Consistent with the receipt.

## Decision

- **No kernel change.** Making Linux bit-match Metal would require
  emulating Metal's reduced-precision accumulation — deliberately adding
  error, non-portable across Metal versions, and a strict regression versus
  the current bit-exact-to-f64 behavior. Q4 paths and stock are untouched.
- **Recommendation**: adopt RNE(f64)/CPU-backend bits as the BF16 reference
  oracle for parity screening, and treat Metal-capture deltas within the
  measured bounds (prefill <=2 bits; decode <=1 bit except near-cancellation
  elements) as acceptable. If bit-parity with macOS Metal is ever a hard
  requirement, it is an upstream mlx-metal kernel change, not an omarchy one.

## Provenance

- Linux wheel that produced the receipt under test:
  `mlx_omarchy-0.32.2.dev202609101856+853f1cd-cp314-cp314-linux_aarch64.whl`,
  sha256 `ea609de5dbf488b72880a83cb407799da9687d6ceec5332c48f4cc0e82892703`,
  built from `~/src/mlx-Bf16Chain3` at 853f1cde620404e964b668a60da581ee05e723a1.
- Native capture: `/tmp/bf16chain3-native-fixed/import-2/capture`,
  `capture.json` sha256 `2defcc6ad88268f024279785742da945a81de279b69698daaf849a20581c2fc6`,
  72 tensors, schema `bf16-native-injected-ops/1`, prompt 262 tokens
  (`prompt_token_sha256 ce0327b4…`).
- Scripts: attached in this directory (`bias_check.py`, `worst_element.py`,
  `deviation_scale.py`, `weight_identity.py`, `resolve.py`,
  `cpu_mlx_check.py`, `inspect2.py`, `gpu_probe.py`).
- Isolated branch `wave/Bf16RootCause` off b6d662a8; no merge to main.

## GPU confirmation

`gpu_probe.py` re-ran the injection on the M1 Linux GPU with the exact
receipt wheel (0.32.2.dev202609101856+853f1cd, the wheel built at
853f1cde620404e964b668a60da581ee05e723a1) under `flock /tmp/m1-gpu.lock`
(`gpu-probe.json`, full records):

```
op               gpu vs native         gpu vs RNE(f64) truth
decode.q_proj    106 mism, max 20      0 mismatches  (bit-exact)
decode.k_proj     13 mism, max  7      0 mismatches  (bit-exact)
decode.v_proj     46 mism, max 14328   0 mismatches  (bit-exact)
decode.o_proj      0 mism, max  0      0 mismatches  (bit-exact)
prefill.q_proj   34 mism, max  1       31 (all <=1 bit RNE ties)
prefill.k_proj    0 mism, max  0       0
prefill.v_proj    3 mism, max  2       4  (<=2 bit RNE ties)
prefill.o_proj    0 mism, max  0       35 (<=13 bits, near-cancellation rows)
```

The gpu-vs-native column reproduces the receipt numbers element-for-element,
and the Linux GPU output equals RNE(f64) exact arithmetic bit-for-bit on all
four decode legs. Caveat: the probe's in-process "CPU stream" comparison is
invalid (device-stream aliasing in a GPU-default process); the authoritative
CPU-backend evidence is `bias_check.py`, which ran in a dedicated CPU process.

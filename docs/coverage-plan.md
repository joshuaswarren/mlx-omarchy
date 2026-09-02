# Coverage plan: 30.1% to 100%

Target: every primitive upstream MLX defines is implemented or composed on the
Omarchy Vulkan backend, and anchored by a test. Coverage is measured by
`tools/gen-compat-matrix.py`, not by claim.

Start state, 2026-09-01, commit `491a048`: 40 of 133 primitives covered.
Gap: 64 primitives raise a named error, 24 have no backend entry, 5 have an
implementation path but no test.

Each wave below is one pushable unit: implementation, tests, regenerated
matrix, and a green gate battery. Waves land on `main` as they finish.

## Wave 1: shape and layout primitives

`AsStrided AsType Broadcast BroadcastAxes Concatenate Contiguous Copy
CustomTransforms Depends DynamicSlice DynamicSliceUpdate ExpandDims Flatten
Full NumberOfElements Pad Reshape Slice Split Squeeze StopGradient Transpose
Unflatten View`

24 primitives. Most already work through the shared `backend/gpu` path or our
copy engine, so this wave is mainly proof: a test anchor per primitive, an
implementation where one is genuinely missing, and a generator status that
tells "handled by the shared GPU path" apart from "nothing runs".

## Wave 2: comparison, logical, and integer elementwise

`Less LessEqual Greater NotEqual LogicalAnd LogicalNot BitwiseBinary
BitwiseInvert DivMod Remainder Power Sign Abs`

13 primitives. `compare.comp` and `int_elementwise.comp` already carry the
word-packed bool transport and the integer path; this extends both.

## Wave 3: transcendental and rounding unaries

`ArcCos ArcCosh ArcSin ArcSinh ArcTan ArcTan2 ArcTanh Cosh Sinh Tan Tanh Erf
ErfInv Expm1 Log1p LogAddExp Ceil Floor Round Conjugate Imag Real`

22 primitives, the largest single block. Most are one case each in the
elementwise switch. `ErfInv` alone unblocks 3,085 upstream python assertions.

## Wave 4: reduction and scan completion

Integer dtypes for `Sum Prod Min Max Any All`, arbitrary-axis reduction to
replace the suffix-only kernel, `cumprod cummax cummin` and reverse scans, and
`Hadamard`.

Integer Sum alone accounts for 4,011 upstream python assertions.

## Wave 5: indexing and scatter

`Scatter ScatterAxis GatherAxis MaskedScatter`, plus wide-row `ArgPartition`
so top-k sampling works over a full vocabulary.

## Wave 6: matmul family

`BlockMaskedMM GatherMM SegmentedMM GatherQMM GatherQQMM QQMatmul`

## Wave 7: linear algebra

`Cholesky Inverse SVD QRF LUF Eig Eigh`

Real algorithms, not wrappers. The largest engineering wave.

## Wave 8: FFT

`FFT`, including the inverse and real variants upstream routes through it.

## Wave 9: fused and custom kernels

`fast::CrossEntropyVJP fast::LayerNormVJP fast::RMSNormVJP fast::ConvertFP8
fast::CustomKernel`, plus native `fast::RMSNorm` and `fast::LayerNorm` to
replace the composed fallbacks, and test anchors for the four composed-untested
primitives.

## Wave 10: distributed

`distributed::AllReduce AllGather Send Recv ReduceScatter`. Single-process
semantics first, which is what upstream reduces to at one rank.

## Wave 11: compiled tape

Root-cause the bfloat16 corruption that the tape currently refuses, then widen
the interpreted subset beyond elementwise.

## Blocked on the owner

These do not gate any wave above and are skipped until Joshua acts.

- ANE bundle execution on Linux needs `/dev/accel` on jwm1: a verified T8103
  device-tree node, a recoverable boot entry, and a KMD load.
- The `eiln/ane` patch series needs approval to open the PR.
- The upstream MLX report on `mx.save` extension handling needs approval to
  file.

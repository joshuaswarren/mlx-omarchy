// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Exact compatibility errors for every primitive this slice does not
// implement (plan R7 / AE2). Shape-manipulation primitives with real
// buffer-level paths are defined in the shared backend/gpu directory; the
// rest must fail with the primitive name, dtype, and shape — never a silent
// CPU fallback.

#include "mlx/backend/omarchy/unsupported.h"

#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"

#define OMARCHY_UNSUPPORTED(func)                                     \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    omarchy::unsupported(#func, out);                                 \
  }

#define OMARCHY_UNSUPPORTED_MULTI(func)                                \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    omarchy::unsupported(#func, outputs.at(0));                        \
  }

#define OMARCHY_USE_FALLBACK(func)    \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  OMARCHY_UNSUPPORTED_MULTI(func)

namespace mlx::core {

OMARCHY_UNSUPPORTED(Abs)
OMARCHY_UNSUPPORTED(Add)
OMARCHY_UNSUPPORTED(AddMM)
OMARCHY_UNSUPPORTED(Arange)
OMARCHY_UNSUPPORTED(ArcCos)
OMARCHY_UNSUPPORTED(ArcCosh)
OMARCHY_UNSUPPORTED(ArcSin)
OMARCHY_UNSUPPORTED(ArcSinh)
OMARCHY_UNSUPPORTED(ArcTan)
OMARCHY_UNSUPPORTED(ArcTan2)
OMARCHY_UNSUPPORTED(ArcTanh)
OMARCHY_UNSUPPORTED(ArgPartition)
OMARCHY_UNSUPPORTED(ArgReduce)
OMARCHY_UNSUPPORTED(ArgSort)
OMARCHY_UNSUPPORTED(BitwiseBinary)
OMARCHY_UNSUPPORTED(BitwiseInvert)
OMARCHY_UNSUPPORTED(BlockMaskedMM)
OMARCHY_UNSUPPORTED(Ceil)
OMARCHY_UNSUPPORTED(Cholesky)
OMARCHY_UNSUPPORTED_MULTI(Compiled)
OMARCHY_UNSUPPORTED(Conjugate)
OMARCHY_UNSUPPORTED(Convolution)
OMARCHY_UNSUPPORTED(Cos)
OMARCHY_UNSUPPORTED(Cosh)
OMARCHY_UNSUPPORTED(Divide)
OMARCHY_UNSUPPORTED_MULTI(DivMod)
OMARCHY_UNSUPPORTED(Equal)
OMARCHY_UNSUPPORTED(Erf)
OMARCHY_UNSUPPORTED(ErfInv)
OMARCHY_UNSUPPORTED(Exp)
OMARCHY_UNSUPPORTED(Expm1)
OMARCHY_UNSUPPORTED(FFT)
OMARCHY_UNSUPPORTED(Floor)
OMARCHY_UNSUPPORTED(Gather)
OMARCHY_UNSUPPORTED(GatherAxis)
OMARCHY_UNSUPPORTED(GatherMM)
OMARCHY_UNSUPPORTED(GatherQMM)
OMARCHY_UNSUPPORTED(GatherQQMM)
OMARCHY_UNSUPPORTED(Greater)
OMARCHY_UNSUPPORTED(GreaterEqual)
OMARCHY_UNSUPPORTED(Hadamard)
OMARCHY_UNSUPPORTED(Imag)
OMARCHY_UNSUPPORTED(Inverse)
OMARCHY_UNSUPPORTED(Less)
OMARCHY_UNSUPPORTED(LessEqual)
OMARCHY_UNSUPPORTED(Load)
OMARCHY_UNSUPPORTED(Log)
OMARCHY_UNSUPPORTED(Log1p)
OMARCHY_UNSUPPORTED(LogicalAnd)
OMARCHY_UNSUPPORTED(LogicalNot)
OMARCHY_UNSUPPORTED(LogicalOr)
OMARCHY_UNSUPPORTED(LogAddExp)
OMARCHY_UNSUPPORTED(LogSumExp)
OMARCHY_UNSUPPORTED_MULTI(LUF)
OMARCHY_UNSUPPORTED(Matmul)
OMARCHY_UNSUPPORTED(Maximum)
OMARCHY_UNSUPPORTED(MaskedScatter)
OMARCHY_UNSUPPORTED(Minimum)
OMARCHY_UNSUPPORTED(Multiply)
OMARCHY_UNSUPPORTED(Negative)
OMARCHY_UNSUPPORTED(NotEqual)
OMARCHY_UNSUPPORTED(Partition)
OMARCHY_UNSUPPORTED(Power)
OMARCHY_UNSUPPORTED_MULTI(QRF)
OMARCHY_UNSUPPORTED(QuantizedMatmul)
OMARCHY_UNSUPPORTED(QQMatmul)
OMARCHY_UNSUPPORTED(RandomBits)
OMARCHY_UNSUPPORTED(Real)
OMARCHY_UNSUPPORTED(Reduce)
OMARCHY_UNSUPPORTED(Remainder)
OMARCHY_UNSUPPORTED(Round)
OMARCHY_UNSUPPORTED(Scan)
OMARCHY_UNSUPPORTED(Scatter)
OMARCHY_UNSUPPORTED(ScatterAxis)
OMARCHY_UNSUPPORTED(SearchSorted)
OMARCHY_UNSUPPORTED(Select)
OMARCHY_UNSUPPORTED(SegmentedMM)
OMARCHY_UNSUPPORTED(Sigmoid)
OMARCHY_UNSUPPORTED(Sign)
OMARCHY_UNSUPPORTED(Sin)
OMARCHY_UNSUPPORTED(Sinh)
OMARCHY_UNSUPPORTED(SliceUpdate)
OMARCHY_UNSUPPORTED(Softmax)
OMARCHY_UNSUPPORTED(Sort)
OMARCHY_UNSUPPORTED(Square)
OMARCHY_UNSUPPORTED(Sqrt)
OMARCHY_UNSUPPORTED(Subtract)
OMARCHY_UNSUPPORTED_MULTI(SVD)
OMARCHY_UNSUPPORTED(Tan)
OMARCHY_UNSUPPORTED(Tanh)
OMARCHY_UNSUPPORTED_MULTI(Eig)
OMARCHY_UNSUPPORTED_MULTI(Eigh)

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    bool force_fused,
    Stream s) {
  if (force_fused) {
    throw std::invalid_argument(
        "[scaled_dot_product_attention] force_fused=True but no fused "
        "kernel is available in the Omarchy backend.");
  }
  return true;
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  return true;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

OMARCHY_USE_FALLBACK(CrossEntropy)
OMARCHY_UNSUPPORTED_MULTI(CrossEntropyVJP)
OMARCHY_USE_FALLBACK(LayerNorm)
OMARCHY_UNSUPPORTED_MULTI(LayerNormVJP)
OMARCHY_USE_FALLBACK(RMSNorm)
OMARCHY_UNSUPPORTED_MULTI(RMSNormVJP)
OMARCHY_USE_FALLBACK(RoPE)
OMARCHY_UNSUPPORTED_MULTI(ScaledDotProductAttention)
OMARCHY_UNSUPPORTED_MULTI(ScaledDotProductAttentionVJP)
OMARCHY_UNSUPPORTED_MULTI(ConvertFP8)
OMARCHY_UNSUPPORTED_MULTI(Quantize)
OMARCHY_UNSUPPORTED_MULTI(CustomKernel)

} // namespace fast

namespace distributed {
OMARCHY_UNSUPPORTED_MULTI(AllReduce)
OMARCHY_UNSUPPORTED_MULTI(AllGather)
OMARCHY_UNSUPPORTED_MULTI(Send)
OMARCHY_UNSUPPORTED_MULTI(Recv)
OMARCHY_UNSUPPORTED_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core

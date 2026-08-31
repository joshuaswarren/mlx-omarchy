// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/unsupported.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

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

namespace {

enum ElementwiseOperation : uint32_t {
  AddOperation,
  MultiplyOperation,
  DivideOperation,
  MaximumOperation,
  ExpOperation,
  SigmoidOperation,
  SquareOperation,
  SqrtOperation,
  RsqrtOperation,
};

allocator::Buffer allocate_omarchy(size_t size) {
  return omarchy::allocator().malloc(size);
}

uint32_t checked_u32(size_t value, const std::string& name, const array& out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    omarchy::unsupported(name + " with more than UINT32_MAX elements", out);
  }
  return static_cast<uint32_t>(value);
}

void require_float_dtype(
    const std::string& name,
    const array& input,
    const array& out,
    omarchy::CommandEncoder& encoder) {
  if ((input.dtype() != float16 && input.dtype() != float32) ||
      input.dtype() != out.dtype()) {
    omarchy::unsupported(name + " dtype", out);
  }
  if (input.dtype() == float16 &&
      (!encoder.device().capabilities().shader_float16 ||
       !encoder.device().capabilities().storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
}

uint32_t checked_item_offset(
    const array& value,
    size_t count,
    const std::string& name,
    const array& out) {
  if (value.offset() % value.itemsize() != 0) {
    omarchy::unsupported(name + " byte offset", out);
  }
  uint64_t offset = value.offset() / value.itemsize();
  if (!omarchy::compute_index_span_fits(offset, count)) {
    omarchy::unsupported(name + " index span", out);
  }
  return static_cast<uint32_t>(offset);
}

omarchy::ComputeBinding binding(const array& value) {
  auto* buffer =
      static_cast<const omarchy::VulkanBuffer*>(value.buffer().ptr());
  return {buffer->buffer, 0, buffer->size};
}

bool is_trailing_broadcast(const array& input, const array& out) {
  if (input.data_size() == 1) {
    return true;
  }
  if (input.ndim() == out.ndim() && input.shape() == out.shape()) {
    size_t expected_stride = 1;
    int axis = input.ndim() - 1;
    for (; axis >= 0 && input.strides()[axis] != 0; --axis) {
      if (input.strides()[axis] != expected_stride) {
        return false;
      }
      expected_stride *= input.shape(axis);
    }
    for (; axis >= 0; --axis) {
      if (input.strides()[axis] != 0) {
        return false;
      }
    }
    return expected_stride == input.data_size();
  }
  if (!input.flags().row_contiguous) {
    return false;
  }
  int first_axis = 0;
  while (first_axis < input.ndim() && input.shape(first_axis) == 1) {
    first_axis++;
  }
  int rank = input.ndim() - first_axis;
  if (rank > out.ndim()) {
    return false;
  }
  for (int axis = 0; axis < rank; ++axis) {
    if (input.shape(first_axis + axis) != out.shape(out.ndim() - rank + axis)) {
      return false;
    }
  }
  return true;
}

uint32_t matrix_group_count(uint32_t dimension) {
  constexpr uint32_t tile_size = 16;
  uint32_t groups = dimension / tile_size + (dimension % tile_size != 0);
  return std::min(groups, omarchy::kMaxComputeGroupCountX);
}

void dispatch_matmul(
    const std::string& name,
    const std::vector<array>& inputs,
    array& out,
    float alpha,
    float beta,
    bool use_c) {
  const array& a = inputs.at(0);
  const array& b = inputs.at(1);
  const array& c = use_c ? inputs.at(2) : out;
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype(name, a, out, encoder);
  require_float_dtype(name, b, out, encoder);
  if (use_c) {
    require_float_dtype(name, c, out, encoder);
  }

  if (a.ndim() < 2 || b.ndim() != 2 || !a.flags().row_contiguous) {
    omarchy::unsupported("matrix layout " + name, out);
  }
  size_t k = a.shape(-1);
  size_t n = b.shape(-1);
  if (b.shape(-2) != k) {
    omarchy::unsupported("matrix dimensions " + name, out);
  }
  bool b_transposed =
      b.strides()[0] == 1 && b.strides()[1] == b.shape(-2);
  bool b_row_contiguous =
      b.strides()[0] == b.shape(-1) && b.strides()[1] == 1;
  if (!b_transposed && !b_row_contiguous) {
    omarchy::unsupported("matrix layout " + name, out);
  }
  if (use_c && !is_trailing_broadcast(c, out)) {
    omarchy::unsupported("broadcast " + name, out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  if (n == 0 || out.size() % n != 0) {
    omarchy::unsupported("matrix dimensions " + name, out);
  }
  size_t m = out.size() / n;
  if ((k != 0 && a.size() != m * k) || b.size() != k * n) {
    omarchy::unsupported("matrix dimensions " + name, out);
  }

  uint32_t output_size = checked_u32(out.size(), name, out);
  omarchy::ComputeParams params;
  params.count = output_size;
  params.lhs_size = checked_u32(a.size(), name, out);
  params.rhs_size = checked_u32(b.size(), name, out);
  params.reduce_size = checked_u32(k, name, out);
  params.output_size = output_size;
  params.aux_size = use_c ? checked_u32(c.data_size(), name, out) : 0;
  params.matrix_m = checked_u32(m, name, out);
  params.matrix_n = checked_u32(n, name, out);
  params.matrix_k = params.reduce_size;
  params.flags = (b_transposed ? 1u : 0u) | (use_c ? 2u : 0u);
  params.alpha = alpha;
  params.beta = beta;

  const array& bound_a = a.size() == 0 ? out : a;
  const array& bound_b = b.size() == 0 ? out : b;
  const array& bound_c = use_c ? c : out;
  params.lhs_offset = checked_item_offset(
      bound_a, bound_a.size(), name, out);
  params.rhs_offset = checked_item_offset(
      bound_b, bound_b.size(), name, out);
  params.aux_offset = checked_item_offset(
      bound_c, use_c ? params.aux_size : out.size(), name, out);
  params.output_offset = checked_item_offset(out, out.size(), name, out);

  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(bound_a), binding(bound_b), binding(bound_c), binding(out)};
  auto kernel = out.dtype() == float16
      ? omarchy::ComputeKernel::MatmulF16
      : omarchy::ComputeKernel::MatmulF32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      matrix_group_count(params.matrix_n),
      matrix_group_count(params.matrix_m));
}

void dispatch_elementwise(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out) {
  const array& lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& rhs = binary ? inputs.at(1) : lhs;
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype(name, lhs, out, encoder);
  require_float_dtype(name, rhs, out, encoder);

  if (!lhs.flags().contiguous || !rhs.flags().contiguous) {
    omarchy::unsupported("non-contiguous " + name, out);
  }

  if (binary) {
    auto binary_type = get_binary_op_type(lhs, rhs);
    if (!is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out)) {
      omarchy::unsupported("broadcast " + name, out);
    }
    set_binary_op_output_data(
        lhs, rhs, out, binary_type, allocate_omarchy);
  } else {
    set_unary_output_data(lhs, out, allocate_omarchy);
  }

  if (out.size() == 0) {
    return;
  }

  uint32_t count = checked_u32(out.size(), name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(lhs), binding(rhs), binding(out)};
  auto kernel = out.dtype() == float16
      ? omarchy::ComputeKernel::ElementwiseF16
      : omarchy::ComputeKernel::ElementwiseF32;
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
}

} // namespace

#define OMARCHY_BINARY(func, operation)                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_elementwise(#func, operation, inputs, out);              \
  }

#define OMARCHY_UNARY(func, operation)                                \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_elementwise(#func, operation, inputs, out);              \
  }

OMARCHY_UNSUPPORTED(Abs)
OMARCHY_BINARY(Add, AddOperation)
void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [alpha, beta] = state();
  dispatch_matmul(name(), inputs, out, alpha, beta, true);
}
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
OMARCHY_BINARY(Divide, DivideOperation)
OMARCHY_UNSUPPORTED_MULTI(DivMod)
OMARCHY_UNSUPPORTED(Equal)
OMARCHY_UNSUPPORTED(Erf)
OMARCHY_UNSUPPORTED(ErfInv)
OMARCHY_UNARY(Exp, ExpOperation)
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
void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_matmul(name(), inputs, out, 1.0f, 0.0f, false);
}
OMARCHY_BINARY(Maximum, MaximumOperation)
OMARCHY_UNSUPPORTED(MaskedScatter)
OMARCHY_UNSUPPORTED(Minimum)
OMARCHY_BINARY(Multiply, MultiplyOperation)
OMARCHY_UNSUPPORTED(Negative)
OMARCHY_UNSUPPORTED(NotEqual)
OMARCHY_UNSUPPORTED(Partition)
OMARCHY_UNSUPPORTED(Power)
OMARCHY_UNSUPPORTED_MULTI(QRF)
OMARCHY_UNSUPPORTED(QuantizedMatmul)
OMARCHY_UNSUPPORTED(QQMatmul)
OMARCHY_UNSUPPORTED(RandomBits)
OMARCHY_UNSUPPORTED(Real)

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto [reduce_type, axes] = state();
  std::string operation_name = name();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype(operation_name, input, out, encoder);

  if (!input.flags().row_contiguous || axes.empty()) {
    omarchy::unsupported("non-contiguous " + operation_name, out);
  }
  int first_axis = input.ndim() - static_cast<int>(axes.size());
  for (int index = 0; index < axes.size(); ++index) {
    if (axes[index] != first_axis + index) {
      omarchy::unsupported("non-suffix " + operation_name, out);
    }
  }
  if (reduce_type != Reduce::Sum && reduce_type != Reduce::Max) {
    omarchy::unsupported(operation_name, out);
  }

  size_t reduce_size = 1;
  for (int axis : axes) {
    reduce_size *= input.shape(axis);
  }
  if (reduce_type == Reduce::Max && reduce_size == 0) {
    omarchy::unsupported("empty Max", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  uint32_t output_size = checked_u32(out.size(), operation_name, out);
  const array& bound_input = input.size() == 0 ? out : input;
  omarchy::ComputeParams params;
  params.count = output_size;
  params.operation = reduce_type == Reduce::Sum ? 0 : 1;
  params.reduce_size = checked_u32(reduce_size, operation_name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(
      bound_input, input.size(), operation_name, out);
  params.output_offset = checked_item_offset(
      out, out.size(), operation_name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(bound_input), binding(bound_input), binding(out)};
  auto kernel = out.dtype() == float16
      ? omarchy::ComputeKernel::ReduceF16
      : omarchy::ComputeKernel::ReduceF32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(output_size));
}

OMARCHY_UNSUPPORTED(Remainder)
OMARCHY_UNSUPPORTED(Round)
OMARCHY_UNSUPPORTED(Scan)
OMARCHY_UNSUPPORTED(Scatter)
OMARCHY_UNSUPPORTED(ScatterAxis)
OMARCHY_UNSUPPORTED(SearchSorted)
OMARCHY_UNSUPPORTED(Select)
OMARCHY_UNSUPPORTED(SegmentedMM)
OMARCHY_UNARY(Sigmoid, SigmoidOperation)
OMARCHY_UNSUPPORTED(Sign)
OMARCHY_UNSUPPORTED(Sin)
OMARCHY_UNSUPPORTED(Sinh)
OMARCHY_UNSUPPORTED(SliceUpdate)
OMARCHY_UNSUPPORTED(Softmax)
OMARCHY_UNSUPPORTED(Sort)
OMARCHY_UNARY(Square, SquareOperation)
void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_elementwise(
      name(), state() ? RsqrtOperation : SqrtOperation, inputs, out);
}
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

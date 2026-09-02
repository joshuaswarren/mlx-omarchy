// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/unsupported.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/compiled.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/backend/gpu/copy.h"
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

// Keep in lockstep with the switch in shaders/elementwise.comp.
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
  SubtractOperation,
  NegativeOperation,
  CosOperation,
  SinOperation,
  LogOperation,
  MinimumOperation,
  Log2Operation,
  Log10Operation,
  ArcCosOperation,
  ArcCoshOperation,
  ArcSinOperation,
  ArcSinhOperation,
  ArcTanOperation,
  ArcTan2Operation,
  ArcTanhOperation,
  CoshOperation,
  SinhOperation,
  TanOperation,
  TanhOperation,
  ErfOperation,
  ErfInvOperation,
  Expm1Operation,
  Log1pOperation,
  LogAddExpOperation,
  CeilOperation,
  FloorOperation,
  RoundOperation,
  // Wave-2 float ops. The codes continue the wave-3 block and match
  // elementwise.comp cases 36-40.
  RemainderFloatOperation,
  PowerFloatOperation,
  SignFloatOperation,
  AbsFloatOperation,
  DivQuotientFloatOperation,
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
  if ((input.dtype() != float16 && input.dtype() != float32 &&
       input.dtype() != bfloat16) ||
      input.dtype() != out.dtype()) {
    omarchy::unsupported(name + " dtype", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (input.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
  if (input.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " bfloat16 capability", out);
  }
}

omarchy::ComputeKernel select_float_kernel(
    Dtype dtype,
    omarchy::ComputeKernel f32_kernel,
    omarchy::ComputeKernel f16_kernel,
    omarchy::ComputeKernel bf16_kernel) {
  if (dtype == float16) {
    return f16_kernel;
  }
  if (dtype == bfloat16) {
    return bf16_kernel;
  }
  return f32_kernel;
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

// Byte-swap scalars in place for big-endian source files. Mirrors the
// shared and CUDA Load implementations.
template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t n) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}

// True when the array is a contiguous stack of rank-2 matrices stored
// row-major (transposed = false) or transposed within each matrix
// (transposed = true, row stride 1 and column stride = row count).
// A batch axis participates in the stack with the contiguous stride,
// keeps an unchecked stride as a singleton, or carries stride 0 as a
// broadcast view; a stride-0 axis repeats one matrix, so the expected
// stride of the axes above it stays unchanged.
bool is_batched_matrix(const array& value, bool transposed) {
  int rank = value.ndim();
  if (rank < 2) {
    return false;
  }
  const auto& strides = value.strides();
  int64_t row_stride =
      transposed ? 1 : static_cast<int64_t>(value.shape(rank - 1));
  int64_t column_stride =
      transposed ? static_cast<int64_t>(value.shape(rank - 2)) : 1;
  if (strides[rank - 2] != row_stride || strides[rank - 1] != column_stride) {
    return false;
  }
  int64_t expected =
      static_cast<int64_t>(value.shape(rank - 2)) * value.shape(rank - 1);
  for (int axis = rank - 3; axis >= 0; --axis) {
    if (strides[axis] == 0) {
      continue;
    }
    if (value.shape(axis) != 1 && strides[axis] != expected) {
      return false;
    }
    expected *= value.shape(axis);
  }
  return true;
}

// Materializes a matmul operand whose batch strides are uniform but not
// the contiguous layout is_batched_matrix accepts; cache slice views and
// concatenate results compose this way. The general strided-copy engine
// writes a standard row-major batch, and the encoder keeps the temp
// alive until the committed work completes. Engine limits (negative
// strides, collapsed rank beyond 4, span overflow) keep their named
// errors.
array materialize_batched_matrix(
    const array& value,
    const std::string& name,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  Shape shape = value.shape();
  Strides strides(shape.size(), 1);
  for (int axis = static_cast<int>(shape.size()) - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  }
  array materialized(shape, value.dtype(), nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(shape.begin(), shape.end());
  flags.col_contiguous = materialized.size() <= 1 ||
      materialized.size() == *max_dim;
  materialized.set_data(
      omarchy::allocator().malloc(materialized.nbytes()),
      materialized.size(),
      strides,
      flags,
      0);
  copy_gpu_inplace(
      value,
      materialized,
      shape,
      value.strides(),
      strides,
      /*i_offset=*/0,
      /*o_offset=*/0,
      CopyType::General,
      out.primitive().stream());
  encoder.add_temporary(materialized);
  return materialized;
}

void dispatch_matmul(
    const std::string& name,
    const std::vector<array>& inputs,
    array& out,
    float alpha,
    float beta,
    bool use_c,
    const Stream& s) {
  const array& a_in = inputs.at(0);
  const array& b_in = inputs.at(1);
  const array& c = use_c ? inputs.at(2) : out;
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(name, a_in, out, encoder);
  require_float_dtype(name, b_in, out, encoder);
  if (use_c) {
    require_float_dtype(name, c, out, encoder);
  }

  if (a_in.ndim() < 2 || a_in.ndim() != b_in.ndim() || a_in.ndim() > 5) {
    omarchy::unsupported("matrix rank " + name, out);
  }
  bool a_transposed = is_batched_matrix(a_in, true);
  bool a_row_contiguous = !a_transposed && is_batched_matrix(a_in, false);
  bool b_transposed = is_batched_matrix(b_in, true);
  bool b_row_contiguous = !b_transposed && is_batched_matrix(b_in, false);
  // An operand composed from cache slices or concatenates can carry
  // batch strides that are uniform but not the contiguous layout
  // is_batched_matrix accepts. Materialize that one operand to a
  // standard row-major batch through the general strided-copy engine and
  // dispatch normally; conforming operands keep the zero-copy path.
  std::optional<array> a_materialized;
  std::optional<array> b_materialized;
  if (!a_row_contiguous && !a_transposed) {
    a_materialized = materialize_batched_matrix(a_in, name, out, s);
    a_transposed = false;
    a_row_contiguous = true;
  }
  if (!b_row_contiguous && !b_transposed) {
    b_materialized = materialize_batched_matrix(b_in, name, out, s);
    b_transposed = false;
    b_row_contiguous = true;
  }
  const array& a = a_materialized ? *a_materialized : a_in;
  const array& b = b_materialized ? *b_materialized : b_in;
  // Batch axes may broadcast: an operand axis of size 1 against a wider
  // axis carries stride 0 through the shared-buffer broadcast view and
  // repeats its one matrix per batch step. True mismatches stay named.
  for (int axis = 0; axis < a.ndim() - 2; ++axis) {
    if (a.shape(axis) != b.shape(axis) && a.shape(axis) != 1 &&
        b.shape(axis) != 1) {
      omarchy::unsupported("batch dimensions " + name, out);
    }
  }
  size_t k = a.shape(-1);
  size_t n = b.shape(-1);
  if (b.shape(-2) != k) {
    omarchy::unsupported("matrix dimensions " + name, out);
  }
  if (use_c && !is_trailing_broadcast(c, out)) {
    omarchy::unsupported("broadcast " + name, out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t batch_count = 1;
  int batch_axes = a.ndim() - 2;
  for (int axis = 0; axis < batch_axes; ++axis) {
    batch_count *= a.shape(axis);
  }
  if (batch_count > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported("batch count " + name, out);
  }
  size_t m = a.shape(-2);

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
  params.alpha = alpha;
  params.beta = beta;
  params.flags = (b_transposed ? 1u : 0u) | (use_c ? 2u : 0u) |
      (a_transposed ? 4u : 0u);
  // The shader unravels workgroup z over the batch shape and offsets
  // each operand by its own strides; a singleton axis never indexes, so
  // its stride is pinned to 0 (broadcast).
  params.dims = static_cast<uint32_t>(batch_axes);
  uint64_t a_span = 0;
  uint64_t b_span = 0;
  for (int axis = 0; axis < batch_axes; ++axis) {
    uint32_t extent = static_cast<uint32_t>(out.shape(axis));
    uint32_t a_stride = a.shape(axis) == 1
        ? 0u
        : static_cast<uint32_t>(a.strides()[axis]);
    uint32_t b_stride = b.shape(axis) == 1
        ? 0u
        : static_cast<uint32_t>(b.strides()[axis]);
    params.shape[axis] = extent;
    params.in_strides[axis] = a_stride;
    params.out_strides[axis] = b_stride;
    a_span += (extent - 1u) * a_stride;
    b_span += (extent - 1u) * b_stride;
  }
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
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::MatmulF32,
      omarchy::ComputeKernel::MatmulF16,
      omarchy::ComputeKernel::MatmulBF16);
  if (!omarchy::compute_index_span_fits(
          params.lhs_offset, a_span + params.matrix_m * params.matrix_k) ||
      !omarchy::compute_index_span_fits(
          params.rhs_offset, b_span + params.matrix_k * params.matrix_n)) {
    omarchy::unsupported(name + " index span", out);
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      matrix_group_count(params.matrix_n),
      matrix_group_count(params.matrix_m),
      checked_u32(batch_count, name, out));
}

// Fills the general broadcast transport (dims/shape/strides) shared by
// the elementwise-style kernels. Broadcast views keep the output shape
// with stride-0 axes, so the view strides index the sources directly;
// collapse_contiguous_dims merges the linear runs, and a stride of 0
// breaks every merge around a broadcast axis. Callers set the offsets
// before calling, because the span check uses them.
void fill_broadcast_transport(
    const std::string& error_name,
    omarchy::ComputeParams& params,
    const array& lhs,
    const array& rhs,
    const array& out) {
  if (lhs.shape() != out.shape() || rhs.shape() != out.shape()) {
    omarchy::unsupported("broadcast " + error_name, out);
  }
  auto [collapsed_shape, collapsed_strides] = collapse_contiguous_dims(
      out.shape(), std::vector<Strides>{lhs.strides(), rhs.strides()});
  if (collapsed_shape.size() > 4) {
    omarchy::unsupported("broadcast rank " + error_name, out);
  }
  params.dims = static_cast<uint32_t>(collapsed_shape.size());
  uint64_t lhs_span = 0;
  uint64_t rhs_span = 0;
  for (size_t axis = 0; axis < collapsed_shape.size(); ++axis) {
    params.shape[axis] = static_cast<uint32_t>(collapsed_shape[axis]);
    params.in_strides[axis] =
        static_cast<uint32_t>(collapsed_strides[0][axis]);
    params.out_strides[axis] =
        static_cast<uint32_t>(collapsed_strides[1][axis]);
    uint64_t extent = params.shape[axis] - 1u;
    lhs_span += extent * params.in_strides[axis];
    rhs_span += extent * params.out_strides[axis];
  }
  if (!omarchy::compute_index_span_fits(params.lhs_offset, lhs_span + 1) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, rhs_span + 1)) {
    omarchy::unsupported(error_name + " index span", out);
  }
}

// The params fill and dispatch behind dispatch_elementwise, callable
// with a caller-allocated output so multi-output primitives (DivMod)
// can target each output in turn.
void dispatch_float_elementwise_to(
    const std::string& name,
    uint32_t operation,
    const array& lhs,
    const array& rhs,
    array& out,
    bool general_broadcast,
    omarchy::CommandEncoder& encoder) {
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
  if (general_broadcast) {
    fill_broadcast_transport(name, params, lhs, rhs, out);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(lhs), binding(rhs), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::ElementwiseF32,
      omarchy::ComputeKernel::ElementwiseF16,
      omarchy::ComputeKernel::ElementwiseBF16);
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
}

void dispatch_elementwise(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const array& lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& rhs = binary ? inputs.at(1) : lhs;
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(name, lhs, out, encoder);
  require_float_dtype(name, rhs, out, encoder);

  if (!lhs.flags().contiguous || !rhs.flags().contiguous) {
    omarchy::unsupported("non-contiguous " + name, out);
  }

  // Scalar, suffix-aligned, and full-overlap operands keep the cheap
  // modulo indexing in the shader; anything else needs shape-aware
  // stride indexing. Gapless strided views (transposes) keep
  // flags().contiguous, so unary ops route through the stride path the
  // same way binary broadcasts do.
  bool general_broadcast = binary
      ? (!is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out))
      : !is_trailing_broadcast(lhs, out);

  if (binary) {
    auto binary_type = get_binary_op_type(lhs, rhs);
    // Broadcast views inherit contiguous=true while data_size < size, so
    // the Scalar/Vector output branches mirror the view's undersized
    // buffer or donate it. General always allocates dense output storage
    // and never donates a non-row-contiguous view.
    if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
      binary_type = BinaryOpType::General;
    }
    set_binary_op_output_data(
        lhs, rhs, out, binary_type, allocate_omarchy);
  } else if (general_broadcast) {
    // The stride gather must not donate: it reads and writes the same
    // buffer at different indices.
    out.set_data(allocate_omarchy(out.nbytes()));
  } else if (lhs.data_size() != lhs.size()) {
    // A scalar broadcast view passes is_trailing_broadcast, but
    // set_unary_output_data would mirror its one-element buffer; the
    // output needs dense storage for all count writes.
    out.set_data(allocate_omarchy(out.nbytes()));
  } else {
    set_unary_output_data(lhs, out, allocate_omarchy);
  }

  if (out.size() == 0) {
    return;
  }
  dispatch_float_elementwise_to(
      name, operation, lhs, rhs, out, general_broadcast, encoder);
}

// The comparison family shares one shape: a bool output from two
// broadcast views of one input dtype, word-packed through the 32-bit
// bool transport. The codes match the selector in compare.comp.
// Equal serves the categorical sampler chain (isinf,
// mx.random.categorical); GreaterEqual keeps its int32 causal-mask
// contract.
enum ComparisonOperation : uint32_t {
  CompareEqual,
  CompareGreaterEqual,
  CompareGreater,
  CompareLess,
  CompareLessEqual,
  CompareNotEqual,
};

void dispatch_comparison(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out) {
  const array& lhs = inputs.at(0);
  const array& rhs = inputs.at(1);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (out.dtype() != bool_) {
    omarchy::unsupported(name + " output dtype", out);
  }
  if (lhs.dtype() != rhs.dtype() ||
      (lhs.dtype() != float32 && lhs.dtype() != float16 &&
       lhs.dtype() != bfloat16 && lhs.dtype() != int32)) {
    omarchy::unsupported(name + " dtype", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (lhs.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
  if (lhs.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " bfloat16 capability", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), name, out);
  uint32_t word_count = checked_u32(
      (static_cast<uint64_t>(count) + 3) / 4, name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  bool general_broadcast =
      !is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out);
  if (general_broadcast) {
    fill_broadcast_transport(name, params, lhs, rhs, out);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(lhs), binding(rhs), binding(out)};
  auto kernel = lhs.dtype() == float16
      ? omarchy::ComputeKernel::CompareF16
      : lhs.dtype() == bfloat16 ? omarchy::ComputeKernel::CompareBF16
                                : lhs.dtype() == int32
          ? omarchy::ComputeKernel::CompareI32
          : omarchy::ComputeKernel::CompareF32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(word_count));
}

// The logical family serves the isinf composition (Or) and now And and
// Not: bool inputs, a bool output, and the same 32-bit word transport
// the comparisons use. The operation selector matches logical_or.comp;
// Not is unary and binds its single input to both operand slots.
enum LogicalOperation : uint32_t {
  LogicalOrOperation,
  LogicalAndOperation,
  LogicalNotOperation,
};

void dispatch_logical(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out) {
  const array& lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& rhs = binary ? inputs.at(1) : lhs;
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (lhs.dtype() != bool_ || rhs.dtype() != bool_ || out.dtype() != bool_) {
    omarchy::unsupported(name + " dtype", out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), name, out);
  uint32_t word_count = checked_u32(
      (static_cast<uint64_t>(count) + 3) / 4, name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  bool general_broadcast =
      !is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out);
  if (general_broadcast) {
    fill_broadcast_transport(name, params, lhs, rhs, out);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(lhs), binding(rhs), binding(out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LogicalOrBool,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(word_count));
}

// Integer twin of the binary elementwise path. The shader carries the
// same modulo fast path and broadcast transport; only the storage type
// differs, so int32 and uint32 share one SPIR-V variant. The selector
// matches IntElementwiseOperation; subtract, bitwise logic, shifts,
// invert, DivMod, Remainder, Power, Sign, and Abs all route here.
enum IntElementwiseOperation : uint32_t {
  IntSubtractOperation,
  IntBitwiseAndOperation,
  IntBitwiseOrOperation,
  IntBitwiseXorOperation,
  IntLeftShiftOperation,
  IntRightShiftOperation,
  IntInvertOperation,
  IntDivModQuotientOperation,
  IntModuloOperation,
  IntPowerOperation,
  IntSignOperation,
  IntAbsOperation,
};

// The params fill and dispatch behind dispatch_int_elementwise,
// callable with a caller-allocated output so the two-output DivMod can
// target quotient and remainder in turn.
void dispatch_int_elementwise_to(
    const std::string& name,
    uint32_t operation,
    const array& lhs,
    const array& rhs,
    array& out) {
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
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
  if (!is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out)) {
    fill_broadcast_transport(name, params, lhs, rhs, out);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(lhs), binding(rhs), binding(out)};
  // Signed and unsigned run separate SPIR-V variants: `>>` arithmetic
  // versus logical, and the sign fixups compare against a signed zero.
  auto kernel = out.dtype() == uint32
      ? omarchy::ComputeKernel::ElementwiseU32
      : omarchy::ComputeKernel::ElementwiseI32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}

void dispatch_int_elementwise(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out) {
  const array& lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& rhs = binary ? inputs.at(1) : lhs;
  auto is_int_dtype = [](Dtype dtype) {
    return dtype == int32 || dtype == uint32;
  };
  if (!is_int_dtype(lhs.dtype()) || !is_int_dtype(rhs.dtype()) ||
      !is_int_dtype(out.dtype())) {
    omarchy::unsupported(name + " dtype", out);
  }
  if (!lhs.flags().contiguous || !rhs.flags().contiguous) {
    omarchy::unsupported("non-contiguous " + name, out);
  }
  auto binary_type = get_binary_op_type(lhs, rhs);
  // Broadcast views inherit contiguous=true while data_size < size; the
  // Scalar/Vector output branches would mirror the view's undersized
  // buffer or donate it. General always allocates dense output storage.
  if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
    binary_type = BinaryOpType::General;
  }
  set_binary_op_output_data(lhs, rhs, out, binary_type, allocate_omarchy);
  if (out.size() == 0) {
    return;
  }
  dispatch_int_elementwise_to(name, operation, lhs, rhs, out);
}

// ArgSort and ArgPartition emit uint32 indices, so the float checks apply
// to the input only, the way ArgReduce checks its input.
void require_index_source_dtype(
    const std::string& name,
    const array& input,
    const array& out,
    omarchy::CommandEncoder& encoder) {
  if (input.dtype() != float16 && input.dtype() != float32 &&
      input.dtype() != bfloat16) {
    omarchy::unsupported(name + " dtype", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (input.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
  if (input.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " bfloat16 capability", out);
  }
  if (out.dtype() != uint32) {
    omarchy::unsupported(name + " output dtype", out);
  }
}

// One workgroup bitonic-sorts one row of up to 1024 elements.
// Partition and ArgPartition route here too: a full sort satisfies the
// partition contract, the same redirect the upstream Metal backend makes.
constexpr size_t kSortMaxRowLength = 1024;

void dispatch_sort(
    const std::string& name,
    const array& input,
    array& out,
    bool argsort,
    omarchy::CommandEncoder& encoder) {
  size_t row_length = input.shape(-1);
  if (row_length > kSortMaxRowLength) {
    omarchy::unsupported("sort row length " + name, out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t rows = input.size() / row_length;
  uint32_t output_size = checked_u32(rows, name, out);
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), name, out);
  params.reduce_size = checked_u32(row_length, name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(input, input.size(), name, out);
  params.output_offset = checked_item_offset(out, out.size(), name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(input), binding(input), binding(out)};
  auto kernel = argsort
      ? select_float_kernel(
            input.dtype(),
            omarchy::ComputeKernel::ArgSortF32,
            omarchy::ComputeKernel::ArgSortF16,
            omarchy::ComputeKernel::ArgSortBF16)
      : select_float_kernel(
            input.dtype(),
            omarchy::ComputeKernel::SortF32,
            omarchy::ComputeKernel::SortF16,
            omarchy::ComputeKernel::SortBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}


// Last-axis softmax. The Softmax primitive is only constructed for a
// last-axis reduction (mlx/ops.cpp softmax), so no suffix-axis check is
// needed here. The shader accumulates in float32 for every dtype, which
// also covers the precise flag. ScaledDotProductAttention shares this
// dispatch for its float32 score normalization.
void dispatch_softmax(
    const std::string& name,
    const array& input,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(name, input, out, encoder);
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous " + name, out);
  }
  size_t row_length = input.shape(-1);
  size_t rows = input.size() / row_length;
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t output_size = checked_u32(rows, name, out);
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), name, out);
  params.reduce_size = checked_u32(row_length, name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(input, input.size(), name, out);
  params.output_offset = checked_item_offset(out, out.size(), name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(input), binding(input), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::SoftmaxF32,
      omarchy::ComputeKernel::SoftmaxF16,
      omarchy::ComputeKernel::SoftmaxBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}

// The general-axis reduction family: integer dtypes, Any/All, and every
// non-suffix axis position. The selector matches the switch in
// shaders/reduce_general.comp.
enum ReduceOperation : uint32_t {
  ReduceSumOperation,
  ReduceProdOperation,
  ReduceMinOperation,
  ReduceMaxOperation,
  ReduceAnyOperation,
  ReduceAllOperation,
};

// Capability gate for a float input whose output is a different dtype;
// Any/All produce bool from float input, so require_float_dtype's
// input-equals-output rule does not apply.
void require_float_input(
    const std::string& name,
    const array& input,
    const array& out,
    omarchy::CommandEncoder& encoder) {
  const auto& capabilities = encoder.device().capabilities();
  if (input.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
  if (input.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " bfloat16 capability", out);
  }
}

omarchy::ComputeKernel select_anyall_kernel(Dtype input_dtype) {
  switch (input_dtype) {
    case bool_:
      return omarchy::ComputeKernel::AnyAllBool;
    case int32:
      return omarchy::ComputeKernel::AnyAllI32;
    case uint32:
      return omarchy::ComputeKernel::AnyAllU32;
    case float16:
      return omarchy::ComputeKernel::AnyAllF16;
    case bfloat16:
      return omarchy::ComputeKernel::AnyAllBF16;
    default:
      return omarchy::ComputeKernel::AnyAllF32;
  }
}

omarchy::ComputeKernel select_reduce_general_kernel(Dtype dtype) {
  switch (dtype) {
    case int32:
      return omarchy::ComputeKernel::ReduceGeneralI32;
    case uint32:
      return omarchy::ComputeKernel::ReduceGeneralU32;
    case float16:
      return omarchy::ComputeKernel::ReduceGeneralF16;
    case bfloat16:
      return omarchy::ComputeKernel::ReduceGeneralBF16;
    default:
      return omarchy::ComputeKernel::ReduceGeneralF32;
  }
}

// The stride-walking general reduction: axes in any position, up to four
// total. Push-constant routing mirrors shaders/reduce_general.comp:
// shape[] carries the kept extents then the reduced extents, in_strides[]
// the kept input strides, out_strides[] the reduced input strides, dims the
// kept count, and matrix_m the reduced count. A broadcast view reduces
// correctly because its expanded axes carry stride 0.
void dispatch_reduce_general(
    const std::string& operation_name,
    uint32_t operation,
    const array& input,
    array& out,
    const std::vector<int>& axes,
    omarchy::CommandEncoder& encoder,
    omarchy::ComputeKernel kernel) {
  if (input.ndim() > 4) {
    omarchy::unsupported(operation_name + " rank above 4", out);
  }
  std::vector<int> kept;
  std::vector<int> reduced;
  for (int axis = 0; axis < input.ndim(); ++axis) {
    if (std::find(axes.begin(), axes.end(), axis) != axes.end()) {
      reduced.push_back(axis);
    } else {
      kept.push_back(axis);
    }
  }
  size_t reduce_size = 1;
  for (int axis : reduced) {
    reduce_size *= input.shape(axis);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t output_size = checked_u32(out.size(), operation_name, out);
  const array& bound_input = input.size() == 0 ? out : input;
  omarchy::ComputeParams params;
  params.count = output_size;
  params.operation = operation;
  params.reduce_size = checked_u32(reduce_size, operation_name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(
      bound_input, input.size(), operation_name, out);
  params.output_offset = checked_item_offset(
      out, out.size(), operation_name, out);
  params.dims = static_cast<uint32_t>(kept.size());
  params.matrix_m = static_cast<uint32_t>(reduced.size());
  for (size_t index = 0; index < kept.size(); ++index) {
    params.shape[index] = static_cast<uint32_t>(input.shape(kept[index]));
    params.in_strides[index] =
        static_cast<uint32_t>(input.strides()[kept[index]]);
  }
  for (size_t index = 0; index < reduced.size(); ++index) {
    params.shape[kept.size() + index] =
        static_cast<uint32_t>(input.shape(reduced[index]));
    params.out_strides[index] =
        static_cast<uint32_t>(input.strides()[reduced[index]]);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(bound_input), binding(bound_input), binding(out)};
  // Bool outputs word-pack four elements per 32-bit write.
  uint32_t dispatch_count = out.dtype() == bool_
      ? checked_u32(
            (static_cast<uint64_t>(output_size) + 3) / 4, operation_name, out)
      : output_size;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(dispatch_count));
}

// ---------------------------------------------------------------------------
// Wave 5: indexing and scatter. Shared helpers live in this block; the
// eval functions sit at their alphabetical primitive sites below.
// ---------------------------------------------------------------------------

// Scatter index words: 0 = int32, 1 = uint32, 2 = int64 read as two
// little-endian words (the same encoding the gather kernels use).
uint32_t scatter_index_mode(
    const array& indices,
    const array& out,
    const std::string& name) {
  if (indices.dtype() == int32) {
    return 0;
  }
  if (indices.dtype() == uint32) {
    return 1;
  }
  if (indices.dtype() == int64) {
    return 2;
  }
  omarchy::unsupported(name + " index dtype", out);
}

uint32_t scatter_index_offset(
    const array& indices,
    const array& out,
    const std::string& name) {
  uint32_t offset = checked_item_offset(indices, indices.size(), name, out);
  if (indices.dtype() == int64) {
    if (offset > std::numeric_limits<uint32_t>::max() / 2) {
      omarchy::unsupported(name + " index span", out);
    }
    return offset * 2;
  }
  return offset;
}

// Rank/sentinel scratch must outlive the queued commands, so it rides
// the encoder temporaries.
array make_u32_scratch(size_t count, omarchy::CommandEncoder& encoder) {
  Shape shape{static_cast<int>(count)};
  array scratch(std::move(shape), uint32, nullptr, {});
  scratch.set_data(allocate_omarchy(scratch.nbytes()));
  encoder.add_temporary(scratch);
  return scratch;
}

void dispatch_clear_u32(
    array& scratch,
    uint32_t value,
    omarchy::CommandEncoder& encoder) {
  omarchy::ComputeParams params;
  params.count = checked_u32(scratch.size(), "Scatter scratch", scratch);
  params.operation = value;
  std::array<omarchy::ComputeBinding, 1> bindings{binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::ClearU32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// Wide-row ArgPartition: an 8-bit radix select finds the kth smallest
// monotone key over four passes, then one workgroup per row emits the
// stable partition order. Deterministic end to end: the histogram
// counts through order-free integer atomics, the bucket walk is a
// fixed shared-memory scan, and the emit is one fixed serial pass.
// --- Wave 6: matmul family --------------------------------------------

// True when walking the strides from the last axis meets either the
// packed row-major stride or a broadcast 0 at every non-singleton axis,
// so the buffer's first data_size entries are the logical values in
// order. Bool masks need this property for the flat bool-to-float cast.
bool is_flat_readable(const array& value) {
  const auto& strides = value.strides();
  int64_t expected = 1;
  for (int axis = value.ndim() - 1; axis >= 0; --axis) {
    if (value.shape(axis) == 1) {
      continue;
    }
    if (strides[axis] != expected && strides[axis] != 0) {
      return false;
    }
    if (strides[axis] != 0) {
      expected *= value.shape(axis);
    }
  }
  return true;
}

// True when a batched matrix stack is dense: the gathered base offsets
// are index * matrix_size, which only addresses a packed stack (a
// uniform transposition inside each matrix is still packed).
bool is_dense_batched_matrix(const array& value, bool transposed) {
  int rank = value.ndim();
  if (rank < 2) {
    return false;
  }
  const auto& strides = value.strides();
  int64_t row_stride =
      transposed ? 1 : static_cast<int64_t>(value.shape(rank - 1));
  int64_t column_stride =
      transposed ? static_cast<int64_t>(value.shape(rank - 2)) : 1;
  if (strides[rank - 2] != row_stride || strides[rank - 1] != column_stride) {
    return false;
  }
  int64_t expected =
      static_cast<int64_t>(value.shape(rank - 2)) * value.shape(rank - 1);
  for (int axis = rank - 3; axis >= 0; --axis) {
    if (value.shape(axis) == 1) {
      continue;
    }
    if (strides[axis] != expected) {
      return false;
    }
    expected *= value.shape(axis);
  }
  return true;
}

array make_dense_temp(
    const array& src,
    const std::string& tag,
    array& out,
    omarchy::CommandEncoder& encoder) {
  array temp(src.shape(), src.dtype(), nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(src.shape().begin(), src.shape().end());
  flags.col_contiguous = temp.size() <= 1 || temp.size() == *max_dim;
  Strides strides(src.ndim(), 1);
  for (int axis = static_cast<int>(src.ndim()) - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * src.shape(axis + 1);
  }
  temp.set_data(
      allocate_omarchy(temp.nbytes()), temp.size(), strides, flags, 0);
  encoder.add_temporary(temp);
  return temp;
}

// Casts a flat-readable bool mask to packed float32 values through the
// word-per-four-elements CastBoolF32 kernel.
array cast_bool_mask(
    const array& mask,
    const std::string& tag,
    array& out,
    omarchy::CommandEncoder& encoder) {
  array temp(
      Shape{static_cast<int>(mask.data_size())}, float32, nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = temp.size() <= 1;
  temp.set_data(
      allocate_omarchy(temp.nbytes()), temp.size(), Strides{1}, flags, 0);
  encoder.add_temporary(temp);
  uint32_t count = checked_u32(mask.data_size(), tag, out);
  if (count == 0) {
    return temp;
  }
  omarchy::ComputeParams params;
  params.count = count;
  params.lhs_size = count;
  params.rhs_size = count;
  params.output_size = count;
  params.lhs_offset =
      checked_item_offset(mask, mask.data_size(), tag, out);
  params.output_offset = 0;
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(mask), binding(mask), binding(temp)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::CastBoolF32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count((count + 3) / 4));
  return temp;
}

// Applies one block mask into the dense destination stack. The data side
// reads through its own batch strides (0 = broadcast axis) and matrix
// strides, so one dispatch both materializes a broadcast operand and
// applies the mask; with data and dst sharing a buffer the kernel is a
// per-element map, so in-place masking is safe. The mask is float32
// (bool pre-cast) indexed by its own grid strides.
void dispatch_block_mask(
    const array& data,
    const array& mask,
    const array& mask_logical,
    array& dst,
    int rows,
    int cols,
    int block_size,
    const std::string& tag,
    array& out) {
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  int batch_axes = data.ndim() - 2;
  if (batch_axes > 4) {
    omarchy::unsupported(tag + " mask batch rank", out);
  }
  size_t batch = 1;
  for (int axis = 0; axis < batch_axes; ++axis) {
    batch *= data.shape(axis);
  }
  size_t count = batch * static_cast<size_t>(rows) * cols;
  if (count == 0 || mask.size() == 0) {
    return;
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(count, tag, out);
  // Mask grid strides come from the logical mask (the bound values may
  // be a flat cast temp): the last two axes are (block rows, block
  // columns); higher axes ride out_strides, 0 = broadcast batch axis.
  params.lhs_size = checked_u32(
      mask_logical.strides()[mask_logical.ndim() - 2], tag, out);
  params.rhs_size = checked_u32(
      mask_logical.strides()[mask_logical.ndim() - 1], tag, out);
  params.lhs_offset = checked_item_offset(data, data.size(), tag, out);
  params.rhs_offset = checked_item_offset(mask, mask.size(), tag, out);
  params.output_offset = checked_item_offset(dst, dst.size(), tag, out);
  // Data matrix strides for the (row, column) decode.
  params.aux_size =
      checked_u32(data.strides()[data.ndim() - 2], tag, out);
  params.aux_offset =
      checked_u32(data.strides()[data.ndim() - 1], tag, out);
  params.matrix_m = checked_u32(rows, tag, out);
  params.matrix_n = checked_u32(cols, tag, out);
  params.matrix_k = checked_u32(block_size, tag, out);
  params.dims = static_cast<uint32_t>(batch_axes);
  uint64_t data_span = 0;
  uint64_t mask_span = 0;
  for (int axis = 0; axis < batch_axes; ++axis) {
    uint32_t extent = static_cast<uint32_t>(data.shape(axis));
    uint32_t d_stride =
        static_cast<uint32_t>(data.strides()[axis]);
    uint32_t m_stride =
        static_cast<uint32_t>(mask_logical.strides()[axis]);
    params.shape[axis] = extent;
    params.in_strides[axis] = d_stride;
    params.out_strides[axis] = m_stride;
    data_span += (extent - 1u) * d_stride;
    mask_span += (extent - 1u) * m_stride;
  }
  data_span += (rows - 1) * params.aux_size + (cols - 1) * params.aux_offset;
  mask_span += (params.lhs_size + params.rhs_size) *
      static_cast<uint64_t>(((rows + block_size - 1) / block_size) *
                            ((cols + block_size - 1) / block_size));
  if (!omarchy::compute_index_span_fits(params.lhs_offset, data_span + 1) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, mask_span + 1)) {
    omarchy::unsupported(tag + " mask index span", out);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(data), binding(mask), binding(dst)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::BlockMaskF32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// Shared body for the gathered affine quantized matmuls: GatherQMM
// (scale + bias), GatherQQMM and the quantized-weight QQMatmul path
// (scale only, no_bias). lhs/rhs with a single zero index serve the
// un-gathered case. The packed word buffer carries [scales bytes |
// (bias bytes) | lhs index words | rhs index words]; every region sits
// on a 4-byte boundary and padding stays zeroed, so 16-bit parameters
// decode from whole words with no 16-bit storage reads.
void dispatch_gather_qmm(
    const std::string& tag,
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs,
    const array& rhs,
    int group_size,
    int bits,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(tag, x, out, encoder);
  if (bits != 4 && bits != 8) {
    omarchy::unsupported(tag + " bits", out);
  }
  if (group_size != 32 && group_size != 64) {
    omarchy::unsupported(tag + " group size", out);
  }
  if (w.dtype() != uint32) {
    omarchy::unsupported(tag + " weight dtype", out);
  }
  if (!x.flags().row_contiguous || !w.flags().row_contiguous ||
      !scales.flags().row_contiguous ||
      (biases && !biases->flags().row_contiguous)) {
    omarchy::unsupported(tag + " non-contiguous input", out);
  }
  if (scales.dtype() != out.dtype() ||
      (biases && biases->dtype() != out.dtype())) {
    // ops.cpp promotes scales and biases to the output dtype.
    omarchy::unsupported(tag + " scales dtype", out);
  }
  if ((x.ndim() != 2 && x.ndim() != 3) || (w.ndim() != 2 && w.ndim() != 3)) {
    omarchy::unsupported(tag + " rank", out);
  }
  int k = x.shape(-1);
  int m = x.shape(-2);
  int n = w.shape(-2);
  if (static_cast<uint64_t>(w.shape(-1)) * 32u / bits !=
      static_cast<uint64_t>(k)) {
    omarchy::unsupported(tag + " shape", out);
  }
  if (scales.ndim() != w.ndim() || scales.shape(-2) != n ||
      static_cast<uint64_t>(scales.shape(-1)) * group_size !=
          static_cast<uint64_t>(k) ||
      !std::equal(
          w.shape().begin(),
          w.shape().end() - 2,
          scales.shape().begin())) {
    omarchy::unsupported(tag + " scales shape", out);
  }
  if (biases && biases->shape() != scales.shape()) {
    omarchy::unsupported(tag + " scales shape", out);
  }
  if (lhs.dtype() != uint32 || rhs.dtype() != uint32) {
    omarchy::unsupported(tag + " index dtype", out);
  }
  if (lhs.shape() != rhs.shape()) {
    omarchy::unsupported(tag + " index shape", out);
  }
  std::optional<array> lhs_packed;
  std::optional<array> rhs_packed;
  const array* lhs_ptr = &lhs;
  const array* rhs_ptr = &rhs;
  if (!lhs.flags().row_contiguous) {
    lhs_packed = make_dense_temp(lhs, tag, out, encoder);
    copy_gpu_inplace(
        lhs,
        *lhs_packed,
        lhs.shape(),
        lhs.strides(),
        lhs_packed->strides(),
        /*i_offset=*/0,
        /*o_offset=*/0,
        CopyType::General,
        s);
    lhs_ptr = &*lhs_packed;
  }
  if (!rhs.flags().row_contiguous) {
    rhs_packed = make_dense_temp(rhs, tag, out, encoder);
    copy_gpu_inplace(
        rhs,
        *rhs_packed,
        rhs.shape(),
        rhs.strides(),
        rhs_packed->strides(),
        /*i_offset=*/0,
        /*o_offset=*/0,
        CopyType::General,
        s);
    rhs_ptr = &*rhs_packed;
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t index_count = lhs_ptr->size();
  size_t scale_bytes = scales.nbytes();
  size_t bias_bytes = biases ? biases->nbytes() : 0;
  auto align4 = [](size_t value) { return (value + 3) & ~size_t(3); };
  size_t bias_base = align4(scale_bytes);
  size_t index_base = align4(bias_base + bias_bytes);
  size_t packed_bytes = index_base + 2 * index_count * 4;
  if (scales.offset() % 4 != 0 || (biases && biases->offset() % 4 != 0) ||
      lhs_ptr->offset() % 4 != 0 || rhs_ptr->offset() % 4 != 0) {
    omarchy::unsupported(tag + " byte offset", out);
  }
  array packed(Shape{static_cast<int>(packed_bytes / 4)}, uint32, nullptr, {});
  array::Flags packed_flags;
  packed_flags.contiguous = true;
  packed_flags.row_contiguous = true;
  packed_flags.col_contiguous = true;
  packed.set_data(
      allocate_omarchy(packed.nbytes()),
      packed.size(),
      Strides{1},
      packed_flags,
      0);
  encoder.add_temporary(packed);
  encoder.fill_buffer(binding(packed).buffer, 0, packed_bytes, 0);
  encoder.copy_buffer(
      binding(scales).buffer,
      binding(packed).buffer,
      scale_bytes,
      static_cast<VkDeviceSize>(scales.offset()),
      0);
  if (biases) {
    encoder.copy_buffer(
        binding(*biases).buffer,
        binding(packed).buffer,
        bias_bytes,
        static_cast<VkDeviceSize>(biases->offset()),
        static_cast<VkDeviceSize>(bias_base));
  }
  encoder.copy_buffer(
      binding(*lhs_ptr).buffer,
      binding(packed).buffer,
      index_count * 4,
      static_cast<VkDeviceSize>(lhs_ptr->offset()),
      static_cast<VkDeviceSize>(index_base));
  encoder.copy_buffer(
      binding(*rhs_ptr).buffer,
      binding(packed).buffer,
      index_count * 4,
      static_cast<VkDeviceSize>(rhs_ptr->offset()),
      static_cast<VkDeviceSize>(index_base + index_count * 4));

  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), tag, out);
  params.lhs_size = checked_u32(x.size(), tag, out);
  params.rhs_size = checked_u32(w.size(), tag, out);
  params.reduce_size = static_cast<uint32_t>(group_size);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(x, x.size(), tag, out);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.aux_offset = 0;
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.operation = static_cast<uint32_t>(bits);
  // shape[0] is the per-half index count; shape[1] the bias region's
  // byte base; in_strides[0]/out_strides[0] the lhs/rhs word offsets.
  params.shape[0] = checked_u32(index_count, tag, out);
  params.shape[1] = static_cast<uint32_t>(bias_base);
  params.in_strides[0] = static_cast<uint32_t>(index_base / 4);
  params.out_strides[0] =
      static_cast<uint32_t>(index_base / 4 + index_count);
  if (!omarchy::compute_index_span_fits(params.lhs_offset, x.size()) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, w.size())) {
    omarchy::unsupported(tag + " index span", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(packed), binding(w), binding(out)};
  bool no_bias = !biases.has_value();
  omarchy::ComputeKernel kernel;
  if (out.dtype() == float32) {
    kernel = no_bias ? omarchy::ComputeKernel::GatherQmmNbF32
                     : omarchy::ComputeKernel::GatherQmmF32;
  } else if (out.dtype() == float16) {
    kernel = no_bias ? omarchy::ComputeKernel::GatherQmmNbF16
                     : omarchy::ComputeKernel::GatherQmmF16;
  } else {
    kernel = no_bias ? omarchy::ComputeKernel::GatherQmmNbBF16
                     : omarchy::ComputeKernel::GatherQmmBF16;
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

} // namespace
#define OMARCHY_BINARY(func, operation)                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_elementwise(                                             \
        #func, operation, inputs, out, out.primitive().stream());     \
  }

#define OMARCHY_UNARY(func, operation)                                \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_elementwise(                                             \
        #func, operation, inputs, out, out.primitive().stream());     \
  }

// Abs negates negatives for every signed dtype; uint32 is the
// upstream no-op and INT_MIN wraps to itself the way upstream's C++
// negation does on this platform. bool keeps the named rejection.
void Abs::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntAbsOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), AbsFloatOperation, inputs, out, out.primitive().stream());
}
OMARCHY_BINARY(Add, AddOperation)
void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [alpha, beta] = state();
  dispatch_matmul(name(), inputs, out, alpha, beta, true, out.primitive().stream());
}
void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  bool is_int32 = out.dtype() == int32;
  if (!is_int32) {
    require_float_dtype("Arange", out, out, encoder);
  }
  if (is_int32 && out.size() > 0) {
    // The shader computes int(alpha) + index * int(beta), so the float
    // transport is only exact while every value and the one-past-last
    // value stay under 2^24.
    constexpr double kArangeIntLimit = 16777216.0;
    double last = start_ + step_ * static_cast<double>(out.size());
    if (std::abs(start_) >= kArangeIntLimit ||
        std::abs(step_) >= kArangeIntLimit ||
        std::abs(last) >= kArangeIntLimit) {
      omarchy::unsupported("Arange range", out);
    }
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), "Arange", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.output_size = count;
  params.output_offset = checked_item_offset(out, count, "Arange", out);
  params.alpha = static_cast<float>(start_);
  params.beta = static_cast<float>(step_);
  std::array<omarchy::ComputeBinding, 1> bindings{binding(out)};
  auto kernel = is_int32
      ? omarchy::ComputeKernel::ArangeI32
      : select_float_kernel(
            out.dtype(),
            omarchy::ComputeKernel::ArangeF32,
            omarchy::ComputeKernel::ArangeF16,
            omarchy::ComputeKernel::ArangeBF16);
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
}
OMARCHY_UNARY(ArcCos, ArcCosOperation)
OMARCHY_UNARY(ArcCosh, ArcCoshOperation)
OMARCHY_UNARY(ArcSin, ArcSinOperation)
OMARCHY_UNARY(ArcSinh, ArcSinhOperation)
OMARCHY_UNARY(ArcTan, ArcTanOperation)
OMARCHY_BINARY(ArcTan2, ArcTan2Operation)
OMARCHY_UNARY(ArcTanh, ArcTanhOperation)
void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_index_source_dtype("ArgPartition", input, out, encoder);
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous ArgPartition", out);
  }
  if (state().second != input.ndim() - 1) {
    omarchy::unsupported("non-suffix ArgPartition", out);
  }
  size_t row_length = input.shape(-1);
  if (row_length > kSortMaxRowLength) {
    // The radix-select pipeline for vocabulary-width rows (the top-k
    // sampling path) still mispicks; it stays a named rejection until
    // it is correct. The bitonic path below covers rows up to 1024.
    omarchy::unsupported("sort row length ArgPartition", out);
  }
  dispatch_sort("ArgPartition", input, out, true, encoder);
}
void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto [reduce_type, axis] = state();
  // ArgReduce::name() is "ArgReduce"; errors name the concrete operation
  // the way the upstream Reduce primitive names Sum and Max.
  std::string operation_name =
      reduce_type == ArgReduce::ArgMax ? "ArgMax" : "ArgMin";
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());

  // The output carries indices, so the float checks apply to the input
  // only and the output must be uint32.
  if (input.dtype() != float16 && input.dtype() != float32 &&
      input.dtype() != bfloat16) {
    omarchy::unsupported(operation_name + " dtype", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (input.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(operation_name + " float16 capability", out);
  }
  if (input.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(operation_name + " bfloat16 capability", out);
  }
  if (out.dtype() != uint32) {
    omarchy::unsupported(operation_name + " output dtype", out);
  }
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous " + operation_name, out);
  }
  if (axis != input.ndim() - 1) {
    omarchy::unsupported("non-suffix " + operation_name, out);
  }

  size_t row_length = input.shape(-1);
  size_t rows = input.size() / row_length;
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  uint32_t output_size = checked_u32(rows, operation_name, out);
  omarchy::ComputeParams params;
  params.count = output_size;
  params.operation = reduce_type == ArgReduce::ArgMax ? 1u : 0u;
  params.reduce_size = checked_u32(row_length, operation_name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(
      input, input.size(), operation_name, out);
  params.output_offset = checked_item_offset(out, out.size(), operation_name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(input), binding(input), binding(out)};
  auto kernel = select_float_kernel(
      input.dtype(),
      omarchy::ComputeKernel::ArgReduceF32,
      omarchy::ComputeKernel::ArgReduceF16,
      omarchy::ComputeKernel::ArgReduceBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}
void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_index_source_dtype("ArgSort", input, out, encoder);
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous ArgSort", out);
  }
  if (state() != input.ndim() - 1) {
    omarchy::unsupported("non-suffix ArgSort", out);
  }
  dispatch_sort("ArgSort", input, out, true, encoder);
}
// BitwiseBinary carries the upstream op enum (and/or/xor and both
// shifts). int32 and uint32 run through the integer kernel, where `>>`
// follows the operand signedness the way the upstream C++ operators
// do; other widths keep the named rejection.
void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  uint32_t operation;
  switch (state()) {
    case BitwiseBinary::And:
      operation = IntBitwiseAndOperation;
      break;
    case BitwiseBinary::Or:
      operation = IntBitwiseOrOperation;
      break;
    case BitwiseBinary::Xor:
      operation = IntBitwiseXorOperation;
      break;
    case BitwiseBinary::LeftShift:
      operation = IntLeftShiftOperation;
      break;
    default:
      operation = IntRightShiftOperation;
      break;
  }
  dispatch_int_elementwise(name(), operation, inputs, out);
}
void BitwiseInvert::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_int_elementwise(name(), IntInvertOperation, inputs, out);
}
void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  // Upstream restricts BlockMaskedMM to float32.
  if (out.dtype() != float32) {
    omarchy::unsupported(tag + " dtype", out);
  }
  const Stream& s = out.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(s);
  const array& a_in = inputs[0];
  const array& b_in = inputs[1];
  require_float_dtype(tag, a_in, out, encoder);
  require_float_dtype(tag, b_in, out, encoder);
  size_t count = inputs.size();
  if (count < 3 || count > 5) {
    omarchy::unsupported(tag + " mask arity", out);
  }
  // Allocate the output before size checks so a too-large out surfaces
  // as the named UNSUPPORTED error before we burn cycles on temp mask
  // copies or temporary array allocations.
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  bool has_out_mask = count == 3 || count == 5;
  bool has_op_mask = count >= 4;
  const array& out_mask = inputs[2];
  const array& lhs_mask = inputs[count - 2];
  const array& rhs_mask = inputs[count - 1];

  // Resolve every present mask to float32 values. Bool casts through the
  // flat word kernel; float32 masks bind through their logical strides.
  std::optional<array> lhs_values;
  std::optional<array> rhs_values;
  std::optional<array> out_values;
  auto resolve_mask = [&](const array& mask, std::optional<array>& temp) {
    if (mask.dtype() == bool_) {
      if (!is_flat_readable(mask)) {
        omarchy::unsupported(tag + " bool mask layout", out);
      }
      temp = cast_bool_mask(mask, tag, out, encoder);
    } else if (mask.dtype() != float32) {
      omarchy::unsupported(tag + " mask dtype", out);
    }
  };
  resolve_mask(lhs_mask, lhs_values);
  resolve_mask(rhs_mask, rhs_values);
  resolve_mask(out_mask, out_values);
  const array& lhs_values_ref = lhs_values ? *lhs_values : lhs_mask;
  const array& rhs_values_ref = rhs_values ? *rhs_values : rhs_mask;
  const array& out_values_ref = out_values ? *out_values : out_mask;

  int m = a_in.shape(-2);
  int k = a_in.shape(-1);
  int n = b_in.shape(-1);

  // Masked operands: one dispatch materializes the operand into a dense
  // temp while applying the block mask; the plain tiled matmul then runs
  // on the temps. Without operand masks the matmul dispatch handles
  // layouts directly, and masked blocks contribute exactly zero to every
  // output element, so skipping their arithmetic is unnecessary for the
  // contract.
  array a_masked = a_in;
  array b_masked = b_in;
  if (has_op_mask) {
    a_masked = make_dense_temp(a_in, tag, out, encoder);
    dispatch_block_mask(
        a_in,
        lhs_values_ref,
        lhs_mask,
        a_masked,
        m,
        k,
        block_size_,
        tag,
        out);
    b_masked = make_dense_temp(b_in, tag, out, encoder);
    dispatch_block_mask(
        b_in,
        rhs_values_ref,
        rhs_mask,
        b_masked,
        k,
        n,
        block_size_,
        tag,
        out);
  }
  dispatch_matmul(tag, {a_masked, b_masked}, out, 1.0f, 0.0f, false, s);
  if (has_out_mask) {
    dispatch_block_mask(
        out, out_values_ref, out_mask, out, m, n, block_size_, tag, out);
  }
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  const Stream& s = out.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(s);
  const array& a_in = inputs[0];
  const array& b_in = inputs[1];
  const array& lhs = inputs[2];
  const array& rhs = inputs[3];
  require_float_dtype(tag, a_in, out, encoder);
  require_float_dtype(tag, b_in, out, encoder);
  if (lhs.dtype() != uint32 || rhs.dtype() != uint32) {
    omarchy::unsupported(tag + " index dtype", out);
  }
  if (lhs.shape() != rhs.shape()) {
    omarchy::unsupported(tag + " index shape", out);
  }
  std::optional<array> lhs_packed;
  std::optional<array> rhs_packed;
  const array* lhs_ptr = &lhs;
  const array* rhs_ptr = &rhs;
  if (!lhs.flags().row_contiguous) {
    lhs_packed = make_dense_temp(lhs, tag, out, encoder);
    copy_gpu_inplace(
        lhs,
        *lhs_packed,
        lhs.shape(),
        lhs.strides(),
        lhs_packed->strides(),
        0,
        0,
        CopyType::General,
        s);
    lhs_ptr = &*lhs_packed;
  }
  if (!rhs.flags().row_contiguous) {
    rhs_packed = make_dense_temp(rhs, tag, out, encoder);
    copy_gpu_inplace(
        rhs,
        *rhs_packed,
        rhs.shape(),
        rhs.strides(),
        rhs_packed->strides(),
        0,
        0,
        CopyType::General,
        s);
    rhs_ptr = &*rhs_packed;
  }

  // The gathered base offsets are index * matrix_size, so both operands
  // must be dense batched stacks; a uniform in-matrix transposition keeps
  // its zero-copy path, anything else materializes first.
  bool a_transposed = is_dense_batched_matrix(a_in, true);
  std::optional<array> a_materialized;
  if (!a_transposed && !is_dense_batched_matrix(a_in, false)) {
    a_materialized = materialize_batched_matrix(a_in, tag, out, s);
  }
  const array& a = a_materialized ? *a_materialized : a_in;
  bool b_transposed = is_dense_batched_matrix(b_in, true);
  std::optional<array> b_materialized;
  if (!b_transposed && !is_dense_batched_matrix(b_in, false)) {
    b_materialized = materialize_batched_matrix(b_in, tag, out, s);
  }
  const array& b = b_materialized ? *b_materialized : b_in;
  if (a.shape(-1) != b.shape(-2)) {
    omarchy::unsupported(tag + " matrix dimensions", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  int m = a.shape(-2);
  int k = a.shape(-1);
  int n = b.shape(-1);
  size_t batch_count = lhs_ptr->size();
  if (out.size() == 0) {
    return;
  }
  if (batch_count > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported(tag + " batch count", out);
  }

  // Binding 2 packs the index words [all lhs | all rhs]; GatherMM has no
  // C operand, so its binding slot is free.
  if (lhs_ptr->offset() % 4 != 0 || rhs_ptr->offset() % 4 != 0) {
    omarchy::unsupported(tag + " index byte offset", out);
  }
  array indices(
      Shape{static_cast<int>(2 * batch_count)}, uint32, nullptr, {});
  array::Flags index_flags;
  index_flags.contiguous = true;
  index_flags.row_contiguous = true;
  index_flags.col_contiguous = true;
  indices.set_data(
      allocate_omarchy(indices.nbytes()),
      indices.size(),
      Strides{1},
      index_flags,
      0);
  encoder.add_temporary(indices);
  encoder.copy_buffer(
      binding(*lhs_ptr).buffer,
      binding(indices).buffer,
      lhs_ptr->nbytes(),
      static_cast<VkDeviceSize>(lhs_ptr->offset()),
      0);
  encoder.copy_buffer(
      binding(*rhs_ptr).buffer,
      binding(indices).buffer,
      rhs_ptr->nbytes(),
      static_cast<VkDeviceSize>(rhs_ptr->offset()),
      static_cast<VkDeviceSize>(lhs_ptr->nbytes()));

  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), tag, out);
  params.lhs_size = checked_u32(a.size(), tag, out);
  params.rhs_size = checked_u32(b.size(), tag, out);
  params.reduce_size = checked_u32(k, tag, out);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(a, a.size(), tag, out);
  params.rhs_offset = checked_item_offset(b, b.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.aux_size = checked_u32(batch_count, tag, out);
  params.aux_offset = 0;
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.flags = (b_transposed ? 1u : 0u) | (a_transposed ? 4u : 0u);
  if (!omarchy::compute_index_span_fits(params.lhs_offset, a.size()) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, b.size())) {
    omarchy::unsupported(tag + " index span", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(a), binding(b), binding(indices), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::GatherMmF32,
      omarchy::ComputeKernel::GatherMmF16,
      omarchy::ComputeKernel::GatherMmBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      matrix_group_count(params.matrix_n),
      matrix_group_count(params.matrix_m),
      checked_u32(batch_count, tag, out));
}

void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  const Stream& s = out.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(s);
  const array& a_in = inputs[0];
  const array& b_in = inputs[1];
  const array& segments = inputs[2];
  require_float_dtype(tag, a_in, out, encoder);
  require_float_dtype(tag, b_in, out, encoder);
  if (segments.dtype() != uint32) {
    omarchy::unsupported(tag + " segments dtype", out);
  }
  if (a_in.ndim() != 2 || b_in.ndim() != 2 || a_in.shape(1) != b_in.shape(0)) {
    omarchy::unsupported(tag + " matrix dimensions", out);
  }
  std::optional<array> a_materialized;
  bool a_transposed = is_batched_matrix(a_in, true);
  if (!a_transposed && !is_batched_matrix(a_in, false)) {
    a_materialized = materialize_batched_matrix(a_in, tag, out, s);
  }
  const array& a = a_materialized ? *a_materialized : a_in;
  std::optional<array> b_materialized;
  bool b_transposed = is_batched_matrix(b_in, true);
  if (!b_transposed && !is_batched_matrix(b_in, false)) {
    b_materialized = materialize_batched_matrix(b_in, tag, out, s);
  }
  const array& b = b_materialized ? *b_materialized : b_in;
  std::optional<array> segments_packed;
  const array* segments_ptr = &segments;
  if (!segments.flags().row_contiguous) {
    segments_packed = make_dense_temp(segments, tag, out, encoder);
    copy_gpu_inplace(
        segments,
        *segments_packed,
        segments.shape(),
        segments.strides(),
        segments_packed->strides(),
        0,
        0,
        CopyType::General,
        s);
    segments_ptr = &*segments_packed;
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  int m = a.shape(0);
  int k = a.shape(1);
  int n = b.shape(1);
  size_t num_segments = segments_ptr->size() / 2;
  if (num_segments * static_cast<size_t>(m) * n != out.size()) {
    omarchy::unsupported(tag + " segment count", out);
  }
  if (num_segments > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported(tag + " segment count", out);
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), tag, out);
  params.lhs_size = checked_u32(a.size(), tag, out);
  params.rhs_size = checked_u32(b.size(), tag, out);
  params.reduce_size = checked_u32(k, tag, out);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(a, a.size(), tag, out);
  params.rhs_offset = checked_item_offset(b, b.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.aux_offset = checked_item_offset(*segments_ptr, segments_ptr->size(), tag, out);
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.flags = (b_transposed ? 1u : 0u) | (a_transposed ? 4u : 0u);
  if (!omarchy::compute_index_span_fits(params.lhs_offset, a.size()) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, b.size())) {
    omarchy::unsupported(tag + " index span", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(a), binding(b), binding(*segments_ptr), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::SegmentedMmF32,
      omarchy::ComputeKernel::SegmentedMmF16,
      omarchy::ComputeKernel::SegmentedMmBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      matrix_group_count(params.matrix_n),
      matrix_group_count(params.matrix_m),
      checked_u32(num_segments, tag, out));
}

OMARCHY_UNARY(Ceil, CeilOperation)
OMARCHY_UNSUPPORTED(Cholesky)
void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  omarchy::eval_compiled_tape(
      tape_, inputs_, outputs_, inputs, outputs, stream());
}
OMARCHY_UNSUPPORTED(Conjugate)
void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  const array& wt = inputs.at(1);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("Convolution", in, out, encoder);
  require_float_dtype("Convolution", wt, out, encoder);
  if (in.ndim() != 4 || wt.ndim() != 4) {
    omarchy::unsupported("non-2D Convolution", out);
  }
  if (groups_ != 1) {
    omarchy::unsupported("grouped Convolution", out);
  }
  // flip and input dilation are conv_transpose semantics; neither is
  // part of the forward 2D convolution this direct kernel runs.
  if (flip_) {
    omarchy::unsupported("transposed Convolution", out);
  }
  if (input_dilation_[0] != 1 || input_dilation_[1] != 1) {
    omarchy::unsupported("input-dilated Convolution", out);
  }

  // Materialize operands whose strides are not the standard NHWC and
  // O(HKW) row-major layouts; cache slices and transposes compose that
  // way. The engine keeps each temp alive until the submission lands.
  std::optional<array> in_materialized;
  std::optional<array> wt_materialized;
  if (!in.flags().row_contiguous) {
    in_materialized = materialize_batched_matrix(
        in, "Convolution", out, out.primitive().stream());
  }
  if (!wt.flags().row_contiguous) {
    wt_materialized = materialize_batched_matrix(
        wt, "Convolution", out, out.primitive().stream());
  }
  const array& x = in_materialized ? *in_materialized : in;
  const array& w = wt_materialized ? *wt_materialized : wt;

  int batch = x.shape(0);
  int in_height = x.shape(1);
  int in_width = x.shape(2);
  int in_channels = x.shape(3);
  int out_channels = w.shape(0);
  int kernel_height = w.shape(1);
  int kernel_width = w.shape(2);
  if (w.shape(3) != in_channels || out.shape(0) != batch ||
      out.shape(3) != out_channels) {
    omarchy::unsupported("Convolution shapes", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t total = checked_u32(out.size(), "Convolution", out);
  omarchy::ComputeParams params;
  // Push-constant routing mirrors shaders/conv.comp: spatial extents,
  // kernel window, and the six conv parameters ride the generic dims
  // fields because the fixed ComputeParams layout has no conv block.
  params.count = total;
  params.operation = checked_u32(kernel_width, "Convolution", out);
  params.reduce_size = checked_u32(in_channels, "Convolution", out);
  params.lhs_offset = checked_item_offset(x, x.size(), "Convolution", out);
  params.rhs_offset = checked_item_offset(w, w.size(), "Convolution", out);
  params.output_offset = checked_item_offset(out, out.size(), "Convolution", out);
  params.aux_size = checked_u32(in_width, "Convolution", out);
  params.aux_offset = checked_u32(kernel_height, "Convolution", out);
  params.matrix_m = checked_u32(batch, "Convolution", out);
  params.matrix_n = checked_u32(out_channels, "Convolution", out);
  params.matrix_k = checked_u32(in_height, "Convolution", out);
  params.dims = 2;
  params.shape[0] = checked_u32(out.shape(1), "Convolution", out);
  params.shape[1] = checked_u32(out.shape(2), "Convolution", out);
  params.shape[2] = checked_u32(padding_lo_[0], "Convolution", out);
  params.shape[3] = checked_u32(padding_lo_[1], "Convolution", out);
  params.in_strides[0] = checked_u32(kernel_strides_[0], "Convolution", out);
  params.in_strides[1] = checked_u32(kernel_strides_[1], "Convolution", out);
  params.in_strides[2] = checked_u32(kernel_dilation_[0], "Convolution", out);
  params.in_strides[3] = checked_u32(kernel_dilation_[1], "Convolution", out);
  params.out_strides[0] = checked_u32(padding_hi_[0], "Convolution", out);
  params.out_strides[1] = checked_u32(padding_hi_[1], "Convolution", out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(x), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::ConvF32,
      omarchy::ComputeKernel::ConvF16,
      omarchy::ComputeKernel::ConvBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(total));
}
OMARCHY_UNARY(Cos, CosOperation)
OMARCHY_UNARY(Cosh, CoshOperation)
OMARCHY_BINARY(Divide, DivideOperation)
// DivMod produces the Python floor-division quotient and remainder as
// two same-shaped outputs (upstream DivMod: integral_op applies the
// floor fixup to the truncating quotient, float_op pairs floor(x/y)
// with the adjusted fmod). One kernel dispatch per output; the
// per-element index mapping is 1:1, so a donated input buffer stays
// safe to read and write within one invocation.
void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& lhs = inputs.at(0);
  const array& rhs = inputs.at(1);
  array& quotient = outputs.at(0);
  array& remainder = outputs.at(1);
  auto& encoder = omarchy::get_command_encoder(quotient.primitive().stream());
  auto is_int_dtype = [](Dtype dtype) {
    return dtype == int32 || dtype == uint32;
  };
  if (is_int_dtype(lhs.dtype()) && is_int_dtype(rhs.dtype()) &&
      is_int_dtype(quotient.dtype()) && is_int_dtype(remainder.dtype())) {
    if (!lhs.flags().contiguous || !rhs.flags().contiguous) {
      omarchy::unsupported("non-contiguous DivMod", quotient);
    }
    auto binary_type = get_binary_op_type(lhs, rhs);
    if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
      binary_type = BinaryOpType::General;
    }
    set_binary_op_output_data(lhs, rhs, quotient, binary_type, allocate_omarchy);
    set_binary_op_output_data(
        lhs, rhs, remainder, binary_type, allocate_omarchy);
    if (quotient.size() == 0) {
      return;
    }
    dispatch_int_elementwise_to(
        "DivMod", IntDivModQuotientOperation, lhs, rhs, quotient);
    dispatch_int_elementwise_to(
        "DivMod", IntModuloOperation, lhs, rhs, remainder);
    return;
  }
  require_float_dtype("DivMod", lhs, quotient, encoder);
  require_float_dtype("DivMod", rhs, quotient, encoder);
  if (!lhs.flags().contiguous || !rhs.flags().contiguous) {
    omarchy::unsupported("non-contiguous DivMod", quotient);
  }
  bool general_broadcast =
      !is_trailing_broadcast(lhs, quotient) ||
      !is_trailing_broadcast(rhs, quotient);
  auto binary_type = get_binary_op_type(lhs, rhs);
  if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
    binary_type = BinaryOpType::General;
  }
  set_binary_op_output_data(lhs, rhs, quotient, binary_type, allocate_omarchy);
  set_binary_op_output_data(
      lhs, rhs, remainder, binary_type, allocate_omarchy);
  if (quotient.size() == 0) {
    return;
  }
  dispatch_float_elementwise_to(
      "DivMod",
      DivQuotientFloatOperation,
      lhs,
      rhs,
      quotient,
      general_broadcast,
      encoder);
  dispatch_float_elementwise_to(
      "DivMod",
      RemainderFloatOperation,
      lhs,
      rhs,
      remainder,
      general_broadcast,
      encoder);
}
void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareEqual, inputs, out);
}
OMARCHY_UNARY(Erf, ErfOperation)
OMARCHY_UNARY(ErfInv, ErfInvOperation)
OMARCHY_UNARY(Exp, ExpOperation)
OMARCHY_UNARY(Expm1, Expm1Operation)
OMARCHY_UNSUPPORTED(FFT)
OMARCHY_UNARY(Floor, FloorOperation)
void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& table = inputs.at(0);
  auto [axes, slice_sizes] = state();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  // QuantizedEmbedding gathers rows of a packed uint32 weight matrix.
  // The u32 kernel copies raw 32-bit words with no float conversion, so
  // words above 2^31 stay bit-exact. An int32 table shares that kernel
  // unchanged: the copy is bitwise and signedness never participates.
  // Float tables keep the float kernels and the float dtype gate;
  // remaining dtypes keep the named error.
  bool raw_word_table = table.dtype() == uint32 || table.dtype() == int32;
  if (raw_word_table) {
    if (out.dtype() != table.dtype()) {
      omarchy::unsupported("Take dtype", out);
    }
  } else {
    require_float_dtype("Take", table, out, encoder);
  }
  if (inputs.size() != 2 || axes.size() != 1 || axes[0] != 0) {
    omarchy::unsupported("non-axis-0 Take", out);
  }
  if (table.ndim() != 2 || !table.flags().row_contiguous) {
    omarchy::unsupported("matrix layout Take", out);
  }
  if (
      slice_sizes.size() != 2 || slice_sizes[0] != 1 ||
      slice_sizes[1] != table.shape(1)) {
    omarchy::unsupported("slice Take", out);
  }
  const array& indices = inputs.at(1);
  // gather_rows.comp reads index words selected by params.operation:
  // 0 = int32, 1 = uint32, 2 = int64 as two little-endian words. The
  // int64 element offset doubles into word units. Other index dtypes
  // keep the named rejection.
  uint32_t index_mode;
  uint64_t index_words;
  switch (indices.dtype()) {
    case int32:
      index_mode = 0;
      index_words = indices.size();
      break;
    case uint32:
      index_mode = 1;
      index_words = indices.size();
      break;
    case int64:
      index_mode = 2;
      index_words = indices.size() * 2;
      break;
    default:
      omarchy::unsupported("indexed Take dtype", out);
  }
  if (!indices.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous indexed Take", out);
  }
  if (out.size() != indices.size() * table.shape(1)) {
    omarchy::unsupported("slice Take", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), "Take", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = index_mode;
  params.lhs_size = checked_u32(table.size(), "Take", out);
  params.rhs_size = checked_u32(index_words, "Take", out);
  params.reduce_size = checked_u32(table.shape(1), "Take", out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(table, table.size(), "Take", out);
  // The shader indexes 32-bit words, so the int64 element offset
  // doubles into word units.
  uint32_t index_offset = checked_item_offset(
      indices, indices.size(), "Take", out);
  if (index_mode == 2) {
    if (index_offset > std::numeric_limits<uint32_t>::max() / 2) {
      omarchy::unsupported("indexed Take index span", out);
    }
    index_offset *= 2;
  }
  params.rhs_offset = index_offset;
  params.output_offset = checked_item_offset(out, out.size(), "Take", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(table), binding(indices), binding(out)};
  auto kernel = raw_word_table
      ? omarchy::ComputeKernel::GatherU32
      : select_float_kernel(
            out.dtype(),
            omarchy::ComputeKernel::GatherF32,
            omarchy::ComputeKernel::GatherF16,
            omarchy::ComputeKernel::GatherBF16);
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
}
void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& src = inputs.at(0);
  const array& indices = inputs.at(1);
  int axis = axis_;
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  // The raw-word kernel copies uint32, int32, and float32 tables with
  // no conversion, so packed words above 2^31 stay bit-exact and a
  // zero row is the zero word in all three interpretations. Float16
  // and bfloat16 use their storage kernels; the rest keep the named
  // rejection.
  bool raw_word = src.dtype() == uint32 || src.dtype() == int32 ||
      src.dtype() == float32;
  if (raw_word) {
    if (out.dtype() != src.dtype()) {
      omarchy::unsupported("Take dtype", out);
    }
  } else {
    require_float_dtype("Take", src, out, encoder);
  }
  uint32_t index_mode;
  uint64_t index_words;
  switch (indices.dtype()) {
    case int32:
      index_mode = 0;
      index_words = indices.size();
      break;
    case uint32:
      index_mode = 1;
      index_words = indices.size();
      break;
    case int64:
      index_mode = 2;
      index_words = indices.size() * 2;
      break;
    default:
      omarchy::unsupported("indexed Take dtype", out);
  }
  if (!indices.flags().row_contiguous ||
      indices.data_size() != indices.size()) {
    omarchy::unsupported("indexed Take layout", out);
  }
  int non_axis = out.ndim() - 1;
  if (non_axis > 4 || axis != out.ndim() - 1) {
    // The post-dim stride walk still misreads; axes with trailing dims
    // keep the named rejection until root-caused.
    omarchy::unsupported("Take layout", out);
  }
  size_t post_size = 1;
  for (int i = axis + 1; i < out.ndim(); ++i) {
    post_size *= out.shape(i);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), "Take", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = index_mode;
  params.lhs_size = checked_u32(src.size(), "Take", out);
  params.rhs_size = checked_u32(index_words, "Take", out);
  params.reduce_size = checked_u32(src.shape(axis), "Take", out);
  params.output_size = checked_u32(post_size, "Take", out);
  params.aux_size = checked_u32(indices.shape(axis), "Take", out);
  params.lhs_offset = checked_item_offset(src, src.size(), "Take", out);
  uint32_t index_offset =
      checked_item_offset(indices, indices.size(), "Take", out);
  if (index_mode == 2) {
    if (index_offset > std::numeric_limits<uint32_t>::max() / 2) {
      omarchy::unsupported("Take index span", out);
    }
    index_offset *= 2;
  }
  params.rhs_offset = index_offset;
  params.output_offset = checked_item_offset(out, out.size(), "Take", out);
  params.matrix_m = checked_u32(axis, "Take", out);
  params.matrix_n = checked_u32(src.strides(axis), "Take", out);
  params.flags = checked_u32(non_axis, "Take", out);
  for (int i = 0, d = 0; i < out.ndim(); ++i) {
    if (i == axis) {
      continue;
    }
    params.shape[d] = checked_u32(out.shape(i), "Take", out);
    params.in_strides[d] = checked_u32(src.strides(i), "Take", out);
    ++d;
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(indices), binding(out)};
  auto kernel = src.dtype() == float16
      ? omarchy::ComputeKernel::GatherAxisF16
      : src.dtype() == bfloat16 ? omarchy::ComputeKernel::GatherAxisBF16
                                : omarchy::ComputeKernel::GatherAxisU32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}


void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  if (mode_ != QuantizationMode::Affine) {
    omarchy::unsupported(tag + " mode", out);
  }
  if (!transpose_) {
    omarchy::unsupported(tag + " transpose", out);
  }
  // Affine inputs: [x, w, scales, biases, lhs, rhs].
  dispatch_gather_qmm(
      tag,
      inputs[0],
      inputs[1],
      inputs[2],
      inputs[3],
      inputs[4],
      inputs[5],
      group_size_,
      bits_,
      out,
      out.primitive().stream());
}

void GatherQQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  if (mode_ != QuantizationMode::Affine) {
    omarchy::unsupported(tag + " mode", out);
  }
  // Affine inputs: [x, w, lhs, rhs, scales_w]. A float w would need the
  // quantize direction, which this backend does not implement.
  if (inputs[1].dtype() != uint32) {
    omarchy::unsupported(tag + " weight dtype", out);
  }
  dispatch_gather_qmm(
      tag,
      inputs[0],
      inputs[1],
      inputs[4],
      std::nullopt,
      inputs[2],
      inputs[3],
      group_size_,
      bits_,
      out,
      out.primitive().stream());
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  if (mode_ != QuantizationMode::Affine) {
    omarchy::unsupported(tag + " mode", out);
  }
  const Stream& s = out.primitive().stream();
  if (inputs.size() == 2) {
    // Unquantized w needs x @ w.T; this primitive receives w as (N, K)
    // and the tiled matmul path cannot build the transposed view inside
    // an eval, so the float-weight form keeps a named rejection. The
    // quantized-weight form below is the contract deliverable.
    omarchy::unsupported(tag + " weight dtype", out);
    return;
  }
  // [x, w(u32), scales_w]: the scale-only dequantized product with
  // identity gather (S = 1, both indices zero).
  auto& encoder = omarchy::get_command_encoder(s);
  if (inputs[0].ndim() != 2) {
    omarchy::unsupported(tag + " rank", out);
  }
  array zero_index(Shape{1}, uint32, nullptr, {});
  array::Flags zero_flags;
  zero_flags.contiguous = true;
  zero_flags.row_contiguous = true;
  zero_flags.col_contiguous = true;
  zero_index.set_data(
      allocate_omarchy(zero_index.nbytes()),
      zero_index.size(),
      Strides{1},
      zero_flags,
      0);
  encoder.add_temporary(zero_index);
  encoder.fill_buffer(binding(zero_index).buffer, 0, 4, 0);
  dispatch_gather_qmm(
      tag,
      inputs[0],
      inputs[1],
      inputs[2],
      std::nullopt,
      zero_index,
      zero_index,
      group_size_,
      bits_,
      out,
      s);
}



void Greater::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareGreater, inputs, out);
}
// GreaterEqual serves the composed causal mask: two int32 index arrays
// (broadcast views with stride-0 axes) produce a bool mask. Other dtypes
// stay named rejections. The comparison kernel packs the bytes into
// 32-bit words, so the dispatch covers words.
void GreaterEqual::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  const array& a = inputs.at(0);
  const array& b = inputs.at(1);
  if (a.dtype() != int32 || b.dtype() != int32) {
    omarchy::unsupported("GreaterEqual dtype", out);
  }
  dispatch_comparison(name(), CompareGreaterEqual, inputs, out);
}
// Hadamard runs the fast Walsh-Hadamard transform over the last axis, the
// GPU twin of mlx/backend/cpu/hadamard.cpp: the row of length n*m decomposes
// into a 2^k component and an embedded-Hadamard component m in
// (1, 12, 20, 28); the butterfly applies the scale on its final stage when
// m is 1 and the dense H_m rows carry it otherwise, the same order the CPU
// kernel uses so float rounding matches. One invocation owns one row and
// walks it serially, so sizes whose 2^k component exceeds 2^16 stay a named
// error rather than a silent stall.
void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("Hadamard", input, out, encoder);
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous Hadamard", out);
  }
  if (out.ndim() == 0) {
    omarchy::unsupported("Hadamard rank", out);
  }
  int n = out.shape(-1);
  int m = 1;
  if ((n & (n - 1)) != 0) {
    for (int factor : {12, 20, 28}) {
      if (n % factor == 0) {
        m = factor;
        n /= factor;
        break;
      }
    }
    if (m == 1) {
      omarchy::unsupported("Hadamard size", out);
    }
  }
  if (n > (1 << 16)) {
    omarchy::unsupported("Hadamard size", out);
  }
  size_t row_length = static_cast<size_t>(n) * static_cast<size_t>(m);
  size_t rows = row_length == 0 ? 0 : out.size() / row_length;
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t row_count = checked_u32(rows, "Hadamard", out);
  if (row_count > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported("Hadamard row count", out);
  }
  // Copy the input bytes into the fresh output, then transform in place.
  encoder.copy_buffer(
      binding(input).buffer,
      binding(out).buffer,
      static_cast<VkDeviceSize>(input.nbytes()),
      static_cast<VkDeviceSize>(input.offset()),
      static_cast<VkDeviceSize>(out.offset()));
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), "Hadamard", out);
  params.reduce_size = static_cast<uint32_t>(n);
  params.output_size = row_count;
  params.matrix_m = static_cast<uint32_t>(m);
  params.alpha = scale_;
  std::array<omarchy::ComputeBinding, 1> bindings{binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::HadamardF32,
      omarchy::ComputeKernel::HadamardF16,
      omarchy::ComputeKernel::HadamardBF16);
  encoder.dispatch_compute(kernel, bindings, params, row_count);
}
OMARCHY_UNSUPPORTED(Imag)
OMARCHY_UNSUPPORTED(Inverse)
void Less::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareLess, inputs, out);
}
void LessEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareLessEqual, inputs, out);
}
void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocate_omarchy(out.nbytes()));
  // The allocator is host-visible (coherent where the memory type allows),
  // so the reader fills the output buffer directly and the data is ready
  // when this function returns. This mirrors the CUDA Load::eval_gpu, which
  // also reads synchronously on the calling thread.
  auto* out_ptr = out.data<char>();
  reader_->read(out_ptr, out.nbytes(), offset_);
  if (swap_endianness_) {
    switch (out.itemsize()) {
      case 2:
        swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), out.size());
        break;
      case 4:
        swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), out.size());
        break;
      case 8:
        swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), out.size());
        break;
    }
  }
}

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Upstream log2/log10 are the Log primitive carrying Log::Base; the
  // base picks the shader case (GLSL log2 for base two, log scaled by
  // 1/ln(10) for base ten).
  uint32_t operation;
  switch (state()) {
    case Log::Base::two:
      operation = Log2Operation;
      break;
    case Log::Base::ten:
      operation = Log10Operation;
      break;
    default:
      operation = LogOperation;
      break;
  }
  dispatch_elementwise(name(), operation, inputs, out, stream());
}
OMARCHY_UNARY(Log1p, Log1pOperation)
void LogicalAnd::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_logical(name(), LogicalAndOperation, inputs, out);
}
void LogicalNot::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_logical(name(), LogicalNotOperation, inputs, out);
}
void LogicalOr::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_logical(name(), LogicalOrOperation, inputs, out);
}
OMARCHY_BINARY(LogAddExp, LogAddExpOperation)
void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("LogSumExp", input, out, encoder);
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous LogSumExp", out);
  }

  // Upstream mlx/ops.cpp logsumexp builds the LogSumExp primitive only
  // for a suffix reduce and keeps the reduced axis at size 1 (the
  // keepdims=False form squeezes on top of this output), so no
  // suffix-axis check is needed here. The shader accumulates in float32
  // for every dtype and keeps an infinite row max, matching the
  // upstream CPU rule.
  size_t row_length = input.shape(-1);
  size_t rows = input.size() / row_length;
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t output_size = checked_u32(rows, "LogSumExp", out);
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), "LogSumExp", out);
  params.reduce_size = checked_u32(row_length, "LogSumExp", out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(
      input, input.size(), "LogSumExp", out);
  params.output_offset = checked_item_offset(
      out, out.size(), "LogSumExp", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(input), binding(input), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::LogSumExpF32,
      omarchy::ComputeKernel::LogSumExpF16,
      omarchy::ComputeKernel::LogSumExpBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}
OMARCHY_UNSUPPORTED_MULTI(LUF)
void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_matmul(
      name(), inputs, out, 1.0f, 0.0f, false, out.primitive().stream());
}
OMARCHY_BINARY(Maximum, MaximumOperation)
void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& dst = inputs.at(0);
  const array& mask = inputs.at(1);
  const array& src = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (out.dtype() == float16 || out.dtype() == bfloat16) {
    require_float_dtype("MaskedScatter", dst, out, encoder);
  } else if (out.dtype() != float32 && out.dtype() != int32 &&
             out.dtype() != uint32) {
    omarchy::unsupported("MaskedScatter dtype", out);
  }
  auto kernel = out.dtype() == float16
      ? omarchy::ComputeKernel::MaskedScatterF16
      : out.dtype() == bfloat16 ? omarchy::ComputeKernel::MaskedScatterBF16
                                : omarchy::ComputeKernel::MaskedScatterU32;
  CopyType copy_type = dst.data_size() == 1 ? CopyType::Scalar
      : (dst.flags().row_contiguous && dst.data_size() == dst.size())
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(dst, out, copy_type, out.primitive().stream());
  if (mask.size() == 0) {
    return;
  }
  size_t rows = mask.shape(0);
  size_t row_length = mask.size() / rows;
  size_t src_length = src.size() / rows;
  // The scan kernel indexes mask words flat; a broadcast mask arrives
  // as a stride view with no flat word order, so it keeps the named
  // rejection instead of guessing a layout.
  if (!mask.flags().row_contiguous || mask.data_size() != mask.size()) {
    omarchy::unsupported("MaskedScatter mask layout", out);
  }
  if (rows > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported("MaskedScatter row count", out);
  }
  uint32_t mask_word_offset =
      checked_item_offset(mask, mask.size(), "MaskedScatter", out);
  if (mask_word_offset % 4 != 0) {
    omarchy::unsupported("MaskedScatter mask alignment", out);
  }
  // A broadcast value materializes into a dense source segment first.
  const array* src_dense = &src;
  std::optional<array> materialized;
  if (src.data_size() != src.size()) {
    // A broadcast value materializes by scalar broadcast; Vector would
    // flat-copy past the single source element.
    materialized = array(src.shape(), src.dtype(), nullptr, {});
    copy_gpu(src, *materialized, CopyType::Scalar, out.primitive().stream());
    encoder.add_temporary(*materialized);
    src_dense = &*materialized;
  } else if (!src.flags().row_contiguous) {
    materialized = array(src.shape(), src.dtype(), nullptr, {});
    copy_gpu(src, *materialized, CopyType::Vector, out.primitive().stream());
    encoder.add_temporary(*materialized);
    src_dense = &*materialized;
  }
  uint32_t count = checked_u32(mask.size(), "MaskedScatter", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.reduce_size = checked_u32(row_length, "MaskedScatter", out);
  params.aux_size = checked_u32(src_length, "MaskedScatter", out);
  params.lhs_offset = checked_item_offset(
      *src_dense, src_dense->size(), "MaskedScatter", out);
  params.rhs_offset = mask_word_offset / 4;
  params.output_offset =
      checked_item_offset(out, out.size(), "MaskedScatter", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(out), binding(mask), binding(*src_dense)};
  encoder.dispatch_compute(
      kernel, bindings, params, checked_u32(rows, "MaskedScatter", out));
}
OMARCHY_BINARY(Minimum, MinimumOperation)
OMARCHY_BINARY(Multiply, MultiplyOperation)
OMARCHY_UNARY(Negative, NegativeOperation)
void NotEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareNotEqual, inputs, out);
}
void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("Partition", input, out, encoder);
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous Partition", out);
  }
  if (state().second != input.ndim() - 1) {
    omarchy::unsupported("non-suffix Partition", out);
  }
  dispatch_sort("Partition", input, out, false, encoder);
}
// Power promotes base and exponent to one dtype upstream, so a float
// dtype runs the float pow and an integer dtype runs the upstream
// exponentiation-by-squaring (negative signed exponent yields 0).
// Float bases below zero need the sign handling GLSL's pow refuses:
// non-integer exponents produce NaN, odd integer exponents negate.
void Power::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntPowerOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), PowerFloatOperation, inputs, out, out.primitive().stream());
}
OMARCHY_UNSUPPORTED_MULTI(QRF)
void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  // Non-affine modes pass three inputs, so the mode check must land
  // before the biases operand is bound.
  if (mode_ != QuantizationMode::Affine) {
    omarchy::unsupported(tag + " mode", out);
  }
  const array& x = inputs[0];
  const array& w = inputs[1];
  const array& scales = inputs[2];
  const array& biases = inputs[3];
  // First-class scope: the mlx-lm Linear shape. transpose=true reads the
  // packed w rows [N, K * bits / 32] as the dequantized matrix [N, K]
  // and computes x @ w.T; other affine shapes keep the named rejection.
  if (bits_ != 4 && bits_ != 8) {
    omarchy::unsupported(tag + " bits", out);
  }
  if (group_size_ != 32 && group_size_ != 64) {
    omarchy::unsupported(tag + " group size", out);
  }
  if (!transpose_) {
    omarchy::unsupported(tag + " transpose", out);
  }
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype(tag, x, out, encoder);
  if (w.dtype() != uint32 || w.ndim() != 2) {
    omarchy::unsupported(tag + " weight layout", out);
  }
  if (scales.dtype() != out.dtype() || biases.dtype() != out.dtype()) {
    // ops.cpp promotes x, scales, and biases to one affine dtype.
    omarchy::unsupported(tag + " scales dtype", out);
  }
  if (scales.ndim() != 2 || scales.shape() != biases.shape() ||
      scales.shape(0) != w.shape(0)) {
    omarchy::unsupported(tag + " scales shape", out);
  }
  if (!x.flags().row_contiguous || !w.flags().row_contiguous ||
      !scales.flags().row_contiguous || !biases.flags().row_contiguous) {
    omarchy::unsupported(tag + " non-contiguous input", out);
  }
  int k = x.shape(-1);
  int n = out.shape(-1);
  size_t m = x.size() / k;
  if (w.shape(0) != n ||
      static_cast<uint64_t>(w.shape(1)) * 32u / bits_ !=
          static_cast<uint64_t>(k) ||
      static_cast<uint64_t>(scales.shape(1)) * group_size_ !=
          static_cast<uint64_t>(k)) {
    omarchy::unsupported(tag + " shape", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  // Binding 2 packs the group parameters as two halves of one buffer:
  // [all scales | all biases] with K / group_size values per w row. Two
  // device copies build it with no extra kernel, and the encoder keeps
  // the temp alive until the dispatch completes. vkCmdCopyBuffer needs
  // 4-byte-aligned source offsets, so odd 16-bit views stay named.
  if (scales.offset() % 4 != 0 || biases.offset() % 4 != 0) {
    omarchy::unsupported(tag + " scales byte offset", out);
  }
  array combined(
      Shape{static_cast<int>(2 * scales.size())}, scales.dtype(), nullptr, {});
  array::Flags combined_flags;
  combined_flags.contiguous = true;
  combined_flags.row_contiguous = true;
  combined_flags.col_contiguous = true;
  combined.set_data(
      allocate_omarchy(combined.nbytes()),
      combined.size(),
      Strides{1},
      combined_flags,
      0);
  encoder.add_temporary(combined);
  VkDeviceSize scale_bytes = static_cast<VkDeviceSize>(scales.nbytes());
  encoder.copy_buffer(
      binding(scales).buffer,
      binding(combined).buffer,
      scale_bytes,
      static_cast<VkDeviceSize>(scales.offset()),
      0);
  encoder.copy_buffer(
      binding(biases).buffer,
      binding(combined).buffer,
      scale_bytes,
      static_cast<VkDeviceSize>(biases.offset()),
      scale_bytes);

  // Push-constant routing for the qmm shader: operation carries bits,
  // reduce_size carries the group size (see shaders/qmm.comp).
  uint64_t total = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
  omarchy::ComputeParams params;
  params.count = checked_u32(total, tag, out);
  params.operation = static_cast<uint32_t>(bits_);
  params.reduce_size = static_cast<uint32_t>(group_size_);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(x, x.size(), tag, out);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.aux_size = checked_u32(combined.size(), tag, out);
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(combined), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::QmmF32,
      omarchy::ComputeKernel::QmmF16,
      omarchy::ComputeKernel::QmmBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& keys = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  // mlx-lm sampling only consumes width 4 (uniform, gumbel, categorical,
  // and key splits all create RandomBits with width 4), so the shader
  // implements the uint32 case and other widths keep the named rejection.
  if (width_ != 4 || out.dtype() != uint32) {
    omarchy::unsupported("RandomBits width", out);
  }
  if (keys.dtype() != uint32) {
    omarchy::unsupported("RandomBits key dtype", out);
  }
  if (!keys.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous RandomBits keys", out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0 || keys.size() == 0) {
    return;
  }
  // Upstream layout: keys (N1, ..., NK, 2) and out
  // (N1, ..., NK, M1, M2, ...), so every key owns an equal number of
  // output words.
  size_t num_keys = keys.size() / 2;
  if (keys.size() % 2 != 0 || out.size() % num_keys != 0) {
    omarchy::unsupported("RandomBits shape", out);
  }
  size_t words_per_key = out.size() / num_keys;
  uint32_t count = checked_u32(out.size(), "RandomBits", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.lhs_size = checked_u32(keys.size(), "RandomBits", out);
  params.reduce_size = checked_u32(words_per_key, "RandomBits", out);
  params.output_size = checked_u32(num_keys, "RandomBits", out);
  params.lhs_offset = checked_item_offset(keys, keys.size(), "RandomBits", out);
  params.output_offset = checked_item_offset(out, out.size(), "RandomBits", out);
  std::array<omarchy::ComputeBinding, 2> bindings{
      binding(keys), binding(out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::RandomBitsU32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}
OMARCHY_UNSUPPORTED(Real)

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto [reduce_type, axes] = state();
  std::string operation_name = name();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (axes.empty()) {
    omarchy::unsupported(operation_name + " axes", out);
  }

  // mx.all / mx.any: ReduceType::And / Or produce a bool output from any
  // input dtype; float inputs keep their float capability gate.
  if (reduce_type == Reduce::And || reduce_type == Reduce::Or) {
    if (out.dtype() != bool_) {
      omarchy::unsupported(operation_name + " output dtype", out);
    }
    switch (input.dtype()) {
      case bool_:
      case int32:
      case uint32:
      case float32:
        break;
      case float16:
      case bfloat16:
        require_float_input(operation_name, input, out, encoder);
        break;
      default:
        omarchy::unsupported(operation_name + " dtype", out);
    }
    uint32_t operation =
        reduce_type == Reduce::Or ? ReduceAnyOperation : ReduceAllOperation;
    dispatch_reduce_general(
        operation_name,
        operation,
        input,
        out,
        axes,
        encoder,
        select_anyall_kernel(input.dtype()));
    return;
  }

  if (reduce_type != Reduce::Sum && reduce_type != Reduce::Prod &&
      reduce_type != Reduce::Min && reduce_type != Reduce::Max) {
    omarchy::unsupported(operation_name, out);
  }

  // Float Sum and Max over a row-contiguous suffix keep the dedicated
  // suffix kernel; every empty reduction routes to the general kernel,
  // whose accumulator seed supplies the upstream identity value.
  bool float_dtype = input.dtype() == float32 || input.dtype() == float16 ||
      input.dtype() == bfloat16;
  bool suffix_fast_path = false;
  size_t reduce_size = 1;
  if (float_dtype && out.dtype() == input.dtype() &&
      (reduce_type == Reduce::Sum || reduce_type == Reduce::Max) &&
      input.flags().row_contiguous) {
    int first_axis = input.ndim() - static_cast<int>(axes.size());
    suffix_fast_path = true;
    for (int index = 0; index < static_cast<int>(axes.size()); ++index) {
      if (axes[index] != first_axis + index) {
        suffix_fast_path = false;
        break;
      }
    }
    for (int axis : axes) {
      reduce_size *= input.shape(axis);
    }
    if (reduce_size == 0) {
      suffix_fast_path = false;
    }
  }

  if (suffix_fast_path) {
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
    auto kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::ReduceF32,
        omarchy::ComputeKernel::ReduceF16,
        omarchy::ComputeKernel::ReduceBF16);
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(output_size));
    return;
  }

  // The general path: integer dtypes, Prod, Min, and every non-suffix or
  // strided layout. Float outputs must equal their input; int32 and
  // uint32 outputs likewise (the op layer promotes bool and small
  // integers before the primitive, so those never arrive here).
  if (input.dtype() != out.dtype() ||
      !(input.dtype() == float32 || input.dtype() == float16 ||
        input.dtype() == bfloat16 || input.dtype() == int32 ||
        input.dtype() == uint32)) {
    omarchy::unsupported(operation_name + " dtype", out);
  }
  if (float_dtype) {
    require_float_input(operation_name, input, out, encoder);
  }
  uint32_t operation;
  switch (reduce_type) {
    case Reduce::Sum:
      operation = ReduceSumOperation;
      break;
    case Reduce::Prod:
      operation = ReduceProdOperation;
      break;
    case Reduce::Min:
      operation = ReduceMinOperation;
      break;
    default:
      operation = ReduceMaxOperation;
      break;
  }
  dispatch_reduce_general(
      operation_name,
      operation,
      input,
      out,
      axes,
      encoder,
      select_reduce_general_kernel(input.dtype()));
}

// Remainder is the Python-style modulo: the truncating remainder
// adjusted by one divisor when its sign differs from the divisor's.
// The integer kernel and the float kernel carry the same fixup, which
// reproduces the upstream integral_op / remainder value contract
// (result takes the divisor's sign) for both operand orders.
void Remainder::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntModuloOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), RemainderFloatOperation, inputs, out, out.primitive().stream());
}
OMARCHY_UNARY(Round, RoundOperation)
// Scan serves cumsum, cumprod, cummax, and cummin over any axis, direction,
// and inclusivity: one invocation owns one line along the scan axis and
// walks it serially, which keeps the shared-buffer transport simple. The
// accumulator stays float32 for every float dtype; int32 and uint32 scan in
// their own width with wrapping arithmetic. The suffix float kernel keeps
// Sum so the categorical sampler path is byte-for-byte unchanged.
void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis, reverse, inclusive] = state();
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (reduce_type == Scan::LogAddExp) {
    omarchy::unsupported("Scan LogAddExp", out);
  }
  if (input.dtype() == float32 || input.dtype() == float16 ||
      input.dtype() == bfloat16) {
    require_float_dtype("Scan", input, out, encoder);
  } else if (
      (input.dtype() == int32 || input.dtype() == uint32) &&
      out.dtype() == input.dtype()) {
    // Integers scan in their own width; bool and 64-bit stay rejected.
  } else {
    omarchy::unsupported("Scan dtype", out);
  }
  if (input.ndim() == 0 || input.ndim() > 4 || axis < 0 ||
      axis >= input.ndim()) {
    omarchy::unsupported("Scan rank", out);
  }

  bool suffix_float_sum = reduce_type == Scan::Sum && !reverse &&
      input.dtype() != int32 && input.dtype() != uint32 &&
      input.flags().row_contiguous && axis == input.ndim() - 1;
  if (suffix_float_sum) {
    size_t row_length = input.shape(-1);
    size_t rows = row_length == 0 ? 0 : input.size() / row_length;
    out.set_data(allocate_omarchy(out.nbytes()));
    if (out.size() == 0) {
      return;
    }
    uint32_t output_size = checked_u32(rows, "Scan", out);
    if (output_size > omarchy::kMaxComputeGroupCountX) {
      omarchy::unsupported("Scan row count", out);
    }
    omarchy::ComputeParams params;
    params.count = checked_u32(out.size(), "Scan", out);
    params.operation = inclusive ? 0u : 1u;
    params.reduce_size = checked_u32(row_length, "Scan", out);
    params.output_size = output_size;
    params.lhs_offset = checked_item_offset(input, input.size(), "Scan", out);
    params.output_offset = checked_item_offset(out, out.size(), "Scan", out);
    std::array<omarchy::ComputeBinding, 3> bindings{
        binding(input), binding(input), binding(out)};
    auto kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::ScanF32,
        omarchy::ComputeKernel::ScanF16,
        omarchy::ComputeKernel::ScanBF16);
    encoder.dispatch_compute(kernel, bindings, params, output_size);
    return;
  }

  // The general stride-walking scan. Push-constant routing mirrors
  // shaders/scan_general.comp: matrix_m carries the scan axis, flags bit 0
  // the reverse flag, reduce_size the operation selector, shape[] the
  // extents, in_strides[] the input strides, and out_strides[] the output
  // strides of the fresh row-contiguous output.
  uint32_t operation_selector;
  switch (reduce_type) {
    case Scan::Sum:
      operation_selector = 0u;
      break;
    case Scan::Prod:
      operation_selector = 1u;
      break;
    case Scan::Min:
      operation_selector = 2u;
      break;
    default:
      operation_selector = 3u;
      break;
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t axis_length = input.shape(axis);
  size_t lines = out.size() / axis_length;
  uint32_t output_size = checked_u32(lines, "Scan", out);
  if (output_size > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported("Scan row count", out);
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), "Scan", out);
  params.operation = inclusive ? 0u : 1u;
  params.reduce_size = operation_selector;
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(input, input.size(), "Scan", out);
  params.output_offset = checked_item_offset(out, out.size(), "Scan", out);
  params.matrix_m = static_cast<uint32_t>(axis);
  params.flags = reverse ? 1u : 0u;
  params.dims = static_cast<uint32_t>(input.ndim());
  uint32_t output_stride = 1;
  for (int dim = input.ndim() - 1; dim >= 0; --dim) {
    params.shape[dim] = static_cast<uint32_t>(input.shape(dim));
    params.in_strides[dim] =
        static_cast<uint32_t>(input.strides()[dim]);
    params.out_strides[dim] = output_stride;
    output_stride *= static_cast<uint32_t>(input.shape(dim));
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(input), binding(input), binding(out)};
  auto kernel = input.dtype() == int32
      ? omarchy::ComputeKernel::ScanGeneralI32
      : input.dtype() == uint32
      ? omarchy::ComputeKernel::ScanGeneralU32
      : select_float_kernel(
            out.dtype(),
            omarchy::ComputeKernel::ScanGeneralF32,
            omarchy::ComputeKernel::ScanGeneralF16,
            omarchy::ComputeKernel::ScanGeneralBF16);
  encoder.dispatch_compute(kernel, bindings, params, output_size);
}
void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axes] = state();
  const array& src = inputs.at(0);
  const array& indices = inputs.at(1);
  const array& updates = inputs.back();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (axes.size() != 1) {
    // Each index array is a descriptor binding; the shared layout caps
    // a dispatch at four buffers and scatter needs out, updates, the
    // index array, and rank/sentinel scratch.
    omarchy::unsupported("multi-index Scatter", out);
  }
  if (reduce_type == Scatter::Prod) {
    // No atomic multiply on this stack; a CAS-loop product depends on
    // the scheduling order of duplicate indices, so it is rejected
    // rather than delivered nondeterministic.
    omarchy::unsupported("Scatter Prod", out);
  }
  bool is_sum = reduce_type == Scatter::Sum;
  // Multi-slot dispatches only land their first slot write on this
  // device; until that is root-caused every Scatter mode keeps the
  // named rejection rather than silently dropping updates.
  omarchy::unsupported("Scatter", out);
  if (out.dtype() == float16 || out.dtype() == bfloat16) {
    require_float_dtype("Scatter", src, out, encoder);
  }
  uint32_t map_code;
  omarchy::ComputeKernel kernel;
  switch (out.dtype()) {
    case float32:
      map_code = 0;
      kernel = omarchy::ComputeKernel::ScatterU32;
      break;
    case int32:
      map_code = 1;
      kernel = omarchy::ComputeKernel::ScatterU32;
      break;
    case uint32:
      map_code = 2;
      kernel = omarchy::ComputeKernel::ScatterU32;
      break;
    case float16:
      map_code = 0;
      kernel = omarchy::ComputeKernel::ScatterF16;
      break;
    case bfloat16:
      map_code = 0;
      kernel = omarchy::ComputeKernel::ScatterBF16;
      break;
    default:
      omarchy::unsupported("Scatter dtype", out);
  }
  if (is_sum && out.dtype() != int32 && out.dtype() != uint32) {
    // Float Sum has no deterministic implementation here: GLSL offers
    // no float atomicAdd, and duplicate-index float accumulation
    // through a CAS loop is scheduling-order dependent.
    omarchy::unsupported("Scatter Sum dtype", out);
  }
  CopyType copy_type = src.data_size() == 1 ? CopyType::Scalar
      : (src.flags().row_contiguous && src.data_size() == src.size())
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(src, out, copy_type, out.primitive().stream());
  if (indices.size() == 0) {
    return;
  }
  if (indices.size() > (size_t{1} << 31)) {
    omarchy::unsupported("Scatter slot count", out);
  }
  if (!indices.flags().row_contiguous ||
      indices.data_size() != indices.size()) {
    omarchy::unsupported("Scatter index layout", out);
  }
  if (!updates.flags().row_contiguous) {
    omarchy::unsupported("Scatter updates layout", out);
  }
  uint32_t index_mode = scatter_index_mode(indices, out, "Scatter");
  size_t update_ndim = updates.ndim() - indices.ndim();
  if (update_ndim > 4) {
    omarchy::unsupported("Scatter update rank", out);
  }
  uint32_t count = checked_u32(indices.size(), "Scatter", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.reduce_size = checked_u32(out.shape(axes[0]), "Scatter", out);
  params.output_size =
      checked_u32(updates.size() / indices.size(), "Scatter", out);
  params.matrix_m = checked_u32(out.strides(axes[0]), "Scatter", out);
  params.rhs_offset = scatter_index_offset(indices, out, "Scatter");
  params.output_offset = checked_item_offset(out, out.size(), "Scatter", out);
  params.aux_size = index_mode;
  params.aux_offset = map_code;
  params.flags = checked_u32(update_ndim, "Scatter", out);
  for (size_t i = 0; i < update_ndim; ++i) {
    int dim = out.ndim() - static_cast<int>(update_ndim) +
        static_cast<int>(i);
    params.shape[i] = checked_u32(updates.shape(dim), "Scatter", out);
    params.in_strides[i] = checked_u32(out.strides(dim), "Scatter", out);
  }
  if (is_sum) {
    // Phase 6: atomicAdd. Integer addition is associative, so the
    // result is deterministic even under duplicate indices.
    std::array<omarchy::ComputeBinding, 3> bindings{
        binding(out), binding(updates), binding(indices)};
    params.operation = 6;
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(count));
    return;
  }
  uint32_t phase1 = 0;
  uint32_t phase2 = 0;
  uint32_t clear_value = 0;
  if (reduce_type == Scatter::None) {
    phase1 = 0;
    phase2 = 1;
  } else if (reduce_type == Scatter::Max) {
    phase1 = 2;
    phase2 = 4;
  } else {
    phase1 = 3;
    phase2 = 5;
    clear_value = 0xFFFFFFFFu;
  }
  array scratch = make_u32_scratch(out.size(), encoder);
  dispatch_clear_u32(scratch, clear_value, encoder);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(out), binding(updates), binding(indices), binding(scratch)};
  params.operation = phase1;
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
  // None pass 2 walks slots; Max/Min finalize walks output elements.
  if (reduce_type == Scatter::None) {
    params.count = count;
  } else {
    params.count = checked_u32(out.size(), "Scatter", out);
  }
  params.operation = phase2;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}
void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  const array& src = inputs.at(0);
  const array& indices = inputs.at(1);
  const array& updates = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  bool is_sum = reduce_type == ScatterAxis::Sum;
  // The slot-writer phases still no-op on the device; until that is
  // root-caused, ScatterAxis keeps the named rejection rather than
  // silently dropping a caller's updates.
  omarchy::unsupported("ScatterAxis", out);
  if (out.dtype() == float16 || out.dtype() == bfloat16) {
    require_float_dtype("ScatterAxis", src, out, encoder);
  }
  omarchy::ComputeKernel kernel;
  switch (out.dtype()) {
    case float32:
    case int32:
    case uint32:
      kernel = omarchy::ComputeKernel::ScatterAxisU32;
      break;
    case float16:
      kernel = omarchy::ComputeKernel::ScatterAxisF16;
      break;
    case bfloat16:
      kernel = omarchy::ComputeKernel::ScatterAxisBF16;
      break;
    default:
      omarchy::unsupported("ScatterAxis dtype", out);
  }
  if (is_sum && out.dtype() != int32 && out.dtype() != uint32) {
    // Same determinism evidence as general Scatter Sum.
    omarchy::unsupported("ScatterAxis Sum dtype", out);
  }
  CopyType copy_type = src.data_size() == 1 ? CopyType::Scalar
      : (src.flags().row_contiguous && src.data_size() == src.size())
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(src, out, copy_type, out.primitive().stream());
  if (indices.size() == 0) {
    return;
  }
  if (indices.size() > (size_t{1} << 31)) {
    omarchy::unsupported("ScatterAxis slot count", out);
  }
  if (!indices.flags().row_contiguous ||
      indices.data_size() != indices.size()) {
    omarchy::unsupported("ScatterAxis index layout", out);
  }
  int non_axis = out.ndim() - 1;
  if (non_axis > 4) {
    omarchy::unsupported("ScatterAxis rank", out);
  }
  size_t post_size = 1;
  for (int i = axis + 1; i < out.ndim(); ++i) {
    post_size *= out.shape(i);
  }
  uint32_t count = checked_u32(indices.size(), "ScatterAxis", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.reduce_size = checked_u32(src.shape(axis), "ScatterAxis", out);
  params.output_size = checked_u32(post_size, "ScatterAxis", out);
  params.aux_size = checked_u32(indices.shape(axis), "ScatterAxis", out);
  params.aux_offset = scatter_index_mode(indices, out, "ScatterAxis");
  // lhs_offset carries the update base element offset; the shader
  // walks update addresses through out_strides plus matrix_n.
  params.lhs_offset =
      checked_item_offset(updates, updates.size(), "ScatterAxis", out);
  params.rhs_offset = scatter_index_offset(indices, out, "ScatterAxis");
  params.output_offset = checked_item_offset(out, out.size(), "ScatterAxis", out);
  params.matrix_m = checked_u32(src.strides(axis), "ScatterAxis", out);
  params.matrix_n = checked_u32(updates.strides(axis), "ScatterAxis", out);
  params.matrix_k = checked_u32(axis, "ScatterAxis", out);
  params.flags = checked_u32(non_axis, "ScatterAxis", out);
  for (int i = 0, d = 0; i < out.ndim(); ++i) {
    if (i == axis) {
      continue;
    }
    params.shape[d] = checked_u32(out.shape(i), "ScatterAxis", out);
    params.in_strides[d] = checked_u32(src.strides(i), "ScatterAxis", out);
    params.out_strides[d] = checked_u32(updates.strides(i), "ScatterAxis", out);
    ++d;
  }
  if (is_sum) {
    std::array<omarchy::ComputeBinding, 3> bindings{
        binding(out), binding(indices), binding(updates)};
    params.operation = 6;
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(count));
    return;
  }
  // None: rank-max scratch, then the winning slot rewrites the target,
  // which reproduces the CPU's sequential last-write-wins order.
  array scratch = make_u32_scratch(out.size(), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(out), binding(indices), binding(updates), binding(scratch)};
  params.operation = 0;
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
  params.operation = 1;
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
}
// SearchSorted serves the categorical sampler: one binary search over
// a sorted 1-D row per value, with both operands already promoted to
// the same dtype by the op layer. The right flag picks between the
// first index at or after the value (left) and the first index after
// it (right), matching the upstream CPU comparator.
void SearchSorted::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& sequence = inputs.at(0);
  const array& values = inputs.at(1);
  bool right = state();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (out.dtype() != uint32 || sequence.dtype() != values.dtype()) {
    omarchy::unsupported("SearchSorted dtype", out);
  }
  if (sequence.dtype() != float32 && sequence.dtype() != float16 &&
      sequence.dtype() != bfloat16 && sequence.dtype() != int32 &&
      sequence.dtype() != uint32) {
    omarchy::unsupported("SearchSorted dtype", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (sequence.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported("SearchSorted float16 capability", out);
  }
  if (sequence.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported("SearchSorted bfloat16 capability", out);
  }
  if (sequence.ndim() != 1 || !sequence.flags().row_contiguous ||
      !values.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous SearchSorted", out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(values.size(), "SearchSorted", out);
  params.operation = right ? 1u : 0u;
  params.reduce_size = checked_u32(sequence.size(), "SearchSorted", out);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(
      sequence, sequence.size(), "SearchSorted", out);
  params.rhs_offset = checked_item_offset(
      values, values.size(), "SearchSorted", out);
  params.output_offset = checked_item_offset(out, out.size(), "SearchSorted", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(sequence), binding(values), binding(out)};
  auto kernel = sequence.dtype() == float16
      ? omarchy::ComputeKernel::SearchSortedF16
      : sequence.dtype() == bfloat16
      ? omarchy::ComputeKernel::SearchSortedBF16
      : sequence.dtype() == int32 ? omarchy::ComputeKernel::SearchSortedI32
                                  : sequence.dtype() == uint32
          ? omarchy::ComputeKernel::SearchSortedU32
          : omarchy::ComputeKernel::SearchSortedF32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// Select serves the composed causal mask: a strided bool condition view
// picks between a row-contiguous value and a scalar floor. The sampler
// chain also selects between two row-contiguous values under a scalar
// condition (where(isinf(m), eq, exp)), so the false operand accepts
// the same suffix-aligned shapes as the true operand. Other layouts
// stay named rejections.
void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& condition = inputs.at(0);
  const array& truthy = inputs.at(1);
  const array& falsy = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (condition.dtype() != bool_) {
    omarchy::unsupported("Select dtype", out);
  }
  require_float_dtype(name(), truthy, out, encoder);
  require_float_dtype(name(), falsy, out, encoder);
  if (!truthy.flags().row_contiguous || !is_trailing_broadcast(falsy, out)) {
    omarchy::unsupported("Select layout", out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // The Honeykrisp bool word read only executes correctly in the
  // straight-line select.comp form, whose condition address requires a
  // flat condition. A broadcast or strided condition view is
  // materialized through the logical_or pipeline, whose dual-buffer
  // word reads are correct on that driver; binding the same buffer as
  // both inputs makes the OR an identity copy. The encoder keeps the
  // temporary alive until the committed work completes.
  std::vector<array> materialized;
  const array* condition_ptr = &condition;
  if (!condition.flags().row_contiguous ||
      condition.data_size() != out.size()) {
    materialized.push_back(array(out.shape(), bool_, nullptr, {}));
    array& flat_condition = materialized.back();
    flat_condition.set_data(allocate_omarchy(flat_condition.nbytes()));
    uint32_t material_count = checked_u32(flat_condition.size(), name(), out);
    uint32_t material_words = checked_u32(
        (static_cast<uint64_t>(material_count) + 3) / 4, name(), out);
    omarchy::ComputeParams material_params;
    material_params.count = material_count;
    material_params.lhs_size =
        checked_u32(condition.data_size(), name(), out);
    material_params.rhs_size = material_params.lhs_size;
    material_params.output_size = material_count;
    material_params.lhs_offset = checked_item_offset(
        condition, material_params.lhs_size, name(), out);
    material_params.rhs_offset = material_params.lhs_offset;
    material_params.output_offset = checked_item_offset(
        flat_condition, material_count, name(), out);
    fill_broadcast_transport(name(), material_params, condition, condition,
                             flat_condition);
    std::array<omarchy::ComputeBinding, 3> material_bindings{
        binding(condition), binding(condition), binding(flat_condition)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LogicalOrBool,
        material_bindings,
        material_params,
        omarchy::compute_dispatch_group_count(material_words));
    encoder.add_temporary(flat_condition);
    condition_ptr = &flat_condition;
  }
  uint32_t count = checked_u32(out.size(), name(), out);
  uint32_t word_count = checked_u32(
      (static_cast<uint64_t>(count) + 3) / 4, name(), out);
  omarchy::ComputeParams params;
  params.count = count;
  params.lhs_size = checked_u32(condition_ptr->data_size(), name(), out);
  params.rhs_size = checked_u32(truthy.data_size(), name(), out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(
      *condition_ptr, params.lhs_size, name(), out);
  params.rhs_offset = checked_item_offset(truthy, params.rhs_size, name(), out);
  params.aux_size = checked_u32(falsy.data_size(), name(), out);
  params.aux_offset = checked_item_offset(falsy, falsy.data_size(), name(), out);
  params.output_offset = checked_item_offset(out, count, name(), out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(*condition_ptr), binding(truthy), binding(falsy), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::SelectF32,
      omarchy::ComputeKernel::SelectF16,
      omarchy::ComputeKernel::SelectBF16);
  // The kernel processes one output word (four elements) per thread
  // with no grid-stride loop, so very large outputs dispatch in
  // back-to-back offset chunks. The aux offset stays absolute: the
  // shader indexes the false operand by output index.
  constexpr uint64_t kMaxWordsPerDispatch =
      static_cast<uint64_t>(omarchy::kMaxComputeGroupCountX) *
      omarchy::kComputeThreadsPerGroup;
  uint32_t lhs_base = params.lhs_offset;
  uint32_t rhs_base = params.rhs_offset;
  uint32_t output_base = params.output_offset;
  uint64_t words_done = 0;
  while (words_done < word_count) {
    uint32_t chunk_words = static_cast<uint32_t>(
        std::min<uint64_t>(word_count - words_done, kMaxWordsPerDispatch));
    uint64_t chunk_elements = 4ull * words_done;
    if (!omarchy::compute_index_span_fits(
            lhs_base + chunk_elements, 4ull * chunk_words) ||
        !omarchy::compute_index_span_fits(
            rhs_base + chunk_elements, 4ull * chunk_words) ||
        !omarchy::compute_index_span_fits(
            output_base + chunk_elements, 4ull * chunk_words)) {
      omarchy::unsupported("Select index span", out);
    }
    params.lhs_offset = checked_u32(lhs_base + chunk_elements, name(), out);
    params.rhs_offset = checked_u32(rhs_base + chunk_elements, name(), out);
    params.output_offset =
        checked_u32(output_base + chunk_elements, name(), out);
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(chunk_words));
    words_done += chunk_words;
  }
}

OMARCHY_UNARY(Sigmoid, SigmoidOperation)
// Sign keeps the upstream three-way rule (-1, 0, 1 by comparison with
// zero, NaN mapping to 0) for float dtypes, and the integer rule
// (unsigned 0/1) through the integer kernel. Everything else keeps the
// named float-dtype rejection from dispatch_elementwise.
void Sign::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntSignOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), SignFloatOperation, inputs, out, out.primitive().stream());
}
OMARCHY_UNARY(Sin, SinOperation)
OMARCHY_UNARY(Sinh, SinhOperation)
void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_softmax(name(), inputs.at(0), out, stream());
}
void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.size() == 0) {
    return;
  }

  const auto& in = inputs[0];
  const auto& upd = inputs[1];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  // Only the None reduce mode is a plain paste; every other mode needs
  // read-modify-write shaders.
  if (reduce_type_ != SliceUpdate::None) {
    omarchy::unsupported("SliceUpdate reduce", out);
  }

  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());

  auto [data_offset, out_strides] =
      prepare_slice(out, start_indices_, strides_);

  copy_gpu_inplace(
      upd,
      out,
      upd.shape(),
      upd.strides(),
      out_strides,
      /* i_offset = */ 0,
      /* o_offset = */ data_offset,
      CopyType::GeneralGeneral,
      stream());
}
void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("Sort", input, out, encoder);
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous Sort", out);
  }
  if (state() != input.ndim() - 1) {
    omarchy::unsupported("non-suffix Sort", out);
  }
  dispatch_sort("Sort", input, out, false, encoder);
}
OMARCHY_UNARY(Square, SquareOperation)
void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_elementwise(
      name(),
      state() ? RsqrtOperation : SqrtOperation,
      inputs,
      out,
      out.primitive().stream());
}
// The categorical sampler subtracts one from uint32 searchsorted
// indices, so the subtract family also covers int32 and uint32 through
// the integer elementwise kernel. Other integer operations stay named
// rejections.
void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntSubtractOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), SubtractOperation, inputs, out, out.primitive().stream());
}
OMARCHY_UNSUPPORTED_MULTI(SVD)
OMARCHY_UNARY(Tan, TanOperation)
OMARCHY_UNARY(Tanh, TanhOperation)
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
  // Training with a logsumexp output needs the VJP, which stays a
  // named rejection, so that one case keeps the composed graph.
  return output_logsumexp;
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
void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  if (outputs.size() != 1) {
    omarchy::unsupported(tag + " logsumexp output", out);
  }
  if (has_sinks_ || inputs.size() > 4) {
    omarchy::unsupported(tag + " sinks", out);
  }
  if (inputs.size() == 4 && do_causal_) {
    omarchy::unsupported(tag + " causal mask with array mask", out);
  }
  const array& q = inputs.at(0);
  const array& k = inputs.at(1);
  const array& v = inputs.at(2);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(tag, q, out, encoder);
  require_float_dtype(tag, k, out, encoder);
  require_float_dtype(tag, v, out, encoder);
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    omarchy::unsupported("attention rank " + tag, out);
  }
  int batch = q.shape(0);
  int heads = q.shape(1);
  int q_len = q.shape(2);
  int head_dim = q.shape(3);
  int kv_heads = k.shape(1);
  int k_len = k.shape(2);
  int v_dim = v.shape(3);
  if (
      k.shape(0) != batch || v.shape(0) != batch || k.shape(3) != head_dim ||
      v.shape(1) != kv_heads || v.shape(2) != k_len ||
      out.shape() != Shape{batch, heads, q_len, v_dim}) {
    omarchy::unsupported("attention shapes " + tag, out);
  }
  if (kv_heads == 0 || heads % kv_heads != 0) {
    omarchy::unsupported("attention head split " + tag, out);
  }
  int repeats = heads / kv_heads;
  if (out.size() == 0) {
    out.set_data(allocate_omarchy(out.nbytes()));
    return;
  }

  // The validated f32-score composition (the M1 4-bit degeneracy fix,
  // docs/2026-09-01-m1-4bit-greedy-sdpa-f16-scores.md): cast to
  // float32, scale, express GQA through the unflatten/expand_dims
  // shapes instead of mx.repeat, add the mask as a float32 additive
  // term, and keep every intermediate in float32 so score magnitudes
  // far beyond float16 stay finite. Only the result narrows back to
  // the output dtype.

  // The backend allocator is host-visible, so constants and the causal
  // mask are written straight into fresh allocations (the Load idiom)
  // before any command references them.
  auto to_f32 = [&](const array& x) {
    array wide(x.shape(), float32, nullptr, {});
    if (x.flags().row_contiguous) {
      copy_gpu(x, wide, CopyType::Vector, s);
    } else {
      array dense = contiguous_copy_gpu(x, s);
      encoder.add_temporary(dense);
      copy_gpu(dense, wide, CopyType::Vector, s);
    }
    encoder.add_temporary(wide);
    return wide;
  };
  auto broadcast_view = [&](const array& base, Shape shape) {
    array view(std::move(shape), base.dtype(), nullptr, {});
    view.copy_shared_buffer(base, Strides(view.ndim(), 0), {true, false, false}, base.size());
    encoder.add_temporary(view);
    return view;
  };

  array q32 = to_f32(q);
  array scale_arr(scale_);
  scale_arr.set_data(allocate_omarchy(scale_arr.nbytes()));
  scale_arr.data<float>()[0] = scale_;
  encoder.add_temporary(scale_arr);
  array qs(q32.shape(), float32, nullptr, {});
  dispatch_elementwise(
      tag,
      MultiplyOperation,
      {q32, broadcast_view(scale_arr, q32.shape())},
      qs,
      s);
  encoder.add_temporary(qs);

  array k32 = to_f32(k);
  array v32 = to_f32(v);
  if (repeats > 1) {
    qs = reshape_in_eval(
        qs, Shape{batch, kv_heads, repeats, q_len, head_dim}, s);
    k32 = reshape_in_eval(k32, Shape{batch, kv_heads, 1, k_len, head_dim}, s);
    v32 = reshape_in_eval(v32, Shape{batch, kv_heads, 1, k_len, v_dim}, s);
    encoder.add_temporary(qs);
    encoder.add_temporary(k32);
    encoder.add_temporary(v32);
  }

  Shape score_shape = qs.shape();
  score_shape.back() = k_len;
  array scores(score_shape, float32, nullptr, {});
  array keys_t = swapaxes_in_eval(k32, -1, -2);
  encoder.add_temporary(keys_t);
  dispatch_matmul(tag, {qs, keys_t}, scores, 1.0f, 0.0f, false, s);
  encoder.add_temporary(scores);

  std::optional<array> masked;
  if (do_causal_) {
    if (k_len < q_len) {
      omarchy::unsupported("causal offset " + tag, out);
    }
    // The additive causal mask holds 0 for attended positions and
    // -1e30 elsewhere: the same float32 tensor the validated
    // composition built from arange/greater_equal and
    // (1 - cast) * -1e30, without Select or repeat.
    array mask(Shape{q_len, k_len}, float32, nullptr, {});
    mask.set_data(allocate_omarchy(mask.nbytes()));
    float* values = mask.data<float>();
    int offset = k_len - q_len;
    for (int row = 0; row < q_len; ++row) {
      for (int col = 0; col < k_len; ++col) {
        values[row * k_len + col] = offset + row >= col ? 0.0f : -1e30f;
      }
    }
    encoder.add_temporary(mask);
    masked = array(scores.shape(), float32, nullptr, {});
    dispatch_elementwise(tag, AddOperation, {scores, mask}, *masked, s);
  } else if (inputs.size() == 4) {
    // Upstream pre-broadcasts an array mask to
    // [B, heads, q_len, k_len] in the output dtype and converts bool
    // masks to additive values before the primitive runs.
    array mask = to_f32(inputs.at(3));
    if (repeats > 1) {
      mask = reshape_in_eval(
          mask, Shape{batch, kv_heads, repeats, q_len, k_len}, s);
      encoder.add_temporary(mask);
    }
    if (mask.shape() != scores.shape()) {
      omarchy::unsupported("attention mask shape " + tag, out);
    }
    masked = array(scores.shape(), float32, nullptr, {});
    dispatch_elementwise(tag, AddOperation, {scores, mask}, *masked, s);
  }
  const array& logits = masked ? *masked : scores;
  encoder.add_temporary(logits);

  array probs(logits.shape(), float32, nullptr, {});
  dispatch_softmax(tag, logits, probs, s);
  encoder.add_temporary(probs);

  Shape result_shape = probs.shape();
  result_shape.back() = v_dim;
  array result(result_shape, float32, nullptr, {});
  dispatch_matmul(tag, {probs, v32}, result, 1.0f, 0.0f, false, s);
  encoder.add_temporary(result);
  if (result.dtype() == out.dtype()) {
    out.copy_shared_buffer(result);
  } else {
    copy_gpu(result, out, CopyType::Vector, s);
  }
}

OMARCHY_UNSUPPORTED_MULTI(ScaledDotProductAttentionVJP)
OMARCHY_UNSUPPORTED_MULTI(ConvertFP8)
void Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  // One primitive serves both directions; only the dequantize side is
  // implemented. The quantize direction and every non-affine mode keep
  // the named rejection.
  if (!dequantize_) {
    omarchy::unsupported(tag + " direction", out);
  }
  if (mode_ != QuantizationMode::Affine) {
    omarchy::unsupported(tag + " mode", out);
  }
  if (bits_ != 4 && bits_ != 8) {
    omarchy::unsupported(tag + " bits", out);
  }
  if (group_size_ != 32 && group_size_ != 64) {
    omarchy::unsupported(tag + " group size", out);
  }
  const array& w = inputs.at(0);
  const array& scales = inputs.at(1);
  const array& biases = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  // The output dtype is the promoted scales dtype (ops.cpp), and the
  // kernel reads the group parameters in that dtype, so mixed dtypes
  // stay a named rejection. The dequant kernels ship f32 and f16
  // variants only: bfloat16 and float64 parameters keep the named
  // error.
  if (
      scales.dtype() != out.dtype() || biases.dtype() != out.dtype() ||
      (scales.dtype() != float16 && scales.dtype() != float32)) {
    omarchy::unsupported(tag + " scales dtype", out);
  }
  require_float_dtype(tag, scales, out, encoder);
  if (w.dtype() != uint32) {
    omarchy::unsupported(tag + " weight dtype", out);
  }
  if (
      w.shape().size() != scales.shape().size() ||
      scales.shape() != biases.shape()) {
    omarchy::unsupported(tag + " scales shape", out);
  }
  size_t words_per_row = w.shape(-1);
  uint64_t out_columns =
      static_cast<uint64_t>(words_per_row) * 32u / bits_;
  uint64_t groups_per_row =
      static_cast<uint64_t>(scales.shape(-1));
  if (
      groups_per_row * group_size_ != out_columns ||
      !std::equal(
          w.shape().begin(),
          w.shape().end() - 1,
          scales.shape().begin())) {
    omarchy::unsupported(tag + " shape", out);
  }
  if (
      !w.flags().row_contiguous || !scales.flags().row_contiguous ||
      !biases.flags().row_contiguous) {
    omarchy::unsupported(tag + " non-contiguous input", out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0 || w.size() == 0) {
    return;
  }
  // One thread owns one packed word. Every word in a row-contiguous
  // weight maps to a unique output span, so the word load is linear in
  // the thread index and the group parameters reuse one address for
  // the whole word.
  omarchy::ComputeParams params;
  params.count = checked_u32(w.size(), tag, out);
  params.operation = static_cast<uint32_t>(bits_);
  params.lhs_size = params.count;
  params.rhs_size = checked_u32(scales.size(), tag, out);
  params.reduce_size = static_cast<uint32_t>(group_size_);
  params.output_size = checked_u32(out.size(), tag, out);
  params.lhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.rhs_offset = checked_item_offset(scales, scales.size(), tag, out);
  params.aux_size = checked_u32(biases.size(), tag, out);
  params.aux_offset = checked_item_offset(biases, biases.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.matrix_n = checked_u32(words_per_row, tag, out);
  params.matrix_k = checked_u32(groups_per_row, tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(w), binding(scales), binding(biases), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::DequantF32,
      omarchy::ComputeKernel::DequantF16,
      omarchy::ComputeKernel::DequantF32);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

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

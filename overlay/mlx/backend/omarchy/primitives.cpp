// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/unsupported.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
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
    set_binary_op_output_data(
        lhs, rhs, out, binary_type, allocate_omarchy);
  } else if (general_broadcast) {
    // The stride gather must not donate: it reads and writes the same
    // buffer at different indices.
    out.set_data(allocate_omarchy(out.nbytes()));
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

// The comparison family shares one shape: a bool output from two
// broadcast views of one input dtype, word-packed through the 32-bit
// bool transport. Equal serves the categorical sampler chain (isinf,
// mx.random.categorical); GreaterEqual keeps its int32 causal-mask
// contract.
enum ComparisonOperation : uint32_t {
  CompareEqual,
  CompareGreaterEqual,
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

// LogicalOr serves the isinf composition: bool inputs, a bool output,
// and the same 32-bit word transport the comparisons use.
void dispatch_logical_or(
    const std::string& name,
    const std::vector<array>& inputs,
    array& out) {
  const array& lhs = inputs.at(0);
  const array& rhs = inputs.at(1);
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
// differs, so int32 and uint32 share one SPIR-V variant.
void dispatch_int_elementwise(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out) {
  const array& lhs = inputs.at(0);
  const array& rhs = inputs.at(1);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
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
  set_binary_op_output_data(lhs, rhs, out, binary_type, allocate_omarchy);
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
  if (!is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out)) {
    fill_broadcast_transport(name, params, lhs, rhs, out);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(lhs), binding(rhs), binding(out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::ElementwiseI32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
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

OMARCHY_UNSUPPORTED(Abs)
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
OMARCHY_UNSUPPORTED(ArcCos)
OMARCHY_UNSUPPORTED(ArcCosh)
OMARCHY_UNSUPPORTED(ArcSin)
OMARCHY_UNSUPPORTED(ArcSinh)
OMARCHY_UNSUPPORTED(ArcTan)
OMARCHY_UNSUPPORTED(ArcTan2)
OMARCHY_UNSUPPORTED(ArcTanh)
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
OMARCHY_UNSUPPORTED(BitwiseBinary)
OMARCHY_UNSUPPORTED(BitwiseInvert)
OMARCHY_UNSUPPORTED(BlockMaskedMM)
OMARCHY_UNSUPPORTED(Ceil)
OMARCHY_UNSUPPORTED(Cholesky)
OMARCHY_UNSUPPORTED_MULTI(Compiled)
OMARCHY_UNSUPPORTED(Conjugate)
OMARCHY_UNSUPPORTED(Convolution)
OMARCHY_UNARY(Cos, CosOperation)
OMARCHY_UNSUPPORTED(Cosh)
OMARCHY_BINARY(Divide, DivideOperation)
OMARCHY_UNSUPPORTED_MULTI(DivMod)
void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareEqual, inputs, out);
}
OMARCHY_UNSUPPORTED(Erf)
OMARCHY_UNSUPPORTED(ErfInv)
OMARCHY_UNARY(Exp, ExpOperation)
OMARCHY_UNSUPPORTED(Expm1)
OMARCHY_UNSUPPORTED(FFT)
OMARCHY_UNSUPPORTED(Floor)
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
OMARCHY_UNSUPPORTED(GatherAxis)
OMARCHY_UNSUPPORTED(GatherMM)
OMARCHY_UNSUPPORTED(GatherQMM)
OMARCHY_UNSUPPORTED(GatherQQMM)
OMARCHY_UNSUPPORTED(Greater)
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
OMARCHY_UNSUPPORTED(Hadamard)
OMARCHY_UNSUPPORTED(Imag)
OMARCHY_UNSUPPORTED(Inverse)
OMARCHY_UNSUPPORTED(Less)
OMARCHY_UNSUPPORTED(LessEqual)
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
OMARCHY_UNARY(Log, LogOperation)
OMARCHY_UNSUPPORTED(Log1p)
OMARCHY_UNSUPPORTED(LogicalAnd)
OMARCHY_UNSUPPORTED(LogicalNot)
void LogicalOr::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_logical_or(name(), inputs, out);
}
OMARCHY_UNSUPPORTED(LogAddExp)
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
OMARCHY_UNSUPPORTED(MaskedScatter)
OMARCHY_BINARY(Minimum, MinimumOperation)
OMARCHY_BINARY(Multiply, MultiplyOperation)
OMARCHY_UNARY(Negative, NegativeOperation)
OMARCHY_UNSUPPORTED(NotEqual)
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
OMARCHY_UNSUPPORTED(Power)
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
OMARCHY_UNSUPPORTED(QQMatmul)
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
}

OMARCHY_UNSUPPORTED(Remainder)
OMARCHY_UNSUPPORTED(Round)
// Scan serves the categorical sampler's exclusive cdf prefix sums: a
// float suffix scan over row-contiguous rows. Sum is the only reduce
// type, reverse scans stay named rejections, and one invocation owns a
// whole row in a serial pass, which keeps the shared-buffer transport
// simple. The accumulator stays float32 for every dtype.
void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis, reverse, inclusive] = state();
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("Scan", input, out, encoder);
  if (reduce_type != Scan::Sum) {
    omarchy::unsupported("Scan reduce", out);
  }
  if (reverse) {
    omarchy::unsupported("reverse Scan", out);
  }
  if (!input.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous Scan", out);
  }
  if (axis != input.ndim() - 1) {
    omarchy::unsupported("non-suffix Scan", out);
  }
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
}
OMARCHY_UNSUPPORTED(Scatter)
OMARCHY_UNSUPPORTED(ScatterAxis)
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
OMARCHY_UNSUPPORTED(SegmentedMM)
OMARCHY_UNARY(Sigmoid, SigmoidOperation)
OMARCHY_UNSUPPORTED(Sign)
OMARCHY_UNARY(Sin, SinOperation)
OMARCHY_UNSUPPORTED(Sinh)
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
    dispatch_int_elementwise(name(), SubtractOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), SubtractOperation, inputs, out, out.primitive().stream());
}
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

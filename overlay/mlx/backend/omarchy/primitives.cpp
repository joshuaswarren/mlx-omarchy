// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/unsupported.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <numeric>
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
#include "mlx/ops.h"

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

// Consumer-boundary dense normalization. Returns |value| itself when
// |dense_enough| holds, so a dense operand takes the caller's existing
// path with no allocation, no header copy, and no rebind. Otherwise the
// general strided-copy engine materializes one dense same-dtype temp;
// the caller owns |temp| across its dispatches and the encoder keeps
// the buffer alive until the committed work completes.
const array& ensure_dense(
    const array& value,
    bool dense_enough,
    std::optional<array>& temp,
    omarchy::CommandEncoder& encoder,
    const Stream& s) {
  if (dense_enough) {
    return value;
  }
  temp = contiguous_copy_gpu(value, s);
  encoder.add_temporary(*temp);
  return *temp;
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


// Materializes a matmul operand whose inner 2-D layout fits neither the
// row-major nor the column-major gap form; concatenate results compose
// this way. The general strided-copy engine writes a standard row-major
// batch, and the encoder keeps the temp alive until the committed work
// completes. Engine limits (negative strides, collapsed rank beyond 4,
// span overflow) keep their named errors.
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
  if (getenv("MLX_OMARCHY_TRACE_MATERIALIZE")) {
    std::fprintf(
        stderr,
        "[materialize] %s shape=[", name.c_str());
    for (int i = 0; i < value.ndim(); ++i) {
      std::fprintf(stderr, "%s%d", i ? "," : "", value.shape(i));
    }
    std::fprintf(stderr, "] strides=[");
    for (int i = 0; i < value.ndim(); ++i) {
      std::fprintf(stderr, "%s%lld", i ? "," : "",
          static_cast<long long>(value.strides()[i]));
    }
    std::fprintf(stderr, "]\n");
  }
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

// Classifies a matmul operand's inner 2-D layout: row-major (column
// stride 1, row gap = the axis -2 stride) or column-major (row stride
// 1, column gap = the axis -1 stride). A size-1 inner axis never
// dereferences its stride. Batch axes ride their actual strides - the
// shader unravels workgroup z over them, and a stride-0 batch axis
// broadcasts - so nothing materializes for batch strides alone. Gaps
// are in elements and must fit uint32; strides outside that, or an
// inner layout fitting neither form, materialize to a dense row-major
// batch first.
void classify_matmul_operand(
    const array& value,
    bool& transposed,
    uint32_t& gap,
    std::optional<array>& materialized,
    const std::string& name,
    array& out,
    const Stream& s) {
  int rank = value.ndim();
  int64_t s_prev = value.strides()[rank - 2];
  int64_t s_last = value.strides()[rank - 1];
  constexpr int64_t kMaxGap = std::numeric_limits<uint32_t>::max();
  if ((s_last == 1 || value.shape(rank - 1) == 1) && s_prev >= 0 &&
      s_prev <= kMaxGap) {
    transposed = false;
    gap = static_cast<uint32_t>(s_prev);
  } else if ((s_prev == 1 || value.shape(rank - 2) == 1) && s_last >= 0 &&
             s_last <= kMaxGap) {
    transposed = true;
    gap = static_cast<uint32_t>(s_last);
  } else {
    materialized = materialize_batched_matrix(value, name, out, s);
    transposed = false;
    gap = static_cast<uint32_t>(value.shape(rank - 1));
  }
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
  // Each operand's inner 2-D matrix rides the shader's gap indexing
  // and its batch axes ride their actual strides, so a cache slice
  // consumes zero copies; only an inner layout fitting neither the
  // row-major nor the column-major form materializes to a dense batch.
  std::optional<array> a_materialized;
  std::optional<array> b_materialized;
  uint32_t a_gap = 0;
  uint32_t b_gap = 0;
  bool a_transposed = false;
  bool b_transposed = false;
  classify_matmul_operand(
      a_in, a_transposed, a_gap, a_materialized, name, out, s);
  classify_matmul_operand(
      b_in, b_transposed, b_gap, b_materialized, name, out, s);
  const array& a = a_materialized ? *a_materialized : a_in;
  const array& b = b_materialized ? *b_materialized : b_in;
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
  params.lhs_gap = a_gap;
  params.rhs_gap = b_gap;
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
  // Inner spans use the actual gaps: (rows - 1) * gap + cols elements
  // past the operand offset. A degenerate inner dimension (m, n, or k
  // == 0) touches no elements, but the uint32 subtraction would wrap
  // zero into a huge span and spuriously refuse valid empty-K
  // matmuls, so those terms are zeroed outright. The shader's
  // per-element tile guards then load nothing and accumulate the
  // zero fill the empty-K contract requires.
  uint64_t a_inner = params.matrix_m == 0u || params.matrix_k == 0u
      ? 0u
      : (a_transposed
            ? static_cast<uint64_t>(params.matrix_k - 1) * a_gap +
                params.matrix_m
            : static_cast<uint64_t>(params.matrix_m - 1) * a_gap +
                params.matrix_k);
  uint64_t b_inner = params.matrix_k == 0u || params.matrix_n == 0u
      ? 0u
      : (b_transposed
            ? static_cast<uint64_t>(params.matrix_n - 1) * b_gap +
                params.matrix_k
            : static_cast<uint64_t>(params.matrix_k - 1) * b_gap +
                params.matrix_n);
  if (!omarchy::compute_index_span_fits(
          params.lhs_offset, a_span + a_inner) ||
      !omarchy::compute_index_span_fits(
          params.rhs_offset, b_span + b_inner)) {
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
  const array& in_lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& in_rhs = binary ? inputs.at(1) : in_lhs;
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(name, in_lhs, out, encoder);
  require_float_dtype(name, in_rhs, out, encoder);

  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs =
      ensure_dense(in_lhs, in_lhs.flags().contiguous, lhs_temp, encoder, s);
  const array& rhs = binary
      ? ensure_dense(in_rhs, in_rhs.flags().contiguous, rhs_temp, encoder, s)
      : lhs;


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

// Forward declaration: dispatch_logical is defined after this function but
// must be callable from the bool-input branch below.
void dispatch_logical(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out);

void dispatch_comparison(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out,
    uint32_t flags = 0) {
  const array& lhs = inputs.at(0);
  const array& rhs = inputs.at(1);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (out.dtype() != bool_) {
    omarchy::unsupported(name + " output dtype", out);
  }
  // Bool inputs run through compare_bool.comp, a dedicated kernel for
  // the comparison sextet. It uses a plain word store (no atomicOr, no
  // zero-fill requirement) and a 6-op selector, avoiding both the
  // Honeykrisp wide-selector store-coalescing miscompile and the
  // shared-accumulator hazard a logical_or.comp extension would carry.
  // The 6 masked C++ cases (test array basics, test array types, gguf
  // metadata, is close, random split, vmap comparison ops) route here.
  if (lhs.dtype() == bool_) {
    // Map ComparisonOperation onto compare_bool selector codes.
    // ComparisonOperation: Equal=0, GreaterEqual=1, Greater=2, Less=3,
    // LessEqual=4, NotEqual=5. compare_bool: Equal=0, NotEqual=1,
    // Greater=2, Less=3, GreaterEqual=4, LessEqual=5.
    uint bool_op;
    switch (operation) {
      case CompareEqual:        bool_op = 0u; break;
      case CompareNotEqual:     bool_op = 1u; break;
      case CompareGreater:      bool_op = 2u; break;
      case CompareLess:         bool_op = 3u; break;
      case CompareGreaterEqual: bool_op = 4u; break;
      case CompareLessEqual:    bool_op = 5u; break;
      default: omarchy::unsupported(name + " bool selector", out);
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
    params.operation = bool_op;
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
        omarchy::ComputeKernel::CompareBool,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(word_count));
    return;
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
  params.flags = flags;
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
  // The logical kernel accumulates each canonical output byte with
  // atomicOr, so the destination must start zeroed - a fresh
  // allocation holds whatever the allocator recycled. Probe-proven on
  // llvmpipe: isinf([0, 1, NaN]) returned [0, 0, 128] with a stale
  // 0x80 byte, poisoning the composed isclose chain. Same pattern as
  // the scatter bool materialization path.
  {
    uint32_t zero_words = checked_u32(
        (static_cast<uint64_t>(out.size()) + 3) / 4, name, out);
    omarchy::ComputeParams clear_params;
    clear_params.count = zero_words;
    std::array<omarchy::ComputeBinding, 1> clear_bindings{binding(out)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::ClearU32,
        clear_bindings,
        clear_params,
        omarchy::compute_dispatch_group_count(zero_words));
  }
  // One thread per output word. Honeykrisp coalesces adjacent word
  // stores into a 16-byte vector store and drops the inner three
  // lanes' values; atomicOr on the output word bypasses that path.
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
  IntAddOperation,
  IntMultiplyOperation,
  IntSquareOperation,
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
  const array& in_lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& in_rhs = binary ? inputs.at(1) : in_lhs;
  auto is_int_dtype = [](Dtype dtype) {
    return dtype == int32 || dtype == uint32;
  };
  if (!is_int_dtype(in_lhs.dtype()) || !is_int_dtype(in_rhs.dtype()) ||
      !is_int_dtype(out.dtype())) {
    omarchy::unsupported(name + " dtype", out);
  }
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs = ensure_dense(
      in_lhs,
      in_lhs.flags().contiguous,
      lhs_temp,
      encoder,
      out.primitive().stream());
  const array& rhs = binary
      ? ensure_dense(
            in_rhs,
            in_rhs.flags().contiguous,
            rhs_temp,
            encoder,
            out.primitive().stream())
      : lhs;
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

// Sort and ArgSort accept float32/float16/bfloat16 plus int32/uint32.
// ArgSort and ArgPartition emit uint32 indices, so the output must be
// uint32 and the dtype checks apply to the input only, the way ArgReduce
// checks its input. The value variants Sort and Partition keep the input
// dtype in the output.
void require_sort_dtype(
    const std::string& name,
    const array& input,
    const array& out,
    bool argsort,
    omarchy::CommandEncoder& encoder) {
  if (input.dtype() != float16 && input.dtype() != float32 &&
      input.dtype() != bfloat16 && input.dtype() != int32 &&
      input.dtype() != uint32) {
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
  if (argsort) {
    if (out.dtype() != uint32) {
      omarchy::unsupported(name + " output dtype", out);
    }
  } else if (input.dtype() != out.dtype()) {
    omarchy::unsupported(name + " dtype", out);
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
  std::optional<array> dense_temp;
  const array& src = ensure_dense(
      input,
      input.flags().row_contiguous,
      dense_temp,
      encoder,
      out.primitive().stream());
  size_t row_length = src.shape(-1);
  if (row_length > kSortMaxRowLength) {
    omarchy::unsupported("sort row length " + name, out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t rows = src.size() / row_length;
  uint32_t output_size = checked_u32(rows, name, out);
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), name, out);
  params.reduce_size = checked_u32(row_length, name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(src, src.size(), name, out);
  params.output_offset = checked_item_offset(out, out.size(), name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(src), binding(out)};
  omarchy::ComputeKernel kernel;
  if (input.dtype() == int32 || input.dtype() == uint32) {
    // 32-bit integer storage buffers need no capability extension.
    kernel = argsort ? (input.dtype() == int32
                            ? omarchy::ComputeKernel::ArgSortI32
                            : omarchy::ComputeKernel::ArgSortU32)
                     : (input.dtype() == int32
                            ? omarchy::ComputeKernel::SortI32
                            : omarchy::ComputeKernel::SortU32);
  } else if (argsort) {
    kernel = select_float_kernel(
        input.dtype(),
        omarchy::ComputeKernel::ArgSortF32,
        omarchy::ComputeKernel::ArgSortF16,
        omarchy::ComputeKernel::ArgSortBF16);
  } else {
    kernel = select_float_kernel(
        input.dtype(),
        omarchy::ComputeKernel::SortF32,
        omarchy::ComputeKernel::SortF16,
        omarchy::ComputeKernel::SortBF16);
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}
// Wide-row ArgPartition. One workgroup per row runs a binary search over
// the monotone unsigned key (with canonical -0.0 and NaN handling) and
// serially emits the indices. The shader handles any row length; the
// dispatch caps the row count at kMaxComputeGroupCountX (batches spill
// into a second batch loop, but no vocabulary-width model needs more than
// one batch of 65535 rows).


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
  std::optional<array> dense_temp;
  const array& src =
      ensure_dense(input, input.flags().row_contiguous, dense_temp, encoder, s);
  size_t row_length = src.shape(-1);
  size_t rows = src.size() / row_length;
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t output_size = checked_u32(rows, name, out);
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), name, out);
  params.reduce_size = checked_u32(row_length, name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(src, src.size(), name, out);
  params.output_offset = checked_item_offset(out, out.size(), name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(src), binding(out)};
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
// the kept input strides, out_strides[] the reduced input strides, dims
// the kept count, and matrix_m the reduced count. A broadcast view
// reduces correctly because its expanded axes carry stride 0.
//
// Single-invocation serial loops in the shader are bounded by the driver
// (the lavapipe runtime stops after roughly 2^15 trips for this kernel
// shape, depending on local size), so reduce_size is split across chunks
// when it exceeds kReduceGeneralChunkTrips. The kernel runs in two
// phases: the partial phase writes one accumulator per (output element,
// chunk) to a scratch buffer at binding 1; the combine phase reads that
// scratch and emits the final typed output (or word-packed bool output).
// When the whole reduce fits in a single chunk the partial phase skips
// scratch and writes the accumulator straight to the output, matching the
// pre-tiling behaviour for small reductions.
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
  uint32_t reduce_size_u32 =
      checked_u32(reduce_size, operation_name, out);
  constexpr uint32_t kReduceGeneralChunkTrips = 4096;
  uint32_t chunks = reduce_size_u32 == 0
      ? 1u
      : (reduce_size_u32 + kReduceGeneralChunkTrips - 1u) /
          kReduceGeneralChunkTrips;
  if (chunks == 0) {
    chunks = 1;
  }
  // The scratch buffer holds one partial per (output element, chunk);
  // the bool AnyAll variant uses 0/1 uints, every other variant uses
  // the input width (with float variants accumulating in float32, the
  // rule the suffix reduce and scan kernels follow).
  bool bool_out = out.dtype() == bool_;
  Dtype scratch_dtype = bool_out ? uint32 : input.dtype();
  uint64_t scratch_elems =
      static_cast<uint64_t>(output_size) * chunks;
  // 64 MiB partials cap keeps any single dispatch's working memory
  // within the device's reachable resident set on lavapipe; shapes that
  // demand more are refused by name so the failure is loud.
  if (scratch_elems > (1ull << 25)) {
    omarchy::unsupported(
        operation_name + " reduction split exceeds scratch budget", out);
  }
  array scratch(
      Shape{static_cast<int>(scratch_elems)}, scratch_dtype, nullptr, {});
  scratch.set_data(allocate_omarchy(scratch.nbytes()));
  encoder.add_temporary(scratch);

  omarchy::ComputeParams params;
  params.count = output_size;
  params.operation = operation;
  params.reduce_size = reduce_size_u32;
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(
      bound_input, input.size(), operation_name, out);
  params.output_offset = checked_item_offset(
      out, out.size(), operation_name, out);
  params.dims = static_cast<uint32_t>(kept.size());
  params.matrix_m = static_cast<uint32_t>(reduced.size());
  params.matrix_n = chunks;
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

  auto run_phase = [&](uint32_t flags, uint32_t dispatch_count) {
    params.flags = flags;
    std::array<omarchy::ComputeBinding, 3> bindings{
        binding(bound_input), binding(scratch), binding(out)};
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(dispatch_count));
  };

  // Bool AnyAll combines partials in a second phase so the output words
  // pack correctly; every other dtype combines per-element typed output.
  if (bool_out) {
    uint32_t word_count =
        checked_u32(
            (static_cast<uint64_t>(output_size) + 3) / 4, operation_name, out);
    if (chunks == 1) {
      run_phase(0u | 2u, word_count);
    } else {
      run_phase(0u, checked_u32(scratch_elems, operation_name, out));
      run_phase(1u, word_count);
    }
  } else {
    if (chunks == 1) {
      run_phase(0u | 2u, output_size);
    } else {
      run_phase(0u, checked_u32(scratch_elems, operation_name, out));
      run_phase(1u, output_size);
    }
  }
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
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> scales_temp;
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  std::optional<array> bias_temp;
  const array& x_d =
      ensure_dense(x, x.flags().row_contiguous, x_temp, encoder, s);
  const array& w_d =
      ensure_dense(w, w.flags().row_contiguous, w_temp, encoder, s);
  const array& scales_d = ensure_dense(
      scales, scales.flags().row_contiguous, scales_temp, encoder, s);
  const array* biases_d = biases.has_value()
      ? &ensure_dense(
              *biases,
              biases->flags().row_contiguous,
              bias_temp,
              encoder,
              s)
      : nullptr;
  const array& lhs_d =
      ensure_dense(lhs, lhs.flags().row_contiguous, lhs_temp, encoder, s);
  const array& rhs_d =
      ensure_dense(rhs, rhs.flags().row_contiguous, rhs_temp, encoder, s);

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t index_count = lhs_d.size();
  size_t scale_bytes = scales.nbytes();
  size_t bias_bytes = biases ? biases->nbytes() : 0;
  auto align4 = [](size_t value) { return (value + 3) & ~size_t(3); };
  size_t bias_base = align4(scale_bytes);
  size_t index_base = align4(bias_base + bias_bytes);
  size_t packed_bytes = index_base + 2 * index_count * 4;
  if (scales_d.offset() % 4 != 0 ||
      (biases_d && biases_d->offset() % 4 != 0) ||
      lhs_d.offset() % 4 != 0 || rhs_d.offset() % 4 != 0) {
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
      binding(scales_d).buffer,
      binding(packed).buffer,
      scale_bytes,
      static_cast<VkDeviceSize>(scales_d.offset()),
      0);
  if (biases_d) {
    encoder.copy_buffer(
        binding(*biases_d).buffer,
        binding(packed).buffer,
        bias_bytes,
        static_cast<VkDeviceSize>(biases_d->offset()),
        static_cast<VkDeviceSize>(bias_base));
  }
  encoder.copy_buffer(
      binding(lhs_d).buffer,
      binding(packed).buffer,
      index_count * 4,
      static_cast<VkDeviceSize>(lhs_d.offset()),
      static_cast<VkDeviceSize>(index_base));
  encoder.copy_buffer(
      binding(rhs_d).buffer,
      binding(packed).buffer,
      index_count * 4,
      static_cast<VkDeviceSize>(rhs_d.offset()),
      static_cast<VkDeviceSize>(index_base + index_count * 4));

  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), tag, out);
  params.lhs_size = checked_u32(x.size(), tag, out);
  params.rhs_size = checked_u32(w.size(), tag, out);
  params.reduce_size = static_cast<uint32_t>(group_size);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(x_d, x_d.size(), tag, out);
  params.rhs_offset = checked_item_offset(w_d, w_d.size(), tag, out);
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
      binding(x_d), binding(packed), binding(w_d), binding(out)};
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

// Complex64Transport. Keep in lockstep with the switch in
// shaders/complex_elementwise.comp.
enum ComplexOperation : uint32_t {
  ComplexConjugate,
  ComplexAdd,
  ComplexSubtract,
  ComplexMultiply,
  ComplexDivide,
  ComplexNegative,
};

// The params fill and dispatch behind the complex64 elementwise
// kernel, mirroring dispatch_float_elementwise_to: a vec2 element per
// item, the same broadcast transport, and the same contiguity rule.
// Unary callers alias lhs and rhs; the shader's lhs_size/rhs_size
// handling then keeps the modulo fast path honest.
void dispatch_complex_elementwise_to(
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
  encoder.dispatch_compute(
      omarchy::ComputeKernel::ComplexElementwise,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}

void dispatch_complex(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const array& in_lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& in_rhs = binary ? inputs.at(1) : in_lhs;
  auto& encoder = omarchy::get_command_encoder(s);
  if (in_lhs.dtype() != complex64 || in_rhs.dtype() != complex64 ||
      out.dtype() != complex64) {
    omarchy::unsupported(name + " complex64 dtype", out);
  }
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs =
      ensure_dense(in_lhs, in_lhs.flags().contiguous, lhs_temp, encoder, s);
  const array& rhs = binary
      ? ensure_dense(in_rhs, in_rhs.flags().contiguous, rhs_temp, encoder, s)
      : lhs;
  bool general_broadcast = binary
      ? (!is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out))
      : !is_trailing_broadcast(lhs, out);
  if (binary) {
    auto binary_type = get_binary_op_type(lhs, rhs);
    if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
      binary_type = BinaryOpType::General;
    }
    set_binary_op_output_data(
        lhs, rhs, out, binary_type, allocate_omarchy);
  } else if (general_broadcast) {
    out.set_data(allocate_omarchy(out.nbytes()));
  } else if (lhs.data_size() != lhs.size()) {
    out.set_data(allocate_omarchy(out.nbytes()));
  } else {
    set_unary_output_data(lhs, out, allocate_omarchy);
  }
  if (out.size() == 0) {
    return;
  }
  dispatch_complex_elementwise_to(
      name, operation, lhs, rhs, out, general_broadcast, encoder);
}

// Complex64 component extraction to float32 (ComplexReal and
// ComplexImag kernels; operation 0 keeps the real part, 1 the
// imaginary part). The input offset is a complex64 item offset and
// the output offset a float32 item offset, which is what the
// per-array checked_item_offset calls already produce.
void dispatch_complex_extract(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const array& in = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(s);
  if (in.dtype() != complex64 || out.dtype() != float32) {
    omarchy::unsupported(name + " dtype", out);
  }
  uint32_t count = checked_u32(out.size(), name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.output_size = count;
  params.lhs_offset = checked_item_offset(in, in.data_size(), name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(in), binding(in), binding(out)};
  auto kernel = operation == 0
      ? omarchy::ComputeKernel::ComplexReal
      : omarchy::ComputeKernel::ComplexImag;
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
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
void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexAdd, inputs, out, out.primitive().stream());
    return;
  }
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntAddOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), AddOperation, inputs, out, out.primitive().stream());
}
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
  auto [kth, axis] = state();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_sort_dtype("ArgPartition", input, out, true, encoder);
  if (axis != input.ndim() - 1) {
    omarchy::unsupported("non-suffix ArgPartition", out);
  }
  // Ops layer (mlx/ops.cpp argpartition) already validated kth in
  // [0, axis_size) so the row axis is non-empty.
  size_t row_length = input.shape(-1);
  if (row_length > kSortMaxRowLength) {
    // Wide-row vocabulary widths (top-k sampling for real language
    // models) need a selection algorithm rather than a full sort. A
    // radix-select kernel was in flight on 2026-09-02 but its shader
    // source was lost before it computed correct values, so this
    // refuses by name rather than returning a wrong kth.
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
  if (axis != input.ndim() - 1) {
    omarchy::unsupported("non-suffix " + operation_name, out);
  }
  std::optional<array> dense_temp;
  const array& src = ensure_dense(
      input,
      input.flags().row_contiguous,
      dense_temp,
      encoder,
      out.primitive().stream());
  size_t row_length = src.shape(-1);
  size_t rows = src.size() / row_length;
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
      src, src.size(), operation_name, out);
  params.output_offset = checked_item_offset(out, out.size(), operation_name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(src), binding(out)};
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
  require_sort_dtype("ArgSort", input, out, true, encoder);
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
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs_d =
      ensure_dense(lhs, lhs.flags().row_contiguous, lhs_temp, encoder, s);
  const array& rhs_d =
      ensure_dense(rhs, rhs.flags().row_contiguous, rhs_temp, encoder, s);

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
  size_t batch_count = lhs_d.size();
  if (out.size() == 0) {
    return;
  }
  if (batch_count > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported(tag + " batch count", out);
  }

  // Binding 2 packs the index words [all lhs | all rhs]; GatherMM has no
  // C operand, so its binding slot is free.
  if (lhs_d.offset() % 4 != 0 || rhs_d.offset() % 4 != 0) {
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
      binding(lhs_d).buffer,
      binding(indices).buffer,
      lhs_d.nbytes(),
      static_cast<VkDeviceSize>(lhs_d.offset()),
      0);
  encoder.copy_buffer(
      binding(rhs_d).buffer,
      binding(indices).buffer,
      rhs_d.nbytes(),
      static_cast<VkDeviceSize>(rhs_d.offset()),
      static_cast<VkDeviceSize>(lhs_d.nbytes()));

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
  // The segmented shader indexes rows by matrix_k, so only the natural
  // gaps pass through; anything else materializes first.
  std::optional<array> a_materialized;
  uint32_t a_gap = 0;
  bool a_transposed = false;
  classify_matmul_operand(
      a_in, a_transposed, a_gap, a_materialized, tag, out, s);
  uint32_t a_natural = a_transposed ? a_in.shape(0) : a_in.shape(1);
  if (!a_materialized && a_gap != a_natural) {
    a_materialized = materialize_batched_matrix(a_in, tag, out, s);
    a_transposed = false;
  }
  const array& a = a_materialized ? *a_materialized : a_in;
  std::optional<array> b_materialized;
  uint32_t b_gap = 0;
  bool b_transposed = false;
  classify_matmul_operand(
      b_in, b_transposed, b_gap, b_materialized, tag, out, s);
  uint32_t b_natural = b_transposed ? b_in.shape(0) : b_in.shape(1);
  if (!b_materialized && b_gap != b_natural) {
    b_materialized = materialize_batched_matrix(b_in, tag, out, s);
    b_transposed = false;
  }
  const array& b = b_materialized ? *b_materialized : b_in;
  std::optional<array> segments_temp;
  const array& segments_d = ensure_dense(
      segments, segments.flags().row_contiguous, segments_temp, encoder, s);

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  int m = a.shape(0);
  int k = a.shape(1);
  int n = b.shape(1);
  size_t num_segments = segments_d.size() / 2;
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
  params.aux_offset =
      checked_item_offset(segments_d, segments_d.size(), tag, out);
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.flags = (b_transposed ? 1u : 0u) | (a_transposed ? 4u : 0u);
  if (!omarchy::compute_index_span_fits(params.lhs_offset, a.size()) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, b.size())) {
    omarchy::unsupported(tag + " index span", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(a), binding(b), binding(segments_d), binding(out)};
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
namespace {

// Wave 7: linear algebra. Every factorization primitive runs one
// workgroup per batch matrix in float32 only, matching the upstream CPU
// dtype contract (float32/float64; float64 has no Omarchy transport).
// Iterative kernels report failure through a u32 status word that the
// host checks behind a synchronize, so a degenerate input raises a named
// error instead of returning a silently wrong factorization.

size_t linalg_batch_count(const Shape& shape) {
  size_t batch = 1;
  int batch_rank = static_cast<int>(shape.size()) - 2;
  for (int axis = 0; axis < batch_rank; ++axis) {
    batch *= static_cast<size_t>(shape[axis]);
  }
  return batch;
}

void linalg_require_f32(
    const std::string& name,
    const array& in,
    const array& out) {
  if (in.dtype() != float32) {
    omarchy::unsupported(name + " dtype", out);
  }
}

void linalg_copy_dense(
    const array& in,
    array& work,
    Stream stream) {
  copy_gpu_inplace(
      in,
      work,
      in.shape(),
      in.strides(),
      work.strides(),
      0,
      0,
      in.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream);
}

// Join pending work and turn any kernel-pinned status word into the
// named failure for this primitive.
void linalg_check_status(
    array& scratch,
    omarchy::CommandEncoder& encoder,
    const char* message,
    const array& out) {
  encoder.synchronize();
  const uint32_t* words = scratch.data<uint32_t>();
  for (size_t i = 0; i < scratch.size(); ++i) {
    if (words[i] != 0) {
      throw std::runtime_error(message);
    }
  }
}

} // namespace

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  linalg_require_f32(name(), in, out);
  if (in.shape(-1) != in.shape(-2)) {
    omarchy::unsupported(std::string("non-square ") + name(), out);
  }
  const uint32_t n = checked_u32(in.shape(-1), name(), out);
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), out);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  linalg_copy_dense(in, out, out.primitive().stream());
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  omarchy::ComputeParams params;
  params.matrix_n = n;
  params.output_size = batch;
  params.flags = state() ? 1u : 0u;
  std::array<omarchy::ComputeBinding, 2> bindings{
      binding(out), binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgCholeskyF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      "[Cholesky::eval_gpu] Cholesky decomposition requires a positive"
      " definite matrix.",
      out);
}
void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  omarchy::eval_compiled_tape(
      tape_, inputs_, outputs_, inputs, outputs, stream());
}
void Conjugate::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Upstream conjugate() returns real input unchanged before a
  // primitive is built (mlx/ops.cpp), so eval only ever sees
  // complex64 here. The op negates the imaginary component.
  dispatch_complex(
      name(),
      ComplexConjugate,
      inputs,
      out,
      out.primitive().stream());
}
// General direct convolution for the channels-last layouts upstream
// hands this primitive: input [N, (H,) W, C] and weight
// [O, (kH,) kW, C_per_group], both row-major. One thread owns one
// output element; a float32 accumulator walks the kernel window and
// index guards contribute zero for padding. The kernel covers every
// combination the public ops build: groups including the depthwise
// case, the flip that conv_transpose uses, input dilation, kernel
// dilation, strides, and asymmetric padding, over 1D and 2D spatial
// ranks. 1D rides the same kernel as a degenerate height of one,
// matching the upstream CPU reference where slow_conv_1D is
// slow_conv_2D with an extent-one height axis.
void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  const array& wt = inputs.at(1);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("Convolution", in, out, encoder);
  require_float_dtype("Convolution", wt, out, encoder);
  const int spatial = in.ndim() - 2;
  if (spatial > 2) {
    omarchy::unsupported("3-D Convolution", out);
  }
  // The flip (conv_transpose) path stays gated: the one-hot probe in
  // test_conv_general.cpp showed the kernel reading the wrong input
  // channels once flip and input dilation combine, and a wrong number
  // is never acceptable. Forward convolutions - grouped, depthwise,
  // 1-D, input-dilated, kernel-dilated, strided - run the general
  // kernel below.
  if (flip_) {
    omarchy::unsupported("transposed (flip) Convolution", out);
  }
  if (in.ndim() != wt.ndim() || spatial < 1) {
    omarchy::unsupported("Convolution shapes", out);
  }
  const bool one_d = spatial == 1;
  const int width_axis = one_d ? 1 : 2;
  const int channel_axis = one_d ? 2 : 3;

  // Materialize operands whose strides are not the standard
  // channels-last and O(HKW) row-major layouts; cache slices and
  // transposes compose that way. The engine keeps each temp alive
  // until the submission lands.
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

  const int batch = x.shape(0);
  const int in_height = one_d ? 1 : x.shape(1);
  const int in_width = x.shape(width_axis);
  const int in_channels = x.shape(channel_axis);
  const int out_channels = w.shape(0);
  const int kernel_height = one_d ? 1 : w.shape(1);
  const int kernel_width = w.shape(width_axis);
  const int in_channels_per_group = w.shape(channel_axis);
  if (groups_ <= 0 || in_channels_per_group * groups_ != in_channels ||
      out_channels % groups_ != 0 || out.shape(0) != batch ||
      out.shape(channel_axis) != out_channels) {
    omarchy::unsupported("Convolution shapes", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t total = checked_u32(out.size(), "Convolution", out);
  omarchy::ComputeParams params;
  // Push-constant routing mirrors shaders/conv.comp: spatial extents,
  // kernel window, and the conv parameters ride the generic dims
  // fields because the fixed ComputeParams layout has no conv block.
  // 1D packs a degenerate height of one, so every axis stays 2D.
  params.count = total;
  params.operation = checked_u32(kernel_width, "Convolution", out);
  params.reduce_size = checked_u32(in_channels_per_group, "Convolution", out);
  params.lhs_offset = checked_item_offset(x, x.size(), "Convolution", out);
  params.rhs_offset = checked_item_offset(w, w.size(), "Convolution", out);
  params.output_offset =
      checked_item_offset(out, out.size(), "Convolution", out);
  params.aux_size = checked_u32(in_width, "Convolution", out);
  params.aux_offset = checked_u32(kernel_height, "Convolution", out);
  params.matrix_m = checked_u32(batch, "Convolution", out);
  params.matrix_n = checked_u32(out_channels, "Convolution", out);
  params.matrix_k = checked_u32(in_height, "Convolution", out);
  // lhs_size/rhs_size carry the input dilation (transposed convs
  // express their stride this way), flags the kernel flip, and
  // out_strides[2] the group count.
  params.lhs_size =
      checked_u32(one_d ? 1 : input_dilation_[0], "Convolution", out);
  params.rhs_size = checked_u32(
      one_d ? input_dilation_[0] : input_dilation_[1], "Convolution", out);
  params.flags = flip_ ? 1u : 0u;
  params.dims = 2;
  params.shape[0] = checked_u32(one_d ? 1 : out.shape(1), "Convolution", out);
  params.shape[1] =
      checked_u32(one_d ? out.shape(1) : out.shape(2), "Convolution", out);
  params.shape[2] = checked_u32(one_d ? 0 : padding_lo_[0], "Convolution", out);
  params.shape[3] =
      checked_u32(one_d ? padding_lo_[0] : padding_lo_[1], "Convolution", out);
  params.in_strides[0] =
      checked_u32(one_d ? 1 : kernel_strides_[0], "Convolution", out);
  params.in_strides[1] = checked_u32(
      one_d ? kernel_strides_[0] : kernel_strides_[1], "Convolution", out);
  params.in_strides[2] =
      checked_u32(one_d ? 1 : kernel_dilation_[0], "Convolution", out);
  params.in_strides[3] = checked_u32(
      one_d ? kernel_dilation_[0] : kernel_dilation_[1], "Convolution", out);
  params.out_strides[0] =
      checked_u32(one_d ? 0 : padding_hi_[0], "Convolution", out);
  params.out_strides[1] = checked_u32(
      one_d ? padding_hi_[0] : padding_hi_[1], "Convolution", out);
  params.out_strides[2] = checked_u32(groups_, "Convolution", out);
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
// The GLSL built-in sin/cos keep upstream-grade accuracy only for
// arguments the driver's range reduction survives. Measured on the M1
// Honeykrisp (scalar sweep, 2026-09-02): built-in error 2.8e-5 at 1e3,
// 4.5e-4 at 12345, 4.8e-3 at 123457, then 1e-2 and worse toward 1e6
// and total collapse from there; llvmpipe stays accurate far higher,
// but the gate is one device-independent contract. kTrigArgumentLimit
// is 1e5, chosen for the consumer that matters: fast::RoPE is a
// fallback composition of exactly these sin/cos calls with
// inv_freq[0] = 1.0, so the limit is the positional ceiling - 1e5
// covers Qwen-class 32k contexts with 3x margin while refusing the
// band where the built-in returns garbage (error 1e-2 and collapsing).
// An in-shader Payne-Hanek fallback was probed on the same device and
// returns garbage of magnitude 1e15+ (its carry chain rides the
// dynamic-indexing shapes this driver miscompiles), so it stays dead
// code. Above the limit the op refuses by name; the compatibility
// matrix counts Sin/Cos as partial with the magnitude named error.
constexpr float kTrigArgumentLimit = 1.0e5f;

void trig_argument_gate(
    const std::string& name,
    const std::vector<array>& inputs,
    const array& out) {
  Stream stream = out.primitive().stream();
  array magnitude = astype(
      max(abs(inputs.at(0), stream), stream), float32, stream);
  magnitude.eval();
  // The magnitude is read on the host, so the stream must be ordered
  // here: array::item() is eval() plus an immediate mapped read with no
  // completion wait, and an unordered read races this gate's own
  // submission — on hardware it returned recycled-page garbage
  // (~1e9) instead of the real magnitude, aborting compiled 4-bit
  // generation (observed 2026-09-03, first hardware run of the compiled
  // path; llvmpipe executes synchronously and masked it). Same pattern
  // as the reduce host checks: host reads behind a synchronize.
  omarchy::get_command_encoder(stream).synchronize();
  float worst = magnitude.item<float>();
  if (worst > kTrigArgumentLimit) {
    throw std::runtime_error(
        "[omarchy] " + name + " argument magnitude " +
        std::to_string(worst) + " exceeds the built-in accuracy limit " +
        std::to_string(kTrigArgumentLimit) +
        " on this backend: the Vulkan driver's sin/cos range reduction"
        " is untrusted above it, the software Payne-Hanek fallback"
        " miscompiles on this driver, and no other accurate kernel"
        " exists. No silent wrong value and no silent CPU fallback"
        " occurs. Run it on an explicit CPU stream to use the CPU"
        " implementation.");
  }
}
void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  trig_argument_gate(name(), inputs, out);
  dispatch_elementwise(
      name(), CosOperation, inputs, out, out.primitive().stream());
}
OMARCHY_UNARY(Cosh, CoshOperation)
void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexDivide, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), DivideOperation, inputs, out, out.primitive().stream());
}
// DivMod produces the Python floor-division quotient and remainder as
// two same-shaped outputs (upstream DivMod: integral_op applies the
// floor fixup to the truncating quotient, float_op pairs floor(x/y)
// with the adjusted fmod). One kernel dispatch per output; the
// per-element index mapping is 1:1, so a donated input buffer stays
// safe to read and write within one invocation.
void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in_lhs = inputs.at(0);
  const array& in_rhs = inputs.at(1);
  array& quotient = outputs.at(0);
  array& remainder = outputs.at(1);
  auto& encoder = omarchy::get_command_encoder(quotient.primitive().stream());
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs = ensure_dense(
      in_lhs,
      in_lhs.flags().contiguous,
      lhs_temp,
      encoder,
      quotient.primitive().stream());
  const array& rhs = ensure_dense(
      in_rhs,
      in_rhs.flags().contiguous,
      rhs_temp,
      encoder,
      quotient.primitive().stream());
  auto is_int_dtype = [](Dtype dtype) {
    return dtype == int32 || dtype == uint32;
  };
  if (is_int_dtype(lhs.dtype()) && is_int_dtype(rhs.dtype()) &&
      is_int_dtype(quotient.dtype()) && is_int_dtype(remainder.dtype())) {
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
  // equal_nan_ lives on the Equal primitive; ops.cpp sets it from
  // mx.array_equal(..., equal_nan=True) and mx.allclose(..., equal_nan=True),
  // and clears it whenever the dtype is not inexact. The kernel honors
  // params.flags bit 0 in compare.comp.
  dispatch_comparison(name(), CompareEqual, inputs, out, equal_nan_ ? 1u : 0u);
}
OMARCHY_UNARY(Erf, ErfOperation)
OMARCHY_UNARY(ErfInv, ErfInvOperation)
OMARCHY_UNARY(Exp, ExpOperation)
OMARCHY_UNARY(Expm1, Expm1Operation)
namespace {

// Wave 8 FFT, extended to general lengths. The FftF32 kernel runs a radix-2
// Cooley-Tukey decimation-in-time pass over shared memory: one workgroup
// computes one 1-D transform of a power-of-two length up to 2048 (16 KiB of
// shared state, the Vulkan minimum guarantee). Longer and non-power-of-two
// lengths decompose at the C++ level:
//   - a composite n factors into stages of the radix-2 pass, Cooley-Tukey
//     style (strided pass, twiddle multiply, recurse), which lifts the old
//     shared-memory cap without any larger footprint; and
//   - a length with no divisor in 2..2048 (a prime, in practice) embeds in a
//     power-of-two circular convolution, Bluestein's chirp-z transform,
//     built from the same radix-2 pass plus elementwise stage kernels.
// Pass flags mirror what the CPU primitive does per call
// (mlx/backend/cpu/fft.cpp): every inverse pass divides by its own axis
// length, so a multi-axis inverse accumulates 1/(n0*n1*...) exactly like the
// upstream scale = 1/nelem.
constexpr uint32_t kFftFlagInverse = 1u;
constexpr uint32_t kFftFlagInputReal = 2u;
constexpr uint32_t kFftFlagInputHalf = 4u;
constexpr uint32_t kFftFlagOutputHalf = 8u;
// Longest single radix-2 pass: bounded by the 16 KiB shared-memory floor,
// not by workgroup size (the kernel strides threads over any length).
constexpr uint32_t kFftMaxDirectLength = 2048;
// Longest Bluestein length: the chirp reduces k*k mod 2n in u32, which
// stays exact only while k < 65536, i.e. while the padded convolution
// length next_pow2(2n-1) stays at or under 65536.
constexpr uint32_t kFftMaxBluesteinLength = 32768;
// Longest decomposed length: Cooley-Tukey twiddle phases are products
// m0*T < n, which must stay exactly representable in float32.
constexpr uint64_t kFftMaxTotalLength = 1ull << 24;

// (sample stride, transform count) of one axis pass over a dense
// row-major buffer of the given shape.
std::pair<uint32_t, uint32_t> fft_pass_geometry(
    const Shape& shape,
    int axis,
    uint32_t length,
    const std::string& name,
    array& out) {
  uint64_t total = 1;
  uint64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    total *= shape[i];
    if (i > axis) {
      stride *= shape[i];
    }
  }
  uint64_t count = total / length;
  if (stride > 0xffffffffull || count > 0xffffffffull) {
    omarchy::unsupported(name + " transform geometry", out);
  }
  return {static_cast<uint32_t>(stride), static_cast<uint32_t>(count)};
}

// Fresh dense row-major buffer for intermediate spectra.
array make_fft_temp(Shape shape, Dtype dtype, const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  Strides strides(shape.size(), 1);
  for (int axis = static_cast<int>(shape.size()) - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  }
  array temp(std::move(shape), dtype, nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(temp.shape().begin(), temp.shape().end());
  flags.col_contiguous = temp.size() <= 1 || temp.size() == *max_dim;
  temp.set_data(
      omarchy::allocator().malloc(temp.nbytes()),
      temp.size(),
      strides,
      flags,
      0);
  encoder.add_temporary(temp);
  return temp;
}

// Dense row-major copy of a float32 FFT input (sliced or transposed
// views). complex64 views keep their named error: the strided copy engine
// has no complex64 path.
array make_fft_input_dense(
    const array& src,
    const std::string& name,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  Shape shape = src.shape();
  Strides strides(shape.size(), 1);
  for (int axis = static_cast<int>(shape.size()) - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  }
  array dense(shape, src.dtype(), nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(shape.begin(), shape.end());
  flags.col_contiguous = dense.size() <= 1 || dense.size() == *max_dim;
  dense.set_data(
      omarchy::allocator().malloc(dense.nbytes()),
      dense.size(),
      strides,
      flags,
      0);
  copy_gpu_inplace(
      src,
      dense,
      shape,
      src.strides(),
      strides,
      /*i_offset=*/0,
      /*o_offset=*/0,
      CopyType::General,
      s);
  encoder.add_temporary(dense);
  return dense;
}

// One radix-2 pass: transform_count workgroups, each transforming
// `length` samples spaced `stride` elements apart inside a dense buffer.
void dispatch_fft_pass(
    const array& src,
    array& dst,
    uint32_t length,
    uint32_t stride,
    uint32_t transform_count,
    uint32_t flags,
    const std::string& name,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  omarchy::ComputeParams params;
  params.count = transform_count;
  params.operation = flags;
  params.lhs_size = length;
  params.rhs_size = stride;
  params.lhs_offset = checked_item_offset(src, 0, name, out);
  params.output_offset = checked_item_offset(dst, 0, name, out);
  uint32_t group_x =
      std::min<uint32_t>(transform_count, omarchy::kMaxComputeGroupCountX);
  uint32_t remaining = (transform_count + group_x - 1) / group_x;
  uint32_t group_y =
      std::min<uint32_t>(remaining, omarchy::kMaxComputeGroupCountX);
  uint32_t group_z = (remaining + group_y - 1) / group_y;
  if (group_z > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported(name + " transform count", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(src),
      omarchy::ComputeBinding{},
      binding(dst),
      omarchy::ComputeBinding{}};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::FftF32,
      bindings,
      params,
      group_x,
      group_y,
      group_z);
}

struct FftStage {
  enum class Kind : uint8_t {
    Pass,     // radix-2 pass of `length` points on samples spaced
              // axis_stride * stride_mul apart
    Twiddle,  // pointwise w_n^(m0*T) multiply for the split n = (above)*inner
    Bluestein,// whole-length chirp-z convolution for a prime-class length
    Permute   // undo the level's factor transpose: slot j1 + inner*k0 holds
              // the bin for k0 + (length/inner)*j1
  };
  Kind kind;
  uint32_t length;     // Pass/Bluestein/Permute: transform length; Twiddle: n
  uint32_t inner;      // Twiddle/Permute: inner extent M of the split
  uint32_t stride_mul; // Pass/Twiddle/Permute: multiplier on the axis stride
};

// Modes of the fft_stage_f32 elementwise kernel (reduce_size push constant).
constexpr uint32_t kFftStageTwiddle = 0;
constexpr uint32_t kFftStageBluesteinA = 1;
constexpr uint32_t kFftStageBluesteinB = 2;
constexpr uint32_t kFftStageMultiply = 3;
constexpr uint32_t kFftStageBluesteinY = 4;
constexpr uint32_t kFftStagePermute = 5;
constexpr uint32_t kFftStageRealEdge = 6;
// Append the stage list for one n-point transform whose samples sit at
// axis_stride * stride_mul apart. Refuses by name any length the two
// mechanisms cannot carry.
void plan_fft_stages(
    uint64_t n,
    uint32_t stride_mul,
    std::vector<FftStage>& stages,
    const std::string& name,
    array& out) {
  if (n <= kFftMaxDirectLength && (n & (n - 1)) == 0) {
    stages.push_back(
        {FftStage::Kind::Pass,
         static_cast<uint32_t>(n),
         0,
         stride_mul});
    return;
  }
  // Largest factor that splits n into two strictly smaller transforms.
  uint64_t factor = 0;
  uint64_t limit = std::min<uint64_t>(kFftMaxDirectLength, n / 2);
  for (uint64_t d = limit; d >= 2; --d) {
    if (n % d == 0) {
      factor = d;
      break;
    }
  }
  if (factor != 0) {
    uint64_t rest = n / factor;
    // Outer factor first: transform along it, twiddle, then the rest.
    // The twiddle addresses this level's transform-local elements, whose
    // dense spacing carries this level's stride multiplier. The level
    // leaves its output transposed (slot j1 + rest*k0 holds the bin for
    // k0 + factor*j1), so it ends with its own permute stage.
    plan_fft_stages(factor, stride_mul * rest, stages, name, out);
    stages.push_back(
        {FftStage::Kind::Twiddle,
         static_cast<uint32_t>(n),
         static_cast<uint32_t>(rest),
         stride_mul});
    plan_fft_stages(rest, stride_mul, stages, name, out);
    stages.push_back(
        {FftStage::Kind::Permute,
         static_cast<uint32_t>(n),
         static_cast<uint32_t>(rest),
         stride_mul});
    return;
  }
  if (n <= kFftMaxBluesteinLength) {
    stages.push_back(
        {FftStage::Kind::Bluestein,
         static_cast<uint32_t>(n),
         0,
         stride_mul});
    return;
  }
  omarchy::unsupported(
      name + " transform length " + std::to_string(n) +
          " (composite lengths to 2^24 decompose into radix-2 passes;"
          " prime-class lengths to 32768 embed in a Bluestein chirp-z"
          " convolution; this length has neither a divisor in 2..2048 nor"
          " Bluestein chirp headroom)",
      out);
}

std::vector<FftStage> plan_fft_axis(uint64_t n, const std::string& name, array& out) {
  std::vector<FftStage> stages;
  if (n == 0) {
    return stages;
  }
  if (n > kFftMaxTotalLength) {
    omarchy::unsupported(
        name + " transform length " + std::to_string(n) +
            " (decomposed Cooley-Tukey twiddle phases need lengths of at"
            " most 2^24 to stay exact in float32)",
        out);
  }
  plan_fft_stages(n, 1, stages, name, out);
  return stages;
}

// One elementwise fft_stage_f32 dispatch: one thread per element.
void dispatch_fft_stage(
    const array& src,
    array& dst,
    uint64_t elements,
    uint32_t mode,
    uint32_t flags,
    uint32_t n,
    uint32_t aux,
    uint32_t stride,
    const std::string& name,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  omarchy::ComputeParams params;
  params.count = checked_u32(elements, name, out);
  params.operation = flags;
  params.lhs_size = n;
  params.rhs_size = stride;
  params.reduce_size = mode;
  params.output_size = aux;
  params.lhs_offset = checked_item_offset(src, 0, name, out);
  params.output_offset = checked_item_offset(dst, 0, name, out);
  uint32_t workgroups = checked_u32((elements + 255) / 256, name, out);
  uint32_t group_x =
      std::min<uint32_t>(workgroups, omarchy::kMaxComputeGroupCountX);
  uint32_t remaining = (workgroups + group_x - 1) / group_x;
  uint32_t group_y =
      std::min<uint32_t>(remaining, omarchy::kMaxComputeGroupCountX);
  uint32_t group_z = (remaining + group_y - 1) / group_y;
  if (group_z > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported(name + " stage element count", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(src),
      omarchy::ComputeBinding{},
      binding(dst),
      omarchy::ComputeBinding{}};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::FftStageF32,
      bindings,
      params,
      group_x,
      group_y,
      group_z);
}

// Bluestein chirp-z: embeds the n-point transform in a circular
// convolution of power-of-two length m = next_pow2(2n-1) built from the
// radix-2 pass. X = w * IDFT(DFT(a) * DFT(b)) with a = x*w and b the
// conjugate chirp; the epilogue applies w and the direction scale. The
// convolution buffers are packed, one m-row per transform.
void run_fft_bluestein(
    const array& src,
    array& dst,
    uint32_t n,
    uint32_t stride,
    uint32_t base_flags,
    uint32_t first_extra,
    uint32_t last_extra,
    uint64_t full_elements,
    const std::string& name,
    array& out,
    const Stream& s) {
  uint64_t transforms = full_elements / n;
  uint64_t m = 1;
  while (m < 2 * uint64_t(n) - 1) {
    m <<= 1;
  }
  uint64_t conv_elements = transforms * m;
  // Shape is int-backed, so refuse batches whose padding outgrows it.
  if (conv_elements > 0x7ffffffull) {
    omarchy::unsupported(name + " Bluestein batch", out);
  }
  Shape conv_shape{static_cast<int>(conv_elements)};
  std::vector<FftStage> conv_plan;
  plan_fft_stages(m, 1, conv_plan, name, out);

  array chirped = make_fft_temp(conv_shape, complex64, s);
  array chirp_kernel = make_fft_temp(conv_shape, complex64, s);
  array spectrum = make_fft_temp(conv_shape, complex64, s);
  array conv = make_fft_temp(conv_shape, complex64, s);

  // One full m-point pipeline (pass/twiddle stages on a packed buffer,
  // axis stride 1) from `from` into `to` under `flags`.
  auto run_conv_pipeline = [&](const array& from, array& to, uint32_t flags) {
    const array* cur = &from;
    std::optional<array> ping;
    for (size_t i = 0; i < conv_plan.size(); ++i) {
      const FftStage& stage = conv_plan[i];
      bool last = i + 1 == conv_plan.size();
      array* target;
      if (last) {
        target = &to;
      } else {
        if (!ping) {
          ping = make_fft_temp(conv_shape, complex64, s);
        }
        target = &*ping;
      }
      if (stage.kind == FftStage::Kind::Pass) {
        dispatch_fft_pass(
            *cur,
            *target,
            stage.length,
            stage.stride_mul,
            checked_u32(conv_elements / stage.length, name, out),
            flags,
            name,
            out,
            s);
      } else if (stage.kind == FftStage::Kind::Twiddle) {
        dispatch_fft_stage(
            *cur,
            *target,
            conv_elements,
            kFftStageTwiddle,
            flags,
            stage.length,
            stage.inner,
            stage.stride_mul,
            name,
            out,
            s);
      } else {
        dispatch_fft_stage(
            *cur,
            *target,
            conv_elements,
            kFftStagePermute,
            flags,
            stage.length,
            stage.inner,
            stage.stride_mul,
            name,
            out,
            s);
      }
      cur = target;
    }
  };

  dispatch_fft_stage(
      src,
      chirped,
      conv_elements,
      kFftStageBluesteinA,
      base_flags | first_extra,
      n,
      static_cast<uint32_t>(m),
      stride,
      name,
      out,
      s);
  dispatch_fft_stage(
      chirped,
      chirp_kernel,
      conv_elements,
      kFftStageBluesteinB,
      base_flags,
      n,
      static_cast<uint32_t>(m),
      stride,
      name,
      out,
      s);
  run_conv_pipeline(chirped, spectrum, base_flags);
  run_conv_pipeline(chirp_kernel, chirped, base_flags);
  // Multiply the spectra in place (per-element read-then-write): the
  // product lands in `chirped`.
  dispatch_fft_stage(
      spectrum,
      chirped,
      conv_elements,
      kFftStageMultiply,
      base_flags,
      n,
      static_cast<uint32_t>(m),
      stride,
      name,
      out,
      s);
  // Inverse of the transform direction: flip the direction/scale bit.
  // Forward pipelines need the scaled inverse pass (total 1/m); inverse
  // pipelines need the unscaled forward pass, with the m/n folding left
  // for the epilogue.
  run_conv_pipeline(chirped, conv, base_flags ^ kFftFlagInverse);
  dispatch_fft_stage(
      conv,
      dst,
      transforms * n,
      kFftStageBluesteinY,
      base_flags | last_extra,
      n,
      static_cast<uint32_t>(m),
      stride,
      name,
      out,
      s);
}

// Run one planned axis pipeline from src into dst. first_extra flags the
// pipeline's first dispatch (a real or half-spectrum input side), last_extra
// its last (a packed half-spectrum output side). full_elements is the dense
// geometry the stages address; scratch ping buffers take scratch_shape.
void run_fft_axis(
    const array& src,
    array& dst,
    const std::vector<FftStage>& stages,
    uint32_t n,
    uint32_t stride,
    uint32_t base_flags,
    uint32_t first_extra,
    uint32_t last_extra,
    uint64_t full_elements,
    const Shape& scratch_shape,
    const std::string& name,
    array& out,
    const Stream& s) {
  const array* cur = &src;
  std::optional<array> ping[2];
  for (size_t i = 0; i < stages.size(); ++i) {
    const FftStage& stage = stages[i];
    bool first = i == 0;
    bool last = i + 1 == stages.size();
    array* target;
    if (last) {
      target = &dst;
    } else {
      auto& slot = ping[i % 2];
      if (!slot) {
        slot = make_fft_temp(scratch_shape, complex64, s);
      }
      target = &*slot;
    }
    switch (stage.kind) {
      case FftStage::Kind::Pass: {
        uint32_t flags = base_flags;
        if (first) {
          flags |= first_extra;
        }
        if (last) {
          flags |= last_extra;
        }
        // Transforms tile the buffer exactly once (the pass kernel's base
        // formula interleaves them across the sample spacing), so the
        // count is elements over transform length regardless of stride.
        uint64_t count = full_elements / stage.length;
        dispatch_fft_pass(
            *cur,
            *target,
            stage.length,
            stride * stage.stride_mul,
            checked_u32(count, name, out),
            flags,
            name,
            out,
            s);
        break;
      }
      case FftStage::Kind::Twiddle:
        dispatch_fft_stage(
            *cur,
            *target,
            full_elements,
            kFftStageTwiddle,
            base_flags,
            stage.length,
            stage.inner,
            stride * stage.stride_mul,
            name,
            out,
            s);
        break;
      case FftStage::Kind::Bluestein:
        run_fft_bluestein(
            *cur,
            *target,
            stage.length,
            stride * stage.stride_mul,
            base_flags,
            first ? first_extra : 0u,
            last ? last_extra : 0u,
            full_elements,
            name,
            out,
            s);
        break;
      case FftStage::Kind::Permute: {
        uint32_t flags = base_flags;
        if (last) {
          flags |= last_extra;
        }
        // A packed half-spectrum destination gathers only bins 0..n/2
        // per transform; otherwise every element moves exactly once.
        uint64_t elements = full_elements;
        if (last && (last_extra & kFftFlagOutputHalf) != 0u) {
          elements = full_elements / stage.length * (stage.length / 2u + 1u);
        }
        dispatch_fft_stage(
            *cur,
            *target,
            elements,
            kFftStagePermute,
            flags,
            stage.length,
            stage.inner,
            stride * stage.stride_mul,
            name,
            out,
            s);
        break;
      }
    }
    cur = target;
  }
}

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  const std::string tag = name();
  Stream s = out.primitive().stream();

  // The op layer (mlx/fft.cpp fft_impl) builds exactly three dtype
  // combinations; anything else keeps the named error.
  Dtype in_type = real_ && !inverse_ ? float32 : complex64;
  Dtype out_type = real_ && inverse_ ? float32 : complex64;
  if (in.dtype() != in_type || out.dtype() != out_type) {
    omarchy::unsupported(tag + " dtype combination", out);
  }

  checked_u32(in.size(), tag, out);
  checked_u32(out.size(), tag, out);

  int rank = in.ndim();
  for (size_t axis : axes_) {
    if (static_cast<int>(axis) >= rank) {
      omarchy::unsupported(tag + " axis out of range", out);
    }
  }

  // Transform length per axis. For the real variants the trailing axis of
  // the axes list is special: rfft reads n samples and writes n/2+1 bins,
  // irfft reads n/2+1 bins and writes n samples. The other axes carry the
  // same length on both sides (the op layer guarantees it; the checks
  // keep a caller bug from turning into silent garbage).
  std::vector<uint32_t> lengths;
  lengths.reserve(axes_.size());
  for (size_t i = 0; i < axes_.size(); ++i) {
    size_t axis = axes_[i];
    bool real_axis = real_ && i + 1 == axes_.size();
    if (!real_axis) {
      if (in.shape(axis) != out.shape(axis)) {
        omarchy::unsupported(tag + " shape mismatch", out);
      }
      lengths.push_back(static_cast<uint32_t>(in.shape(axis)));
    } else if (!inverse_) {
      lengths.push_back(static_cast<uint32_t>(in.shape(axis)));
      if (out.shape(axis) != lengths.back() / 2 + 1) {
        omarchy::unsupported(tag + " rfft output shape", out);
      }
    } else {
      lengths.push_back(static_cast<uint32_t>(out.shape(axis)));
      if (in.shape(axis) != lengths.back() / 2 + 1) {
        omarchy::unsupported(tag + " irfft input shape", out);
      }
    }
  }
  // Plan every axis before touching memory: a length class this backend
  // refuses must fail here, by name, before any allocation.
  std::vector<std::vector<FftStage>> plans;
  plans.reserve(lengths.size());
  for (uint32_t length : lengths) {
    plans.push_back(plan_fft_axis(length, tag, out));
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  // The kernels address dense row-major buffers. Sliced or transposed
  // inputs of either dtype materialize through the strided copy
  // engine, which now carries complex64 (Complex64Transport).
  const array* src = &in;
  std::optional<array> dense_input;
  if (!(in.flags().row_contiguous && in.data_size() == in.size())) {
    dense_input = make_fft_input_dense(in, tag, out, s);
    src = &*dense_input;
  }

  if (!real_) {
    // Complex-to-complex: one planned pipeline per axis, with the last
    // pipeline landing in the output.
    const array* cur = src;
    std::optional<array> temps[2];
    for (size_t i = 0; i < axes_.size(); ++i) {
      size_t axis = axes_[i];
      auto [stride, count] =
          fft_pass_geometry(in.shape(), axis, lengths[i], tag, out);
      (void)count;
      array* dst;
      if (i + 1 == axes_.size()) {
        dst = &out;
      } else {
        auto& slot = temps[i % 2];
        if (!slot) {
          slot = make_fft_temp(Shape(in.shape()), complex64, s);
        }
        dst = &*slot;
      }
      run_fft_axis(
          *cur,
          *dst,
          plans[i],
          lengths[i],
          stride,
          inverse_ ? kFftFlagInverse : 0u,
          0,
          0,
          in.size(),
          Shape(in.shape()),
          tag,
          out,
          s);
      cur = dst;
    }
  } else if (!inverse_) {
    // Real-to-complex: promote-and-forward on the real axis (kept to
    // bins 0..n/2), then forward pipelines over the remaining axes of
    // the half-spectrum. A single radix-2 pass carries the real-promote
    // and packed-half store in-flight; a multi-stage pipeline promotes
    // up front with an edge kernel, runs plain complex, and truncates
    // with a second edge kernel, because the packed half-spectrum only
    // aligns with the pass layout for a natural-ordered single pass.
    size_t real_axis = axes_.back();
    auto [real_stride, _] = fft_pass_geometry(
        in.shape(), real_axis, lengths.back(), tag, out);
    const std::vector<FftStage>& real_plan = plans.back();
    bool single_pass_real = real_plan.size() == 1 &&
        real_plan.front().kind == FftStage::Kind::Pass;
    array real_owned = make_fft_temp(Shape(out.shape()), complex64, s);
    array& real_dst = axes_.size() == 1 ? out : real_owned;
    if (single_pass_real) {
      run_fft_axis(
          *src,
          real_dst,
          real_plan,
          lengths.back(),
          real_stride,
          0,
          kFftFlagInputReal,
          kFftFlagOutputHalf,
          in.size(),
          Shape(in.shape()),
          tag,
          out,
          s);
    } else {
      array promoted = make_fft_temp(Shape(in.shape()), complex64, s);
      dispatch_fft_stage(
          *src,
          promoted,
          in.size(),
          kFftStageRealEdge,
          kFftFlagInputReal,
          lengths.back(),
          0,
          real_stride,
          tag,
          out,
          s);
      array real_full = make_fft_temp(Shape(in.shape()), complex64, s);
      run_fft_axis(
          promoted,
          real_full,
          real_plan,
          lengths.back(),
          real_stride,
          0,
          0,
          0,
          in.size(),
          Shape(in.shape()),
          tag,
          out,
          s);
      dispatch_fft_stage(
          real_full,
          real_dst,
          out.size(),
          kFftStageRealEdge,
          0,
          lengths.back(),
          0,
          real_stride,
          tag,
          out,
          s);
    }
    const array* cur = &real_dst;
    if (axes_.size() > 1) {
      std::optional<array> temps[2];
      for (size_t i = 0; i + 1 < axes_.size(); ++i) {
        size_t axis = axes_[i];
        auto [stride, count] =
            fft_pass_geometry(out.shape(), axis, lengths[i], tag, out);
        (void)count;
        array* dst;
        bool final_axis = i + 2 == axes_.size();
        if (final_axis) {
          dst = &out;
        } else {
          auto& slot = temps[i % 2];
          if (!slot) {
            slot = make_fft_temp(Shape(out.shape()), complex64, s);
          }
          dst = &*slot;
        }
        run_fft_axis(
            *cur,
            *dst,
            plans[i],
            lengths[i],
            stride,
            0,
            0,
            0,
            out.size(),
            Shape(out.shape()),
            tag,
            out,
            s);
        cur = dst;
      }
    }
  } else {
    // Complex-to-real: inverse pipelines on the non-real axes over the
    // half-spectrum shape, then the real-axis pipeline synthesizes the
    // full Hermitian spectrum (INPUT_HALF) and inverts it; the real
    // part of the result is extracted to the float32 output.
    const array* cur = src;
    std::optional<array> temps[2];
    for (size_t i = 0; i + 1 < axes_.size(); ++i) {
      size_t axis = axes_[i];
      auto [stride, count] =
          fft_pass_geometry(in.shape(), axis, lengths[i], tag, out);
      (void)count;
      auto& slot = temps[i % 2];
      if (!slot) {
        slot = make_fft_temp(Shape(in.shape()), complex64, s);
      }
      run_fft_axis(
          *cur,
          *slot,
          plans[i],
          lengths[i],
          stride,
          kFftFlagInverse,
          0,
          0,
          in.size(),
          Shape(in.shape()),
          tag,
          out,
          s);
      cur = &*slot;
    }
    size_t real_axis = axes_.back();
    auto [real_stride, _] =
        fft_pass_geometry(out.shape(), real_axis, lengths.back(), tag, out);
    array full = make_fft_temp(Shape(out.shape()), complex64, s);
    const std::vector<FftStage>& real_plan = plans.back();
    if (real_plan.size() == 1 && real_plan.front().kind == FftStage::Kind::Pass) {
      // Single radix-2 pass: the packed half-spectrum input synthesizes
      // to the full Hermitian spectrum in-flight at load.
      run_fft_axis(
          *cur,
          full,
          real_plan,
          lengths.back(),
          real_stride,
          kFftFlagInverse,
          kFftFlagInputHalf,
          0,
          out.size(),
          Shape(out.shape()),
          tag,
          out,
          s);
    } else {
      // Multi-stage pipeline: expand the packed half spectrum to the
      // full Hermitian spectrum with an edge kernel, then run plain
      // complex. The edge kernel writes every full slot exactly once.
      array herm = make_fft_temp(Shape(out.shape()), complex64, s);
      dispatch_fft_stage(
          *cur,
          herm,
          out.size(),
          kFftStageRealEdge,
          kFftFlagInputHalf,
          lengths.back(),
          0,
          real_stride,
          tag,
          out,
          s);
      run_fft_axis(
          herm,
          full,
          real_plan,
          lengths.back(),
          real_stride,
          kFftFlagInverse,
          0,
          0,
          out.size(),
          Shape(out.shape()),
          tag,
          out,
          s);
    }
    if (out.size() > 65535ull * 256ull) {
      omarchy::unsupported(tag + " output size", out);
    }
    omarchy::ComputeParams extract_params;
    extract_params.count = checked_u32(out.size(), tag, out);
    extract_params.lhs_offset = checked_item_offset(full, 0, tag, out);
    extract_params.output_offset = checked_item_offset(out, 0, tag, out);
    std::array<omarchy::ComputeBinding, 4> extract_bindings{
        binding(full),
        omarchy::ComputeBinding{},
        binding(out),
        omarchy::ComputeBinding{}};
    auto& encoder = omarchy::get_command_encoder(s);
    encoder.dispatch_compute(
        omarchy::ComputeKernel::FftRealF32,
        extract_bindings,
        extract_params,
        omarchy::compute_dispatch_group_count(extract_params.count));
  }
}
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
  if (table.ndim() != 2) {
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
  std::optional<array> table_temp;
  std::optional<array> indices_temp;
  const array& table_d = ensure_dense(
      table, table.flags().row_contiguous, table_temp, encoder, stream());
  const array& indices_d = ensure_dense(
      indices,
      indices.flags().row_contiguous,
      indices_temp,
      encoder,
      stream());
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
  params.lhs_offset = checked_item_offset(table_d, table_d.size(), "Take", out);
  // The shader indexes 32-bit words, so the int64 element offset
  // doubles into word units.
  uint32_t index_offset = checked_item_offset(
      indices_d, indices_d.size(), "Take", out);
  if (index_mode == 2) {
    if (index_offset > std::numeric_limits<uint32_t>::max() / 2) {
      omarchy::unsupported("indexed Take index span", out);
    }
    index_offset *= 2;
  }
  params.rhs_offset = index_offset;
  params.output_offset = checked_item_offset(out, out.size(), "Take", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(table_d), binding(indices_d), binding(out)};
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
  std::optional<array> indices_temp;
  const array& indices_d = ensure_dense(
      indices,
      indices.flags().row_contiguous && indices.data_size() == indices.size(),
      indices_temp,
      encoder,
      out.primitive().stream());
  int non_axis = out.ndim() - 1;
  if (non_axis > 4) {
    // shape[] and in_strides[] cap at four slots for the non-axis walk.
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
      checked_item_offset(indices_d, indices_d.size(), "Take", out);
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
      binding(src), binding(indices_d), binding(out)};
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
  std::optional<array> dense_temp;
  const array& src = ensure_dense(
      input,
      input.flags().row_contiguous,
      dense_temp,
      encoder,
      out.primitive().stream());
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
      binding(src).buffer,
      binding(out).buffer,
      static_cast<VkDeviceSize>(src.nbytes()),
      static_cast<VkDeviceSize>(src.offset()),
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
void Imag::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Upstream imag() returns zeros_like for real input before a
  // primitive is built (mlx/ops.cpp), so eval only ever sees
  // complex64 here; the op extracts the imaginary component.
  dispatch_complex_extract(name(), 1, inputs, out, out.primitive().stream());
}
void Inverse::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  linalg_require_f32(name(), in, out);
  if (in.shape(-1) != in.shape(-2)) {
    omarchy::unsupported(std::string("non-square ") + name(), out);
  }
  const uint32_t n = checked_u32(in.shape(-1), name(), out);
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), out);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // The kernel consumes a dense copy of A in a scratch and builds the
  // inverse in `out` starting from identity.
  array work(in.shape(), in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  linalg_copy_dense(in, work, out.primitive().stream());
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  auto [tri, upper] = state();
  omarchy::ComputeParams params;
  params.operation = tri ? 1u : 0u;
  params.matrix_n = n;
  params.output_size = batch;
  params.flags = upper ? 1u : 0u;
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(work), binding(out), binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgInverseF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      tri
          ? "[Inverse::eval_gpu] triangular inverse requires a nonzero"
            " diagonal."
          : "[Inverse::eval_gpu] matrix is singular to working precision.",
      out);
}
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
  std::optional<array> dense_temp;
  const array& src = ensure_dense(
      input,
      input.flags().row_contiguous,
      dense_temp,
      encoder,
      out.primitive().stream());


  // Upstream mlx/ops.cpp logsumexp builds the LogSumExp primitive only
  // for a suffix reduce and keeps the reduced axis at size 1 (the
  // keepdims=False form squeezes on top of this output), so no
  // suffix-axis check is needed here. The shader accumulates in float32
  // for every dtype and keeps an infinite row max, matching the
  // upstream CPU rule.
  size_t row_length = src.shape(-1);
  size_t rows = src.size() / row_length;
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
      src, src.size(), "LogSumExp", out);
  params.output_offset = checked_item_offset(
      out, out.size(), "LogSumExp", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(src), binding(out)};
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
void LUF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& lu = outputs.at(0);
  auto& pivots = outputs.at(1);
  auto& row_indices = outputs.at(2);
  auto& encoder = omarchy::get_command_encoder(lu.primitive().stream());
  linalg_require_f32(name(), in, lu);
  const int m = in.shape(-2);
  const int n = in.shape(-1);
  const int k_count = std::min(m, n);
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), lu);
  lu.set_data(allocate_omarchy(lu.nbytes()));
  pivots.set_data(allocate_omarchy(pivots.nbytes()));
  row_indices.set_data(allocate_omarchy(row_indices.nbytes()));
  if (m == 0 || n == 0 || batch == 0) {
    // Nothing to factorize; upstream leaves the empty outputs.
    return;
  }
  // Factorize in place: the packed-LU output starts as a dense copy of
  // A, exactly like the CPU path copies A into lu before getrf.
  linalg_copy_dense(in, lu, lu.primitive().stream());
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  omarchy::ComputeParams params;
  params.matrix_m = checked_u32(m, name(), lu);
  params.matrix_n = checked_u32(n, name(), lu);
  params.matrix_k = checked_u32(k_count, name(), lu);
  params.output_size = batch;
  // Kernel binding order: 0 packed LU, 1 pivots, 2 row_indices, 3 status.
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(lu),
      binding(pivots),
      binding(row_indices),
      binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgLuF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      "[LUF::eval_gpu] LU factorization encountered a zero pivot"
      " (singular matrix); refusing rather than returning a partial"
      " factorization.",
      lu);
}
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
  std::optional<array> mask_temp;
  const array& mask_d = ensure_dense(
      mask,
      mask.flags().row_contiguous && mask.data_size() == mask.size(),
      mask_temp,
      encoder,
      out.primitive().stream());
  if (rows > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported("MaskedScatter row count", out);
  }
  uint32_t mask_word_offset =
      checked_item_offset(mask_d, mask_d.size(), "MaskedScatter", out);
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
      binding(out), binding(mask_d), binding(*src_dense)};
  encoder.dispatch_compute(
      kernel, bindings, params, checked_u32(rows, "MaskedScatter", out));
}
OMARCHY_BINARY(Minimum, MinimumOperation)
void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexMultiply, inputs, out, out.primitive().stream());
    return;
  }
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntMultiplyOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), MultiplyOperation, inputs, out, out.primitive().stream());
}
void Negative::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexNegative, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), NegativeOperation, inputs, out, out.primitive().stream());
}
void NotEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareNotEqual, inputs, out);
}
void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_sort_dtype("Partition", input, out, false, encoder);
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
void QRF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& q = outputs.at(0);
  // Batched inputs are gated: single-matrix factors verify against
  // Q^T Q = I and QR = A, but the batch>1 path returns garbage whose
  // root cause is not yet identified. Refuse by name instead.
  if (linalg_batch_count(in.shape()) != 1) {
    omarchy::unsupported(
        "[QRF] batched input gated pending numeric verification", q);
  }
  auto& r = outputs.at(1);
  auto& encoder = omarchy::get_command_encoder(q.primitive().stream());
  linalg_require_f32(name(), in, q);
  const int m = in.shape(-2);
  const int n = in.shape(-1);
  const int k = std::min(m, n);
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), q);
  q.set_data(allocate_omarchy(q.nbytes()));
  r.set_data(allocate_omarchy(r.nbytes()));
  if (m == 0 || n == 0 || batch == 0) {
    // Nothing to factorize; upstream leaves the empty outputs.
    return;
  }
  array work(in.shape(), in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  linalg_copy_dense(in, work, q.primitive().stream());
  omarchy::ComputeParams params;
  params.matrix_m = checked_u32(m, name(), q);
  params.matrix_n = checked_u32(n, name(), q);
  params.matrix_k = checked_u32(k, name(), q);
  params.output_size = batch;
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(work), binding(q), binding(r)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgQrF32, bindings, params, batch);
}
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
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> scales_temp;
  std::optional<array> biases_temp;
  const array& x_d =
      ensure_dense(x, x.flags().row_contiguous, x_temp, encoder, stream());
  const array& w_d =
      ensure_dense(w, w.flags().row_contiguous, w_temp, encoder, stream());
  const array& scales_d = ensure_dense(
      scales, scales.flags().row_contiguous, scales_temp, encoder, stream());
  const array& biases_d = ensure_dense(
      biases, biases.flags().row_contiguous, biases_temp, encoder, stream());
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
  if (scales_d.offset() % 4 != 0 || biases_d.offset() % 4 != 0) {
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
      binding(scales_d).buffer,
      binding(combined).buffer,
      scale_bytes,
      static_cast<VkDeviceSize>(scales_d.offset()),
      0);
  encoder.copy_buffer(
      binding(biases_d).buffer,
      binding(combined).buffer,
      scale_bytes,
      static_cast<VkDeviceSize>(biases_d.offset()),
      scale_bytes);

  // Push-constant routing for the qmm shader: operation carries bits,
  // reduce_size carries the group size (see shaders/qmm.comp).
  uint64_t total = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
  omarchy::ComputeParams params;
  params.count = checked_u32(total, tag, out);
  params.operation = static_cast<uint32_t>(bits_);
  params.reduce_size = static_cast<uint32_t>(group_size_);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(x_d, x_d.size(), tag, out);
  params.rhs_offset = checked_item_offset(w_d, w_d.size(), tag, out);
  params.aux_size = checked_u32(combined.size(), tag, out);
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x_d), binding(w_d), binding(combined), binding(out)};
  // DecodeGemv dispatch: when lhs has a single row, the per-row GEMV
  // path replaces the 16x16 tile. The subgroup-reduction variant is
  // picked when the device reports subgroupSize == 32 AND the ARITHMETIC
  // subgroup feature bit is set, both queried at device init and held
  // on the capability report. The default fall-through path is the
  // general Qmm kernel (unaffected by this addition).
  //
  // The subgroup variant replaces the five-round workgroup-shared
  // tree with one subgroupAdd per 32-lane slot. The microbenchmark
  // tools/subgroup-bench decides whether the trade pays; if the
  // device lacks subgroup support, this gate is a no-op and the
  // general path runs unchanged. See PROTOCOL.md for the keep rule.
  //
  // Gemv group count: COLUMNS_PER_GROUP output columns per workgroup,
  // matching the lane split in shaders/qmm_vec.comp.
  constexpr uint32_t kGemvColumnsPerGroup = 8u;
  auto n_groups_qmm_vec = (params.matrix_n + kGemvColumnsPerGroup - 1u) /
      kGemvColumnsPerGroup;
  if (params.matrix_m == 1u) {
    const auto& caps = encoder.device().capabilities();
    bool subgroup_ready =
        caps.subgroup_size == 32u &&
        (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    auto vec_kernel = subgroup_ready
        ? select_float_kernel(
              out.dtype(),
              omarchy::ComputeKernel::QmmVecSubgroupF32,
              omarchy::ComputeKernel::QmmVecSubgroupF16,
              omarchy::ComputeKernel::QmmVecSubgroupBF16)
        : select_float_kernel(
              out.dtype(),
              omarchy::ComputeKernel::QmmVecF32,
              omarchy::ComputeKernel::QmmVecF16,
              omarchy::ComputeKernel::QmmVecBF16);
    encoder.dispatch_compute(
        vec_kernel,
        bindings,
        params,
        std::min(n_groups_qmm_vec, omarchy::kMaxComputeGroupCountX),
        1u,
        1u);
    return;
  }
  if (const char* tile_env = std::getenv("MLX_OMARCHY_QMM_TILE");
      tile_env == nullptr || std::strcmp(tile_env, "0") != 0) {
    auto tile_kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::QmmTileF32,
        omarchy::ComputeKernel::QmmTileF16,
        omarchy::ComputeKernel::QmmTileBF16);
    uint32_t m_groups = (params.matrix_m + 15u) / 16u;
    uint32_t n_groups = (params.matrix_n + 15u) / 16u;
    encoder.dispatch_compute(
        tile_kernel,
        bindings,
        params,
        std::min(n_groups, omarchy::kMaxComputeGroupCountX),
        std::min(m_groups, omarchy::kMaxComputeGroupCountX),
        1u);
    return;
  }
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
  std::optional<array> keys_temp;
  const array& keys_d = ensure_dense(
      keys, keys.flags().row_contiguous, keys_temp, encoder, stream());
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
  params.lhs_offset = checked_item_offset(keys_d, keys_d.size(), "RandomBits", out);
  params.output_offset = checked_item_offset(out, out.size(), "RandomBits", out);
  std::array<omarchy::ComputeBinding, 2> bindings{
      binding(keys_d), binding(out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::RandomBitsU32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}
void Real::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Upstream real() returns its input for real dtypes before a
  // primitive is built (mlx/ops.cpp), so eval only ever sees
  // complex64 here; the op extracts the real component, matching
  // complex64_t::operator float().
  dispatch_complex_extract(name(), 0, inputs, out, out.primitive().stream());
}

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
    uint32_t reduce_size_u32 =
        checked_u32(reduce_size, operation_name, out);
    const array& bound_input = input.size() == 0 ? out : input;
    // The suffix kernel runs the reduction as a single per-invocation
    // serial loop; the driver caps that loop at a few thousand trips
    // (roughly 2^16 on lavapipe for local_size_x = 256), so reduce_size
    // is split across chunks when it exceeds the chunk-trip threshold.
    constexpr uint32_t kReduceSuffixChunkTrips = 4096;
    uint32_t chunks = reduce_size_u32 == 0
        ? 1u
        : (reduce_size_u32 + kReduceSuffixChunkTrips - 1u) /
            kReduceSuffixChunkTrips;
    if (chunks == 0) {
      chunks = 1;
    }
    bool bool_out = out.dtype() == bool_;
    Dtype scratch_dtype = bool_out ? uint32 : input.dtype();
    array scratch(
        Shape{static_cast<int>(static_cast<uint64_t>(output_size) * chunks)},
        scratch_dtype,
        nullptr,
        {});
    scratch.set_data(allocate_omarchy(scratch.nbytes()));
    encoder.add_temporary(scratch);
    omarchy::ComputeParams params;
    params.count = output_size;
    params.operation = reduce_type == Reduce::Sum ? 0u : 3u;
    params.reduce_size = reduce_size_u32;
    params.output_size = output_size;
    params.lhs_offset = checked_item_offset(
        bound_input, input.size(), operation_name, out);
    params.output_offset =
        checked_item_offset(out, out.size(), operation_name, out);
    params.matrix_n = chunks;
    auto kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::ReduceF32,
        omarchy::ComputeKernel::ReduceF16,
        omarchy::ComputeKernel::ReduceBF16);
    std::array<omarchy::ComputeBinding, 3> bindings{
        binding(bound_input), binding(scratch), binding(out)};
    auto run_phase = [&](uint32_t flags, uint32_t dispatch_count) {
      params.flags = flags;
      encoder.dispatch_compute(
          kernel,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(dispatch_count));
    };
    if (chunks == 1) {
      run_phase(0u | 2u, output_size);
    } else {
      run_phase(
          0u,
          checked_u32(
              static_cast<uint64_t>(output_size) * chunks,
              operation_name,
              out));
      run_phase(1u, output_size);
    }
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
  if (axes.size() > 2) {
    omarchy::unsupported(
        "multi-index Scatter with " + std::to_string(axes.size()) +
            " index arrays",
        out);
  }
  const bool multi_index = axes.size() == 2;
  bool is_sum = reduce_type == Scatter::Sum;
  bool is_prod = reduce_type == Scatter::Prod;
  const bool float_reduce =
      out.dtype() == float32 || out.dtype() == float16 ||
      out.dtype() == bfloat16;
  if (multi_index) {
    // The two-index kernel binds out, updates, both index arrays, and
    // the rank/key or accumulation scratch (only integer Sum drops the
    // scratch): four or five slots against the budget the device
    // reported at initialization.
    uint32_t needed = (is_sum && !float_reduce && out.dtype() != bool_)
        ? 4u
        : 5u;
    uint32_t allowed = encoder.device().compute().binding_limit();
    if (needed > allowed) {
      omarchy::unsupported(
          "multi-index Scatter needs " + std::to_string(needed) +
              " storage-buffer bindings; this device allows " +
              std::to_string(allowed),
          out);
    }
  }
  // Upstream grounds the float Sum/Prod path: the Metal backend
  // scatters float32 through native float atomics (is_metal_atomic
  // includes float) and float16/bfloat16 through a packed CAS loop, so
  // duplicate-index accumulation order is nondeterministic upstream
  // too and no ordering contract exists to preserve. Hardware fp32
  // atomicAdd (op 11) needs shaderBufferFloat32AtomicAdd: llvmpipe
  // reports it, the M1 Honeykrisp does not advertise the extension at
  // all, so devices without it run the FCAS kernel variants whose
  // op-17 compare-exchange add is the exact-arithmetic twin of op 11.
  // Prod's op-13 CAS multiply never needed the extension.
  const bool hw_atomic_add =
      encoder.device().capabilities().shader_atomic_float_add;
  if (multi_index && (is_sum || is_prod) &&
      (out.dtype() == float16 || out.dtype() == bfloat16)) {
    // Only fp32 has a multi-index FADD blob; fp16/bf16 accumulation
    // rides the single-index kernel.
    omarchy::unsupported(
        "multi-index Scatter float16/bfloat16 Sum/Prod dtype",
        out);
  }
  // The second index array in the multi-index case; upstream's op layer
  // broadcasts and dtype-promotes every index array to one shape.
  const array& indices_b = multi_index ? inputs.at(2) : indices;
  if (out.dtype() == float16 || out.dtype() == bfloat16) {
    require_float_dtype("Scatter", src, out, encoder);
  }
  uint32_t map_code;
  omarchy::ComputeKernel kernel;
  switch (out.dtype()) {
    case float32:
      map_code = 0;
      if (is_sum || is_prod) {
        kernel = multi_index
            ? (hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddMultiF32
                             : omarchy::ComputeKernel::ScatterFCasMultiF32)
            : (hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddF32
                             : omarchy::ComputeKernel::ScatterFCasF32);
      } else {
        kernel = multi_index ? omarchy::ComputeKernel::ScatterMultiU32
                             : omarchy::ComputeKernel::ScatterU32;
      }
      break;
    case int32:
      map_code = 1;
      kernel = multi_index ? omarchy::ComputeKernel::ScatterMultiU32
                           : omarchy::ComputeKernel::ScatterU32;
      break;
    case uint32:
      map_code = 2;
      kernel = multi_index ? omarchy::ComputeKernel::ScatterMultiU32
                           : omarchy::ComputeKernel::ScatterU32;
      break;
    case float16:
      map_code = 0;
      if (is_sum || is_prod) {
        kernel = hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddF16
                               : omarchy::ComputeKernel::ScatterFCasF16;
      } else {
        kernel = multi_index ? omarchy::ComputeKernel::ScatterMultiF16
                             : omarchy::ComputeKernel::ScatterF16;
      }
      break;
    case bfloat16:
      map_code = 0;
      if (is_sum || is_prod) {
        kernel = hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddBF16
                               : omarchy::ComputeKernel::ScatterFCasBF16;
      } else {
        kernel = multi_index ? omarchy::ComputeKernel::ScatterMultiBF16
                             : omarchy::ComputeKernel::ScatterBF16;
      }
      break;
    case bool_:
      map_code = 1;
      kernel = multi_index ? omarchy::ComputeKernel::ScatterBoolMulti
                           : omarchy::ComputeKernel::ScatterBool;
      break;
    default:
      omarchy::unsupported("Scatter dtype", out);
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
  // Materialize non-dense index and update views. Test data_size
  // against size rather than trusting the flags: a broadcast view can
  // carry a contiguous flag while holding fewer elements than the
  // shape names (the lying-contiguous-flag defect family).
  const array* idx = &indices;
  std::optional<array> idx_mat;
  if (!indices.flags().row_contiguous ||
      indices.data_size() != indices.size()) {
    idx_mat = array(indices.shape(), indices.dtype(), nullptr, {});
    copy_gpu(indices, *idx_mat, CopyType::General, out.primitive().stream());
    encoder.add_temporary(*idx_mat);
    idx = &*idx_mat;
  }
  if (multi_index) {
    // Upstream promotes all index arrays together before the primitive
    // sees them, so the second array shares the first's dtype and
    // shape; its layout is materialized like the first.
    if (indices_b.size() != indices.size()) {
      omarchy::unsupported("Scatter index shape", out);
    }
    if (indices_b.dtype() != indices.dtype()) {
      omarchy::unsupported("Scatter index dtype", out);
    }
  }
  const array* idx_b = &indices_b;
  std::optional<array> idx_b_mat;
  if (multi_index &&
      (!indices_b.flags().row_contiguous ||
       indices_b.data_size() != indices_b.size())) {
    idx_b_mat =
        array(indices_b.shape(), indices_b.dtype(), nullptr, {});
    copy_gpu(
        indices_b, *idx_b_mat, CopyType::General, out.primitive().stream());
    encoder.add_temporary(*idx_b_mat);
    idx_b = &*idx_b_mat;
  }
  const array* upd = &updates;
  std::optional<array> upd_mat;
  if (updates.data_size() != updates.size()) {
    // Broadcast update: Scalar for the single-source form, General for
    // the strided multi-source form; Vector would flat-copy past the
    // source elements.
    upd_mat = array(updates.shape(), updates.dtype(), nullptr, {});
    copy_gpu(
        updates,
        *upd_mat,
        updates.data_size() == 1 ? CopyType::Scalar : CopyType::General,
        out.primitive().stream());
    encoder.add_temporary(*upd_mat);
    upd = &*upd_mat;
  } else if (!updates.flags().row_contiguous) {
    upd_mat = array(updates.shape(), updates.dtype(), nullptr, {});
    copy_gpu(updates, *upd_mat, CopyType::Vector, out.primitive().stream());
    encoder.add_temporary(*upd_mat);
    upd = &*upd_mat;
  }
  uint32_t index_mode = scatter_index_mode(*idx, out, "Scatter");
  size_t update_ndim = updates.ndim() - indices.ndim();
  if (update_ndim > 4) {
    omarchy::unsupported("Scatter update rank", out);
  }
  uint32_t count = checked_u32(indices.size(), "Scatter", out);
  omarchy::ComputeParams params;
  // Output element bound for the shader's target guard: update blocks
  // whose trailing dims overflow the output are skipped instead of
  // writing out of range (upstream leaves that undefined).
  params.dims = checked_u32(out.size(), "Scatter", out);
  params.reduce_size = checked_u32(out.shape(axes[0]), "Scatter", out);
  params.output_size =
      checked_u32(updates.size() / indices.size(), "Scatter", out);
  if (multi_index) {
    // Axis-1 addressing rides fields the single-index kernel never
    // reads: the axis dim in matrix_n, the axis stride in matrix_k,
    // and the second array's word offset in lhs_size.
    params.matrix_n = checked_u32(out.shape(axes[1]), "Scatter", out);
    params.matrix_k = checked_u32(out.strides(axes[1]), "Scatter", out);
    params.lhs_size = scatter_index_offset(*idx_b, out, "Scatter");
  }
  params.matrix_m = checked_u32(out.strides(axes[0]), "Scatter", out);
  params.rhs_offset = scatter_index_offset(*idx, out, "Scatter");
  params.output_offset = checked_item_offset(out, out.size(), "Scatter", out);
  params.aux_size = index_mode;
  params.aux_offset = map_code;
  params.flags = checked_u32(update_ndim, "Scatter", out);
  for (size_t i = 0; i < update_ndim; ++i) {
    // Update trailing dim i is updates.shape(indices.ndim() + i) and
    // maps onto out dim i; in_strides[i] is that out dim's stride.
    // Indexing updates from dim 0 instead reads the index-prefix dims
    // and misroutes every slot past the first.
    params.shape[i] = checked_u32(
        updates.shape(indices.ndim() + i), "Scatter", out);
    params.in_strides[i] = checked_u32(out.strides(i), "Scatter", out);
  }
  // The kernel walks update elements: slot = t / output_size. The
  // update base offset matters for sliced views.
  params.lhs_offset =
      checked_item_offset(*upd, upd->size(), "Scatter", out);
  if (static_cast<uint64_t>(count) * params.output_size > 0xFFFFFFFFull) {
    omarchy::unsupported("Scatter update span", out);
  }
  uint32_t elements = checked_u32(
      static_cast<uint64_t>(count) * params.output_size, "Scatter", out);
  // The kernel loop bound is the update element count, not the slot
  // count: with multi-element update blocks, slot-count would drop
  // every element past the first block.
  params.count = elements;
  if (is_sum && (out.dtype() == int32 || out.dtype() == uint32)) {
    // Phase 6: integer atomicAdd. Integer addition is associative, so
    // the result is deterministic even under duplicate indices. The
    // multi-index kernel binds both index arrays; the fourth slot of
    // the array stays unbound for the single-index path.
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(out), binding(*upd), binding(*idx)};
    if (multi_index) {
      bindings[3] = binding(*idx_b);
    }
    params.operation = 6;
    encoder.dispatch_compute(
        kernel,
        std::span<const omarchy::ComputeBinding>(
            bindings.data(), multi_index ? 4u : 3u),
        params,
        omarchy::compute_dispatch_group_count(elements));
    return;
  }
  const bool bool_word = out.dtype() == bool_;
  if ((float_reduce && (is_sum || is_prod)) ||
      (bool_word &&
       (is_sum || is_prod || reduce_type == Scatter::Max ||
        reduce_type == Scatter::Min))) {
    // Scratch accumulation paths: float Sum/Prod and bool Sum/Prod/
    // Max/Min. Float Sum runs op 11 (fp32 hardware atomicAdd) or the
    // FCAS op 17 (compare-exchange add) and stores at op 12; float
    // Prod runs op 13 (CAS fp32 multiply) and multiplies into the
    // copied src at op 16. Bool Sum/Max OR the canonical update bit
    // (op 6) and byte-write at op 14; Prod/Min AND (op 7) and
    // byte-write at op 15.
    bool float_sum = float_reduce && is_sum;
    bool float_prod = float_reduce && is_prod;
    bool bool_or = bool_word && (is_sum || reduce_type == Scatter::Max);
    bool bool_and = bool_word && (is_prod || reduce_type == Scatter::Min);
    uint32_t clear_value = float_prod ? 0x3F800000u
        : bool_and ? 0xFFFFFFFFu
        : 0u;
    array scratch = make_u32_scratch(out.size(), encoder);
    dispatch_clear_u32(scratch, clear_value, encoder);
    std::array<omarchy::ComputeBinding, 5> bindings{
        binding(out),
        binding(*upd),
        binding(*idx),
        binding(*idx_b),
        binding(scratch)};
    if (!multi_index) {
      bindings[3] = binding(scratch);
    }
    uint32_t bound = multi_index ? 5u : 4u;
    uint32_t accumulate;
    uint32_t finalize;
    if (float_sum) {
      accumulate = hw_atomic_add ? 11u : 17u;
      finalize = 12;
    } else if (float_prod) {
      accumulate = 13;
      finalize = 16;
    } else if (bool_or) {
      accumulate = 6;
      finalize = 14;
    } else {
      accumulate = 7;
      finalize = 15;
    }
    params.operation = accumulate;
    encoder.dispatch_compute(
        kernel,
        std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
        params,
        omarchy::compute_dispatch_group_count(elements));
    params.count = checked_u32(out.size(), "Scatter", out);
    params.operation = finalize;
    encoder.dispatch_compute(
        kernel,
        std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
        params,
        omarchy::compute_dispatch_group_count(params.count));
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
  } else if (reduce_type == Scatter::Prod) {
    phase1 = 7;
    phase2 = 8;
    clear_value = 1;
  } else {
    phase1 = 3;
    phase2 = 5;
    clear_value = 0xFFFFFFFFu;
  }
  array scratch = make_u32_scratch(out.size(), encoder);
  dispatch_clear_u32(scratch, clear_value, encoder);
  // The multi-index kernel moves scratch to binding 4 and binds the
  // second index array at binding 3; the span width follows the case.
  std::array<omarchy::ComputeBinding, 5> bindings{
      binding(out),
      binding(*upd),
      binding(*idx),
      binding(*idx_b),
      binding(scratch)};
  // The single-index shader keeps the rank scratch at binding 3 (the
  // multi-index variant moves it to 4 for the second index array).
  // Without this rebind, binding 3 carries the indices buffer: pass 1
  // atomically maxes rank words into the indices and pass 2 reads its
  // ranks back out of them, so updates land only where an index value
  // collides with a rank and the indices array is corrupted.
  if (!multi_index) {
    bindings[3] = binding(scratch);
  }
  uint32_t bound = multi_index ? 5u : 4u;
  params.operation = phase1;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
      params,
      omarchy::compute_dispatch_group_count(elements));
  // None pass 2 walks update elements; Max/Min finalize walks output
  // elements.
  if (reduce_type == Scatter::None) {
    params.count = elements;
  } else {
    params.count = checked_u32(out.size(), "Scatter", out);
  }
  params.operation = phase2;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
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
  if (out.dtype() == float16 || out.dtype() == bfloat16) {
    require_float_dtype("ScatterAxis", src, out, encoder);
  }
  const bool float_reduce =
      out.dtype() == float32 || out.dtype() == float16 ||
      out.dtype() == bfloat16;
  // Same upstream grounding as general Scatter: Metal's float axis
  // scatter add is a native atomic fetch add, so duplicate order is
  // nondeterministic upstream and no ordering contract exists to
  // preserve. Devices without shaderBufferFloat32AtomicAdd run the
  // FCAS axis variants (op 17 compare-exchange add) instead of
  // refusing: the M1 Honeykrisp does not advertise
  // VK_EXT_shader_atomic_float at all.
  const bool hw_atomic_add =
      encoder.device().capabilities().shader_atomic_float_add;
  omarchy::ComputeKernel kernel;
  switch (out.dtype()) {
    case float32:
      kernel = !is_sum ? omarchy::ComputeKernel::ScatterAxisU32
          : hw_atomic_add ? omarchy::ComputeKernel::ScatterAxisFAddF32
                          : omarchy::ComputeKernel::ScatterAxisFCasF32;
      break;
    case int32:
    case uint32:
      kernel = omarchy::ComputeKernel::ScatterAxisU32;
      break;
    case float16:
      kernel = !is_sum ? omarchy::ComputeKernel::ScatterAxisF16
          : hw_atomic_add ? omarchy::ComputeKernel::ScatterAxisFAddF16
                          : omarchy::ComputeKernel::ScatterAxisFCasF16;
      break;
    case bfloat16:
      kernel = !is_sum ? omarchy::ComputeKernel::ScatterAxisBF16
          : hw_atomic_add ? omarchy::ComputeKernel::ScatterAxisFAddBF16
                          : omarchy::ComputeKernel::ScatterAxisFCasBF16;
      break;
    case bool_:
      kernel = omarchy::ComputeKernel::ScatterAxisBool;
      break;
    default:
      omarchy::unsupported("ScatterAxis dtype", out);
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
  // Materialize non-dense index views; test data_size against size
  // rather than trusting the flags (lying-contiguous-flag family).
  const array* idx = &indices;
  std::optional<array> idx_mat;
  if (!indices.flags().row_contiguous ||
      indices.data_size() != indices.size()) {
    idx_mat = array(indices.shape(), indices.dtype(), nullptr, {});
    copy_gpu(indices, *idx_mat, CopyType::General, out.primitive().stream());
    encoder.add_temporary(*idx_mat);
    idx = &*idx_mat;
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
  params.aux_offset = scatter_index_mode(*idx, out, "ScatterAxis");
  // lhs_offset carries the update base element offset; the shader
  // walks update addresses through out_strides plus matrix_n.
  params.lhs_offset =
      checked_item_offset(updates, updates.size(), "ScatterAxis", out);
  params.rhs_offset = scatter_index_offset(*idx, out, "ScatterAxis");
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
  // Sum: int rides the direct atomicAdd (op 6); float accumulates in
  // the fp32 scratch (op 11, or the FCAS op-17 compare-exchange add
  // without hardware float atomics) and stores once (op 12); bool
  // ORs the canonical bit (op 6) and byte-writes (op 14). Duplicate
  // order matches upstream GPU semantics.
  if (is_sum) {
    bool int_sum = out.dtype() == int32 || out.dtype() == uint32;
    std::optional<array> scratch;
    if (!int_sum) {
      scratch = make_u32_scratch(out.size(), encoder);
      dispatch_clear_u32(*scratch, 0, encoder);
    }
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(out),
        binding(*idx),
        binding(updates),
        int_sum ? binding(out) : binding(*scratch)};
    uint32_t bound = int_sum ? 3u : 4u;
    params.operation = int_sum || out.dtype() == bool_ ? 6u
        : hw_atomic_add ? 11u
        : 17u;
    encoder.dispatch_compute(
        kernel,
        std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
        params,
        omarchy::compute_dispatch_group_count(count));
    if (!int_sum) {
      params.count = checked_u32(out.size(), "ScatterAxis", out);
      params.operation = out.dtype() == bool_ ? 14u : 12u;
      encoder.dispatch_compute(
          kernel,
          std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
          params,
          omarchy::compute_dispatch_group_count(params.count));
    }
    return;
  }
  // None: rank-max scratch, then the winning slot rewrites the target
  // byte/word, which reproduces the CPU's sequential last-write-wins
  // order.
  array scratch = make_u32_scratch(out.size(), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(out), binding(*idx), binding(updates), binding(scratch)};
  params.operation = 0;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), 4u),
      params,
      omarchy::compute_dispatch_group_count(count));
  params.operation = 1;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), 4u),
      params,
      omarchy::compute_dispatch_group_count(count));
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
  if (sequence.ndim() != 1) {
    omarchy::unsupported("non-1-D SearchSorted", out);
  }
  std::optional<array> sequence_temp;
  std::optional<array> values_temp;
  const array& sequence_d = ensure_dense(
      sequence,
      sequence.flags().row_contiguous,
      sequence_temp,
      encoder,
      stream());
  const array& values_d = ensure_dense(
      values, values.flags().row_contiguous, values_temp, encoder, stream());
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
      sequence_d, sequence_d.size(), "SearchSorted", out);
  params.rhs_offset = checked_item_offset(
      values_d, values_d.size(), "SearchSorted", out);
  params.output_offset = checked_item_offset(out, out.size(), "SearchSorted", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(sequence_d), binding(values_d), binding(out)};
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

// Select serves tril/triu (the where() pair behind composed lu), the
// sampler chain's scalar selects, and the composed causal mask. Value
// dtypes are float32, float16, bfloat16, int32, uint32, and bool, and
// every operand layout routes through one of two transports in
// select.comp. The flat transport keeps the modulo fast path for dense
// operands. The general transport unravels the output coordinate over
// the collapsed broadcast shape and indexes the true and false operands
// through their own strides, where a stride of 0 broadcasts an axis -
// the same transport the compare and elementwise kernels use. Layout
// gates consult data_size and strides directly and never the contiguous
// flags: this session found five arrays whose flags claimed dense while
// data_size was smaller than size, so a flag-trusting gate would read
// out of bounds.
void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& condition = inputs.at(0);
  const array& truthy = inputs.at(1);
  const array& falsy = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (condition.dtype() != bool_ || truthy.dtype() != out.dtype() ||
      falsy.dtype() != out.dtype()) {
    omarchy::unsupported("Select dtype", out);
  }
  omarchy::ComputeKernel kernel;
  switch (out.dtype()) {
    case float32:
      kernel = omarchy::ComputeKernel::SelectF32;
      break;
    case float16:
      kernel = omarchy::ComputeKernel::SelectF16;
      break;
    case bfloat16:
      kernel = omarchy::ComputeKernel::SelectBF16;
      break;
    case int32:
    case uint32:
      kernel = omarchy::ComputeKernel::SelectI32;
      break;
    case bool_:
      kernel = omarchy::ComputeKernel::SelectBool;
      break;
    default:
      omarchy::unsupported("Select dtype", out);
  }
  // Dense row-major means the flat element index is the memory index:
  // data_size equals size and the strides are the exact suffix products.
  // Broadcast views carry a stride of 0 and transposed or sliced views
  // permute the strides, so all of them fail this check and route to
  // the general transport or a materialization.
  auto dense_row_major = [](const array& value) {
    if (value.data_size() != value.size()) {
      return false;
    }
    int64_t expected = 1;
    for (int axis = value.ndim() - 1; axis >= 0; --axis) {
      if (value.strides()[axis] != expected) {
        return false;
      }
      expected *= static_cast<int64_t>(value.shape(axis));
    }
    return true;
  };
  auto is_suffix_shape = [](const Shape& small, const Shape& big) {
    if (small.size() > big.size()) {
      return false;
    }
    size_t offset = big.size() - small.size();
    for (size_t axis = 0; axis < small.size(); ++axis) {
      if (small[axis] != big[offset + axis]) {
        return false;
      }
    }
    return true;
  };
  bool condition_flat =
      dense_row_major(condition) && condition.data_size() == out.size();
  bool truthy_flat = dense_row_major(truthy) && truthy.shape() == out.shape();
  bool falsy_scalar = falsy.data_size() == 1;
  bool falsy_flat = falsy_scalar ||
      (dense_row_major(falsy) &&
       (falsy.shape() == out.shape() ||
        is_suffix_shape(falsy.shape(), out.shape())));
  // The Honeykrisp bool word read only executes correctly in the
  // straight-line select.comp form, whose condition address requires a
  // flat condition. A broadcast or strided condition view is
  // materialized through the logical_or pipeline, whose dual-buffer
  // word reads are correct on that driver; binding the same buffer as
  // both inputs makes the OR an identity copy. The packed-bool output
  // kernel keeps every bool word load in that same straight-line form,
  // so it additionally requires its value operands dense over the full
  // output count and word-aligned; anything else is materialized the
  // same way. The encoder keeps the temporaries alive until the
  // committed work completes.
  auto materialize_bool_operand = [&](const array& value) {
    array dense(value.shape(), bool_, nullptr, {});
    dense.set_data(allocate_omarchy(dense.nbytes()));
    uint32_t material_count = checked_u32(dense.size(), name(), out);
    uint32_t material_words = checked_u32(
        (static_cast<uint64_t>(material_count) + 3) / 4, name(), out);
    // The logical_or kernel accumulates each canonical output byte with
    // atomicOr, so the destination must start zeroed - a fresh
    // allocation holds whatever the allocator recycled.
    omarchy::ComputeParams clear_params;
    clear_params.count = material_words;
    std::array<omarchy::ComputeBinding, 1> clear_bindings{binding(dense)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::ClearU32,
        clear_bindings,
        clear_params,
        omarchy::compute_dispatch_group_count(material_words));
    omarchy::ComputeParams material_params;
    material_params.count = material_count;
    material_params.lhs_size = checked_u32(value.data_size(), name(), out);
    material_params.rhs_size = material_params.lhs_size;
    material_params.output_size = material_count;
    material_params.lhs_offset = checked_item_offset(
        value, material_params.lhs_size, name(), out);
    material_params.rhs_offset = material_params.lhs_offset;
    material_params.output_offset = checked_item_offset(
        dense, material_count, name(), out);
    fill_broadcast_transport(name(), material_params, value, value, dense);
    std::array<omarchy::ComputeBinding, 3> material_bindings{
        binding(value), binding(value), binding(dense)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LogicalOrBool,
        material_bindings,
        material_params,
        omarchy::compute_dispatch_group_count(material_words));
    encoder.add_temporary(dense);
    return dense;
  };
  auto word_aligned = [](const array& value) {
    return value.offset() % value.itemsize() == 0 &&
        (value.offset() / value.itemsize()) % 4 == 0;
  };
  auto bool_kernel_ready = [&](const array& value) {
    return dense_row_major(value) && value.data_size() == out.size() &&
        word_aligned(value);
  };
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  std::vector<array> materialized;
  // Up to three operand materializations (bool output); reserving keeps
  // the .back() pointers below valid across pushes.
  materialized.reserve(3);
  const array* condition_ptr = &condition;
  const array* truthy_ptr = &truthy;
  const array* falsy_ptr = &falsy;
  bool general = false;
  if (out.dtype() == bool_) {
    if (truthy.shape() != out.shape() || falsy.shape() != out.shape()) {
      omarchy::unsupported("Select layout", out);
    }
    if (!condition_flat || !word_aligned(condition)) {
      materialized.push_back(materialize_bool_operand(condition));
      condition_ptr = &materialized.back();
    }
    if (!bool_kernel_ready(truthy)) {
      materialized.push_back(materialize_bool_operand(truthy));
      truthy_ptr = &materialized.back();
    }
    if (!bool_kernel_ready(falsy)) {
      materialized.push_back(materialize_bool_operand(falsy));
      falsy_ptr = &materialized.back();
    }
  } else {
    if (!condition_flat) {
      materialized.push_back(materialize_bool_operand(condition));
      condition_ptr = &materialized.back();
    }
    general = !truthy_flat || !falsy_flat;
    if (general &&
        (truthy.shape() != out.shape() || falsy.shape() != out.shape())) {
      omarchy::unsupported("Select layout", out);
    }
  }
  uint32_t count = checked_u32(out.size(), name(), out);
  uint32_t lhs_offset = checked_item_offset(
      *condition_ptr, condition_ptr->data_size(), name(), out);
  // A byte-misaligned flat condition shifts the per-word element window
  // by lhs_offset % 4, so the final word covers tail elements and the
  // dispatch must grow by the same amount (matching the shader's word
  // count). The materialized or aligned flat path has mis 0.
  uint32_t condition_mis = lhs_offset & 3;
  uint32_t word_count = checked_u32(
      (static_cast<uint64_t>(count) + condition_mis + 3) / 4, name(), out);
  omarchy::ComputeParams params;
  params.count = count;
  params.output_size = count;
  params.lhs_size = checked_u32(condition_ptr->data_size(), name(), out);
  params.rhs_size = checked_u32(truthy_ptr->data_size(), name(), out);
  params.aux_size = checked_u32(falsy_ptr->data_size(), name(), out);
  params.lhs_offset = lhs_offset;
  params.rhs_offset = checked_item_offset(
      *truthy_ptr, params.rhs_size, name(), out);
  params.aux_offset = checked_item_offset(
      *falsy_ptr, params.aux_size, name(), out);
  params.output_offset = checked_item_offset(out, count, name(), out);
  if (general) {
    // Collapse contiguous runs once for both strided operands; a stride
    // of 0 breaks every merge around a broadcast axis, and collapsed
    // rank beyond 4 stays a named refusal exactly like the elementwise
    // kernels.
    auto [collapsed_shape, collapsed_strides] = collapse_contiguous_dims(
        out.shape(),
        std::vector<Strides>{truthy_ptr->strides(), falsy_ptr->strides()});
    if (collapsed_shape.size() > 4) {
      omarchy::unsupported("Select layout", out);
    }
    params.dims = static_cast<uint32_t>(collapsed_shape.size());
    uint64_t true_span = 0;
    uint64_t false_span = 0;
    for (size_t axis = 0; axis < collapsed_shape.size(); ++axis) {
      params.shape[axis] = static_cast<uint32_t>(collapsed_shape[axis]);
      params.out_strides[axis] =
          static_cast<uint32_t>(collapsed_strides[0][axis]);
      params.in_strides[axis] =
          static_cast<uint32_t>(collapsed_strides[1][axis]);
      uint64_t extent = params.shape[axis] - 1u;
      true_span += extent * params.out_strides[axis];
      false_span += extent * params.in_strides[axis];
    }
    if (!omarchy::compute_index_span_fits(params.rhs_offset, true_span + 1) ||
        !omarchy::compute_index_span_fits(
            params.aux_offset, false_span + 1)) {
      omarchy::unsupported("Select index span", out);
    }
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(*condition_ptr),
      binding(*truthy_ptr),
      binding(*falsy_ptr),
      binding(out)};
  // The kernel processes one output word (four elements) per thread
  // with no grid-stride loop, so very large outputs dispatch in
  // back-to-back offset chunks.
  constexpr uint64_t kMaxWordsPerDispatch =
      static_cast<uint64_t>(omarchy::kMaxComputeGroupCountX) *
      omarchy::kComputeThreadsPerGroup;
  uint32_t lhs_base = params.lhs_offset;
  uint32_t rhs_base = params.rhs_offset;
  uint32_t aux_base = params.aux_offset;
  uint32_t output_base = params.output_offset;
  uint64_t words_done = 0;
  while (words_done < word_count) {
    uint32_t chunk_words = static_cast<uint32_t>(
        std::min<uint64_t>(word_count - words_done, kMaxWordsPerDispatch));
    uint64_t chunk_elements = 4ull * words_done;
    // Condition and output stay flat across chunks. In the flat
    // transport the true operand advances linearly with them and the
    // false operand does too whenever it covers the full count; a
    // scalar false operand reads one absolute element. In the general
    // transport the strided operands were span-checked once above, and
    // the shader receives the chunk's first output element in matrix_m.
    if (!omarchy::compute_index_span_fits(
            lhs_base + chunk_elements, 4ull * chunk_words) ||
        !omarchy::compute_index_span_fits(
            output_base + chunk_elements, 4ull * chunk_words) ||
        (!general &&
            (!omarchy::compute_index_span_fits(
                 rhs_base + chunk_elements, 4ull * chunk_words) ||
             (params.aux_size == count &&
              !omarchy::compute_index_span_fits(
                  aux_base + chunk_elements, 4ull * chunk_words))))) {
      omarchy::unsupported("Select index span", out);
    }
    params.lhs_offset = checked_u32(lhs_base + chunk_elements, name(), out);
    params.output_offset =
        checked_u32(output_base + chunk_elements, name(), out);
    if (general) {
      params.matrix_m = checked_u32(chunk_elements, name(), out);
    } else {
      params.rhs_offset = checked_u32(rhs_base + chunk_elements, name(), out);
      if (params.aux_size == count) {
        params.aux_offset =
            checked_u32(aux_base + chunk_elements, name(), out);
      }
    }
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
void Sin::eval_gpu(const std::vector<array>& inputs, array& out) {
  trig_argument_gate(name(), inputs, out);
  dispatch_elementwise(
      name(), SinOperation, inputs, out, out.primitive().stream());
}
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
  require_sort_dtype("Sort", input, out, false, encoder);
  if (state() != input.ndim() - 1) {
    omarchy::unsupported("non-suffix Sort", out);
  }
  dispatch_sort("Sort", input, out, false, encoder);
}
// The float Square keeps its macro body; integer Square squares in the
// operand dtype through the integer kernel, never via float.
void Square::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntSquareOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), SquareOperation, inputs, out, out.primitive().stream());
}
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
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexSubtract, inputs, out, out.primitive().stream());
    return;
  }
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntSubtractOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), SubtractOperation, inputs, out, out.primitive().stream());
}
void SVD::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& encoder =
      omarchy::get_command_encoder(outputs.at(0).primitive().stream());
  linalg_require_f32(name(), in, outputs.at(0));
  const int m = in.shape(-2);
  const int n = in.shape(-1);
  const int k = std::min(m, n);
  if (k > 1024) {
    omarchy::unsupported(std::string(name()) + " matrix size", outputs.at(0));
  }
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), outputs.at(0));
  const bool wide = m < n;
  const bool compute_uv = state();
  // The sweeps always run on an R x C copy with R >= C: tall inputs use
  // A itself, wide inputs use a dense transposed copy.
  Shape work_shape = in.shape();
  if (wide) {
    work_shape[work_shape.size() - 2] = n;
    work_shape[work_shape.size() - 1] = m;
  }
  array work(work_shape, in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  if (wide) {
    auto strides = in.strides();
    std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
    copy_gpu_inplace(
        in,
        work,
        // The walk shape must be the transposed (R x C) work shape. With
        // in.shape() here the kernel walks the source as (m, n) over
        // transposed strides and reads n - m rows past the input buffer;
        // those recycled-page values land in work rows >= k and made
        // every wide SVD (and pinv) silently wrong, varying run to run.
        work.shape(),
        strides,
        work.strides(),
        0,
        0,
        CopyType::GeneralGeneral,
        outputs.at(0).primitive().stream());
  } else {
    linalg_copy_dense(in, work, outputs.at(0).primitive().stream());
  }
  omarchy::ComputeParams params;
  params.matrix_m = checked_u32(wide ? n : m, name(), outputs.at(0));
  params.matrix_n = checked_u32(k, name(), outputs.at(0));
  params.matrix_k = checked_u32(k, name(), outputs.at(0));
  params.output_size = batch;
  params.flags = wide ? 1u : 0u;
  if (compute_uv) {
    auto& u = outputs.at(0);
    auto& s = outputs.at(1);
    auto& vt = outputs.at(2);
    u.set_data(allocate_omarchy(u.nbytes()));
    s.set_data(allocate_omarchy(s.nbytes()));
    vt.set_data(allocate_omarchy(vt.nbytes()));
    if (batch == 0) {
      return;
    }
    auto scratch = make_u32_scratch(batch, encoder);
    dispatch_clear_u32(scratch, 0, encoder);
    params.operation = 0u;
    std::array<omarchy::ComputeBinding, 4> sweep_bindings{
        binding(work), binding(u), binding(work), binding(scratch)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LinalgSvdF32, sweep_bindings, params, batch);
    linalg_check_status(
        scratch,
        encoder,
        "[SVD::eval_gpu] one-sided Jacobi sweep limit (60) exceeded"
        " without convergence.",
        u);
    params.operation = 2u;
    std::array<omarchy::ComputeBinding, 4> final_bindings{
        binding(work), binding(u), binding(vt), binding(s)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LinalgSvdFinalizeF32,
        final_bindings,
        params,
        batch);
  } else {
    auto& s = outputs.at(0);
    s.set_data(allocate_omarchy(s.nbytes()));
    if (batch == 0 || s.size() == 0) {
      return;
    }
    auto scratch = make_u32_scratch(batch, encoder);
    dispatch_clear_u32(scratch, 0, encoder);
    params.operation = 1u;
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(work), binding(work), binding(s), binding(scratch)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LinalgSvdF32, bindings, params, batch);
    linalg_check_status(
        scratch,
        encoder,
        "[SVD::eval_gpu] one-sided Jacobi sweep limit (60) exceeded"
        " without convergence.",
        s);
  }
}
OMARCHY_UNARY(Tan, TanOperation)
OMARCHY_UNARY(Tanh, TanhOperation)
void Eig::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& encoder =
      omarchy::get_command_encoder(outputs.at(0).primitive().stream());
  linalg_require_f32(name(), in, outputs.at(0));
  const int n = in.shape(-1);
  if (in.shape(-2) != n) {
    omarchy::unsupported(std::string("non-square ") + name(), outputs.at(0));
  }
  if (n > 1024) {
    omarchy::unsupported(std::string(name()) + " matrix size", outputs.at(0));
  }
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), outputs.at(0));
  auto& values = outputs.at(0);
  values.set_data(allocate_omarchy(values.nbytes()));
  if (values.size() == 0) {
    return;
  }
  array work(in.shape(), in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  linalg_copy_dense(in, work, outputs.at(0).primitive().stream());
  const bool compute_ev = state();
  if (compute_ev) {
    auto& vectors = outputs.at(1);
    vectors.set_data(allocate_omarchy(vectors.nbytes()));
  }
  // Dense per-matrix accumulation matrix V. The complex64 vectors output
  // spans twice the floats of a dense n*n matrix, so it cannot double as
  // the accumulator; the kernel writes it once in the final interleave.
  // Values-only mode leaves it unused and binds `work` as the dummy.
  array vacc(
      Shape{static_cast<int>(compute_ev ? n * n * batch : 1)},
      float32,
      nullptr,
      {});
  vacc.set_data(allocate_omarchy(vacc.nbytes()));
  if (compute_ev) {
    encoder.add_temporary(vacc);
  }
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  omarchy::ComputeParams params;
  params.operation = compute_ev ? 1u : 0u;
  params.matrix_n = checked_u32(n, name(), values);
  params.output_size = batch;
  // Kernel binding order: 0 work H, 1 vectors output, 2 values output,
  // 3 status, 4 dense V accumulator. Values-only mode binds `work` as
  // the dummy for the unwritten slots.
  std::array<omarchy::ComputeBinding, 5> bindings{
      binding(work),
      compute_ev ? binding(outputs.at(1)) : binding(work),
      binding(values),
      binding(scratch),
      compute_ev ? binding(vacc) : binding(work)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgEigF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      "[Eig::eval_gpu] QR iteration limit (40) exceeded without"
      " convergence.",
      values);
}

void Eigh::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& encoder =
      omarchy::get_command_encoder(outputs.at(0).primitive().stream());
  linalg_require_f32(name(), in, outputs.at(0));
  const int n = in.shape(-1);
  if (in.shape(-2) != n) {
    omarchy::unsupported(std::string("non-square ") + name(), outputs.at(0));
  }
  if (n > 1024) {
    omarchy::unsupported(std::string(name()) + " matrix size", outputs.at(0));
  }
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), outputs.at(0));
  auto [uplo, compute_ev] = state();
  const bool upper = (uplo == "U" || uplo == "upper");
  auto& values = outputs.at(0);
  values.set_data(allocate_omarchy(values.nbytes()));
  if (values.size() == 0) {
    return;
  }
  array work(in.shape(), in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  linalg_copy_dense(in, work, outputs.at(0).primitive().stream());
  if (compute_ev) {
    auto& vectors = outputs.at(1);
    vectors.set_data(allocate_omarchy(vectors.nbytes()));
  }
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  omarchy::ComputeParams params;
  params.operation = compute_ev ? 1u : 0u;
  params.matrix_n = checked_u32(n, name(), values);
  params.output_size = batch;
  params.flags = upper ? 1u : 0u;
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(work),
      compute_ev ? binding(outputs.at(1)) : binding(work),
      binding(values),
      binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgEighF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      "[Eigh::eval_gpu] Jacobi sweep limit (60) exceeded without"
      " convergence.",
      values);
}

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


namespace {

// Wave 9: fused fast-op kernels. Every kernel mirrors the exact fallback
// algebra in mlx/fast.cpp; per-row statistics reduce in shared memory with
// float32 arithmetic for every storage dtype.

void require_norm_input(
    const std::string& tag,
    const array& x,
    array& out,
    omarchy::CommandEncoder& encoder) {
  require_float_dtype(tag, x, out, encoder);
  if (!x.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous " + tag, out);
  }
}

void require_norm_parameter(
    const std::string& tag,
    const array& parameter,
    size_t row_length,
    array& out) {
  if (!parameter.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous " + tag, out);
  }
  if (
      parameter.dtype() != out.dtype() ||
      !(parameter.size() == 1 || parameter.shape(-1) == row_length)) {
    omarchy::unsupported(tag + " parameter shape", out);
  }
}

// A weightless VJP writes a scalar zero, the upstream zeros_like(w) form.
void zero_fill(array& out) {
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() != 0) {
    std::memset(out.data<uint8_t>(), 0, out.nbytes());
  }
}

omarchy::ComputeParams norm_params(
    const array& x,
    size_t row_length,
    float eps,
    const std::string& tag,
    array& out) {
  omarchy::ComputeParams params;
  params.count = checked_u32(x.size(), tag, out);
  params.reduce_size = checked_u32(row_length, tag, out);
  params.output_size = checked_u32(x.size() / row_length, tag, out);
  params.lhs_offset = checked_item_offset(x, x.size(), tag, out);
  params.alpha = eps;
  return params;
}

} // namespace
bool CrossEntropy::use_fallback(Stream s) {
  return false;
}

void CrossEntropy::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_y = inputs.at(1);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> y_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& y =
      ensure_dense(in_y, in_y.flags().row_contiguous, y_temp, encoder, s);
  require_float_dtype(tag, x, x, encoder);
  if (out.dtype() != float32) {
    omarchy::unsupported(tag + " output dtype", out);
  }
  if (y.dtype() != int32) {
    omarchy::unsupported(tag + " targets dtype", out);
  }
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // loss = lse - score per row, one fused row kernel; the output is always
  // float32, matching the upstream astype(loss, float32).
  auto params = norm_params(x, row_length, 0.0f, tag, out);
  params.rhs_offset = checked_item_offset(y, y.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(x), binding(y), binding(out)};
  auto kernel = select_float_kernel(
      x.dtype(),
      omarchy::ComputeKernel::CrossEntropyF32,
      omarchy::ComputeKernel::CrossEntropyF16,
      omarchy::ComputeKernel::CrossEntropyBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

bool RMSNorm::use_fallback(Stream s) {
  return false;
}
void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  require_norm_input(tag, x, out, encoder);
  require_norm_parameter(tag, w, x.shape(-1), out);
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto params = norm_params(x, row_length, eps_, tag, out);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.lhs_size = checked_u32(w.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(w), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::FastRmsNormF32,
      omarchy::ComputeKernel::FastRmsNormF16,
      omarchy::ComputeKernel::FastRmsNormBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

bool LayerNorm::use_fallback(Stream s) {
  return false;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  const array& in_b = inputs.at(2);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> b_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  const array& b =
      ensure_dense(in_b, in_b.flags().row_contiguous, b_temp, encoder, s);
  require_norm_input(tag, x, out, encoder);
  require_norm_parameter(tag, w, x.shape(-1), out);
  require_norm_parameter(tag, b, x.shape(-1), out);
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto params = norm_params(x, row_length, eps_, tag, out);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.aux_offset = checked_item_offset(b, b.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.lhs_size = checked_u32(w.size(), tag, out);
  params.aux_size = checked_u32(b.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(b), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::FastLayerNormF32,
      omarchy::ComputeKernel::FastLayerNormF16,
      omarchy::ComputeKernel::FastLayerNormBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& dx = outputs.at(0);
  array& dw = outputs.at(1);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  const array& in_g = inputs.at(2);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> g_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  const array& g =
      ensure_dense(in_g, in_g.flags().row_contiguous, g_temp, encoder, s);
  require_norm_input(tag, x, dx, encoder);
  require_norm_parameter(tag, w, x.shape(-1), dx);
  size_t row_length = x.shape(-1);
  if (x.size() == 0) {
    dx.set_data(allocate_omarchy(dx.nbytes()));
    zero_fill(dw);
    return;
  }
  // dx = gw * n - x * mean(gw * x) * n^3, one fused row kernel.
  dx.set_data(allocate_omarchy(dx.nbytes()));
  auto params = norm_params(x, row_length, eps_, tag, dx);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, dx);
  params.aux_offset = checked_item_offset(g, g.size(), tag, dx);
  params.output_offset = checked_item_offset(dx, dx.size(), tag, dx);
  params.lhs_size = checked_u32(w.size(), tag, dx);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(g), binding(dx)};
  auto dx_kernel = select_float_kernel(
      dx.dtype(),
      omarchy::ComputeKernel::FastRmsNormVjpDxF32,
      omarchy::ComputeKernel::FastRmsNormVjpDxF16,
      omarchy::ComputeKernel::FastRmsNormVjpDxBF16);
  encoder.dispatch_compute(
      dx_kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
  // dw = sum_rows(g * x * n); a scalar weight keeps the upstream
  // zeros_like(w) result.
  if (w.ndim() == 0) {
    zero_fill(dw);
    return;
  }
  dw.set_data(allocate_omarchy(dw.nbytes()));
  // Stage 1 scatters one row's per-column contributions per workgroup;
  // stage 2 sums the rows. A single workgroup used to loop all rows with
  // a barrier per tile, and llvmpipe stops honoring those barriers past
  // roughly a thousand row iterations (NaN dw at 8x100x1024 and up).
  array dw_partial(Shape{static_cast<int>(x.size())}, float32, nullptr, {});
  dw_partial.set_data(allocate_omarchy(dw_partial.nbytes()));
  encoder.add_temporary(dw_partial);
  auto dw_params = norm_params(x, row_length, eps_, tag, dw_partial);
  dw_params.rhs_offset = checked_item_offset(g, g.size(), tag, dw);
  dw_params.output_offset = 0;
  std::array<omarchy::ComputeBinding, 3> dw_bindings{
      binding(x), binding(g), binding(dw_partial)};
  auto dw_kernel = select_float_kernel(
      dw.dtype(),
      omarchy::ComputeKernel::FastRmsNormVjpDwF32,
      omarchy::ComputeKernel::FastRmsNormVjpDwF16,
      omarchy::ComputeKernel::FastRmsNormVjpDwBF16);
  encoder.dispatch_compute(
      dw_kernel,
      dw_bindings,
      dw_params,
      std::min(dw_params.output_size, omarchy::kMaxComputeGroupCountX));
  omarchy::ComputeParams reduce_params = dw_params;
  reduce_params.lhs_offset = 0;
  reduce_params.reduce_size = dw_params.output_size;
  reduce_params.output_size = checked_u32(row_length, tag, dw);
  reduce_params.output_offset = checked_item_offset(dw, dw.size(), tag, dw);
  std::array<omarchy::ComputeBinding, 2> reduce_bindings{
      binding(dw_partial), binding(dw)};
  auto reduce_kernel = select_float_kernel(
      dw.dtype(),
      omarchy::ComputeKernel::FastRmsNormVjpDwReduceF32,
      omarchy::ComputeKernel::FastRmsNormVjpDwReduceF16,
      omarchy::ComputeKernel::FastRmsNormVjpDwReduceBF16);
  encoder.dispatch_compute(
      reduce_kernel,
      reduce_bindings,
      reduce_params,
      std::min(
          (reduce_params.output_size + 255u) / 256u,
          omarchy::kMaxComputeGroupCountX));
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& dx = outputs.at(0);
  array& dw = outputs.at(1);
  array& db = outputs.at(2);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  const array& in_b = inputs.at(2);
  const array& in_g = inputs.at(3);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> b_temp;
  std::optional<array> g_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  const array& b =
      ensure_dense(in_b, in_b.flags().row_contiguous, b_temp, encoder, s);
  const array& g =
      ensure_dense(in_g, in_g.flags().row_contiguous, g_temp, encoder, s);
  require_norm_input(tag, x, dx, encoder);
  require_norm_parameter(tag, w, x.shape(-1), dx);
  require_norm_parameter(tag, b, x.shape(-1), dx);
  size_t row_length = x.shape(-1);
  if (x.size() == 0) {
    dx.set_data(allocate_omarchy(dx.nbytes()));
    zero_fill(dw);
    zero_fill(db);
    return;
  }
  // dx = (wg - mean(wg)) * n - (x - mu) * mean(wg * (x - mu)) * n^3.
  dx.set_data(allocate_omarchy(dx.nbytes()));
  auto params = norm_params(x, row_length, eps_, tag, dx);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, dx);
  params.aux_offset = checked_item_offset(g, g.size(), tag, dx);
  params.output_offset = checked_item_offset(dx, dx.size(), tag, dx);
  params.lhs_size = checked_u32(w.size(), tag, dx);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(g), binding(dx)};
  auto dx_kernel = select_float_kernel(
      dx.dtype(),
      omarchy::ComputeKernel::FastLayerNormVjpDxF32,
      omarchy::ComputeKernel::FastLayerNormVjpDxF16,
      omarchy::ComputeKernel::FastLayerNormVjpDxBF16);
  encoder.dispatch_compute(
      dx_kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
  // dw = sum_rows(g * (x - mu) * n); db = sum_rows(g) through the general
  // reduction. Scalar parameters keep the upstream zeros_like form.
  if (w.ndim() != 0) {
    dw.set_data(allocate_omarchy(dw.nbytes()));
    // Stage 1 scatters one row's per-column contributions per workgroup;
    // stage 2 sums the rows. A single workgroup used to loop all rows with
    // a barrier per tile, and llvmpipe stops honoring those barriers past
    // roughly a thousand row iterations (NaN dw at 8x100x1024 and up).
    array dw_partial(Shape{static_cast<int>(x.size())}, float32, nullptr, {});
    dw_partial.set_data(allocate_omarchy(dw_partial.nbytes()));
    encoder.add_temporary(dw_partial);
    auto dw_params = norm_params(x, row_length, eps_, tag, dw_partial);
    dw_params.rhs_offset = checked_item_offset(g, g.size(), tag, dw);
    dw_params.output_offset = 0;
    std::array<omarchy::ComputeBinding, 3> dw_bindings{
        binding(x), binding(g), binding(dw_partial)};
    auto dw_kernel = select_float_kernel(
        dw.dtype(),
        omarchy::ComputeKernel::FastLayerNormVjpDwF32,
        omarchy::ComputeKernel::FastLayerNormVjpDwF16,
        omarchy::ComputeKernel::FastLayerNormVjpDwBF16);
    encoder.dispatch_compute(
        dw_kernel,
        dw_bindings,
        dw_params,
        std::min(dw_params.output_size, omarchy::kMaxComputeGroupCountX));
    omarchy::ComputeParams reduce_params = dw_params;
    reduce_params.lhs_offset = 0;
    reduce_params.reduce_size = dw_params.output_size;
    reduce_params.output_size = checked_u32(row_length, tag, dw);
    reduce_params.output_offset = checked_item_offset(dw, dw.size(), tag, dw);
    std::array<omarchy::ComputeBinding, 2> reduce_bindings{
        binding(dw_partial), binding(dw)};
    auto reduce_kernel = select_float_kernel(
        dw.dtype(),
        omarchy::ComputeKernel::FastRmsNormVjpDwReduceF32,
        omarchy::ComputeKernel::FastRmsNormVjpDwReduceF16,
        omarchy::ComputeKernel::FastRmsNormVjpDwReduceBF16);
    encoder.dispatch_compute(
        reduce_kernel,
        reduce_bindings,
        reduce_params,
        std::min(
            (reduce_params.output_size + 255u) / 256u,
            omarchy::kMaxComputeGroupCountX));
  } else {
    zero_fill(dw);
  }
  if (b.ndim() != 0) {
    db.set_data(allocate_omarchy(db.nbytes()));
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    dispatch_reduce_general(
        tag,
        ReduceSumOperation,
        g,
        db,
        axes,
        encoder,
        select_reduce_general_kernel(g.dtype()));
  } else {
    zero_fill(db);
  }
}

void CrossEntropyVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& x = inputs.at(0);
  const array& y = inputs.at(1);
  const array& loss = inputs.at(2);
  const array& g = inputs.at(3);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  require_norm_input(tag, x, out, encoder);
  if (y.dtype() != int32) {
    omarchy::unsupported(tag + " targets dtype", out);
  }
  if (g.dtype() != float32 || loss.dtype() != float32) {
    omarchy::unsupported(tag + " cotangent dtype", out);
  }
  if (!y.flags().row_contiguous || !g.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous " + tag, out);
  }
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // The kernel recomputes the stable row logsumexp from the logits, so p is
  // the exact softmax probability; the loss input only rides along.
  auto params = norm_params(x, row_length, 0.0f, tag, out);
  params.rhs_offset = checked_item_offset(y, y.size(), tag, out);
  params.aux_offset = checked_item_offset(g, g.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(y), binding(g), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::CrossEntropyVjpF32,
      omarchy::ComputeKernel::CrossEntropyVjpF16,
      omarchy::ComputeKernel::CrossEntropyVjpBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

bool RoPE::use_fallback(Stream s) {
  return false;
}

// Fused RoPE absorbs the eager composition's Sin and Cos primitives, so it
// carries their kTrigArgumentLimit gate: known-defects.md, "A correction on
// the record", pins that a path which runs a primitive without going
// through its eval_gpu must carry that primitive's gates. The eager gate
// reduced the materialized theta tensor; this path never builds one, so the
// bound comes from the factors instead:
//
//   max|theta| <= (max_b |off_b| + (T - 1)) * |scale| * max_i |inv_freq_i|
//
// For the non-negative offsets every caller sends this is exact (the
// per-batch positions run off_b .. off_b + T - 1); mixed signs take the
// triangle inequality, which can only refuse marginally earlier, never
// later. max_i |inv_freq_i| is exactly 1.0 for base >= 1 (exp(0) = 1.0 in
// any IEEE exp), a host exp otherwise (an ulp caveat at the refusal
// boundary only), and the reciprocal of the smallest magnitude when freqs
// are given. Only the tiny offset and freqs arrays are read, behind the
// same stream-ordered synchronize trig_argument_gate documents: an
// unordered mapped read raced its own submission on hardware and returned
// recycled-page garbage.
void rope_trig_gate(
    const std::string& name,
    const array& in,
    const array& offset,
    const array* freqs,
    int dims,
    float base,
    float scale,
    const array& out) {
  Stream stream = out.primitive().stream();
  int T = in.shape(-2);
  float worst_offset;
  if (offset.size() == 1) {
    // A scalar constructed from a host int (what an mlx_lm decode step
    // passes) is Status::available from construction with no primitive:
    // nothing on the queue can be writing it, so it is read directly.
    // Anything else may still be in flight and takes the stream-ordered
    // synchronize. is_available() is NOT the test: it promotes an
    // in-flight evaluated array with no event. Draining the queue for
    // every offset cost two host round trips per decoder layer, 48 per
    // token (receipts/2026-09-04-rope-gate-drain.md).
    bool host_constant = offset.status() == array::Status::available &&
        !offset.has_primitive();
    // bf16 still needs this guard: the unexpected scalar writer is unknown.
    // Keep it until the readiness defect in docs/known-defects.md is resolved.
    if (!host_constant || out.dtype() == bfloat16) {
      omarchy::get_command_encoder(stream).synchronize();
    }
    worst_offset = std::abs(static_cast<float>(offset.item<int>()));
  } else {
    array offset_worst =
        astype(max(abs(offset, stream), stream), float32, stream);
    offset_worst.eval();
    omarchy::get_command_encoder(stream).synchronize();
    worst_offset = offset_worst.item<float>();
  }
  float inv_freq_bound;
  if (freqs != nullptr) {
    array freqs_min = min(abs(*freqs, stream), stream);
    freqs_min.eval();
    omarchy::get_command_encoder(stream).synchronize();
    inv_freq_bound = 1.0f / freqs_min.item<float>();
  } else {
    float beta = static_cast<float>(std::log(base) / (dims / 2));
    inv_freq_bound =
        (beta >= 0.0f) ? 1.0f : std::exp(-static_cast<float>(dims / 2 - 1) * beta);
  }
  float bound = (worst_offset + (T - 1)) * std::abs(scale) * inv_freq_bound;
  if (bound > kTrigArgumentLimit) {
    throw std::runtime_error(
        "[omarchy] " + name + " rotational argument magnitude " +
        std::to_string(bound) + " exceeds the built-in accuracy limit " +
        std::to_string(kTrigArgumentLimit) +
        " on this backend: the fused kernel computes sin/cos with the"
        " Vulkan driver's built-in, whose range reduction is untrusted"
        " above it, and the software Payne-Hanek fallback miscompiles on"
        " this driver. No silent wrong value occurs. Run it on an explicit"
        " CPU stream to use the CPU implementation.");
  }
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in = inputs.at(0);
  const array& offset = inputs.at(1);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);

  // FENCE (2026-09-03): two variants have not passed equivalence
  // against the composed fallback - per-batch vector offsets (160 of
  // 320 element checks diverged) and the inverse/VJP path (5 of 160).
  // A fused path must never serve a leg that has not passed, so both
  // ride the composition until each passes real equivalence. Decode,
  // the case worth ~460 primitives per token, is forward with a
  // scalar offset and stays fused. Remove one conjunct per fixed
  // defect, with the equivalence test green.
  if (!forward_ || offset.size() > 1) {
    auto result = fallback_(inputs);
    result[0].eval();
    encoder.synchronize();
    out.copy_shared_buffer(result[0]);
    return;
  }

  require_float_dtype(tag, in, out, encoder);
  if (out.size() == 0) {
    out.set_data(allocate_omarchy(out.nbytes()));
    return;
  }
  bool with_freqs = inputs.size() == 3;
  rope_trig_gate(
      tag,
      in,
      offset,
      with_freqs ? &inputs.at(2) : nullptr,
      dims_,
      base_,
      scale_,
      out);

  // The kernel never rotates in place: the input binding is readonly, the
  // output binding writeonly, and they never alias (an aliased
  // readonly/writeonly pair drops stores on llvmpipe - observed
  // 2026-09-03 - so every case below reads the input buffer directly or
  // through a temporary and writes a fresh output). The stride cases
  // mirror the upstream Metal decision tree (mlx/backend/metal/rope.cpp).
  int ndim = in.ndim();
  int B = in.shape(0);
  int T = in.shape(-2);
  int D = in.shape(-1);
  int half_dims = dims_ / 2;
  bool passthrough = dims_ < D;
  size_t mat_size = static_cast<size_t>(T) * D;
  bool row_contiguous = in.flags().row_contiguous;
  bool head_seq_transpose = false;
  const array* src = &in;
  int64_t strides[3];

  int dispatch_ndim = ndim;
  while (in.shape(-dispatch_ndim) == 1 && dispatch_ndim > 3) {
    dispatch_ndim--;
  }
  int N = 1;
  for (int i = 1; i < (ndim - 2); ++i) {
    N *= in.shape(i);
  }

  if (row_contiguous) {
    strides[0] = mat_size;
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else if (dispatch_ndim == 3) {
    // Handle non-contiguous 3D inputs
    strides[0] = in.strides()[ndim - 3];
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else if (
      ndim == 4 &&
      // batch dim is regularly strided
      in.strides()[0] == static_cast<int64_t>(T) * N * D &&
      // sequence and head dimensions are transposed
      in.strides()[1] == D &&
      in.strides()[2] == static_cast<int64_t>(N) * D) {
    head_seq_transpose = true;
    strides[0] = in.strides()[1];
    strides[1] = in.strides()[2];
    strides[2] = in.strides()[3];
  } else {
    // Copy non-contiguous > 3D inputs into a contiguous temporary and
    // rotate from there.
    array temp(in.shape(), in.dtype(), nullptr, {});
    temp.set_data(allocate_omarchy(temp.nbytes()));
    copy_gpu(in, temp, CopyType::General, s);
    encoder.add_temporary(temp);
    src = &temp;
    strides[0] = mat_size;
    strides[1] = temp.strides()[ndim - 2];
    strides[2] = temp.strides()[ndim - 1];
  }

  // A size-1 offset is a scalar no matter its rank: the composition
  // broadcasts it to every batch. A (1,)-shaped array has stride 1, so
  // borrowing it verbatim would read past the buffer for b > 0 (found
  // 2026-09-03 by the fence's scalar-offset equivalence test, at the
  // first element of batch 1).
  int64_t offset_stride =
      (offset.size() == 1 || offset.ndim() == 0) ? 0 : offset.strides()[0];
  omarchy::ComputeParams params;
  params.count = checked_u32(
      static_cast<size_t>(B) * N * T * (passthrough ? D : half_dims),
      tag,
      out);
  params.dims = checked_u32(half_dims, tag, out);
  params.flags = (forward_ ? 1u : 0u) | (traditional_ ? 2u : 0u) |
      (head_seq_transpose ? 4u : 0u) | (passthrough ? 8u : 0u);
  params.alpha = scale_;
  params.beta = static_cast<float>(std::log(base_) / half_dims);
  params.lhs_offset = checked_item_offset(*src, in.size(), tag, out);
  params.rhs_offset = checked_item_offset(offset, offset.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.matrix_m = checked_u32(offset_stride, tag, out);
  params.shape[0] = checked_u32(N, tag, out);
  params.shape[1] = checked_u32(T, tag, out);
  params.shape[2] = checked_u32(D, tag, out);
  params.in_strides[0] = checked_u32(strides[0], tag, out);
  params.in_strides[1] = checked_u32(strides[1], tag, out);
  params.in_strides[2] = checked_u32(strides[2], tag, out);
  // The output is freshly allocated here, so it is row contiguous.
  params.out_strides[0] = checked_u32(mat_size, tag, out);
  params.out_strides[1] = checked_u32(D, tag, out);
  params.out_strides[2] = 1u;
  if (with_freqs) {
    params.aux_offset =
        checked_item_offset(inputs.at(2), inputs.at(2).size(), tag, out);
  }
  // The no-freqs shader variants declare three bindings; the freqs slot
  // doubles the offset binding like the scalar-weight norm kernels do.
  // The bfloat16 shader legs read the packed bf16 words directly (the
  // proven constant-shift form, commit cf68e7d) and write float32; a
  // uint16_t-typed block compiled from that source returned recycled
  // memory on llvmpipe, and per-element half-word stores of a shared
  // 32-bit word race between adjacent lanes, so the in-kernel bf16
  // store stays unbuilt.
  // Wrapped path (default): one CastBF16F32 up-cast feeds the f32 rope
  // kernel and one CastF32BF16 narrows the output - the proven device
  // capability. Direct path (MLX_OMARCHY_ROPE_BF16_DIRECT, default off
  // until the M1 equivalence leg passes; any value other than "0"
  // enables): the bf16 input binds straight into the packed-word load
  // leg, deleting the up-cast - one dispatch fewer per rope call. The
  // leg's rope_round keeps the per-intermediate bf16 roundings of the
  // eager composition, so its result sits within a derived
  // intermediate-quantization bound of the wrapped result (pinned by
  // the fast-ops bf16 case) and bit-exact against the composed bf16
  // chain.
  bool direct_bf16 = false;
  if (out.dtype() == bfloat16) {
    if (const char* env = std::getenv("MLX_OMARCHY_ROPE_BF16_DIRECT");
        env != nullptr && std::strcmp(env, "0") != 0) {
      direct_bf16 = true;
    }
  }
  const array* rope_input = src;
  std::optional<array> bf16_to_f32;
  if (out.dtype() == bfloat16 && !direct_bf16) {
    bf16_to_f32 = array(in.shape(), float32, nullptr, {});
    bf16_to_f32->set_data(allocate_omarchy(bf16_to_f32->nbytes()));
    encoder.add_temporary(*bf16_to_f32);
    copy_gpu(in, *bf16_to_f32, CopyType::Vector, s);
    rope_input = &*bf16_to_f32;
    params.lhs_offset = 0;
  }
  array rope_output = out;
  std::optional<array> f32_to_bf16;
  if (out.dtype() == bfloat16) {
    f32_to_bf16 = array(out.shape(), float32, nullptr, {});
    f32_to_bf16->set_data(allocate_omarchy(f32_to_bf16->nbytes()));
    encoder.add_temporary(*f32_to_bf16);
    rope_output = *f32_to_bf16;
  } else {
    out.set_data(allocate_omarchy(out.nbytes()));
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(*rope_input),
      binding(rope_output),
      binding(offset),
      binding(with_freqs ? inputs.at(2) : offset)};
  // The direct bf16 bind must select by INPUT dtype: the packed-word
  // load leg pairs a bf16 input with this call's float32 output
  // temporary, so the output-dtype selector would pick the F32 kernel
  // and read the bf16 buffer as float words.
  omarchy::ComputeKernel kernel;
  if (direct_bf16) {
    kernel = with_freqs ? omarchy::ComputeKernel::FastRopeFreqsBF16
                        : omarchy::ComputeKernel::FastRopeBF16;
  } else {
    kernel = select_float_kernel(
        rope_output.dtype(),
        with_freqs ? omarchy::ComputeKernel::FastRopeFreqsF32
                   : omarchy::ComputeKernel::FastRopeF32,
        with_freqs ? omarchy::ComputeKernel::FastRopeFreqsF16
                   : omarchy::ComputeKernel::FastRopeF16,
        with_freqs ? omarchy::ComputeKernel::FastRopeFreqsBF16
                   : omarchy::ComputeKernel::FastRopeBF16);
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
  if (f32_to_bf16) {
    copy_gpu(*f32_to_bf16, out, CopyType::Vector, s);
  }
}
void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
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

  // Shared commit of the attention result into the output: with GQA the
  // result is 5-D (kv, repeat) while out is 4-D (heads). The flat
  // row-major layouts match, so share the buffer under out's own
  // strides: copy_shared_buffer(result) alone would install 5-D strides
  // and a 5-element stride vector on a 4-D array, and every later
  // reader would walk wrong addresses (the upstream fallback flattens
  // axes 1-2 here).
  auto commit_result = [&](const array& result) {
    if (repeats > 1) {
      Strides out_strides(out.ndim(), 1);
      for (int axis = out.ndim() - 2; axis >= 0; --axis) {
        out_strides[axis] = out_strides[axis + 1] * out.shape(axis + 1);
      }
      array::Flags flags;
      flags.contiguous = true;
      flags.row_contiguous = true;
      auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
      flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
      out.copy_shared_buffer(result, out_strides, flags, result.data_size());
    } else {
      out.copy_shared_buffer(result);
    }
  };

  // f16 runs at f16 storage end to end: the scores and probs matmuls
  // keep f16 operands and accumulate in float inside the shader
  // (matmul.comp), the scale rides the scores matmul's alpha, and the
  // softmax runs float math over f16 storage (softmax_suffix.comp). The
  // only new rounding against the f32 composition below is f16 storage
  // of the score and prob intermediates; the qmm decode path already
  // runs this pattern. Deletes the three q/k/v upcasts, the scale
  // broadcast and multiply, and the output downcast per call.
  //
  // bfloat16 rides the same shape under MLX_OMARCHY_SDPA_BF16_FAST
  // (default off until the M1 equivalence leg passes; any value other
  // than "0" enables): MatmulBF16, SoftmaxBF16, and ElementwiseBF16 are
  // the proven uint16_t-typed USE_BF16 legs the projections, the
  // residual adds, and the norm already run - float accumulation inside
  // the shader, bf16 storage with round-to-nearest-even stores on both
  // drivers. Scores store bf16 (2^-8 relative rounding per stored
  // score), the additive causal floor becomes bf16's finite maximum
  // (-3.3895313892515355e38 - overflow-immune where the f16 path caps
  // at 65504), and the output stays bf16 end to end. Deletes the three
  // q/k/v upcasts, the f32 scale multiply, the f32 softmax, and the
  // output downcast per call.
  bool bf16_fast = false;
  if (q.dtype() == bfloat16) {
    if (const char* env = std::getenv("MLX_OMARCHY_SDPA_BF16_FAST");
        env != nullptr && std::strcmp(env, "0") != 0) {
      bf16_fast = true;
    }
  }
  if (q.dtype() == float16 || bf16_fast) {
    const bool bf16 = q.dtype() == bfloat16;
    const Dtype storage_dtype = bf16 ? bfloat16 : float16;
    // GQA regroup as pure stride views, so a non-contiguous cache
    // slice rides its own strides straight into the matmul;
    // reshape_in_eval would copy - it only views row-contiguous
    // inputs, and a cache slice never is one. q splits the heads axis
    // into (kv, repeat); k and v keep their kv heads and insert a
    // size-1 repeat axis the matmul broadcasts (stride pinned 0).
    auto regroup_view = [&](const array& base) {
      bool splits = base.shape(1) != kv_heads;
      int rep = splits ? repeats : 1;
      Shape shape = {
          base.shape(0), kv_heads, rep, base.shape(2), base.shape(3)};
      Strides strides(5);
      strides[0] = base.strides()[0];
      strides[1] = splits ? base.strides()[1] * rep : base.strides()[1];
      strides[2] = splits ? base.strides()[1] : 0;
      strides[3] = base.strides()[2];
      strides[4] = base.strides()[3];
      array view(std::move(shape), base.dtype(), nullptr, {});
      view.copy_shared_buffer(
          base, strides, {false, false, false}, base.size());
      encoder.add_temporary(view);
      return view;
    };
    array qs = repeats > 1 ? regroup_view(q) : q;
    array ks = repeats > 1 ? regroup_view(k) : k;
    array vs = repeats > 1 ? regroup_view(v) : v;
    array keys_t = swapaxes_in_eval(ks, -1, -2);
    encoder.add_temporary(keys_t);
    Shape score_shape = qs.shape();
    score_shape.back() = k_len;
    array scores(score_shape, storage_dtype, nullptr, {});
    dispatch_matmul(tag, {qs, keys_t}, scores, scale_, 0.0f, false, s);
    encoder.add_temporary(scores);

    std::optional<array> masked;
    if (do_causal_) {
      if (k_len < q_len) {
        omarchy::unsupported("causal offset " + tag, out);
      }
      // The same 0 / -1e30 additive shape the f32 path builds, stored
      // in the storage dtype at its finite maximum (f16 -65504, bf16
      // -3.3895313892515355e38), not -inf: softmax still maps masked
      // positions to an exact zero, and a fully masked row (padding
      // masks over padded positions) stays defined - the additive
      // constant cancels in the max subtraction, so the row reduces to
      // softmax over its own scores exactly like the f32 path, instead
      // of inf-minus-inf NaN.
      constexpr float kF16Floor = -65504.0f;
      constexpr float kBF16Floor = -3.3895313892515355e38f;
      array mask(Shape{q_len, k_len}, storage_dtype, nullptr, {});
      mask.set_data(allocate_omarchy(mask.nbytes()));
      int offset = k_len - q_len;
      if (bf16) {
        bfloat16_t* values = mask.data<bfloat16_t>();
        for (int row = 0; row < q_len; ++row) {
          for (int col = 0; col < k_len; ++col) {
            values[row * k_len + col] =
                offset + row >= col ? bfloat16_t(0.0f)
                                    : bfloat16_t(kBF16Floor);
          }
        }
      } else {
        float16_t* values = mask.data<float16_t>();
        for (int row = 0; row < q_len; ++row) {
          for (int col = 0; col < k_len; ++col) {
            values[row * k_len + col] =
                offset + row >= col
                    ? float16_t(0.0f)
                    : float16_t(kF16Floor);
          }
        }
      }
      encoder.add_temporary(mask);
      masked = array(scores.shape(), storage_dtype, nullptr, {});
      dispatch_elementwise(tag, AddOperation, {scores, mask}, *masked, s);
    } else if (inputs.size() == 4) {
      // Upstream delivers the additive mask pre-broadcast in the
      // output dtype, so the f16 and bf16 fast paths consume it
      // without a cast.
      const array& mask = inputs.at(3);
      if (mask.dtype() != storage_dtype) {
        omarchy::unsupported("attention mask dtype " + tag, out);
      }
      if (repeats > 1) {
        masked = reshape_in_eval(
            mask, Shape{batch, kv_heads, repeats, q_len, k_len}, s);
        encoder.add_temporary(*masked);
      } else {
        masked = mask;
      }
      if ((*masked).shape() != scores.shape()) {
        omarchy::unsupported("attention mask shape " + tag, out);
      }
      array added(scores.shape(), storage_dtype, nullptr, {});
      dispatch_elementwise(tag, AddOperation, {scores, *masked}, added, s);
      masked = added;
      encoder.add_temporary(*masked);
    }
    const array& logits = masked ? *masked : scores;
    encoder.add_temporary(logits);

    array probs(logits.shape(), storage_dtype, nullptr, {});
    dispatch_softmax(tag, logits, probs, s);
    encoder.add_temporary(probs);

    Shape result_shape = probs.shape();
    result_shape.back() = v_dim;
    array result(result_shape, storage_dtype, nullptr, {});
    dispatch_matmul(tag, {probs, vs}, result, 1.0f, 0.0f, false, s);
    encoder.add_temporary(result);
    commit_result(result);
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
    commit_result(result);
  } else {
    copy_gpu(result, out, CopyType::Vector, s);
  }
}

OMARCHY_UNSUPPORTED_MULTI(ScaledDotProductAttentionVJP)
void ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> in_temp;
  const array& in =
      ensure_dense(inputs.at(0), inputs.at(0).flags().row_contiguous, in_temp, encoder, s);
  // E4M3 payloads travel as little-endian uint32 word packs of four bytes,
  // so the byte-side array offset must stay 4-byte aligned.
  auto word_aligned = [&](const array& value) {
    return checked_item_offset(value, value.size(), tag, out) % 4 == 0;
  };
  auto kernel = omarchy::ComputeKernel::Fp8ToF32;
  if (to_fp8_) {
    if (out.dtype() != uint8 ||
        (in.dtype() != float32 && in.dtype() != float16 &&
         in.dtype() != bfloat16)) {
      omarchy::unsupported(tag + " dtype", out);
    }
    if (!word_aligned(out)) {
      omarchy::unsupported(tag + " byte alignment", out);
    }
    kernel = select_float_kernel(
        in.dtype(),
        omarchy::ComputeKernel::Fp8FromF32,
        omarchy::ComputeKernel::Fp8FromF16,
        omarchy::ComputeKernel::Fp8FromBF16);
  } else {
    if (in.dtype() != uint8 ||
        (out.dtype() != float32 && out.dtype() != float16 &&
         out.dtype() != bfloat16)) {
      omarchy::unsupported(tag + " dtype", out);
    }
    if (!word_aligned(in)) {
      omarchy::unsupported(tag + " byte alignment", out);
    }
    kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::Fp8ToF32,
        omarchy::ComputeKernel::Fp8ToF16,
        omarchy::ComputeKernel::Fp8ToBF16);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(in.size(), tag, out);
  params.lhs_offset = checked_item_offset(in, in.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(in), binding(in), binding(in), binding(out)};
  uint32_t words = checked_u32((in.size() + 3) / 4, tag, out);
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(words));
}

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
  const array& in_w = inputs.at(0);
  const array& in_scales = inputs.at(1);
  const array& in_biases = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  // The output dtype is the promoted scales dtype (ops.cpp), and the
  // kernel reads the group parameters in that dtype, so mixed dtypes
  // stay a named rejection. The dequant kernels ship f32 and f16
  // variants only: bfloat16 and float64 parameters keep the named
  // error.
  if (
      in_scales.dtype() != out.dtype() ||
      in_biases.dtype() != out.dtype() ||
      (in_scales.dtype() != float16 && in_scales.dtype() != float32)) {
    omarchy::unsupported(tag + " scales dtype", out);
  }
  require_float_dtype(tag, in_scales, out, encoder);
  if (in_w.dtype() != uint32) {
    omarchy::unsupported(tag + " weight dtype", out);
  }
  if (
      in_w.shape().size() != in_scales.shape().size() ||
      in_scales.shape() != in_biases.shape()) {
    omarchy::unsupported(tag + " scales shape", out);
  }
  size_t words_per_row = in_w.shape(-1);
  uint64_t out_columns =
      static_cast<uint64_t>(words_per_row) * 32u / bits_;
  uint64_t groups_per_row =
      static_cast<uint64_t>(in_scales.shape(-1));
  if (
      groups_per_row * group_size_ != out_columns ||
      !std::equal(
          in_w.shape().begin(),
          in_w.shape().end() - 1,
          in_scales.shape().begin())) {
    omarchy::unsupported(tag + " shape", out);
  }
  std::optional<array> w_temp;
  std::optional<array> scales_temp;
  std::optional<array> biases_temp;
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, stream());
  const array& scales = ensure_dense(
      in_scales, in_scales.flags().row_contiguous, scales_temp, encoder, stream());
  const array& biases = ensure_dense(
      in_biases, in_biases.flags().row_contiguous, biases_temp, encoder, stream());
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

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // An honest refusal in place of a half-implementation: fast::CustomKernel
  // carries user-authored Metal shading-language source (or precompiled
  // Metal air) that only Apple's Metal toolchain can compile and load. This
  // backend dispatches SPIR-V to Vulkan, and no Metal-to-SPIR-V translator
  // exists in this stack, so arbitrary user kernel source cannot be
  // validated or executed here. Upstream also refuses CPU execution
  // ("Custom kernels only run on GPU"), so a CPU stream cannot rescue it.
  throw std::runtime_error(
      "[omarchy] fast::CustomKernel is not supported on the Omarchy Vulkan "
      "backend: custom kernels ship Metal shading-language source that only "
      "the Metal backend can compile and load, and this stack has no "
      "Metal-to-SPIR-V translator. Port the kernel to GLSL as a native "
      "Omarchy compute shader instead. No GPU kernel exists for it and"
      " custom kernels cannot run on a CPU stream; no silent CPU fallback"
      " occurs.");
}

} // namespace fast

namespace distributed {
OMARCHY_UNSUPPORTED_MULTI(AllReduce)
OMARCHY_UNSUPPORTED_MULTI(AllGather)
OMARCHY_UNSUPPORTED_MULTI(Send)
OMARCHY_UNSUPPORTED_MULTI(Recv)
OMARCHY_UNSUPPORTED_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core

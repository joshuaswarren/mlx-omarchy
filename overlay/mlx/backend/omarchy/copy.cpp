// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Buffer-level copy path. Transfer commands handle contiguous same-dtype
// ranges. Vulkan compute handles contiguous FP16 and FP32 conversions,
// non-zero scalar fills, and same-dtype strided copies.
// Dtype-converting strided copies keep the exact compatibility error for
// this slice.

#include "mlx/backend/omarchy/unsupported.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/vulkan.h"

namespace mlx::core {

namespace {

VkBuffer buffer_handle(const array& arr) {
  auto* vbuf = static_cast<const omarchy::VulkanBuffer*>(arr.buffer().ptr());
  return vbuf->buffer;
}

bool scalar_is_zero(const array& in, int64_t item_offset) {
  const auto* bytes = in.data<char>() + item_offset * in.itemsize();
  size_t itemsize = in.itemsize();
  for (size_t i = 0; i < itemsize; ++i) {
    if (bytes[i] != 0) {
      return false;
    }
  }
  return true;
}

// Host-side read of a scalar fill value. Callers must synchronize the
// stream first so a GPU-produced scalar is visible.
float scalar_fill_value(const array& in, int64_t item_offset) {
  const auto* bytes = in.data<char>() + item_offset * in.itemsize();
  if (in.dtype() == float32) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
  }
  uint16_t bits = 0;
  std::memcpy(&bits, bytes, sizeof(bits));
  if (in.dtype() == bfloat16) {
    uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
  }
  // Widen the float16 bit pattern to float32.
  uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1fu;
  uint32_t mantissa = bits & 0x3ffu;
  uint32_t wide = sign;
  if (exponent == 0 && mantissa != 0) {
    // Renormalize a subnormal into the float32 exponent field.
    exponent = 113;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x3ffu;
    wide |= (exponent << 23) | (mantissa << 13);
  } else if (exponent == 0x1fu) {
    wide |= 0x7f800000u | (mantissa << 13);
  } else if (exponent != 0) {
    wide |= ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

// Byte offset of a copy start in the backing VkBuffer: the array's own
// buffer offset plus an explicit item offset.
VkDeviceSize byte_offset(const array& arr, int64_t item_offset) {
  return static_cast<VkDeviceSize>(arr.offset() + item_offset * arr.itemsize());
}

uint32_t checked_u32(
    size_t value,
    const std::string& name,
    const array& out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    omarchy::unsupported(name + " with more than UINT32_MAX elements", out);
  }
  return static_cast<uint32_t>(value);
}

uint32_t compute_item_offset(
    const array& value,
    int64_t item_offset,
    const std::string& name,
    const array& out) {
  if (item_offset < 0 || value.offset() % value.itemsize() != 0) {
    omarchy::unsupported(name + " byte offset", out);
  }
  uint64_t base = value.offset() / value.itemsize();
  uint64_t delta = static_cast<uint64_t>(item_offset);
  constexpr uint64_t max_index = std::numeric_limits<uint32_t>::max();
  if (base > max_index || delta > max_index - base) {
    omarchy::unsupported(name + " index span", out);
  }
  return static_cast<uint32_t>(base + delta);
}

omarchy::ComputeBinding compute_binding(const array& value) {
  auto* buffer =
      static_cast<const omarchy::VulkanBuffer*>(value.buffer().ptr());
  return {buffer->buffer, 0, buffer->size};
}

// Zero an allocated output at its destination offset: whole 4-byte words on
// the device with vkCmdFillBuffer, misaligned lead and trailing bytes on
// the host. When edges exist, prior device work is drained first and the
// host memset happens immediately - never in a completion handler, or a
// later same-stream GPU submission could read stale bytes. The next
// submission's flush publishes the host writes.
void fill_zero(const Stream& s, array& out, int64_t o_offset) {
  auto& encoder = omarchy::get_command_encoder(s);
  size_t start = byte_offset(out, o_offset);
  size_t nbytes = out.nbytes();
  // vkCmdFillBuffer requires a 4-byte-aligned offset and size.
  size_t lead = (4 - start % 4) % 4;
  if (lead > nbytes) {
    lead = nbytes;
  }
  size_t words_bytes = (nbytes - lead) & ~size_t(3);
  size_t tail = nbytes - lead - words_bytes;
  if (lead + tail > 0) {
    encoder.synchronize();
    auto* base = static_cast<uint8_t*>(out.buffer().raw_ptr());
    std::memset(base + start, 0, lead);
    std::memset(base + start + lead + words_bytes, 0, tail);
  }
  if (words_bytes > 0) {
    encoder.fill_buffer(buffer_handle(out), 0, words_bytes, start + lead);
  }
}

omarchy::ComputeKernel fill_kernel(Dtype dtype) {
  if (dtype == float16) {
    return omarchy::ComputeKernel::FillF16;
  }
  if (dtype == bfloat16) {
    return omarchy::ComputeKernel::FillBF16;
  }
  return omarchy::ComputeKernel::FillF32;
}

omarchy::ComputeKernel copy_general_kernel(Dtype dtype) {
  if (dtype == float16) {
    return omarchy::ComputeKernel::CopyGeneralF16;
  }
  if (dtype == bfloat16) {
    return omarchy::ComputeKernel::CopyGeneralBF16;
  }
  return omarchy::ComputeKernel::CopyGeneralF32;
}

// Storage-buffer access requirements for a 16-bit float dtype, shared by
// the fill and strided-copy kernels.
void require_float_storage(
    const std::string& name,
    Dtype dtype,
    const array& out,
    omarchy::CommandEncoder& encoder) {
  const auto& capabilities = encoder.device().capabilities();
  if ((dtype == float16 &&
       (!capabilities.shader_float16 ||
        !capabilities.storage_buffer_16bit_access)) ||
      (dtype == bfloat16 &&
       (!capabilities.storage_buffer_16bit_access ||
        !capabilities.shader_int16))) {
    omarchy::unsupported(name + " capability", out);
  }
}

} // namespace

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  if (dynamic_i_offset || dynamic_o_offset) {
    // The offset tensor would have to be read by a shader.
    omarchy::unsupported("DynamicSlice copy", out);
  }

  if (out.nbytes() == 0) {
    return;
  }

  auto& encoder = omarchy::get_command_encoder(s);
  if (ctype == CopyType::Scalar) {
    if (in.has_primitive()) {
      omarchy::unsupported("GPU-in-flight scalar fill", out);
    }
    encoder.synchronize();
    if (scalar_is_zero(in, i_offset)) {
      fill_zero(s, out, o_offset);
      return;
    }
    if (in.dtype() != float32 && in.dtype() != float16 &&
        in.dtype() != bfloat16) {
      omarchy::unsupported("non-zero scalar fill", out);
    }
    require_float_storage("non-zero scalar fill", in.dtype(), out, encoder);
    uint32_t count = checked_u32(out.data_size(), "scalar fill", out);
    omarchy::ComputeParams params;
    params.count = count;
    params.output_size = count;
    params.output_offset =
        compute_item_offset(out, o_offset, "scalar fill", out);
    params.alpha = scalar_fill_value(in, i_offset);
    std::array<omarchy::ComputeBinding, 1> bindings{compute_binding(out)};
    encoder.dispatch_compute(
        fill_kernel(in.dtype()),
        bindings,
        params,
        omarchy::compute_dispatch_group_count(count));
    return;
  }

  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    if (in.dtype() != out.dtype() ||
        (in.dtype() != float32 && in.dtype() != float16 &&
         in.dtype() != bfloat16)) {
      omarchy::unsupported("strided copy", out);
    }
    require_float_storage("strided copy", in.dtype(), out, encoder);
    for (const auto& strides : {i_strides, o_strides}) {
      for (int64_t stride : strides) {
        if (stride < 0) {
          omarchy::unsupported("negative stride copy", out);
        }
      }
    }
    auto [collapsed_shape, collapsed_strides] = collapse_contiguous_dims(
        data_shape, std::vector<Strides>{i_strides, o_strides});
    size_t rank = collapsed_shape.size();
    if (rank > 4) {
      omarchy::unsupported("rank>4 strided copy", out);
    }
    size_t total = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
      total *= static_cast<size_t>(collapsed_shape[axis]);
    }
    uint32_t count = checked_u32(total, "strided copy", out);
    omarchy::ComputeParams params;
    params.count = count;
    params.dims = static_cast<uint32_t>(rank);
    uint64_t in_span = 0;
    uint64_t out_span = 0;
    for (size_t axis = 0; axis < rank; ++axis) {
      params.shape[axis] = static_cast<uint32_t>(collapsed_shape[axis]);
      params.in_strides[axis] =
          static_cast<uint32_t>(collapsed_strides[0][axis]);
      params.out_strides[axis] =
          static_cast<uint32_t>(collapsed_strides[1][axis]);
      uint64_t extent = params.shape[axis] - 1u;
      in_span += extent * params.in_strides[axis];
      out_span += extent * params.out_strides[axis];
    }
    params.lhs_offset =
        compute_item_offset(in, i_offset, "strided copy", out);
    params.output_offset =
        compute_item_offset(out, o_offset, "strided copy", out);
    if (!omarchy::compute_index_span_fits(params.lhs_offset, in_span + 1) ||
        !omarchy::compute_index_span_fits(
            params.output_offset, out_span + 1)) {
      omarchy::unsupported("strided copy index span", out);
    }
    std::array<omarchy::ComputeBinding, 3> bindings{
        compute_binding(in), compute_binding(in), compute_binding(out)};
    encoder.dispatch_compute(
        copy_general_kernel(in.dtype()),
        bindings,
        params,
        omarchy::compute_dispatch_group_count(count));
    return;
  }

  if (in.dtype() != out.dtype()) {
    omarchy::ComputeKernel kernel;
    if (in.dtype() == float16 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastF16F32;
    } else if (in.dtype() == float32 && out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::CastF32F16;
    } else if (in.dtype() == bfloat16 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastBF16F32;
    } else if (in.dtype() == float32 && out.dtype() == bfloat16) {
      kernel = omarchy::ComputeKernel::CastF32BF16;
    } else if (in.dtype() == bfloat16 && out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::CastBF16F16;
    } else if (in.dtype() == float16 && out.dtype() == bfloat16) {
      kernel = omarchy::ComputeKernel::CastF16BF16;
    } else {
      omarchy::unsupported("dtype converting copy", out);
    }

    const auto& capabilities = encoder.device().capabilities();
    if ((in.dtype() == float16 || out.dtype() == float16) &&
        (!capabilities.shader_float16 ||
         !capabilities.storage_buffer_16bit_access)) {
      omarchy::unsupported("dtype converting copy float16 capability", out);
    }
    if ((in.dtype() == bfloat16 || out.dtype() == bfloat16) &&
        (!capabilities.storage_buffer_16bit_access ||
         !capabilities.shader_int16)) {
      omarchy::unsupported("dtype converting copy bfloat16 capability", out);
    }

    uint32_t count =
        checked_u32(out.data_size(), "dtype converting copy", out);
    omarchy::ComputeParams params;
    params.count = count;
    params.lhs_size =
        checked_u32(in.data_size(), "dtype converting copy", out);
    params.rhs_size = params.lhs_size;
    params.output_size = count;
    params.lhs_offset =
        compute_item_offset(in, i_offset, "dtype converting copy", out);
    params.output_offset =
        compute_item_offset(out, o_offset, "dtype converting copy", out);
    if (!omarchy::compute_index_span_fits(params.lhs_offset, count) ||
        !omarchy::compute_index_span_fits(params.output_offset, count)) {
      omarchy::unsupported("dtype converting copy index span", out);
    }
    std::array<omarchy::ComputeBinding, 3> bindings{
        compute_binding(in), compute_binding(in), compute_binding(out)};
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(count));
    return;
  }

  encoder.copy_buffer(
      buffer_handle(in),
      buffer_handle(out),
      out.nbytes(),
      byte_offset(in, i_offset),
      byte_offset(out, o_offset));
}

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  if (out.nbytes() > 0) {
    out.set_data(omarchy::allocator().malloc(out.nbytes()));
  }
  copy_gpu_inplace(
      in, out, in.shape(), in.strides(), out.strides(), 0, 0, ctype, s);
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }
  if (out.nbytes() > 0) {
    out.set_data(omarchy::allocator().malloc(out.nbytes()));
  }
  if (val.has_primitive()) {
    omarchy::unsupported("GPU-in-flight fill", out);
  }
  auto& encoder = omarchy::get_command_encoder(s);
  encoder.synchronize();
  if (!scalar_is_zero(val, 0)) {
    omarchy::unsupported("non-zero fill", out);
  }
  fill_zero(s, out, 0);
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    // Only a contiguous source reshapes as a flat device copy; strided
    // reshapes need a gather shader.
    if (!in.flags().contiguous || in.dtype() != out.dtype()) {
      omarchy::unsupported("strided reshape", out);
    }
    if (out.nbytes() > 0) {
      out.set_data(omarchy::allocator().malloc(out.nbytes()));
    }
    auto& encoder = omarchy::get_command_encoder(s);
    encoder.copy_buffer(
        buffer_handle(in),
        buffer_handle(out),
        out.nbytes(),
        byte_offset(in, 0),
        byte_offset(out, 0));
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

} // namespace mlx::core

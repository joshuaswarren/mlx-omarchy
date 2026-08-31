// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Buffer-level copy path. Transfer commands handle contiguous same-dtype
// ranges. Vulkan compute handles contiguous FP16 and FP32 conversions.
// Strided copies keep the exact compatibility error for this slice.

#include "mlx/backend/omarchy/unsupported.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

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

// Byte offset of a copy start in the backing VkBuffer: the array's own
// buffer offset plus an explicit item offset.
VkDeviceSize byte_offset(const array& arr, int64_t item_offset) {
  return static_cast<VkDeviceSize>(arr.offset() + item_offset * arr.itemsize());
}

uint32_t checked_u32(size_t value, const array& out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    omarchy::unsupported(
        "dtype converting copy with more than UINT32_MAX elements", out);
  }
  return static_cast<uint32_t>(value);
}

uint32_t compute_item_offset(
    const array& value,
    int64_t item_offset,
    const array& out) {
  if (item_offset < 0 || value.offset() % value.itemsize() != 0) {
    omarchy::unsupported("dtype converting copy byte offset", out);
  }
  uint64_t base = value.offset() / value.itemsize();
  uint64_t delta = static_cast<uint64_t>(item_offset);
  constexpr uint64_t max_index = std::numeric_limits<uint32_t>::max();
  if (base > max_index || delta > max_index - base) {
    omarchy::unsupported("dtype converting copy index span", out);
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

} // namespace

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& /*data_shape*/,
    const Strides& /*i_strides*/,
    const Strides& /*o_strides*/,
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
  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    omarchy::unsupported("strided copy", out);
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
    if (!scalar_is_zero(in, i_offset)) {
      omarchy::unsupported("non-zero scalar fill", out);
    }
    fill_zero(s, out, o_offset);
    return;
  }

  if (in.dtype() != out.dtype()) {
    omarchy::ComputeKernel kernel;
    if (in.dtype() == float16 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastF16F32;
    } else if (in.dtype() == float32 && out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::CastF32F16;
    } else {
      omarchy::unsupported("dtype converting copy", out);
    }

    const auto& capabilities = encoder.device().capabilities();
    if (!capabilities.shader_float16 ||
        !capabilities.storage_buffer_16bit_access) {
      omarchy::unsupported("dtype converting copy float16 capability", out);
    }

    uint32_t count = checked_u32(out.data_size(), out);
    omarchy::ComputeParams params;
    params.count = count;
    params.lhs_size = checked_u32(in.data_size(), out);
    params.rhs_size = params.lhs_size;
    params.output_size = count;
    params.lhs_offset = compute_item_offset(in, i_offset, out);
    params.output_offset = compute_item_offset(out, o_offset, out);
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

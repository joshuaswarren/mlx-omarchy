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
  if (dtype == complex64) {
    // Complex64Transport: vec2 element, the scalar's real part rides
    // in alpha and the imaginary part in beta.
    return omarchy::ComputeKernel::FillComplex64;
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
  if (dtype == int32 || dtype == uint32) {
    // Raw-word copies: the same 4-byte stride math as float32 with no
    // float conversion, so packed words stay bit-exact.
    return omarchy::ComputeKernel::CopyGeneralU32;
  }
  if (dtype == complex64) {
    // Complex64Transport: a vec2 element copies bit-exact with no
    // conversion, and the 8-byte stride math is the item math the
    // params already carry.
    return omarchy::ComputeKernel::CopyGeneralComplex64;
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
    if (in.dtype() == int32 || in.dtype() == uint32) {
      if (in.dtype() != out.dtype()) {
        omarchy::unsupported("non-zero scalar fill dtype", out);
      }
      uint32_t count = checked_u32(out.data_size(), "scalar fill", out);
      uint32_t word = 0;
      std::memcpy(
          &word, in.data<char>() + i_offset * in.itemsize(), sizeof(word));
      VkDeviceSize start = static_cast<VkDeviceSize>(
          compute_item_offset(out, o_offset, "scalar fill", out)) *
          sizeof(uint32_t);
      encoder.fill_buffer(
          buffer_handle(out), word, count * sizeof(uint32_t), start);
      return;
    }
    if (in.dtype() != float32 && in.dtype() != float16 &&
        in.dtype() != bfloat16 && in.dtype() != complex64) {
      omarchy::unsupported("non-zero scalar fill", out);
    }
    require_float_storage("non-zero scalar fill", in.dtype(), out, encoder);
    uint32_t count = checked_u32(out.data_size(), "scalar fill", out);
    omarchy::ComputeParams params;
    params.count = count;
    params.output_size = count;
    params.output_offset =
        compute_item_offset(out, o_offset, "scalar fill", out);
    if (in.dtype() == complex64) {
      // Complex64Transport: the fill kernel writes vec2(alpha, beta),
      // so the scalar's two float32 words ride in alpha and beta.
      // scalar_fill_value would decode the real part as float16 bits,
      // so both components come from the raw 8 bytes instead.
      float parts[2] = {0.0f, 0.0f};
      std::memcpy(
          parts,
          in.data<char>() + i_offset * in.itemsize(),
          sizeof(parts));
      params.alpha = parts[0];
      params.beta = parts[1];
    } else {
      params.alpha = scalar_fill_value(in, i_offset);
    }
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
         in.dtype() != bfloat16 && in.dtype() != int32 &&
         in.dtype() != uint32 && in.dtype() != complex64)) {
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
    if (in.dtype() == bool_ && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastBoolF32;
    } else if (in.dtype() == float16 && out.dtype() == float32) {
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
    } else if (in.dtype() == int32 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastI32F32;
    } else if (in.dtype() == float32 && out.dtype() == int32) {
      kernel = omarchy::ComputeKernel::CastF32I32;
    } else if (in.dtype() == int32 && out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::CastI32F16;
    } else if (in.dtype() == float16 && out.dtype() == int32) {
      kernel = omarchy::ComputeKernel::CastF16I32;
    } else if (in.dtype() == int32 && out.dtype() == bfloat16) {
      kernel = omarchy::ComputeKernel::CastI32BF16;
    } else if (in.dtype() == bfloat16 && out.dtype() == int32) {
      kernel = omarchy::ComputeKernel::CastBF16I32;
    } else if (in.dtype() == uint32 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastU32F32;
    } else if (in.dtype() == float32 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastF32Complex64;
    } else if (in.dtype() == int32 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastI32Complex64;
    } else if (in.dtype() == uint32 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastU32Complex64;
    } else if (in.dtype() == bool_ && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastBoolComplex64;
    } else if (in.dtype() == float16 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastF16Complex64;
    } else if (in.dtype() == bfloat16 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastBF16Complex64;
    } else if (in.dtype() == complex64 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastComplex64F32;
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
    // CastBoolF32 processes one output word (four elements) per thread
    // for the same reason as select.comp and compare.comp; every other
    // cast kernel stays per element.
    uint32_t dispatch_count =
        kernel == omarchy::ComputeKernel::CastBoolF32 ||
            kernel == omarchy::ComputeKernel::CastBoolComplex64
        ? checked_u32(
              (static_cast<uint64_t>(count) + 3) / 4,
              "dtype converting copy",
              out)
        : count;
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(dispatch_count));
    return;
  }

  encoder.copy_buffer(
      buffer_handle(in),
      buffer_handle(out),
      out.nbytes(),
      byte_offset(in, i_offset),
      byte_offset(out, o_offset));
}

void copy_gpu(const array& input, array& out, CopyType ctype, const Stream& s) {
  // Vector and dtype-converting copies read the source flat, in storage
  // order. That is only the logical order when the flags honestly
  // describe a dense layout. Two contiguous-flagged views violate it:
  // a transposed view of a contiguous array (data_size == size but
  // storage order differs from logical order, so astype returned the
  // source order), and a stride-0 broadcast view (data_size smaller
  // than size, so full_like with an array fill and a dtype cast read
  // past the one-element buffer and zero-filled everything after the
  // first element). Both report contiguous=true under the span-based
  // definition and row_contiguous=false, so row_contiguous alone
  // decides; the data_size relation differs between the two and must
  // not be part of the test. Materialize through a same-dtype strided
  // copy first; the flat op then reads a dense buffer whose storage
  // order is the logical order and whose allocation covers the whole
  // shape.
  std::optional<array> dense;
  const array* in = &input;
  bool flat_only =
      (ctype == CopyType::Vector || input.dtype() != out.dtype()) &&
      !input.flags().row_contiguous;
  if (flat_only) {
    dense = array(input.shape(), input.dtype(), nullptr, {});
    dense->set_data(omarchy::allocator().malloc(dense->nbytes()));
    // Pin it: this local dies at return while both dispatches that touch
    // it are still queued. Unpinned, the allocator recycled its bytes into
    // the next token's RoPE offset scalar and the strided copy wrote an
    // f32 over it (receipts/2026-09-04-rope-gate-drain.md). The per-call
    // queue drains that used to sit in rope_trig_gate masked it.
    omarchy::get_command_encoder(s).add_temporary(*dense);
    copy_gpu_inplace(
        input,
        *dense,
        input.shape(),
        input.strides(),
        make_contiguous_strides(input.shape()),
        /*i_offset=*/0,
        /*o_offset=*/0,
        CopyType::GeneralGeneral,
        s);
    in = &*dense;
  }
  // Upstream's set_copy_output_data always gives the output a buffer, even
  // for zero-size outputs (malloc(0) yields a valid empty VulkanBuffer).
  // Skipping set_data left array_desc_->data null, and any later
  // buffer_size()/data() access on the eval'd array segfaulted.
  out.set_data(omarchy::allocator().malloc(out.nbytes()));
  copy_gpu_inplace(
      *in, out, in->shape(), in->strides(), out.strides(), 0, 0, ctype, s);
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
  if (!copy_necessary) {
    shared_buffer_reshape(in, out_strides, out);
    return;
  }
  // A strided reshape is a general gather. Broadcast views from
  // mx.repeat and mx.tile carry stride-0 axes, and transposed views
  // permute strides; both report flags().contiguous under the
  // span-based definition while size() exceeds data_size(), so the
  // old flat buffer copy here read past the source allocation. The
  // strided-copy engine expresses both shapes for 4-byte words (floats
  // converted, int32/uint32 raw); rank limits and non-4-byte dtypes
  // keep the named error for the rest.
  if (
      in.dtype() != float32 && in.dtype() != float16 &&
      in.dtype() != bfloat16 && in.dtype() != int32 &&
      in.dtype() != uint32 && in.dtype() != complex64) {
    omarchy::unsupported("strided reshape", out);
  }
  if (out.nbytes() > 0) {
    out.set_data(omarchy::allocator().malloc(out.nbytes()));
  }
  copy_gpu_inplace(
      in,
      out,
      in.shape(),
      in.strides(),
      make_contiguous_strides(in.shape()),
      /*i_offset=*/0,
      /*o_offset=*/0,
      CopyType::General,
      s);
}
} // namespace mlx::core

// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Omarchy slicing support. slice_gpu and pad_gpu come from the shared GPU
// directory; concatenation copies each input into a window of the output
// through the shared strided-copy path. Dynamic offsets still need gather
// shaders and are not part of this slice.

#include "mlx/backend/omarchy/unsupported.h"

#include <numeric>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/omarchy/allocator.h"

namespace mlx::core {

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  if (out.nbytes() > 0) {
    out.set_data(omarchy::allocator().malloc(out.nbytes()));
  }

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;

  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i].size() == 0) {
      continue;
    }
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    auto ctype = CopyType::GeneralGeneral;
    if (axis == 0 && inputs[i].flags().row_contiguous) {
      ctype = CopyType::Vector;
    }
    copy_gpu_inplace(inputs[i], out_slice, ctype, s);
  }
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& /*strides*/,
    const std::vector<int>& /*axes*/,
    const Stream& /*s*/) {
  omarchy::unsupported("dynamic slice offset", indices);
}

} // namespace mlx::core

// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Omarchy slicing support. slice_gpu and pad_gpu come from the shared GPU
// directory; concatenation and dynamic offsets need gather shaders and are
// not part of this slice.

#include "mlx/backend/omarchy/unsupported.h"

#include "mlx/backend/gpu/slicing.h"

namespace mlx::core {

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int /*axis*/,
    const Stream& /*s*/) {
  omarchy::unsupported("Concatenate", out);
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& /*strides*/,
    const std::vector<int>& /*axes*/,
    const Stream& /*s*/) {
  omarchy::unsupported("dynamic slice offset", indices);
}

} // namespace mlx::core

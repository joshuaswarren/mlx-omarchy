// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
#pragma once

// Dense<->tile element placement shared by the libane device backend.
//
// Hardware-proven layout (a9f14124 smoke runtime): linear element index
// maps to plane * plane_stride + row * row_stride + column * element_size
// bytes inside the allocation, with NCHW[6] = [N, C, H, W, plane_stride,
// row_stride]. element_size is 2 for fp16 surfaces; the 2026-09-13 ABI
// extension adds 1-byte bool surfaces (elementDtype field, default
// fp16): logical_bytes stay bytes — a bool surface of N elements is N
// bytes — and only element-derived arithmetic widens from the fp16
// assumption.

#include "mlx/backend/omarchy/ane/manifest.h"

#include <cstddef>
#include <cstdint>

namespace mlx::core::omarchy::ane {

inline size_t ane_element_size(const AneProgramBinding& binding) {
  return binding.dtype == "bool" ? 1 : 2;
}

inline size_t ane_packed_offset(
    const AneProgramBinding& binding, size_t element) {
  const uint64_t height = binding.nchw[2];
  const uint64_t width = binding.nchw[3];
  const uint64_t plane_stride = binding.nchw[4];
  const uint64_t row_stride = binding.nchw[5];
  const uint64_t plane_elements = height * width;
  const uint64_t plane = element / plane_elements;
  const uint64_t within = element % plane_elements;
  const uint64_t row = within / width;
  const uint64_t column = within % width;
  return static_cast<size_t>(
      plane * plane_stride + row * row_stride +
      column * ane_element_size(binding));
}

} // namespace mlx::core::omarchy::ane

// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime_detail.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

using namespace mlx::core::omarchy::ane;

namespace {

AneProgramBinding lane_binding() {
  AneProgramBinding binding;
  binding.tensor = "x";
  binding.channel = 5;
  binding.dtype = "float16";
  binding.shape = {1, 64, 1, 1};
  binding.nchw = {1, 64, 1, 1, 64, 64};
  binding.logical_bytes = 128;
  binding.allocation_bytes = 0x4000;
  binding.element_count = 64;
  binding.physical_elements = 64;
  return binding;
}

std::vector<uint8_t> dense_values(size_t elements) {
  std::vector<uint8_t> dense(elements * 2);
  for (size_t i = 0; i < elements; ++i) {
    dense[i * 2] = static_cast<uint8_t>(i);
    dense[i * 2 + 1] = static_cast<uint8_t>(0x80 + i);
  }
  return dense;
}

} // namespace

TEST_CASE("ANE runtime packs and unpacks qualified 64-byte lanes") {
  const auto binding = lane_binding();
  const auto dense = dense_values(64);

  const auto packed = detail::pack_binding(binding, dense);
  REQUIRE(packed.size() == binding.allocation_bytes);
  for (size_t i = 0; i < 64; ++i) {
    CHECK(packed[i * 64] == dense[i * 2]);
    CHECK(packed[i * 64 + 1] == dense[i * 2 + 1]);
    CHECK(std::all_of(
        packed.begin() + i * 64 + 2,
        packed.begin() + (i + 1) * 64,
        [](uint8_t value) { return value == 0; }));
  }

  std::vector<uint8_t> round_trip(dense.size(), 0xff);
  detail::unpack_binding(binding, packed, round_trip);
  CHECK(round_trip == dense);
}

TEST_CASE("ANE runtime preserves tensor regions outside a sliced binding") {
  auto binding = lane_binding();
  binding.shape = {1, 4, 1, 1};
  binding.logical_bytes = 8;
  binding.element_offset = 2;
  binding.element_count = 4;
  binding.physical_elements = 4;
  binding.nchw = {1, 4, 1, 1, 64, 64};
  const auto dense = dense_values(8);

  const auto packed = detail::pack_binding(binding, dense);
  std::vector<uint8_t> restored(dense.size(), 0x5a);
  detail::unpack_binding(binding, packed, restored);

  CHECK(restored[0] == 0x5a);
  CHECK(restored[3] == 0x5a);
  CHECK(std::equal(restored.begin() + 4, restored.begin() + 12, dense.begin() + 4));
  CHECK(restored[12] == 0x5a);
  CHECK(restored.back() == 0x5a);
}

TEST_CASE("ANE runtime rejects invalid staging and deadlines before device work") {
  const auto binding = lane_binding();
  CHECK_THROWS_WITH_AS(
      detail::pack_binding(binding, std::vector<uint8_t>(126)),
      "[omarchy-ane] runtime: tensor 'x' dense staging byte count is 126, expected at least 128.",
      std::runtime_error);
  CHECK_THROWS_WITH_AS(
      detail::validate_deadline(std::chrono::milliseconds(0)),
      "[omarchy-ane] runtime: deadline must be positive.",
      std::invalid_argument);
}

// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (gpu::is_available()) {
    return true;
  }
  std::cout << "Skipping: no qualifying Vulkan device.\n";
  return false;
}

template <typename T>
std::vector<T> read_values(array value, const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const T* data = value.data<T>();
  return {data, data + value.size()};
}

template <typename T>
void check_axis_contract(
    const array& input,
    const std::vector<T>& logical,
    const Shape& shape,
    int axis,
    int kth,
    const Stream& stream) {
  REQUIRE_EQ(input.shape(), shape);
  const size_t axis_size = static_cast<size_t>(shape[axis]);
  size_t inner = 1;
  for (size_t i = static_cast<size_t>(axis + 1); i < shape.size(); ++i) {
    inner *= static_cast<size_t>(shape[i]);
  }
  const size_t outer = logical.size() / (axis_size * inner);
  std::vector<T> expected_values(logical.size());
  std::vector<uint32_t> expected_indices(logical.size());
  for (size_t o = 0; o < outer; ++o) {
    for (size_t i = 0; i < inner; ++i) {
      std::vector<uint32_t> order(axis_size);
      std::iota(order.begin(), order.end(), 0u);
      auto at = [&](size_t a) {
        return o * axis_size * inner + a * inner + i;
      };
      std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return logical[at(a)] < logical[at(b)] ||
            (logical[at(a)] == logical[at(b)] && a < b);
      });
      for (size_t a = 0; a < axis_size; ++a) {
        expected_values[at(a)] = logical[at(order[a])];
        expected_indices[at(a)] = order[a];
      }
    }
  }

  CHECK_EQ(read_values<T>(sort(input, axis, stream), stream), expected_values);
  CHECK_EQ(
      read_values<uint32_t>(argsort(input, axis, stream), stream),
      expected_indices);

  auto partitioned = read_values<T>(partition(input, kth, axis, stream), stream);
  auto partition_indices =
      read_values<uint32_t>(argpartition(input, kth, axis, stream), stream);
  for (size_t o = 0; o < outer; ++o) {
    for (size_t i = 0; i < inner; ++i) {
      auto at = [&](size_t a) {
        return o * axis_size * inner + a * inner + i;
      };
      const T pivot = expected_values[at(static_cast<size_t>(kth))];
      std::vector<T> value_row(axis_size);
      std::vector<T> indexed_row(axis_size);
      std::vector<bool> seen(axis_size, false);
      for (size_t a = 0; a < axis_size; ++a) {
        value_row[a] = partitioned[at(a)];
        uint32_t source = partition_indices[at(a)];
        REQUIRE(source < axis_size);
        CHECK_FALSE(seen[source]);
        seen[source] = true;
        indexed_row[a] = logical[at(source)];
        if (a < static_cast<size_t>(kth)) {
          CHECK(value_row[a] <= pivot);
          CHECK(indexed_row[a] <= pivot);
        } else if (a > static_cast<size_t>(kth)) {
          CHECK(value_row[a] >= pivot);
          CHECK(indexed_row[a] >= pivot);
        }
      }
      CHECK_EQ(value_row[kth], pivot);
      CHECK_EQ(indexed_row[kth], pivot);
      std::sort(value_row.begin(), value_row.end());
      std::sort(indexed_row.begin(), indexed_row.end());
      std::vector<T> expected_row(axis_size);
      for (size_t a = 0; a < axis_size; ++a) {
        expected_row[a] = expected_values[at(a)];
      }
      CHECK_EQ(value_row, expected_row);
      CHECK_EQ(indexed_row, expected_row);
    }
  }
}

template <typename T>
void check_one_dim_contract(
    const std::vector<T>& values,
    Dtype dtype,
    int kth,
    const Stream& stream) {
  array input(values.begin(), Shape{static_cast<int>(values.size())}, dtype);
  check_axis_contract(input, values, input.shape(), 0, kth, stream);
}

} // namespace

TEST_CASE("int64 and uint64 sorting preserves both words and extrema") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int64_t min = std::numeric_limits<int64_t>::min();
  const int64_t max = std::numeric_limits<int64_t>::max();
  check_one_dim_contract<int64_t>(
      {max, min, INT64_C(0x0000000200000000),
       INT64_C(0x00000001ffffffff), -1, 0, max, min + 1,
       -INT64_C(0x0000000200000000), -INT64_C(0x00000001ffffffff), 42, 42},
      int64,
      5,
      stream);
  check_one_dim_contract<uint64_t>(
      {UINT64_MAX, 0, UINT64_C(0x0000000200000000),
       UINT64_C(0x00000001ffffffff), UINT64_C(0x8000000000000000), 1,
       UINT64_MAX, UINT64_C(0xffffffff00000000), 42, 42},
      uint64,
      4,
      stream);
}

TEST_CASE("int64 sorting handles zero-stride and non-suffix axes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<int64_t> row = {
      1, 0, INT64_C(0x0000010000000000), -1,
      std::numeric_limits<int64_t>::min(), 0,
      std::numeric_limits<int64_t>::max(), 1};
  array base(row.begin(), Shape{8}, int64);
  array input = broadcast_to(base, Shape{4, 8}, stream);
  std::vector<int64_t> logical;
  for (int i = 0; i < 4; ++i) {
    logical.insert(logical.end(), row.begin(), row.end());
  }
  check_axis_contract(input, logical, Shape{4, 8}, 0, 2, stream);
  check_axis_contract(input, logical, Shape{4, 8}, 1, 3, stream);
}

TEST_CASE("uint64 sorting handles strided layouts and a non-suffix axis") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<uint64_t> base_values = {
      UINT64_MAX, 3, UINT64_C(0x100000000), 7,
      90, 80, 70, 60,
      0, UINT64_C(0xffffffff), UINT64_C(0x100000000), 5,
      50, 40, 30, 20,
      UINT64_C(0x8000000000000000), 3, UINT64_MAX, 1};
  array base(base_values.begin(), Shape{5, 4}, uint64);
  array input = slice(base, {0, 0}, {5, 4}, {2, 1}, stream);
  std::vector<uint64_t> logical;
  logical.insert(logical.end(), base_values.begin(), base_values.begin() + 4);
  logical.insert(logical.end(), base_values.begin() + 8, base_values.begin() + 12);
  logical.insert(logical.end(), base_values.begin() + 16, base_values.end());
  check_axis_contract(input, logical, Shape{3, 4}, 0, 1, stream);
}

TEST_CASE("int64 and uint64 sorting crosses the 1024-element merge boundary") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<int64_t> signed_values(2051);
  for (size_t i = 0; i < signed_values.size(); ++i) {
    int64_t high = static_cast<int64_t>((i * 48271u) % 131071u) - 65535;
    uint32_t low = static_cast<uint32_t>(i * 2654435761u);
    signed_values[i] = high * INT64_C(4294967296) + low;
  }
  signed_values[17] = std::numeric_limits<int64_t>::min();
  signed_values[1024] = std::numeric_limits<int64_t>::max();
  signed_values[1025] = signed_values[5];
  signed_values[2050] = std::numeric_limits<int64_t>::max();
  check_one_dim_contract(signed_values, int64, 1024, stream);

  std::vector<uint64_t> unsigned_values(1025);
  for (size_t i = 0; i < unsigned_values.size(); ++i) {
    unsigned_values[i] =
        (uint64_t((i * 65537u) % 1009u) << 32u) |
        uint32_t(i * 2246822519u);
  }
  unsigned_values[3] = 0;
  unsigned_values[511] = UINT64_MAX;
  unsigned_values[1023] = unsigned_values[7];
  unsigned_values[1024] = UINT64_MAX;
  check_one_dim_contract(unsigned_values, uint64, 512, stream);
}

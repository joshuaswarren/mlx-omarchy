// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <iostream>
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

std::vector<float> read(array value, const Stream& stream) {
  auto dense = contiguous(value, false, stream);
  dense.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = dense.data<float>();
  return {data, data + dense.size()};
}

void check_extracts(
    const array& input,
    const std::vector<float>& real_values,
    const std::vector<float>& imag_values,
    const Stream& stream) {
  CHECK(read(real(input, stream), stream) == real_values);
  CHECK(read(imag(input, stream), stream) == imag_values);

  std::vector<float> magnitudes(real_values.size());
  for (size_t i = 0; i < magnitudes.size(); ++i) {
    magnitudes[i] = std::hypot(real_values[i], imag_values[i]);
  }
  auto actual = read(abs(input, stream), stream);
  REQUIRE_EQ(actual.size(), magnitudes.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CHECK(actual[i] == doctest::Approx(magnitudes[i]).epsilon(1e-6));
  }
}

} // namespace

TEST_CASE("complex extraction follows logical layouts") {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device.\n";
    return;
  }
  auto stream = gpu_stream();
  array source(
      {complex64_t{1, 10}, complex64_t{2, 20}, complex64_t{3, 30},
       complex64_t{4, 40}, complex64_t{5, 50}, complex64_t{6, 60}},
      {2, 3});

  check_extracts(
      source,
      {1, 2, 3, 4, 5, 6},
      {10, 20, 30, 40, 50, 60},
      stream);
  check_extracts(
      transpose(source),
      {1, 4, 2, 5, 3, 6},
      {10, 40, 20, 50, 30, 60},
      stream);
  check_extracts(
      slice(reshape(source, {6}), {1}, {6}, {2}),
      {2, 4, 6},
      {20, 40, 60},
      stream);
  check_extracts(
      broadcast_to(slice(source, {0, 0}, {1, 3}), {2, 3}),
      {1, 2, 3, 1, 2, 3},
      {10, 20, 30, 10, 20, 30},
      stream);

  array empty = zeros({0}, complex64, stream);
  auto empty_real = real(empty, stream);
  auto empty_imag = imag(empty, stream);
  auto empty_abs = abs(empty, stream);
  CHECK_NOTHROW(empty_real.eval());
  CHECK_NOTHROW(empty_imag.eval());
  CHECK_NOTHROW(empty_abs.eval());
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(empty_real.buffer_size(), 0);
  CHECK_EQ(empty_imag.buffer_size(), 0);
  CHECK_EQ(empty_abs.buffer_size(), 0);
}

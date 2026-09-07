// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"

using namespace mlx::core;

namespace {

bool compute_available() {
  if (gpu::is_available()) {
    return true;
  }
  std::cout << "Skipping: no qualifying Vulkan device"
               " (set MLX_OMARCHY_ALLOW_NON_APPLE=1).\n";
  return false;
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

std::vector<float> read(array value, const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  return {value.data<float>(), value.data<float>() + value.size()};
}

float tolerance(float reference, float absolute, float relative) {
  return std::max(absolute, relative * std::abs(reference));
}

void check_value(
    const char* operation,
    float input,
    float got,
    float reference,
    float absolute,
    float relative = 0.0f) {
  INFO(operation << "(" << input << "): got=" << got
                 << " reference=" << reference);
  CHECK(std::abs(got - reference) <= tolerance(reference, absolute, relative));
}

} // namespace

TEST_CASE("float32 sin cos and tan cover the full finite magnitude range") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const std::vector<float> inputs{
      0.0f,
      -0.0f,
      0.5f,
      -0.5f,
      1.0f,
      -1.0f,
      12345.0f,
      -12345.0f,
      123456.789f,
      -123456.789f,
      1.0e6f,
      -1.0e6f,
      1.0e10f,
      -1.0e10f,
      1.0e20f,
      -1.0e20f,
      1.0e30f,
      -1.0e30f,
      std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max()};
  array x(inputs.begin(), Shape{static_cast<int>(inputs.size())}, float32);
  auto got_sin = read(sin(x, stream), stream);
  auto got_cos = read(cos(x, stream), stream);
  auto got_tan = read(tan(x, stream), stream);

  for (size_t i = 0; i < inputs.size(); ++i) {
    const float input = inputs[i];
    const float reference_sin = static_cast<float>(std::sin(input));
    const float reference_cos = static_cast<float>(std::cos(input));
    const float reference_tan = static_cast<float>(std::tan(input));
    check_value("sin", input, got_sin[i], reference_sin, 2.0e-6f);
    check_value("cos", input, got_cos[i], reference_cos, 2.0e-6f);
    check_value("tan", input, got_tan[i], reference_tan, 3.0e-5f, 3.0e-5f);
  }
}

TEST_CASE("float32 trig keeps accuracy around quadrant boundaries") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr double half_pi = 1.57079632679489661923132169163975144;
  const std::array<int, 8> quadrants{1, 2, 3, 4, 17, 1024, 65537, 1048576};
  std::vector<float> inputs;
  inputs.reserve(6 * quadrants.size());
  for (int quadrant : quadrants) {
    float center = static_cast<float>(quadrant * half_pi);
    float below = std::nextafter(center, -std::numeric_limits<float>::infinity());
    float above = std::nextafter(center, std::numeric_limits<float>::infinity());
    inputs.insert(inputs.end(), {below, center, above, -below, -center, -above});
  }

  array x(inputs.begin(), Shape{static_cast<int>(inputs.size())}, float32);
  auto got_sin = read(sin(x, stream), stream);
  auto got_cos = read(cos(x, stream), stream);
  auto got_tan = read(tan(x, stream), stream);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const float input = inputs[i];
    const float reference_sin = static_cast<float>(std::sin(input));
    const float reference_cos = static_cast<float>(std::cos(input));
    const float reference_tan = static_cast<float>(std::tan(input));
    check_value("sin", input, got_sin[i], reference_sin, 2.0e-6f);
    check_value("cos", input, got_cos[i], reference_cos, 2.0e-6f);
    check_value("tan", input, got_tan[i], reference_tan, 5.0e-5f, 5.0e-5f);
  }
}

TEST_CASE("float32 trig handles tiny values and non-finite inputs") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const float denorm = std::numeric_limits<float>::denorm_min();
  const float minimum = std::numeric_limits<float>::min();
  const std::vector<float> finite{0.0f, -0.0f, denorm, -denorm, minimum, -minimum};
  array x(finite.begin(), Shape{static_cast<int>(finite.size())}, float32);
  auto got_sin = read(sin(x, stream), stream);
  auto got_cos = read(cos(x, stream), stream);
  auto got_tan = read(tan(x, stream), stream);
  for (size_t i = 0; i < finite.size(); ++i) {
    check_value("sin", finite[i], got_sin[i], finite[i], denorm);
    check_value("cos", finite[i], got_cos[i], 1.0f, 0.0f);
    check_value("tan", finite[i], got_tan[i], finite[i], denorm);
  }
  CHECK(std::signbit(got_sin[1]));
  CHECK(std::signbit(got_tan[1]));

  const std::vector<float> non_finite{
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  array special(
      non_finite.begin(), Shape{static_cast<int>(non_finite.size())}, float32);
  auto special_sin = read(sin(special, stream), stream);
  auto special_cos = read(cos(special, stream), stream);
  auto special_tan = read(tan(special, stream), stream);
  for (size_t i = 0; i < non_finite.size(); ++i) {
    CHECK(std::isnan(special_sin[i]));
    CHECK(std::isnan(special_cos[i]));
    CHECK(std::isnan(special_tan[i]));
  }
}

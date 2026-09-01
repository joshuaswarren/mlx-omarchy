// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <functional>
#include <vector>
#include <numeric>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/linalg.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

void check_values(
    array value,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

void check_int32_values(
    array value,
    const std::vector<int32_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const int32_t* values = value.data<int32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_uint32_values(
    array value,
    const std::vector<uint32_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const uint32_t* values = value.data<uint32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}


// Checks one batch matrix of a rank-5 output against a host vector.
void check_values(
    array value,
    size_t b1,
    size_t b2,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE(value.size() >= (b1 * 2 + b2 + 1) * expected.size());
  const float* values = value.data<float>();
  size_t base = (b1 * 2 + b2) * expected.size();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(
        values[base + index] ==
        doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

// Host threefry2x32 copied from the upstream CPU reference
// mlx/backend/cpu/threefry.cpp: the same rotation constants and 5-round
// key schedule, so GPU words must match bit for bit.
std::pair<uint32_t, uint32_t> host_threefry(
    std::pair<uint32_t, uint32_t> key,
    std::pair<uint32_t, uint32_t> count) {
  constexpr static uint32_t rotations[2][4] = {
      {13, 15, 26, 6}, {17, 29, 16, 24}};

  uint32_t ks[3] = {key.first, key.second, key.first ^ key.second ^ 0x1BD11BDA};

  count.first += ks[0];
  count.second += ks[1];

  for (int i = 0; i < 5; ++i) {
    for (auto r : rotations[i % 2]) {
      count.first += count.second;
      count.second = (count.second << r) | (count.second >> (32 - r));
      count.second ^= count.first;
    }
    count.first += ks[(i + 1) % 3];
    count.second += ks[(i + 2) % 3] + i + 1;
  }

  return count;
}

// Word j of one key's region under the width-4 RandomBits layout, per
// upstream RandomBits::eval_cpu: counters walk (first, second) pairs and
// an odd word count leaves a middle word fed by counter (half, 0).
uint32_t host_random_word(
    std::pair<uint32_t, uint32_t> key,
    uint32_t words,
    uint32_t word) {
  uint32_t half = words / 2;
  bool even = words % 2 == 0;
  std::pair<uint32_t, uint32_t> counter;
  if (word < half) {
    counter = {word, word + half + (even ? 0u : 1u)};
  } else if (even) {
    counter = {word - half, word};
  } else if (word == half) {
    counter = {half, 0u};
  } else {
    counter = {word - half - 1u, word};
  }
  auto bits = host_threefry(key, counter);
  return (word < half || (!even && word == half)) ? bits.first : bits.second;
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

} // namespace

TEST_CASE("FP32 elementwise primitives dispatch through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto& counters = omarchy::trace::counters();
  uint64_t compute_before = counters.vk_compute_dispatches.load();
  uint64_t primitives_before = counters.gpu_primitive_dispatches.load();
  array x({1.0f, 2.0f, 4.0f, 8.0f}, float32);
  array y({2.0f, 4.0f, 2.0f, 4.0f}, float32);

  check_values(add(x, y, stream), {3.0f, 6.0f, 6.0f, 12.0f}, stream);
  check_values(multiply(x, y, stream), {2.0f, 8.0f, 8.0f, 32.0f}, stream);
  check_values(divide(x, y, stream), {0.5f, 0.5f, 2.0f, 2.0f}, stream);
  check_values(maximum(x, y, stream), {2.0f, 4.0f, 4.0f, 8.0f}, stream);
  check_values(add(x, array(1.0f), stream), {2.0f, 3.0f, 5.0f, 9.0f}, stream);
  check_values(subtract(x, y, stream), {-1.0f, -2.0f, 2.0f, 4.0f}, stream);
  check_values(
      subtract(x, array(0.5f), stream), {0.5f, 1.5f, 3.5f, 7.5f}, stream);
  check_values(
      subtract(array(1.0f), y, stream), {-1.0f, -3.0f, -1.0f, -3.0f}, stream);

  CHECK(counters.vk_compute_dispatches.load() >= compute_before + 5);
  CHECK(counters.gpu_primitive_dispatches.load() >= primitives_before + 5);
}

TEST_CASE("FP32 unary primitives dispatch through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x({0.0f, 1.0f, 2.0f}, float32);
  array positive({1.0f, 4.0f, 16.0f}, float32);

  check_values(
      exp(x, stream), {1.0f, std::exp(1.0f), std::exp(2.0f)}, stream, 1e-4);
  check_values(
      sigmoid(x, stream),
      {0.5f, 1.0f / (1.0f + std::exp(-1.0f)),
       1.0f / (1.0f + std::exp(-2.0f))},
      stream,
      1e-4);
  check_values(square(positive, stream), {1.0f, 16.0f, 256.0f}, stream);
  check_values(sqrt(positive, stream), {1.0f, 2.0f, 4.0f}, stream);
  check_values(rsqrt(positive, stream), {1.0f, 0.5f, 0.25f}, stream);
  check_values(negative(x, stream), {0.0f, -1.0f, -2.0f}, stream);
}

TEST_CASE("FP16 casts and elementwise primitives use Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  array source({0.5f, -2.0f, 7.25f}, float32);
  array half = astype(source, float16, stream);
  array round_trip = astype(half, float32, stream);

  CHECK_EQ(half.dtype(), float16);
  check_values(round_trip, {0.5f, -2.0f, 7.25f}, stream, 1e-3);
  check_values(
      astype(add(half, half, stream), float32, stream),
      {1.0f, -4.0f, 14.5f},
      stream,
      1e-3);
  check_values(
      astype(sum(half, std::vector<int>{0}, false, stream), float32, stream),
      {5.75f},
      stream,
      1e-3);
}

TEST_CASE("BF16 primitives use emulated conversion through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access ||
      !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 storage and shader features.");
    return;
  }

  array source({0.5f, -2.0f, 7.25f}, float32);
  array wide = astype(source, bfloat16, stream);
  CHECK_EQ(wide.dtype(), bfloat16);
  check_values(astype(wide, float32, stream), {0.5f, -2.0f, 7.25f}, stream, 8e-3);

  array x({1.5f, 2.5f, -3.25f}, float32);
  array y({0.5f, 1.25f, 0.75f}, float32);
  check_values(
      astype(add(astype(x, bfloat16, stream), astype(y, bfloat16, stream),
                 stream),
             float32, stream),
      {2.0f, 3.75f, -2.5f},
      stream,
      8e-3);

  array grid(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2, 2, 2},
      float32);
  check_values(
      astype(
          sum(astype(grid, bfloat16, stream), std::vector<int>{1, 2}, false,
              stream),
          float32,
          stream),
      {10.0f, 26.0f},
      stream,
      8e-3);

  array a(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f},
      {3, 4},
      float32);
  array b({0.5f, 1.0f, 2.0f, 1.0f, 0.5f, 1.0f, 2.0f, 1.0f}, {4, 2}, float32);
  check_values(
      astype(
          matmul(astype(a, bfloat16, stream), astype(b, bfloat16, stream),
                 stream),
          float32,
          stream),
      {14.0f, 10.0f, 34.0f, 26.0f, 54.0f, 42.0f},
      stream,
      8e-3);

  if (!capabilities.shader_float16) {
    skip("Vulkan device lacks shaderFloat16; skipping the BF16/FP16 cast.");
    return;
  }
  array half = astype(source, float16, stream);
  array half_round_trip = astype(
      astype(astype(half, bfloat16, stream), float16, stream), float32,
      stream);
  check_values(half_round_trip, {0.5f, -2.0f, 7.25f}, stream, 8e-3);
}

TEST_CASE("suffix Sum and Max reductions use Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2, 2, 2},
      float32);

  check_values(sum(x, std::vector<int>{1, 2}, false, stream), {10.0f, 26.0f}, stream);
  check_values(
      max(x, std::vector<int>{2}, false, stream),
      {2.0f, 4.0f, 6.0f, 8.0f},
      stream);
}

TEST_CASE("compute indexing stays inside Vulkan and uint32 limits") {
  constexpr uint32_t max_u32 = std::numeric_limits<uint32_t>::max();

  CHECK_EQ(omarchy::compute_dispatch_group_count(0), 0);
  CHECK_EQ(omarchy::compute_dispatch_group_count(256), 1);
  CHECK_EQ(omarchy::compute_dispatch_group_count(257), 2);
  CHECK_EQ(omarchy::compute_dispatch_group_count(max_u32), 65535);
  CHECK(omarchy::compute_index_span_fits(0, max_u32));
  CHECK(omarchy::compute_index_span_fits(max_u32, 1));
  CHECK_FALSE(omarchy::compute_index_span_fits(max_u32, 2));
  CHECK(omarchy::compute_index_span_fits(1, max_u32));
  CHECK_FALSE(omarchy::compute_index_span_fits(2, max_u32));
}

TEST_CASE("compute shaders handle offsets, multiple workgroups, and NaN") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> values(300);
  std::vector<float> expected(300);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(index);
    expected[index] = values[index] + 1.0f;
  }
  array many(values.begin(), Shape{300}, float32);
  check_values(add(many, array(1.0f), stream), expected, stream);

  array source = add(many, array(0.0f), stream);
  source.eval();
  auto& encoder = omarchy::get_command_encoder(stream);
  encoder.synchronize();
  array output = multiply(source, array(0.0f), stream);
  output.eval();
  encoder.synchronize();
  auto binding = [](const array& value) {
    auto* buffer =
        static_cast<const omarchy::VulkanBuffer*>(value.buffer().ptr());
    return omarchy::ComputeBinding{buffer->buffer, 0, buffer->size};
  };
  omarchy::ComputeParams params;
  params.count = 300;
  params.operation = 0;
  params.lhs_size = 300;
  params.rhs_size = 300;
  params.output_size = 300;
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(source), binding(source), binding(output)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::ElementwiseF32, bindings, params, 1);
  encoder.synchronize();
  CHECK_EQ(output.data<float>()[0], 0.0f);
  CHECK_EQ(output.data<float>()[256], 512.0f);
  CHECK_EQ(output.data<float>()[299], 598.0f);

  array base({-9.0f, 0.0f, 1.0f, 2.0f, 3.0f}, float32);
  array offset = slice(base, {1}, {5}, {1}, stream);
  check_values(square(offset, stream), {0.0f, 1.0f, 4.0f, 9.0f}, stream);

  float nan = std::numeric_limits<float>::quiet_NaN();
  array nan_result = max(
      array({1.0f, nan, 3.0f}, float32),
      std::vector<int>{0},
      false,
      stream);
  nan_result.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK(std::isnan(nan_result.data<float>()[0]));
}

TEST_CASE("FP32 and FP16 Matmul support dense and transposed weights") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}, float32);
  array weights(
      {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f}, {2, 3}, float32);

  check_values(matmul(a, b, stream), {58.0f, 64.0f, 139.0f, 154.0f}, stream);
  check_values(
      matmul(a, transpose(weights, {1, 0}, stream), stream),
      {58.0f, 64.0f, 139.0f, 154.0f},
      stream);

  array stored({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}, float32);
  check_values(
      matmul(
          transpose(stored, {1, 0}, stream),
          array({7.0f, 10.0f, 8.0f, 11.0f, 9.0f, 12.0f}, {3, 2}, float32),
          stream),
      {76.0f, 103.0f, 100.0f, 136.0f},
      stream);

  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    check_values(
        astype(
            matmul(
                astype(a, float16, stream),
                astype(b, float16, stream),
                stream),
            float32,
            stream),
        {58.0f, 64.0f, 139.0f, 154.0f},
        stream,
        1e-3);
    check_values(
        astype(
            addmm(
                astype(array({1.0f, 2.0f}, float32), float16, stream),
                astype(a, float16, stream),
                astype(b, float16, stream),
                2.0f,
                0.5f,
                stream),
            float32,
            stream),
        {116.5f, 129.0f, 278.5f, 309.0f},
        stream,
        1e-3);
  }
}

TEST_CASE("Matmul flattens leading batches and AddMM broadcasts bias") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a(
      {1.0f,
       2.0f,
       3.0f,
       4.0f,
       5.0f,
       6.0f,
       7.0f,
       8.0f,
       9.0f,
       10.0f,
       11.0f,
       12.0f},
      {2, 2, 3},
      float32);
  array b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}, float32);
  array product = matmul(a, b, stream);
  check_values(
      product,
      {58.0f, 64.0f, 139.0f, 154.0f, 220.0f, 244.0f, 301.0f, 334.0f},
      stream);
  check_values(
      add(product, array({1.0f, 2.0f}, float32), stream),
      {59.0f, 66.0f, 140.0f, 156.0f, 221.0f, 246.0f, 302.0f, 336.0f},
      stream);
  check_values(
      addmm(array({1.0f, 2.0f}, float32), a, b, 2.0f, 0.5f, stream),
      {116.5f, 129.0f, 278.5f, 309.0f, 440.5f, 489.0f, 602.5f, 669.0f},
      stream);
}

TEST_CASE("Matmul handles tiled edges, offsets, and empty K") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t m = 17;
  constexpr size_t k = 18;
  constexpr size_t n = 19;
  std::vector<float> a_values(m * k);
  std::vector<float> b_values(k * n);
  std::vector<float> expected(m * n, 0.0f);
  for (size_t index = 0; index < a_values.size(); ++index) {
    a_values[index] = static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  for (size_t index = 0; index < b_values.size(); ++index) {
    b_values[index] = static_cast<float>(static_cast<int>(index % 5) - 2);
  }
  for (size_t row = 0; row < m; ++row) {
    for (size_t column = 0; column < n; ++column) {
      for (size_t inner = 0; inner < k; ++inner) {
        expected[row * n + column] +=
            a_values[row * k + inner] * b_values[inner * n + column];
      }
    }
  }
  check_values(
      matmul(
          array(a_values.begin(), Shape{m, k}, float32),
          array(b_values.begin(), Shape{k, n}, float32),
          stream),
      expected,
      stream);

  std::vector<float> transposed_expected(m * n, 0.0f);
  for (size_t row = 0; row < m; ++row) {
    for (size_t column = 0; column < n; ++column) {
      for (size_t inner = 0; inner < k; ++inner) {
        transposed_expected[row * n + column] +=
            a_values[inner * m + row] * b_values[inner * n + column];
      }
    }
  }
  check_values(
      matmul(
          transpose(array(a_values.begin(), Shape{k, m}, float32), {1, 0},
                    stream),
          array(b_values.begin(), Shape{k, n}, float32),
          stream),
      transposed_expected,
      stream);

  array a_base(
      {99.0f, 99.0f, 99.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {3, 3},
      float32);
  array b_base(
      {99.0f, 99.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
      {4, 2},
      float32);
  array a_offset = slice(a_base, {1, 0}, {3, 3}, {1, 1}, stream);
  array b_offset = slice(b_base, {1, 0}, {4, 2}, {1, 1}, stream);
  check_values(
      matmul(a_offset, b_offset, stream),
      {58.0f, 64.0f, 139.0f, 154.0f},
      stream);

  std::vector<float> empty;
  array empty_a(empty.begin(), Shape{2, 0}, float32);
  array empty_b(empty.begin(), Shape{0, 3}, float32);
  check_values(
      matmul(empty_a, empty_b, stream),
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      stream);
  check_values(
      addmm(
          array({1.0f, 2.0f, 3.0f}, float32),
          empty_a,
          empty_b,
          1.0f,
          2.0f,
          stream),
      {2.0f, 4.0f, 6.0f, 2.0f, 4.0f, 6.0f},
      stream);
}

TEST_CASE("non-zero scalar fills dispatch through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  check_values(
      full({2, 3}, 1.5f, float32, stream),
      std::vector<float>(6, 1.5f),
      stream);
  check_values(full({2}, 0.0f, float32, stream), {0.0f, 0.0f}, stream);

  std::string int_error = evaluation_error(full({2}, 5, int32, stream));
  CHECK(int_error.find("non-zero scalar fill") != std::string::npos);

  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    array half_out = zeros({2, 2}, float16, stream);
    half_out.eval();
    array half_scalar = array(1.5f, float16);
    copy_gpu_inplace(
        half_scalar,
        half_out,
        half_out.shape(),
        half_scalar.strides(),
        half_out.strides(),
        0,
        0,
        CopyType::Scalar,
        stream);
    check_values(
        astype(half_out, float32, stream),
        std::vector<float>(4, 1.5f),
        stream,
        1e-3);
  }
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array bf_out = zeros({3}, bfloat16, stream);
    bf_out.eval();
    array bf_scalar = array(2.5f, bfloat16);
    copy_gpu_inplace(
        bf_scalar,
        bf_out,
        bf_out.shape(),
        bf_scalar.strides(),
        bf_out.strides(),
        0,
        0,
        CopyType::Scalar,
        stream);
    check_values(
        astype(bf_out, float32, stream), {2.5f, 2.5f, 2.5f}, stream, 8e-3);
  }
}

TEST_CASE("general strided copies materialize through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {2, 3}, float32);
  check_values(
      contiguous(transpose(base, stream), false, stream),
      {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f},
      stream);

  array wide(
      {0.0f,
       1.0f,
       2.0f,
       3.0f,
       4.0f,
       5.0f,
       6.0f,
       7.0f,
       8.0f,
       9.0f,
       10.0f,
       11.0f},
      {3, 4},
      float32);
  check_values(
      contiguous(slice(wide, {0, 1}, {3, 4}, {2, 1}, stream), false, stream),
      {1.0f, 2.0f, 3.0f, 9.0f, 10.0f, 11.0f},
      stream);

  array grid(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2, 2, 2},
      float32);
  check_values(
      contiguous(transpose(grid, {2, 0, 1}, stream), false, stream),
      {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f},
      stream);

  array ints({0, 1, 2, 3}, {2, 2}, int32);
  std::string int_error =
      evaluation_error(contiguous(transpose(ints, stream), false, stream));
  CHECK(int_error.find("strided copy") != std::string::npos);
}

TEST_CASE("value_and_grad computes matmul and subtract gradients on device") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x({0.0f, 0.1f, 0.2f, 0.05f, 0.15f, 0.25f}, {2, 3}, float32);
  array w({0.1f, -0.2f, 0.3f, 0.05f, -0.1f, 0.2f}, {3, 2}, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    return sum(exp(matmul(inputs[0], inputs[1], stream), stream), stream);
  };
  auto [value, grads] = value_and_grad(fun, std::vector<int>{0, 1})({x, w});

  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.05f, 0.15f, 0.25f};
  std::vector<float> wv = {0.1f, -0.2f, 0.3f, 0.05f, -0.1f, 0.2f};
  float expected_value = 0.0f;
  float e[2][2];
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      float y = 0.0f;
      for (int inner = 0; inner < 3; ++inner) {
        y += xv[row * 3 + inner] * wv[inner * 2 + column];
      }
      e[row][column] = std::exp(y);
      expected_value += e[row][column];
    }
  }
  std::vector<float> expected_dx(6);
  for (int row = 0; row < 2; ++row) {
    for (int inner = 0; inner < 3; ++inner) {
      float grad = 0.0f;
      for (int column = 0; column < 2; ++column) {
        grad += e[row][column] * wv[inner * 2 + column];
      }
      expected_dx[row * 3 + inner] = grad;
    }
  }
  std::vector<float> expected_dw(6);
  for (int inner = 0; inner < 3; ++inner) {
    for (int column = 0; column < 2; ++column) {
      float grad = 0.0f;
      for (int row = 0; row < 2; ++row) {
        grad += xv[row * 3 + inner] * e[row][column];
      }
      expected_dw[inner * 2 + column] = grad;
    }
  }
  check_values(value, {expected_value}, stream, 1e-4);
  check_values(grads.at(0), expected_dx, stream, 1e-4);
  check_values(grads.at(1), expected_dw, stream, 1e-4);

  auto sub_fun = [&](const std::vector<array>& inputs) {
    return sum(subtract(inputs[0], inputs[1], stream), stream);
  };
  auto [sub_value, sub_grads] = value_and_grad(sub_fun, std::vector<int>{0, 1})(
      {x, reshape(w, {2, 3}, stream)});
  check_values(sub_value, {0.4f}, stream, 1e-5);
  // The positive cotangent is a broadcast view of the scalar seed, so
  // materialize it on device before reading linearly.
  check_values(
      multiply(sub_grads.at(0), array(1.0f), stream),
      std::vector<float>(6, 1.0f),
      stream);
  check_values(sub_grads.at(1), std::vector<float>(6, -1.0f), stream);
}

TEST_CASE("jvp computes forward-mode tangents through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x({0.0f, 0.1f, 0.2f}, float32);
  array tangent({1.0f, -2.0f, 3.0f}, float32);

  auto exp_fun = [&](const array& input) {
    return sum(exp(input, stream), stream);
  };
  auto [exp_value, exp_tangent] = jvp(exp_fun, x, tangent);
  std::vector<float> ex = {
      std::exp(0.0f), std::exp(0.1f), std::exp(0.2f)};
  float expected_value = ex[0] + ex[1] + ex[2];
  float expected_jvp = ex[0] - 2.0f * ex[1] + 3.0f * ex[2];
  check_values(exp_value, {expected_value}, stream, 1e-4);
  check_values(exp_tangent, {expected_jvp}, stream, 1e-4);

  std::vector<float> av = {0.0f, 0.1f, 0.2f, 0.05f, 0.15f, 0.25f};
  std::vector<float> wv = {0.1f, -0.2f, 0.3f, 0.05f, -0.1f, 0.2f};
  std::vector<float> tv = {1.0f, 0.5f, -1.0f, 2.0f, 0.25f, -0.5f};
  array a(av.begin(), Shape{2, 3}, float32);
  array w(wv.begin(), Shape{3, 2}, float32);
  array a_tangent(tv.begin(), Shape{2, 3}, float32);
  auto matmul_fun = [&](const array& input) {
    return matmul(input, w, stream);
  };
  auto [mm_value, mm_tangent] = jvp(matmul_fun, a, a_tangent);
  std::vector<float> expected_product(4);
  std::vector<float> expected_mm_jvp(4);
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      for (int inner = 0; inner < 3; ++inner) {
        expected_product[row * 2 + column] +=
            av[row * 3 + inner] * wv[inner * 2 + column];
        expected_mm_jvp[row * 2 + column] +=
            tv[row * 3 + inner] * wv[inner * 2 + column];
      }
    }
  }
  check_values(mm_value, expected_product, stream, 1e-4);
  check_values(mm_tangent, expected_mm_jvp, stream, 1e-4);
}

TEST_CASE("vmap batches closures over the leading axis through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  array x(xv.begin(), Shape{2, 3}, float32);

  auto batched_exp = vmap(
      [&](const array& input) { return exp(input, stream); });
  check_values(
      batched_exp(x),
      {std::exp(xv[0]),
       std::exp(xv[1]),
       std::exp(xv[2]),
       std::exp(xv[3]),
       std::exp(xv[4]),
       std::exp(xv[5])},
      stream,
      1e-4);

  std::vector<float> yv = {1.0f, -0.5f, 2.0f, 0.25f, -1.0f, 0.75f};
  array y(yv.begin(), Shape{2, 3}, float32);
  auto batched_add = vmap(
      [&](const array& lhs, const array& rhs) {
        return add(lhs, rhs, stream);
      });
  std::vector<float> expected_add(6);
  for (size_t index = 0; index < expected_add.size(); ++index) {
    expected_add[index] = xv[index] + yv[index];
  }
  check_values(batched_add(x, y), expected_add, stream);

  std::vector<float> av = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> bv = {0.5f, 1.0f, 2.0f, 1.0f, 0.5f, 1.0f, 2.0f, 1.0f};
  array batched_a(av.begin(), Shape{2, 2, 2}, float32);
  array batched_b(bv.begin(), Shape{2, 2, 2}, float32);
  auto batched_matmul = vmap(
      [&](const array& lhs, const array& rhs) {
        return matmul(lhs, rhs, stream);
      });
  // Equal leading batch dims dispatch as one batched product now.
  check_values(
      batched_matmul(batched_a, batched_b),
      {4.5f, 3.0f, 9.5f, 7.0f, 14.5f, 11.0f, 19.5f, 15.0f},
      stream);
}

TEST_CASE("mx.compile fuses only with backend support and no_fuse evaluates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.3f};
  std::vector<float> yv = {1.0f, 0.5f, 2.0f, 0.25f};
  array x(xv.begin(), Shape{4}, float32);
  array y(yv.begin(), Shape{4}, float32);
  std::vector<float> expected(4);
  for (size_t index = 0; index < expected.size(); ++index) {
    expected[index] = std::exp(xv[index]) * yv[index];
  }
  using VectorFn = std::function<std::vector<array>(const std::vector<array>&)>;

  set_compile_mode(CompileMode::enabled);
  VectorFn fused_fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        multiply(exp(inputs[0], stream), inputs[1], stream)};
  };
  auto fused = compile(fused_fun);
  auto fused_outputs = fused({x, y});
  // compile_available_for_device reports true for the GPU device, so the
  // fused tape reaches Compiled::eval_gpu and must fail with its name.
  std::string fused_error = evaluation_error(fused_outputs[0]);
  CHECK(fused_error.find("[omarchy] Compiled is not implemented") !=
        std::string::npos);

  set_compile_mode(CompileMode::no_fuse);
  VectorFn unfused_fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        multiply(exp(inputs[0], stream), inputs[1], stream)};
  };
  auto unfused = compile(unfused_fun);
  auto unfused_outputs = unfused({x, y});
  check_values(unfused_outputs[0], expected, stream, 1e-4);
  set_compile_mode(CompileMode::enabled);
}

TEST_CASE("unsupported compute shapes and dtypes fail without CPU fallback") {
  if (!compute_available()) {
    return;
  }
  CHECK_FALSE(is_available(Device::cpu));
  CHECK_EQ(device_count(Device::cpu), 0);
  Stream stream = gpu_stream();

  std::string dtype_error = evaluation_error(add(
      array({1, 2}, int32), array({3, 4}, int32), stream));
  CHECK(dtype_error.find("[omarchy] Add dtype") != std::string::npos);
  CHECK(dtype_error.find("No CPU fallback") != std::string::npos);

  // Slice views with gaps materialize at eval, so elementwise work over
  // them runs. A transpose view keeps its strides and pins the named
  // layout error.
  array base({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  check_values(
      exp(slice(base, {0, 1}, {2, 3}, {1, 2}, stream), stream),
      {std::exp(2.0f), std::exp(5.0f)},
      stream,
      1e-4);
  // A transposed view is gapless and keeps its strides, so the
  // elementwise stride path reads it directly with no copy.
  check_values(
      exp(transpose(base, stream), stream),
      {std::exp(1.0f),
       std::exp(4.0f),
       std::exp(2.0f),
       std::exp(5.0f),
       std::exp(3.0f),
       std::exp(6.0f)},
      stream,
      1e-4);

  // Inner-axis broadcast is supported now; a broadcast pattern whose
  // stride runs still collapse to rank above 4 pins the named rank
  // error.
  array lhs({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
             9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
             17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
             25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f},
            {2, 1, 2, 1, 2, 1, 2, 1, 2}, float32);
  std::vector<float> wide_values(512, 1.0f);
  array rhs(wide_values.begin(), Shape{2, 2, 2, 2, 2, 2, 2, 2, 2}, float32);
  std::string broadcast_error = evaluation_error(add(lhs, rhs, stream));
  CHECK(broadcast_error.find("broadcast rank Add") != std::string::npos);

  array matrix({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  std::string reduction_error =
      evaluation_error(sum(matrix, 0, false, stream));
  CHECK(reduction_error.find("non-suffix Sum") != std::string::npos);

  auto construction_error = [](auto&& build) -> std::string {
    try {
      build();
    } catch (const std::exception& error) {
      return error.what();
    }
    return {};
  };

  std::string float64_error = construction_error([&] {
    add(array({1.0, 2.0}, float64), array({3.0, 4.0}, float64), stream);
  });
  CHECK(
      float64_error.find("float64 is not supported on the GPU") !=
      std::string::npos);

  std::string complex_error = evaluation_error(add(
      array(complex64_t{1.0f, 2.0f}), array(complex64_t{3.0f, 4.0f}), stream));
  CHECK(complex_error.find("[omarchy] Add dtype") != std::string::npos);
  CHECK(complex_error.find("complex64") != std::string::npos);

  array spd({4.0f, 0.0f, 0.0f, 9.0f}, {2, 2}, float32);
  std::string cholesky_error =
      construction_error([&] { linalg::cholesky(spd, false, stream); });
  CHECK(cholesky_error.find("[linalg::cholesky]") != std::string::npos);
  CHECK(
      cholesky_error.find("not yet supported on the GPU") != std::string::npos);

  std::string svd_error =
      construction_error([&] { linalg::svd(spd, true, stream); });
  CHECK(svd_error.find("[linalg::svd]") != std::string::npos);
  CHECK(svd_error.find("not yet supported on the GPU") != std::string::npos);

  std::string inv_error =
      construction_error([&] { linalg::inv(spd, stream); });
  CHECK(inv_error.find("[linalg::inv]") != std::string::npos);
  CHECK(inv_error.find("not yet supported on the GPU") != std::string::npos);
}

// Host reference: numerically stable softmax over the last axis.
std::vector<float> host_softmax(
    const std::vector<float>& values, int rows, int row_length) {
  std::vector<float> expected(values.size());
  for (int row = 0; row < rows; ++row) {
    const float* input = values.data() + row * row_length;
    float* output = expected.data() + row * row_length;
    float maximum = input[0];
    for (int index = 1; index < row_length; ++index) {
      maximum = std::max(maximum, input[index]);
    }
    float normalizer = 0.0f;
    for (int index = 0; index < row_length; ++index) {
      output[index] = std::exp(input[index] - maximum);
      normalizer += output[index];
    }
    for (int index = 0; index < row_length; ++index) {
      output[index] /= normalizer;
    }
  }
  return expected;
}

// Host reference: numerically stable log-sum-exp over the last axis, one
// value per row. An infinite row max is already the answer.
std::vector<float> host_logsumexp(
    const std::vector<float>& values, int rows, int row_length) {
  std::vector<float> expected(rows);
  for (int row = 0; row < rows; ++row) {
    const float* input = values.data() + row * row_length;
    float maximum = input[0];
    for (int index = 1; index < row_length; ++index) {
      maximum = std::max(maximum, input[index]);
    }
    if (std::isinf(maximum)) {
      expected[row] = maximum;
      continue;
    }
    float sum = 0.0f;
    for (int index = 0; index < row_length; ++index) {
      sum += std::exp(input[index] - maximum);
    }
    expected[row] = maximum + std::log(sum);
  }
  return expected;
}

TEST_CASE("softmax normalizes rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  array x(xv.begin(), Shape{2, 3}, float32);
  check_values(
      softmax(x, std::vector<int>{-1}, false, stream),
      host_softmax(xv, 2, 3),
      stream);

  // Rows sum to 1.
  check_values(
      sum(softmax(x, std::vector<int>{-1}, false, stream),
          std::vector<int>{-1},
          false,
          stream),
      {1.0f, 1.0f},
      stream);

  // Large logits stay finite because the kernel subtracts the row max.
  std::vector<float> bigv = {80.0f, 81.0f, 80.0f, 79.0f};
  check_values(
      softmax(array(bigv.begin(), Shape{2, 2}, float32),
              std::vector<int>{-1},
              false,
              stream),
      host_softmax(bigv, 2, 2),
      stream);

  // One flat row longer than one workgroup.
  std::vector<float> wide_values(300);
  for (size_t index = 0; index < wide_values.size(); ++index) {
    wide_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  check_values(
      softmax(array(wide_values.begin(), Shape{1, 300}, float32),
              std::vector<int>{-1},
              false,
              stream),
      host_softmax(wide_values, 1, 300),
      stream,
      2e-5);

  // More rows than one dispatch can name, so workgroups grid-stride.
  std::vector<float> tall_values(70000, 3.0f);
  check_values(
      softmax(array(tall_values.begin(), Shape{70000, 1}, float32),
              std::vector<int>{-1},
              false,
              stream),
      std::vector<float>(70000, 1.0f),
      stream);

  // Slice views with gaps materialize at eval, so softmax over them runs
  // and normalizes the sliced rows. A transpose view keeps its strides
  // and pins the named layout error.
  array base(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
      {3, 3},
      float32);
  check_values(
      softmax(slice(base, {0, 1}, {3, 3}, {1, 1}, stream),
              std::vector<int>{-1},
              false,
              stream),
      {0.26894142f,
       0.73105858f,
       0.26894142f,
       0.73105858f,
       0.26894142f,
       0.73105858f},
      stream,
      1e-5);
  std::string layout_error = evaluation_error(
      softmax(transpose(base, stream), std::vector<int>{-1}, false, stream));
  CHECK(layout_error.find("non-contiguous Softmax") != std::string::npos);

  // Non-suffix axes never build a Softmax primitive; upstream decomposes
  // them into reductions, and the reduction pins the named error.
  array grid(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2, 2, 2},
      float32);
  std::string suffix_error =
      evaluation_error(softmax(grid, std::vector<int>{0, 1}, false, stream));
  CHECK(suffix_error.find("non-suffix Max") != std::string::npos);
}

TEST_CASE("FP16 and BF16 softmax match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  std::vector<float> xv = {0.0f, 1.0f, 2.0f, 1.0f, 3.0f, -1.0f};
  auto expected = host_softmax(xv, 2, 3);
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    array half = astype(
        array(xv.begin(), Shape{2, 3}, float32), float16, stream);
    check_values(
        astype(
            softmax(half, std::vector<int>{-1}, false, stream),
            float32,
            stream),
        expected,
        stream,
        1e-3);
  } else {
    skip("Vulkan device lacks required FP16 shader and storage features.");
  }
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain = astype(
        array(xv.begin(), Shape{2, 3}, float32), bfloat16, stream);
    check_values(
        astype(
            softmax(brain, std::vector<int>{-1}, false, stream),
            float32,
            stream),
        expected,
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}

TEST_CASE("logsumexp reduces last-axis rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Multi-row case at the default float32 tolerance.
  std::vector<float> xv = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                           -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
                           10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
  array x(xv.begin(), Shape{3, 5}, float32);
  check_values(
      logsumexp(x, -1, false, stream),
      host_logsumexp(xv, 3, 5),
      stream);

  // The mlx-lm logprobs epilogue: [1, V] logits reduce keepdims to
  // [1, 1] so a broadcast subtract normalizes every row.
  std::vector<float> logits_values(16);
  for (size_t index = 0; index < logits_values.size(); ++index) {
    logits_values[index] = 0.5f * static_cast<float>(static_cast<int>(index) - 8);
  }
  array logits(logits_values.begin(), Shape{1, 16}, float32);
  array lse = logsumexp(logits, -1, true, stream);
  REQUIRE_EQ(lse.shape(), Shape{1, 1});
  check_values(
      lse,
      host_logsumexp(logits_values, 1, 16),
      stream);

  // Large logits stay finite because the kernel subtracts the row max.
  std::vector<float> bigv = {100.0f, 101.0f, 100.0f, 99.0f,
                             -100.0f, -101.0f, -100.0f, -99.0f};
  check_values(
      logsumexp(array(bigv.begin(), Shape{2, 4}, float32), -1, false, stream),
      host_logsumexp(bigv, 2, 4),
      stream);

  // One flat row longer than one workgroup.
  std::vector<float> wide_values(300);
  for (size_t index = 0; index < wide_values.size(); ++index) {
    wide_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  check_values(
      logsumexp(array(wide_values.begin(), Shape{1, 300}, float32),
                -1,
                false,
                stream),
      host_logsumexp(wide_values, 1, 300),
      stream,
      2e-5);

  // An all -inf row keeps -inf instead of a NaN sum; a row that holds
  // one finite value next to -infs reduces to that value.
  float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> infs = {neg_inf, neg_inf, neg_inf,
                             neg_inf, 3.0f, neg_inf};
  array inf_result = logsumexp(
      array(infs.begin(), Shape{2, 3}, float32), -1, false, stream);
  inf_result.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(inf_result.size(), 2u);
  const float* inf_values = inf_result.data<float>();
  CHECK(std::isinf(inf_values[0]));
  CHECK(inf_values[0] < 0.0f);
  CHECK_EQ(inf_values[1], 3.0f);

  // A transpose view keeps its strides and pins the named layout error.
  std::string layout_error = evaluation_error(
      logsumexp(transpose(x, stream), -1, false, stream));
  CHECK(layout_error.find("non-contiguous LogSumExp") != std::string::npos);

  // FP16 and BF16 match the float32 host reference at their usual
  // tolerances.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    array half = astype(x, float16, stream);
    check_values(
        astype(logsumexp(half, -1, false, stream), float32, stream),
        host_logsumexp(xv, 3, 5),
        stream,
        1e-3);
  } else {
    skip("Vulkan device lacks required FP16 shader and storage features.");
  }
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain = astype(logits, bfloat16, stream);
    check_values(
        astype(logsumexp(brain, -1, true, stream), float32, stream),
        host_logsumexp(logits_values, 1, 16),
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}

TEST_CASE("Log matches host references through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv;
  for (int index = 1; index <= 12; ++index) {
    xv.push_back(0.25f * static_cast<float>(index));
  }
  std::vector<float> expected;
  for (float value : xv) {
    expected.push_back(std::log(value));
  }
  array x(xv.begin(), Shape{3, 4}, float32);
  check_values(log(x, stream), expected, stream, 1e-6);

  // Sampling code negates log probabilities before the cumulative sum.
  check_values(
      negative(log(x, stream), stream),
      [&]() {
        std::vector<float> values;
        for (float value : expected) {
          values.push_back(-value);
        }
        return values;
      }(),
      stream,
      1e-6);

  // BF16 log feeds the mlx-lm sampling path for brain-float models.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    check_values(
        astype(log(astype(x, bfloat16, stream), stream), float32, stream),
        expected,
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}



TEST_CASE("take gathers table rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {
      10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f,
      30.0f, 31.0f, 32.0f, 33.0f};
  array table(tv.begin(), Shape{3, 4}, float32);

  // Exact lookups with a repeated index.
  array indices({2, 0, 2}, int32);
  check_values(
      take(table, indices, 0, stream),
      {30.0f,
       31.0f,
       32.0f,
       33.0f,
       10.0f,
       11.0f,
       12.0f,
       13.0f,
       30.0f,
       31.0f,
       32.0f,
       33.0f},
      stream);

  // Out-of-range and negative indices write zero rows. This deviates from
  // upstream negative-index wrapping and is the documented constraint.
  array bounds({1, 7, -1, 0}, int32);
  check_values(
      take(table, bounds, 0, stream),
      {20.0f,
       21.0f,
       22.0f,
       23.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       10.0f,
       11.0f,
       12.0f,
       13.0f},
      stream);

  // Every integral index dtype outside int32, uint32, and int64 pins the
  // named backend error. Boolean and float indices are rejected one
  // layer up by the shared gather op itself.
  for (Dtype dtype : {int16, uint16, int8, uint8}) {
    std::string error =
        evaluation_error(take(table, array({0}, dtype), 0, stream));
    CHECK(error.find("indexed Take dtype") != std::string::npos);
  }
  // Boolean and float indices are rejected one layer up by the shared
  // gather op itself, at graph build time.
  std::string float_error;
  try {
    take(table, array({0}, float32), 0, stream);
  } catch (const std::exception& error) {
    float_error = error.what();
  }
  CHECK(float_error.find("Indices must be integral") != std::string::npos);
  std::string bool_error;
  try {
    take(table, array({0}, bool_), 0, stream);
  } catch (const std::exception& error) {
    bool_error = error.what();
  }
  CHECK(bool_error.find("Boolean indices") != std::string::npos);
  // Non-zero gather axes pin the named axis error.
  std::string axis_error =
      evaluation_error(take(table, array({0}, int32), 1, stream));
  CHECK(axis_error.find("non-axis-0 Take") != std::string::npos);

  // Higher-rank tables pin the named layout error.
  array cube(tv.begin(), Shape{1, 3, 4}, float32);
  std::string rank_error =
      evaluation_error(take(cube, array({0}, int32), 0, stream));
  CHECK(rank_error.find("matrix layout Take") != std::string::npos);

  const auto& capabilities = omarchy::device(0).capabilities();
  std::vector<float> hv = {0.5f, 1.0f, 2.0f, -1.5f, 0.25f, 4.0f};
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    array half_table = astype(
        array(hv.begin(), Shape{2, 3}, float32), float16, stream);
    check_values(
        astype(
            take(half_table, array({1, 0, 1}, int32), 0, stream),
            float32,
            stream),
        {-1.5f, 0.25f, 4.0f, 0.5f, 1.0f, 2.0f, -1.5f, 0.25f, 4.0f},
        stream,
        1e-3);
  }
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain_table = astype(
        array(hv.begin(), Shape{2, 3}, float32), bfloat16, stream);
    check_values(
        astype(
            take(brain_table, array({0, 1}, int32), 0, stream),
            float32,
            stream),
        {0.5f, 1.0f, 2.0f, -1.5f, 0.25f, 4.0f},
        stream,
        8e-3);
  }
}

TEST_CASE("take gathers N-D index arrays as flat row sequences") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {
      10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f,
      30.0f, 31.0f, 32.0f, 33.0f, 40.0f, 41.0f, 42.0f, 43.0f,
      50.0f, 51.0f, 52.0f, 53.0f};
  array table(tv.begin(), Shape{5, 4}, float32);

  // A 2-D index array gathers rows in its flat row-major order, and the
  // output shape is indices.shape + [cols].
  std::vector<int> iv = {4, 0, 3, 1, 2, 0};
  array indices(iv.begin(), Shape{2, 3}, int32);
  array gathered = take(table, indices, 0, stream);
  CHECK_EQ(gathered.shape().size(), 3);
  CHECK_EQ(gathered.shape(0), 2);
  CHECK_EQ(gathered.shape(1), 3);
  CHECK_EQ(gathered.shape(2), 4);
  check_values(
      gathered,
      {50.0f, 51.0f, 52.0f, 53.0f,
       10.0f, 11.0f, 12.0f, 13.0f,
       40.0f, 41.0f, 42.0f, 43.0f,
       20.0f, 21.0f, 22.0f, 23.0f,
       30.0f, 31.0f, 32.0f, 33.0f,
       10.0f, 11.0f, 12.0f, 13.0f},
      stream);

  // The decode-time index shape [1, 1] keeps one row.
  std::vector<int> dv = {2};
  array decode(dv.begin(), Shape{1, 1}, int32);
  array decoded = take(table, decode, 0, stream);
  CHECK_EQ(decoded.shape().size(), 3);
  CHECK_EQ(decoded.shape(0), 1);
  CHECK_EQ(decoded.shape(1), 1);
  CHECK_EQ(decoded.shape(2), 4);
  check_values(decoded, {30.0f, 31.0f, 32.0f, 33.0f}, stream);

  // A bf16 table through the mlx-lm embedding shape family with 2-D
  // indices.
  const auto& capabilities = omarchy::device(0).capabilities();
  std::vector<float> hv = {0.5f, 1.0f, 2.0f, -1.5f, 0.25f, 4.0f};
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain_table = astype(
        array(hv.begin(), Shape{2, 3}, float32), bfloat16, stream);
    std::vector<int> biv = {1, 0, 0, 1};
    array brain_indices(biv.begin(), Shape{2, 2}, int32);
    check_values(
        astype(
            take(brain_table, brain_indices, 0, stream),
            float32,
            stream),
        {-1.5f,
         0.25f,
         4.0f,
         0.5f,
         1.0f,
         2.0f,
         0.5f,
         1.0f,
         2.0f,
         -1.5f,
         0.25f,
         4.0f},
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }

  // A transposed index view is gapless but not row-contiguous, so it
  // still pins the named error.
  std::vector<int> wv = {1, 2, 3, 4, 5, 6};
  array wide(wv.begin(), Shape{2, 3}, int32);
  std::string layout_error = evaluation_error(
      take(table, transpose(wide, {1, 0}, stream), 0, stream));
  CHECK(layout_error.find("non-contiguous indexed Take") != std::string::npos);
}


TEST_CASE("take gathers uint32 argmax indices through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {
      10.0f, 11.0f, 12.0f,
      20.0f, 21.0f, 22.0f,
      30.0f, 31.0f, 32.0f,
      40.0f, 41.0f, 42.0f};
  array table(tv.begin(), Shape{4, 3}, float32);

  // The mlx-lm greedy decode shape: argmax over [1, V] logits feeds a
  // [1, 1] index array into the embedding take.
  std::vector<float> dv = {0.1f, 3.0f, 0.2f, 0.3f};
  array decode_logits(dv.begin(), Shape{1, 4}, float32);
  array decode_ids = expand_dims(
      argmax(decode_logits, -1, false, stream), 1, stream);
  CHECK_EQ(decode_ids.dtype(), uint32);
  check_values(
      take(table, decode_ids, 0, stream),
      {20.0f, 21.0f, 22.0f},
      stream);

  // A [2, 3, V] batch reduces to [2, 3] uint32 indices, one gather per
  // batch element.
  std::vector<float> lv = {
      0.1f, 0.2f, 0.3f, 2.0f,
      1.5f, 0.2f, 0.1f, 0.4f,
      0.3f, 0.1f, 1.7f, 0.2f,
      0.1f, 2.2f, 0.3f, 0.1f,
      0.0f, 1.9f, 0.5f, 0.2f,
      0.4f, 0.2f, 0.1f, 1.3f};
  array logits(lv.begin(), Shape{2, 3, 4}, float32);
  array ids = argmax(logits, -1, false, stream);
  CHECK_EQ(ids.dtype(), uint32);
  check_uint32_values(ids, {3, 0, 2, 1, 1, 3}, stream);
  check_values(
      take(table, ids, 0, stream),
      {40.0f, 41.0f, 42.0f,
       10.0f, 11.0f, 12.0f,
       30.0f, 31.0f, 32.0f,
       20.0f, 21.0f, 22.0f,
       20.0f, 21.0f, 22.0f,
       40.0f, 41.0f, 42.0f},
      stream);

  // Plain uint32 indices gather directly, and one above the row count
  // writes the zero row.
  std::vector<uint32_t> uv = {0, 2, 2, 9};
  array raw(uv.begin(), Shape{4}, uint32);
  check_values(
      take(table, raw, 0, stream),
      {10.0f, 11.0f, 12.0f,
       30.0f, 31.0f, 32.0f,
       30.0f, 31.0f, 32.0f,
       0.0f, 0.0f, 0.0f},
      stream);
}

TEST_CASE("take gathers int64 indices and zeroes wide values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
  array table(tv.begin(), Shape{3, 2}, float32);

  // Two little-endian words per index: a value above 2^32 and a negative
  // value both have a nonzero high word, so both write the zero row.
  std::vector<int64_t> iv = {2, 0, 5000000000LL, -1};
  array indices(iv.begin(), Shape{4}, int64);
  check_values(
      take(table, indices, 0, stream),
      {30.0f, 31.0f,
       10.0f, 11.0f,
       0.0f, 0.0f,
       0.0f, 0.0f},
      stream);

  // A [2, 2] batch of int64 indices keeps its flat row-major order.
  std::vector<int64_t> bv = {1, 2, 0, 1};
  array batch(bv.begin(), Shape{2, 2}, int64);
  array gathered = take(table, batch, 0, stream);
  CHECK_EQ(gathered.shape().size(), 3);
  CHECK_EQ(gathered.shape(0), 2);
  CHECK_EQ(gathered.shape(1), 2);
  check_values(
      gathered,
      {20.0f, 21.0f,
       30.0f, 31.0f,
       10.0f, 11.0f,
       20.0f, 21.0f},
      stream);
}

TEST_CASE("general broadcast elementwise matches host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // (rows,1) * (rows,cols): the broadcast axis is the last lhs axis, so
  // this pins the shape-aware path.
  std::vector<float> rv = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> mv = {2.0f, 1.0f, 0.5f, 3.0f, -1.0f, 2.0f,
                           0.5f, 4.0f, -2.0f, 1.5f, 0.25f, -0.5f};
  array rows(rv.begin(), Shape{4, 1}, float32);
  array matrix(mv.begin(), Shape{4, 3}, float32);
  std::vector<float> expected(12);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      expected[r * 3 + c] = rv[r] * mv[r * 3 + c];
    }
  }
  check_values(multiply(rows, matrix, stream), expected, stream);

  // (1,cols) * (rows,cols): leading-axis broadcast keeps the trailing
  // modulo fast path.
  std::vector<float> cv = {0.5f, -1.0f, 2.0f};
  array cols(cv.begin(), Shape{1, 3}, float32);
  std::vector<float> expected_leading(12);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      expected_leading[r * 3 + c] = cv[c] * mv[r * 3 + c];
    }
  }
  check_values(multiply(cols, matrix, stream), expected_leading, stream);

  // Scalar broadcast view.
  check_values(
      multiply(matrix, array(2.0f), stream),
      {4.0f, 2.0f, 1.0f, 6.0f, -2.0f, 4.0f,
       1.0f, 8.0f, -4.0f, 3.0f, 0.5f, -1.0f},
      stream);

  // 3D middle-axis broadcast: (2,1,3) * (2,2,3).
  std::vector<float> tv = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  array middle(tv.begin(), Shape{2, 1, 3}, float32);
  std::vector<float> bv = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                           3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f};
  array cube(bv.begin(), Shape{2, 2, 3}, float32);
  std::vector<float> expected_middle(12);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        expected_middle[(i * 2 + j) * 3 + k] =
            tv[i * 3 + k] * bv[(i * 2 + j) * 3 + k];
      }
    }
  }
  check_values(multiply(middle, cube, stream), expected_middle, stream);

  // One bf16 broadcast case through the same shape-aware path.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain_rows = astype(rows, bfloat16, stream);
    array brain_matrix = astype(matrix, bfloat16, stream);
    check_values(
        astype(
            multiply(brain_rows, brain_matrix, stream), float32, stream),
        expected,
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}

TEST_CASE("value_and_grad runs softmax times input through broadcast views") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {
      0.25f, -1.0f, 2.0f, 0.5f, 0.125f, -0.75f, 1.5f, -2.0f};
  array x(xv.begin(), Shape{2, 4}, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    array weights = softmax(inputs[0], std::vector<int>{-1}, false, stream);
    return sum(multiply(weights, inputs[0], stream), stream);
  };
  auto [value, grads] = value_and_grad(fun, std::vector<int>{0})({x});

  // d/dx sum_j softmax(x)_j * x_j = s_j * (1 + x_j - dot(s_row, x_row)).
  float expected_value = 0.0f;
  std::vector<float> expected_dx(8);
  for (int row = 0; row < 2; ++row) {
    const float* xr = xv.data() + row * 4;
    float maximum = xr[0];
    for (int j = 1; j < 4; ++j) {
      maximum = std::max(maximum, xr[j]);
    }
    float weights[4];
    float normalizer = 0.0f;
    for (int j = 0; j < 4; ++j) {
      weights[j] = std::exp(xr[j] - maximum);
      normalizer += weights[j];
    }
    float dot = 0.0f;
    for (int j = 0; j < 4; ++j) {
      weights[j] /= normalizer;
      dot += weights[j] * xr[j];
    }
    expected_value += dot;
    for (int j = 0; j < 4; ++j) {
      expected_dx[row * 4 + j] = weights[j] * (1.0f + xr[j] - dot);
    }
  }
  check_values(value, {expected_value}, stream, 1e-4);
  check_values(grads.at(0), expected_dx, stream, 1e-4);
}

TEST_CASE("batched Matmul matches host references across layouts") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto fill = [](std::vector<float>& values, int modulo, int bias) {
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] =
          static_cast<float>(static_cast<int>(index % modulo) - bias);
    }
  };

  // Rank 3: (2, 3, 4) @ (2, 4, 5).
  constexpr size_t batch3 = 2;
  constexpr size_t m3 = 3;
  constexpr size_t k3 = 4;
  constexpr size_t n3 = 5;
  std::vector<float> a3_values(batch3 * m3 * k3);
  std::vector<float> b3_values(batch3 * k3 * n3);
  fill(a3_values, 7, 3);
  fill(b3_values, 5, 2);
  std::vector<float> expected3(batch3 * m3 * n3, 0.0f);
  for (size_t batch = 0; batch < batch3; ++batch) {
    for (size_t row = 0; row < m3; ++row) {
      for (size_t column = 0; column < n3; ++column) {
        for (size_t inner = 0; inner < k3; ++inner) {
          expected3[batch * m3 * n3 + row * n3 + column] +=
              a3_values[batch * m3 * k3 + row * k3 + inner] *
              b3_values[batch * k3 * n3 + inner * n3 + column];
        }
      }
    }
  }
  array a3(a3_values.begin(), Shape{2, 3, 4}, float32);
  array b3(b3_values.begin(), Shape{2, 4, 5}, float32);
  check_values(matmul(a3, b3, stream), expected3, stream);
  // Per-matrix transposed views of both operands. The stores hold the
  // operands in transposed layout and the transpose views restore the
  // matmul operand shapes.
  std::vector<float> b3_store_values(batch3 * k3 * n3);
  for (size_t batch = 0; batch < batch3; ++batch) {
    for (size_t row = 0; row < k3; ++row) {
      for (size_t column = 0; column < n3; ++column) {
        b3_store_values[batch * n3 * k3 + column * k3 + row] =
            b3_values[batch * k3 * n3 + row * n3 + column];
      }
    }
  }
  array b3_store(b3_store_values.begin(), Shape{2, 5, 4}, float32);
  check_values(
      matmul(a3, transpose(b3_store, {0, 2, 1}, stream), stream),
      expected3,
      stream);
  std::vector<float> a3_store_values(batch3 * k3 * m3);
  for (size_t batch = 0; batch < batch3; ++batch) {
    for (size_t row = 0; row < m3; ++row) {
      for (size_t inner = 0; inner < k3; ++inner) {
        a3_store_values[batch * k3 * m3 + inner * m3 + row] =
            a3_values[batch * m3 * k3 + row * k3 + inner];
      }
    }
  }
  array a3_store(a3_store_values.begin(), Shape{2, 4, 3}, float32);
  check_values(
      matmul(transpose(a3_store, {0, 2, 1}, stream), b3, stream),
      expected3,
      stream);

  // Rank 4: (2, 2, 3, 4) @ (2, 2, 4, 5), dense and transposed-view B.
  constexpr size_t batch4 = 4;
  constexpr size_t m4 = 3;
  constexpr size_t k4 = 4;
  constexpr size_t n4 = 5;
  std::vector<float> a4_values(batch4 * m4 * k4);
  std::vector<float> b4_values(batch4 * k4 * n4);
  fill(a4_values, 6, 3);
  fill(b4_values, 4, 2);
  std::vector<float> expected4(batch4 * m4 * n4, 0.0f);
  for (size_t batch = 0; batch < batch4; ++batch) {
    for (size_t row = 0; row < m4; ++row) {
      for (size_t column = 0; column < n4; ++column) {
        for (size_t inner = 0; inner < k4; ++inner) {
          expected4[batch * m4 * n4 + row * n4 + column] +=
              a4_values[batch * m4 * k4 + row * k4 + inner] *
              b4_values[batch * k4 * n4 + inner * n4 + column];
        }
      }
    }
  }
  array a4(a4_values.begin(), Shape{2, 2, 3, 4}, float32);
  array b4(b4_values.begin(), Shape{2, 2, 4, 5}, float32);
  check_values(matmul(a4, b4, stream), expected4, stream);
  std::vector<float> b4_store_values(batch4 * k4 * n4);
  for (size_t batch = 0; batch < batch4; ++batch) {
    for (size_t row = 0; row < k4; ++row) {
      for (size_t column = 0; column < n4; ++column) {
        b4_store_values[batch * n4 * k4 + column * k4 + row] =
            b4_values[batch * k4 * n4 + row * n4 + column];
      }
    }
  }
  array b4_store(b4_store_values.begin(), Shape{2, 2, 5, 4}, float32);
  check_values(
      matmul(a4, transpose(b4_store, {0, 1, 3, 2}, stream), stream),
      expected4,
      stream);

  // Batched AddMM: scalar, per-row, and full per-batch bias.
  check_values(
      addmm(array(1.0f, float32), a3, b3, 2.0f, 0.5f, stream),
      [&]() {
        std::vector<float> values;
        for (float value : expected3) {
          values.push_back(0.5f + 2.0f * value);
        }
        return values;
      }(),
      stream);
  std::vector<float> bias(n3);
  fill(bias, 3, 1);
  std::vector<float> row_bias_expected;
  for (size_t batch = 0; batch < batch3; ++batch) {
    for (size_t row = 0; row < m3; ++row) {
      for (size_t column = 0; column < n3; ++column) {
        row_bias_expected.push_back(
            2.0f * expected3[batch * m3 * n3 + row * n3 + column] +
            0.5f * bias[column]);
      }
    }
  }
  check_values(
      addmm(array(bias.begin(), Shape{5}, float32), a3, b3, 2.0f, 0.5f, stream),
      row_bias_expected,
      stream);
  std::vector<float> full_bias(batch3 * m3 * n3);
  fill(full_bias, 8, 4);
  std::vector<float> full_bias_expected;
  for (size_t index = 0; index < expected3.size(); ++index) {
    full_bias_expected.push_back(
        2.0f * expected3[index] + 0.5f * full_bias[index]);
  }
  check_values(
      addmm(
          array(full_bias.begin(), Shape{2, 3, 5}, float32),
          a3,
          b3,
          2.0f,
          0.5f,
          stream),
      full_bias_expected,
      stream);

  // Broadcast batch axes run through the stride-0 views: every batch
  // step multiplies the same stored matrix, verified against the host
  // loop. This is the shape the composed GQA attention emits.
  array a3_single(a3_values.begin(), Shape{1, 3, 4}, float32);
  array broadcast_a = broadcast_to(a3_single, {3, 3, 4}, stream);
  std::vector<float> b3_wide_values(3 * 4 * 5);
  fill(b3_wide_values, 5, 2);
  array b3_wide(b3_wide_values.begin(), Shape{3, 4, 5}, float32);
  std::vector<float> broadcast_expected;
  for (size_t batch = 0; batch < 3; ++batch) {
    for (size_t row = 0; row < m3; ++row) {
      for (size_t column = 0; column < n3; ++column) {
        float dot = 0.0f;
        for (size_t inner = 0; inner < k3; ++inner) {
          dot += a3_values[row * k3 + inner] *
              b3_wide_values[inner * n3 + column];
        }
        broadcast_expected.push_back(dot);
      }
    }
  }
  check_values(matmul(broadcast_a, b3_wide, stream), broadcast_expected, stream);

  // Rank 5 passes with dense, stride-0 broadcast, and per-matrix
  // transposed batch operands. Every host reference is computed from
  // the same vectors that back the arrays.
  std::vector<float> a5_values(2 * 2 * 2 * 3 * 4);
  std::vector<float> b5_values(2 * 2 * 2 * 4 * 5);
  fill(a5_values, 6, 3);
  fill(b5_values, 4, 2);
  auto host5 = [&](size_t b1, size_t b2, size_t b2_source) {
    std::vector<float> out(3 * 5);
    for (size_t row = 0; row < 3; ++row) {
      for (size_t column = 0; column < 5; ++column) {
        float dot = 0.0f;
        for (size_t inner = 0; inner < 4; ++inner) {
          dot += a5_values[(b1 * 2 + b2) * 12 + row * 4 + inner] *
              b5_values[(b1 * 2 + b2_source) * 20 + inner * 5 + column];
        }
        out[row * 5 + column] = dot;
      }
    }
    return out;
  };
  array a5(a5_values.begin(), Shape{2, 2, 2, 3, 4}, float32);
  array b5(b5_values.begin(), Shape{2, 2, 2, 4, 5}, float32);
  for (size_t b1 = 0; b1 < 2; ++b1) {
    for (size_t b2 = 0; b2 < 2; ++b2) {
      check_values(matmul(a5, b5, stream), b1, b2, host5(b1, b2, b2), stream);
    }
  }
  // A stride-0 broadcast batch axis repeats one stored matrix per
  // batch step: batches (b1, 0) and (b1, 1) share the (b1, 0) source.
  array b5_single(b5_values.begin(), Shape{2, 2, 1, 4, 5}, float32);
  array b5_broadcast =
      broadcast_to(b5_single, {2, 2, 2, 4, 5}, stream);
  for (size_t b1 = 0; b1 < 2; ++b1) {
    for (size_t b2 = 0; b2 < 2; ++b2) {
      check_values(
          matmul(a5, b5_broadcast, stream), b1, b2, host5(b1, b2, 0), stream);
    }
  }
  // A per-matrix transposed stack works at rank 5 as well.
  std::vector<float> b5_store_values(2 * 2 * 2 * 5 * 4);
  for (size_t b1 = 0; b1 < 2; ++b1) {
    for (size_t b2 = 0; b2 < 2; ++b2) {
      for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 5; ++column) {
          b5_store_values[(b1 * 2 + b2) * 20 + column * 4 + row] =
              b5_values[(b1 * 2 + b2) * 20 + row * 5 + column];
        }
      }
    }
  }
  array b5_store(b5_store_values.begin(), Shape{2, 2, 2, 5, 4}, float32);
  for (size_t b1 = 0; b1 < 2; ++b1) {
    for (size_t b2 = 0; b2 < 2; ++b2) {
      check_values(
          matmul(a5, transpose(b5_store, {0, 1, 2, 4, 3}, stream), stream),
          b1,
          b2,
          host5(b1, b2, b2),
          stream);
    }
  }

  // Rank beyond 5 still reports the named rank error. The operands keep
  // valid matrix dims so the rank check is what rejects them.
  std::vector<float> a6_values(2 * 2 * 2 * 2 * 3 * 4, 0.5f);
  array a6(a6_values.begin(), Shape{2, 2, 2, 2, 3, 4}, float32);
  array b6(a6_values.begin(), Shape{2, 2, 2, 2, 4, 5}, float32);
  std::string rank_error = evaluation_error(matmul(a6, b6, stream));
  CHECK(rank_error.find("matrix rank Matmul") != std::string::npos);
}

TEST_CASE("scaled_dot_product_attention composes through batched matmul") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t heads = 2;
  constexpr size_t queries_length = 4;
  constexpr size_t keys_length = 8;
  constexpr size_t head_dim = 8;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(heads * queries_length * head_dim);
  std::vector<float> k_values(heads * keys_length * head_dim);
  std::vector<float> v_values(heads * keys_length * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  for (size_t index = 0; index < k_values.size(); ++index) {
    k_values[index] = pattern(index + 3);
  }
  for (size_t index = 0; index < v_values.size(); ++index) {
    v_values[index] = pattern(index + 7);
  }
  array q(q_values.begin(), Shape{1, 2, 4, 8}, float32);
  array k(k_values.begin(), Shape{1, 2, 8, 8}, float32);
  array v(v_values.begin(), Shape{1, 2, 8, 8}, float32);
  constexpr float scale = 0.25f;

  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());

  std::vector<float> expected(heads * queries_length * head_dim, 0.0f);
  std::vector<float> scores(keys_length);
  for (size_t head = 0; head < heads; ++head) {
    for (size_t row = 0; row < queries_length; ++row) {
      float max_score = -std::numeric_limits<float>::infinity();
      for (size_t column = 0; column < keys_length; ++column) {
        float dot = 0.0f;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          dot += q_values[head * queries_length * head_dim + row * head_dim +
                          inner] *
              k_values[head * keys_length * head_dim + column * head_dim +
                       inner];
        }
        scores[column] = scale * dot;
        max_score = std::max(max_score, scores[column]);
      }
      float normalizer = 0.0f;
      for (size_t column = 0; column < keys_length; ++column) {
        scores[column] = std::exp(scores[column] - max_score);
        normalizer += scores[column];
      }
      for (size_t inner = 0; inner < head_dim; ++inner) {
        float sum = 0.0f;
        for (size_t column = 0; column < keys_length; ++column) {
          sum += scores[column] / normalizer *
              v_values[head * keys_length * head_dim + column * head_dim +
                       inner];
        }
        expected[head * queries_length * head_dim + row * head_dim + inner] =
            sum;
      }
    }
  }
  check_values(attention, expected, stream, 1e-3);
}

TEST_CASE(
    "scaled_dot_product_attention expands kv heads through broadcast matmul") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t q_heads = 4;
  constexpr size_t kv_heads = 2;
  constexpr size_t seq_length = 8;
  constexpr size_t head_dim = 8;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(q_heads * seq_length * head_dim);
  std::vector<float> kv_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  for (size_t index = 0; index < kv_values.size(); ++index) {
    kv_values[index] = pattern(index + 5);
  }
  array q(q_values.begin(), Shape{1, 4, 8, 8}, float32);
  array k(kv_values.begin(), Shape{1, 2, 8, 8}, float32);
  array v(kv_values.begin(), Shape{1, 2, 8, 8}, float32);
  constexpr float scale = 0.25f;

  // The composed fallback unflattens q to [B, kv_heads, n_rep, L, D]
  // and broadcasts the kv view, so the scores matmul runs at rank 5
  // with a stride-0 batch axis.
  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());

  auto host_reference = [&](const std::vector<float>& qv) {
    std::vector<float> out(q_heads * seq_length * head_dim, 0.0f);
    for (size_t head = 0; head < q_heads; ++head) {
      size_t kv_head = head / (q_heads / kv_heads);
      for (size_t row = 0; row < seq_length; ++row) {
        float max_score = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(seq_length);
        for (size_t column = 0; column < seq_length; ++column) {
          float dot_qk = 0.0f;
          for (size_t inner = 0; inner < head_dim; ++inner) {
            dot_qk += qv[head * seq_length * head_dim + row * head_dim +
                         inner] *
                kv_values[kv_head * seq_length * head_dim + column *
                              head_dim + inner];
          }
          scores[column] = scale * dot_qk;
          max_score = std::max(max_score, scores[column]);
        }
        float normalizer = 0.0f;
        for (size_t column = 0; column < seq_length; ++column) {
          scores[column] = std::exp(scores[column] - max_score);
          normalizer += scores[column];
        }
        for (size_t inner = 0; inner < head_dim; ++inner) {
          float sum = 0.0f;
          for (size_t column = 0; column < seq_length; ++column) {
            sum += scores[column] / normalizer *
                kv_values[kv_head * seq_length * head_dim + column *
                              head_dim + inner];
          }
          out[head * seq_length * head_dim + row * head_dim + inner] = sum;
        }
      }
    }
    return out;
  };
  check_values(attention, host_reference(q_values), stream, 1e-3);

  // The bf16 variant runs the same composition through the bf16
  // matmul, softmax, and select kernels.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 || !capabilities.storage_buffer_16bit_access ||
      !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 shader and storage features.");
    return;
  }
  array q_bf16 = astype(q, bfloat16, stream);
  array k_bf16 = astype(k, bfloat16, stream);
  array v_bf16 = astype(v, bfloat16, stream);
  array attention_bf16 = fast::scaled_dot_product_attention(
      q_bf16, k_bf16, v_bf16, scale, "", std::nullopt, std::nullopt, false,
      stream);
  std::string blocked_bf16 = evaluation_error(attention_bf16);
  REQUIRE(blocked_bf16.empty());
  check_values(
      astype(attention_bf16, float32, stream),
      host_reference(q_values),
      stream,
      1e-2);
}


TEST_CASE("cold-cache GQA decode matmul runs over cache slice views") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Qwen2.5-0.5B geometry: 14 query heads = 2 kv heads x 7 repeats. A
  // fresh cache preallocates, slice_updates one decode step, and reads
  // the state prefix back as a strided view, so the scores matmul sees
  // batch strides that are uniform but not contiguous.
  constexpr size_t kv_heads = 2;
  constexpr size_t n_rep = 7;
  constexpr size_t head_dim = 8;
  constexpr float scale = 0.25f;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  auto new_token = [&](size_t token) {
    std::vector<float> values(kv_heads * head_dim);
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] = pattern(index + token * 17 + 3);
    }
    array update(
        values.begin(),
        Shape{static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
        float32);
    return std::pair<array, std::vector<float>>{update, values};
  };

  array cache = zeros(
      {1,
       static_cast<int>(kv_heads),
       static_cast<int>(16),
       static_cast<int>(head_dim)},
      float32,
      stream);

  // First decode step: state [1, 2, 1, 8] with strides {128, 64, 8, 1}.
  // The scores matmul output is exactly the [1, 2, 7, 1, 1] shape from
  // the M1 smoke blocker.
  auto [update1, k1_values] = new_token(0);
  cache = slice_update(
      cache,
      update1,
      Shape{0, 0, 0, 0},
      Shape{1, static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
      stream);
  array state1 = slice(
      cache,
      {0, 0, 0, 0},
      {1, static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
      stream);

  std::vector<float> q_values(kv_heads * n_rep * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  array q(q_values.begin(), Shape{1, 14, 1, 8}, float32);
  array q5 = unflatten(
      multiply(array(scale, float32), q, stream),
      1,
      Shape{static_cast<int>(kv_heads), static_cast<int>(n_rep)},
      stream);
  array scores1 = matmul(
      q5, swapaxes(expand_dims(state1, 2, stream), -1, -2, stream), stream);
  const Shape scores1_shape{1, 2, 7, 1, 1};
  REQUIRE_EQ(scores1.shape(), scores1_shape);
  std::vector<float> expected_scores1(kv_heads * n_rep, 0.0f);
  for (size_t kv = 0; kv < kv_heads; ++kv) {
    for (size_t rep = 0; rep < n_rep; ++rep) {
      float dot = 0.0f;
      for (size_t inner = 0; inner < head_dim; ++inner) {
        dot += scale * q_values[(kv * n_rep + rep) * head_dim + inner] *
            k1_values[kv * head_dim + inner];
      }
      expected_scores1[kv * n_rep + rep] = dot;
    }
  }
  check_values(scores1, expected_scores1, stream, 1e-5);

  // End to end: the composed causal attention over the same cache view
  // matches the key row (one key makes the softmax trivial). The second
  // step below checks a real two-key softmax.
  array attention1 = fast::scaled_dot_product_attention(
      q,
      state1,
      state1,
      scale,
      "causal",
      std::nullopt,
      std::nullopt,
      false,
      stream);
  std::string blocked1 = evaluation_error(attention1);
  REQUIRE(blocked1.empty());
  std::vector<float> expected_attention1(kv_heads * n_rep * head_dim, 0.0f);
  for (size_t head = 0; head < kv_heads * n_rep; ++head) {
    size_t kv_head = head / n_rep;
    for (size_t inner = 0; inner < head_dim; ++inner) {
      expected_attention1[head * head_dim + inner] =
          k1_values[kv_head * head_dim + inner];
    }
  }
  check_values(attention1, expected_attention1, stream, 1e-5);

  // Second decode step: two keys, still a strided state view, and a
  // non-trivial softmax over the two scores.
  auto [update2, k2_values] = new_token(1);
  cache = slice_update(
      cache,
      update2,
      Shape{0, 0, 1, 0},
      Shape{1, static_cast<int>(kv_heads), 2, static_cast<int>(head_dim)},
      stream);
  array state2 = slice(
      cache,
      {0, 0, 0, 0},
      {1, static_cast<int>(kv_heads), 2, static_cast<int>(head_dim)},
      stream);
  array scores2 = matmul(
      q5, swapaxes(expand_dims(state2, 2, stream), -1, -2, stream), stream);
  const Shape scores2_shape{1, 2, 7, 1, 2};
  REQUIRE_EQ(scores2.shape(), scores2_shape);

  auto host_attention = [&](const std::vector<float>& kv1,
                            const std::vector<float>& kv2) {
    std::vector<float> out(kv_heads * n_rep * head_dim, 0.0f);
    for (size_t kv = 0; kv < kv_heads; ++kv) {
      for (size_t rep = 0; rep < n_rep; ++rep) {
        const float* q_row =
            q_values.data() + (kv * n_rep + rep) * head_dim;
        float s1 = 0.0f;
        float s2 = 0.0f;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          s1 += scale * q_row[inner] * kv1[kv * head_dim + inner];
          s2 += scale * q_row[inner] * kv2[kv * head_dim + inner];
        }
        float maximum = std::max(s1, s2);
        float p1 = std::exp(s1 - maximum);
        float p2 = std::exp(s2 - maximum);
        float normalizer = p1 + p2;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          out[(kv * n_rep + rep) * head_dim + inner] =
              (p1 * kv1[kv * head_dim + inner] +
               p2 * kv2[kv * head_dim + inner]) /
              normalizer;
        }
      }
    }
    return out;
  };
  check_values(
      fast::scaled_dot_product_attention(
          q,
          state2,
          state2,
          scale,
          "causal",
          std::nullopt,
          std::nullopt,
          false,
          stream),
      host_attention(k1_values, k2_values),
      stream,
      1e-5);

  // The bf16 model path runs the same materialized scores matmul.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array scores_bf16 = matmul(
        astype(q5, bfloat16, stream),
        swapaxes(
            expand_dims(astype(state1, bfloat16, stream), 2, stream),
            -1,
            -2,
            stream),
        stream);
    check_values(
        astype(scores_bf16, float32, stream),
        expected_scores1,
        stream,
        1e-2);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}

TEST_CASE("causal scaled_dot_product_attention composes through int32 arange") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t q_heads = 4;
  constexpr size_t kv_heads = 2;
  constexpr size_t seq_length = 32;
  constexpr size_t head_dim = 8;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(q_heads * seq_length * head_dim);
  std::vector<float> kv_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  for (size_t index = 0; index < kv_values.size(); ++index) {
    kv_values[index] = pattern(index + 5);
  }
  array q(q_values.begin(), Shape{1, 4, 32, 8}, float32);
  array k(kv_values.begin(), Shape{1, 2, 32, 8}, float32);
  array v(kv_values.begin(), Shape{1, 2, 32, 8}, float32);
  constexpr float scale = 0.25f;

  // The causal mask materializes int32 aranges (32 > 28 covers the
  // M1 smoke blocker), a GreaterEqual bool mask, and a Select against
  // the dtype minimum.
  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "causal", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());

  std::vector<float> expected(q_heads * seq_length * head_dim, 0.0f);
  for (size_t head = 0; head < q_heads; ++head) {
    size_t kv_head = head / (q_heads / kv_heads);
    for (size_t row = 0; row < seq_length; ++row) {
      float max_score = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(seq_length);
      for (size_t column = 0; column < seq_length; ++column) {
        if (column > row) {
          scores[column] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float dot_qk = 0.0f;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          dot_qk += q_values[head * seq_length * head_dim + row * head_dim +
                             inner] *
              kv_values[kv_head * seq_length * head_dim + column * head_dim +
                        inner];
        }
        scores[column] = scale * dot_qk;
        max_score = std::max(max_score, scores[column]);
      }
      float normalizer = 0.0f;
      for (size_t column = 0; column <= row; ++column) {
        scores[column] = std::exp(scores[column] - max_score);
        normalizer += scores[column];
      }
      for (size_t inner = 0; inner < head_dim; ++inner) {
        float sum = 0.0f;
        for (size_t column = 0; column <= row; ++column) {
          sum += scores[column] / normalizer *
              kv_values[kv_head * seq_length * head_dim + column * head_dim +
                        inner];
        }
        expected[head * seq_length * head_dim + row * head_dim + inner] = sum;
      }
    }
  }
  check_values(attention, expected, stream, 1e-3);
}

namespace {

void check_indices(
    array value,
    const std::vector<uint32_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.dtype(), uint32);
  REQUIRE_EQ(value.size(), expected.size());
  const uint32_t* values = value.data<uint32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

} // namespace

TEST_CASE("argmax reduces last-axis rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Exact indices, a tie row, negative values, NaN skipping, and the
  // all-NaN row that falls back to index 0.
  std::vector<float> xv = {
      1.0f, 3.0f, 2.0f,
      3.0f, 3.0f, 1.0f,
      -5.0f, -1.0f, -3.0f,
      1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  array x(xv.begin(), Shape{5, 3}, float32);
  check_indices(argmax(x, -1, false, stream), {1, 0, 1, 2, 0}, stream);

  // keepdims keeps the reduced axis with size 1.
  array kept = argmax(x, -1, true, stream);
  kept.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(kept.shape().size(), 2);
  CHECK_EQ(kept.shape(0), 5);
  CHECK_EQ(kept.shape(1), 1);
  check_indices(std::move(kept), {1, 0, 1, 2, 0}, stream);

  // A 1000-wide row crosses several 256-thread blocks. The spike at 512
  // ties the one at 768, so the first occurrence wins; the low spike at
  // 257 is the unique row minimum.
  std::vector<float> wide(1000);
  for (size_t index = 0; index < wide.size(); ++index) {
    wide[index] = static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  wide[512] = 3.5f;
  wide[768] = 3.5f;
  wide[257] = -4.0f;
  array w(wide.begin(), Shape{1, 1000}, float32);
  check_indices(argmax(w, -1, false, stream), {512}, stream);
  check_indices(argmin(w, -1, false, stream), {257}, stream);

  // Non-suffix axes are named rejections, not CPU fallbacks.
  array matrix({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  std::string axis_error = evaluation_error(argmax(matrix, 0, false, stream));
  CHECK(axis_error.find("non-suffix ArgMax") != std::string::npos);
  CHECK(axis_error.find("No CPU fallback") != std::string::npos);
}

TEST_CASE("argmin matches first-occurrence ties through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {
      1.0f, 3.0f, 2.0f,
      3.0f, 3.0f, 1.0f,
      -5.0f, -1.0f, -3.0f,
      1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  array x(xv.begin(), Shape{5, 3}, float32);
  check_indices(argmin(x, -1, false, stream), {0, 2, 0, 0, 0}, stream);

  // A tie on the minimum keeps the first occurrence.
  array ties({2.0f, -1.0f, -1.0f, 4.0f}, {4}, float32);
  check_indices(argmin(ties, -1, false, stream), {1}, stream);
}

TEST_CASE("RandomBits matches the host threefry reference bit for bit") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<std::pair<uint32_t, uint32_t>> key_vectors = {
      {0x01234567u, 0x89abcdefu},
      {0x00000000u, 0x00000000u},
      {0xffffffffu, 0xffffffffu},
      {0xdeadbeefu, 0x12345678u}};
  // Word counts cover the even layout, the odd layout with its middle
  // word, and the single-word case.
  for (const auto& shape : {Shape{5}, Shape{4}, Shape{1}, Shape{3}}) {
    for (const auto& key_pair : key_vectors) {
      array key({key_pair.first, key_pair.second}, uint32);
      array bits = random::bits(shape, 4, key, stream);
      bits.eval();
      omarchy::get_command_encoder(stream).synchronize();
      REQUIRE_EQ(bits.size(), shape[0]);
      const uint32_t* words = bits.data<uint32_t>();
      for (uint32_t word = 0; word < bits.size(); ++word) {
        CHECK_EQ(
            words[word],
            host_random_word(key_pair, bits.size(), word));
      }
    }
  }

  // The mx.random.split key shape: one {2} key filling a {2, 2} output,
  // four words through the even layout.
  std::vector<uint32_t> kv = {0x01234567u, 0x89abcdefu};
  array split_key(kv.begin(), Shape{2}, uint32);
  array split_bits = random::bits(Shape{2, 2}, 4, split_key, stream);
  split_bits.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(split_bits.size(), 4u);
  const uint32_t* words = split_bits.data<uint32_t>();
  for (uint32_t word = 0; word < 4; ++word) {
    CHECK_EQ(
        words[word],
        host_random_word({kv[0], kv[1]}, 4, word));
  }
}

TEST_CASE("uniform with a pinned key is deterministic through Vulkan") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array key = random::key(0x5eed1234u);
  auto first = random::uniform(Shape{257}, float32, key, stream);
  auto second = random::uniform(Shape{257}, float32, key, stream);
  first.eval();
  second.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(first.size(), 257u);
  const float* a = first.data<float>();
  const float* b = second.data<float>();
  for (size_t index = 0; index < first.size(); ++index) {
    CHECK_EQ(a[index], b[index]);
    CHECK(a[index] >= 0.0f);
    CHECK(a[index] < 1.0f);
  }
}

TEST_CASE("categorical samples every class through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Uniform logits over four classes: 1000 draws must hit every class.
  array logits = zeros({1000, 4}, float32, stream);
  array samples = random::categorical(logits, -1, std::nullopt, stream);
  samples.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(samples.size(), 1000u);
  CHECK_EQ(samples.dtype(), uint32);
  std::vector<size_t> counts(4, 0);
  const uint32_t* drawn = samples.data<uint32_t>();
  for (size_t index = 0; index < samples.size(); ++index) {
    REQUIRE(drawn[index] < 4u);
    counts[drawn[index]]++;
  }
  for (size_t class_index = 0; class_index < 4; ++class_index) {
    CHECK(counts[class_index] >= 10);
    CHECK(counts[class_index] <= 500);
  }
}

TEST_CASE("RandomBits pins named errors outside the uint32 width") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array key({1u, 2u}, uint32);
  std::string width_error =
      evaluation_error(random::bits(Shape{4}, 2, key, stream));
  CHECK(width_error.find("RandomBits width") != std::string::npos);
  std::string byte_error =
      evaluation_error(random::bits(Shape{4}, 1, key, stream));
  CHECK(byte_error.find("RandomBits width") != std::string::npos);
}

TEST_CASE("FP16 argmax matches host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {1.0f, 3.0f, 2.0f, -2.0f, -2.0f, -1.0f};
  array x(xv.begin(), Shape{2, 3}, float16);
  // Row 1 ties on -2, so argmin keeps the first occurrence.
  check_indices(argmin(x, -1, false, stream), {0, 0}, stream);
}

TEST_CASE("Cos and Sin match host references through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.5f, 1.0f, -0.75f, -2.0f, 3.5f};
  array x(xv.begin(), Shape{static_cast<int>(xv.size())}, float32);
  std::vector<float> cos_expected;
  std::vector<float> sin_expected;
  for (float value : xv) {
    cos_expected.push_back(std::cos(value));
    sin_expected.push_back(std::sin(value));
  }
  check_values(cos(x, stream), cos_expected, stream);
  check_values(sin(x, stream), sin_expected, stream);
}

TEST_CASE("Arange fills start plus step times index through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  check_values(
      arange(0, 10, 1, float32, stream),
      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
      stream);

  // Upstream derives the length from ceil((stop - start) / step), so a
  // negative step over a descending range is a valid request.
  check_values(
      arange(2.5, 0.5, -0.25, float32, stream),
      {2.5f, 2.25f, 2.0f, 1.75f, 1.5f, 1.25f, 1.0f, 0.75f},
      stream);

  // int32 aranges compute int(start) + index * int(step) in exact int
  // arithmetic; the host guards |start|, |step|, and the one-past-last
  // value below 2^24 so the float transport stays exact.
  std::vector<int32_t> int_expected;
  for (int32_t value = 0; value < 10; ++value) {
    int_expected.push_back(value);
  }
  check_int32_values(arange(0, 10, 1, int32, stream), int_expected, stream);
  check_int32_values(
      arange(10, 0, -2, int32, stream), {10, 8, 6, 4, 2}, stream);
  std::string range_error =
      evaluation_error(arange(0, 16777217, 1, int32, stream));
  CHECK(range_error.find("[omarchy] Arange range") != std::string::npos);
  std::string start_error =
      evaluation_error(arange(16777216, 0, -1, int32, stream));
  CHECK(start_error.find("[omarchy] Arange range") != std::string::npos);

  // Other non-float dtypes keep the named dtype error.
  std::string dtype_error =
      evaluation_error(arange(0, 4, 1, int64, stream));
  CHECK(dtype_error.find("[omarchy] Arange dtype") != std::string::npos);
  CHECK(dtype_error.find("No CPU fallback") != std::string::npos);

  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  array half = arange(0, 2, 0.5, float16, stream);
  check_values(
      astype(half, float32, stream), {0.0f, 0.5f, 1.0f, 1.5f}, stream, 1e-2);
}

TEST_CASE("grad of sum sin matches cos through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.1f, 0.4f, -0.9f, 1.3f};
  array x(xv.begin(), Shape{static_cast<int>(xv.size())}, float32);

  // Sin::vjp lowers to Cos and Multiply only, both backend-supported.
  auto fun = [&](const std::vector<array>& inputs) {
    return sum(sin(inputs[0], stream), stream);
  };
  auto [value, grads] = value_and_grad(fun, std::vector<int>{0})({x});

  std::vector<float> expected(xv.size());
  float expected_value = 0.0f;
  for (size_t index = 0; index < xv.size(); ++index) {
    expected[index] = std::cos(xv[index]);
    expected_value += std::sin(xv[index]);
  }
  check_values(value, {expected_value}, stream, 1e-5);
  check_values(grads.at(0), expected, stream, 1e-5);
}

TEST_CASE("sort and argsort order last-axis rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Shuffled rows with duplicates and negatives. Equal keys keep source
  // order, which the argsort indices pin exactly: row 1 ties on zero at
  // indices 0, 1, and 4, and row 2 ties on both -3.5 and 2.5.
  std::vector<float> xv = {
      3.0f, -1.0f, 2.0f, -1.0f, 0.5f,
      0.0f, -0.0f, 5.0f, -7.5f, 0.0f,
      2.5f, -3.5f, 2.5f, -3.5f, 1.0f};
  array x(xv.begin(), Shape{3, 5}, float32);
  check_values(
      sort(x, -1, stream),
      {-1.0f, -1.0f, 0.5f, 2.0f, 3.0f, -7.5f, 0.0f, -0.0f, 0.0f, 5.0f,
       -3.5f, -3.5f, 1.0f, 2.5f, 2.5f},
      stream);
  check_indices(
      argsort(x, -1, stream),
      {1, 3, 4, 2, 0, 3, 0, 1, 4, 2, 1, 3, 4, 0, 2},
      stream);

  // A 1-D row is its own last axis.
  array flat({4.0f, -2.0f, 4.0f, 0.0f}, float32);
  check_values(sort(flat, 0, stream), {-2.0f, 0.0f, 4.0f, 4.0f}, stream);
  check_indices(argsort(flat, 0, stream), {1, 3, 0, 2}, stream);
}

TEST_CASE("sort places NaN after every number through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // The kernel mirrors the upstream CPU comparator: NaN sorts after every
  // number, and two NaN keys order by source index, so the NaN pair at
  // indices 1 and 3 yields 1 then 3.
  float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> xv = {1.0f, nan, -2.0f, nan, 0.0f};
  array x(xv.begin(), Shape{5}, float32);
  array order = argsort(x, -1, stream);
  check_indices(std::move(order), {2, 4, 0, 1, 3}, stream);

  array sorted = sort(x, -1, stream);
  sorted.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(sorted.size(), 5);
  const float* values = sorted.data<float>();
  CHECK_EQ(values[0], -2.0f);
  CHECK_EQ(values[1], 0.0f);
  CHECK_EQ(values[2], 1.0f);
  CHECK(std::isnan(values[3]));
  CHECK(std::isnan(values[4]));
}

TEST_CASE("sort and argsort handle wide rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // A 1000-wide row pads to 1024 inside the kernel. Each row holds a
  // different shuffle of the generator, so the host reference runs
  // stable_sort under the same (value, index) rule once per row.
  std::vector<float> wide(2000);
  for (size_t index = 0; index < wide.size(); ++index) {
    wide[index] =
        static_cast<float>((static_cast<int>(index * 37) % 101) - 50);
  }
  array x(wide.begin(), Shape{2, 1000}, float32);
  std::vector<float> sorted_expected;
  std::vector<uint32_t> order_expected;
  for (int repeat = 0; repeat < 2; ++repeat) {
    std::vector<float> row(
        wide.begin() + repeat * 1000, wide.begin() + (repeat + 1) * 1000);
    std::vector<uint32_t> order(1000);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return row[a] < row[b] || (row[a] == row[b] && a < b);
    });
    std::stable_sort(row.begin(), row.end());
    sorted_expected.insert(sorted_expected.end(), row.begin(), row.end());
    order_expected.insert(order_expected.end(), order.begin(), order.end());
  }
  check_values(sort(x, -1, stream), sorted_expected, stream);
  check_indices(argsort(x, -1, stream), order_expected, stream);

  // An exact 1024-wide descending row skips the padding path.
  std::vector<float> exact(1024);
  for (size_t index = 0; index < exact.size(); ++index) {
    exact[index] = static_cast<float>(1023 - static_cast<int>(index));
  }
  array y(exact.begin(), Shape{1, 1024}, float32);
  std::vector<uint32_t> exact_order(1024);
  std::iota(exact_order.begin(), exact_order.end(), 0u);
  std::reverse(exact_order.begin(), exact_order.end());
  check_indices(argsort(y, -1, stream), exact_order, stream);
}

TEST_CASE("sort rejects long rows and non-float dtypes with named errors") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  std::vector<float> wide(1025, 1.0f);
  array x(wide.begin(), Shape{1, 1025}, float32);
  std::string length_error = evaluation_error(sort(x, -1, stream));
  CHECK(length_error.find("sort row length Sort") != std::string::npos);
  CHECK(length_error.find("No CPU fallback") != std::string::npos);
  std::string index_length_error =
      evaluation_error(argsort(x, -1, stream));
  CHECK(
      index_length_error.find("sort row length ArgSort") != std::string::npos);

  array ints({3, 1, 2}, int32);
  std::string dtype_error = evaluation_error(sort(ints, -1, stream));
  CHECK(dtype_error.find("[omarchy] Sort dtype") != std::string::npos);

  array matrix({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  std::string axis_error = evaluation_error(sort(matrix, 0, stream));
  CHECK(axis_error.find("non-suffix Sort") != std::string::npos);
  std::string index_axis_error =
      evaluation_error(argsort(matrix, 0, stream));
  CHECK(index_axis_error.find("non-suffix ArgSort") != std::string::npos);
}

TEST_CASE("partition redirects to sort and topk returns the right set") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {3.0f, -1.0f, 2.0f, -1.0f, 0.5f};
  array x(xv.begin(), Shape{5}, float32);

  // A full sort satisfies the partition contract, so every position holds
  // the sorted value and argpartition equals argsort exactly.
  check_values(
      partition(x, 2, -1, stream), {-1.0f, -1.0f, 0.5f, 2.0f, 3.0f}, stream);
  check_indices(argpartition(x, 2, -1, stream), {1, 3, 4, 2, 0}, stream);

  // mx.topk lowers to partition plus a tail slice, so a 1-D topk returns
  // the k largest in ascending order through the same path.
  check_values(topk(x, 2, -1, stream), {2.0f, 3.0f}, stream);

  // The same full-sort redirect satisfies the partition contract for
  // several rows at once.
  array m({5.0f, 1.0f, 4.0f, 2.0f, 3.0f, 0.0f}, {2, 3}, float32);
  check_values(
      partition(m, 0, -1, stream),
      {1.0f, 4.0f, 5.0f, 0.0f, 2.0f, 3.0f},
      stream);

  std::string axis_error = evaluation_error(partition(m, 0, 0, stream));
  CHECK(axis_error.find("non-suffix Partition") != std::string::npos);
}

TEST_CASE("strided slice views materialize exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Column 2 of each row: the tail-slice shape that mx.topk lowers to.
  array m({5.0f, 1.0f, 4.0f, 2.0f, 3.0f, 0.0f}, {2, 3}, float32);
  check_values(slice(m, {0, 2}, {2, 3}, stream), {4.0f, 0.0f}, stream);

  // mx.topk(m, 2, -1) is partition plus a strided tail slice.
  check_values(topk(m, 2, -1, stream), {4.0f, 5.0f, 2.0f, 3.0f}, stream);

  // Inner-axis slices leave gaps between rows.
  array p({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
          {2, 4},
          float32);
  check_values(slice(p, {0, 1}, {2, 3}, {1, 1}, stream),
               {1.0f, 2.0f, 5.0f, 6.0f},
               stream);

  // A 1-D stride-2 slice gathers every other element.
  array base({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
             {8},
             float32);
  check_values(
      slice(base, {0}, {7}, {2}, stream), {0.0f, 2.0f, 4.0f, 6.0f}, stream);
}

TEST_CASE("elementwise ops over strided slice views match host values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // The mx.fast.rope half-split pattern: x[..., 0:2] and x[..., 2:4] are
  // strided views over the same parent rows.
  array x({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
          {2, 4},
          float32);
  array x1 = slice(x, {0, 0}, {2, 2}, {1, 1}, stream);
  array x2 = slice(x, {0, 2}, {2, 4}, {1, 1}, stream);

  check_values(multiply(x1, x2, stream), {0.0f, 3.0f, 24.0f, 35.0f}, stream);
  check_values(subtract(x2, x1, stream), {2.0f, 2.0f, 2.0f, 2.0f}, stream);
  check_values(add(x1, x2, stream), {2.0f, 4.0f, 10.0f, 12.0f}, stream);
}

TEST_CASE("FP16 sort and argsort match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }

  // Row ties on -2 at indices 1 and 3, so argsort keeps 1 then 3.
  std::vector<float> xv = {1.0f, -2.0f, 0.5f, -2.0f};
  array x(xv.begin(), Shape{4}, float16);
  check_values(
      astype(sort(x, -1, stream), float32, stream),
      {-2.0f, -2.0f, 0.5f, 1.0f},
      stream,
      1e-2);
  check_indices(argsort(x, -1, stream), {1, 3, 2, 0}, stream);
}

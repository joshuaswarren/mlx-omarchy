// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/device.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"
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

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
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

  array base({1.0f, 2.0f, 3.0f, 4.0f}, float32);
  array strided = slice(base, {0}, {4}, {2}, stream);
  std::string stride_error = evaluation_error(exp(strided, stream));
  CHECK(stride_error.find("non-contiguous Exp") != std::string::npos);

  array lhs({1.0f, 2.0f}, {2, 1}, float32);
  array rhs({3.0f, 4.0f}, {1, 2}, float32);
  std::string broadcast_error = evaluation_error(add(lhs, rhs, stream));
  CHECK(broadcast_error.find("broadcast Add") != std::string::npos);

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

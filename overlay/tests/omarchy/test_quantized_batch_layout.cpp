// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
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

std::vector<float> readback(const Stream& stream, array value) {
  value = astype(value, float32, stream);
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = value.data<float>();
  return {data, data + value.size()};
}

void check_constant_slices(
    const std::vector<float>& values,
    const std::vector<float>& expected,
    size_t slice_size) {
  REQUIRE_EQ(values.size(), expected.size() * slice_size);
  for (size_t slice = 0; slice < expected.size(); ++slice) {
    for (size_t i = 0; i < slice_size; ++i) {
      CHECK(values[slice * slice_size + i] ==
          doctest::Approx(expected[slice]).epsilon(1e-5));
    }
  }
}

array fp_weights(
    int batch,
    int rows,
    int packed_words,
    const std::vector<uint32_t>& words) {
  return array(words.begin(), Shape{batch, rows, packed_words}, uint32);
}

array fp_scales(
    int batch,
    int rows,
    int groups,
    const std::vector<uint8_t>& scales) {
  return array(scales.begin(), Shape{batch, rows, groups}, uint8);
}

struct ReferenceKernelGate {
  ReferenceKernelGate() {
    setenv("MLX_OMARCHY_QMM_TILE", "0", 1);
  }
  ~ReferenceKernelGate() {
    unsetenv("MLX_OMARCHY_QMM_TILE");
  }
};

} // namespace

TEST_CASE("affine qvm accepts a rank-one vector") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  constexpr int k = 32;
  constexpr int n = 32;
  std::vector<float> x_values(k, 1.0f);
  std::vector<uint32_t> words(k * (n * 8 / 32), 0x01010101u);
  std::vector<float> scales(k * (n / 32), 1.0f);
  std::vector<float> biases(scales.size(), 0.0f);

  array out = quantized_matmul(
      array(x_values.begin(), Shape{k}, float32),
      array(words.begin(), Shape{k, n * 8 / 32}, uint32),
      array(scales.begin(), Shape{k, n / 32}, float32),
      array(biases.begin(), Shape{k, n / 32}, float32),
      false,
      32,
      8,
      "affine",
      stream);

  CHECK_EQ(out.shape(), Shape{n});
  check_constant_slices(readback(stream, out), {32.0f}, n);

  constexpr int m = 2;
  std::vector<float> xb(2 * m * k);
  std::fill_n(xb.begin(), m * k, 1.0f);
  std::fill(xb.begin() + m * k, xb.end(), 2.0f);
  std::vector<uint32_t> wb(3 * n * (k * 8 / 32));
  std::fill_n(wb.begin(), n * (k * 8 / 32), 0x01010101u);
  std::fill_n(
      wb.begin() + n * (k * 8 / 32), n * (k * 8 / 32), 0x02020202u);
  std::fill(wb.begin() + 2 * n * (k * 8 / 32), wb.end(), 0x03030303u);
  std::vector<float> sb(3 * n, 1.0f);
  std::vector<float> bb(sb.size(), 0.0f);
  array broadcast = quantized_matmul(
      array(xb.begin(), Shape{2, 1, m, k}, float32),
      array(wb.begin(), Shape{1, 3, n, k * 8 / 32}, uint32),
      array(sb.begin(), Shape{1, 3, n, 1}, float32),
      array(bb.begin(), Shape{1, 3, n, 1}, float32),
      true,
      32,
      8,
      "affine",
      stream);
  CHECK_EQ(broadcast.shape(), Shape{2, 3, m, n});
  check_constant_slices(
      readback(stream, broadcast),
      {32.0f, 64.0f, 96.0f, 64.0f, 128.0f, 192.0f},
      m * n);
}

TEST_CASE("fp vector paths pair each x row with its weight batch") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  constexpr int batch = 2;
  constexpr int k = 32;
  constexpr int n = 32;
  constexpr int words_per_row = k * 4 / 32;
  std::vector<float> x_values(batch * k, 1.0f);
  std::vector<uint32_t> transposed_words(batch * n * words_per_row);
  std::fill_n(transposed_words.begin(), n * words_per_row, 0x22222222u);
  std::fill(transposed_words.begin() + n * words_per_row,
            transposed_words.end(), 0x44444444u);
  std::vector<uint8_t> transposed_scales(batch * n, 127u);

  array qmv = quantized_matmul(
      array(x_values.begin(), Shape{batch, 1, k}, float32),
      fp_weights(batch, n, words_per_row, transposed_words),
      fp_scales(batch, n, 1, transposed_scales),
      std::nullopt,
      true,
      std::nullopt,
      std::nullopt,
      "mxfp4",
      stream);
  CHECK_EQ(qmv.shape(), Shape{batch, 1, n});
  check_constant_slices(readback(stream, qmv), {32.0f, 64.0f}, n);

  array shared_qmv = quantized_matmul(
      array(x_values.begin(), Shape{1, k}, float32),
      array(transposed_words.begin(), Shape{n, words_per_row}, uint32),
      array(transposed_scales.begin(), Shape{n, 1}, uint8),
      std::nullopt,
      true,
      std::nullopt,
      std::nullopt,
      "mxfp4",
      stream);
  CHECK_EQ(shared_qmv.shape(), Shape{1, n});
  check_constant_slices(readback(stream, shared_qmv), {32.0f}, n);

  constexpr int packed_n = n * 4 / 32;
  std::vector<uint32_t> non_transposed_words(batch * k * packed_n);
  std::fill_n(non_transposed_words.begin(), k * packed_n, 0x22222222u);
  std::fill(non_transposed_words.begin() + k * packed_n,
            non_transposed_words.end(), 0x44444444u);
  std::vector<uint8_t> non_transposed_scales(batch * k, 127u);
  array qvm = quantized_matmul(
      array(x_values.begin(), Shape{batch, 1, k}, float32),
      fp_weights(batch, k, packed_n, non_transposed_words),
      fp_scales(batch, k, 1, non_transposed_scales),
      std::nullopt,
      false,
      std::nullopt,
      std::nullopt,
      "mxfp4",
      stream);
  CHECK_EQ(qvm.shape(), Shape{batch, 1, n});
  check_constant_slices(readback(stream, qvm), {32.0f, 64.0f}, n);
}

TEST_CASE("fp tiled path broadcasts arbitrary batch ranks") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  constexpr int k = 32;
  constexpr int m = 3;
  constexpr int n = 32;
  constexpr int words_per_row = k * 4 / 32;
  std::vector<float> x_values(2 * m * k);
  std::fill_n(x_values.begin(), m * k, 1.0f);
  std::fill(x_values.begin() + m * k, x_values.end(), 2.0f);

  std::vector<uint32_t> words(3 * n * words_per_row);
  std::fill_n(words.begin(), n * words_per_row, 0x22222222u);
  std::fill_n(words.begin() + n * words_per_row,
              n * words_per_row, 0x44444444u);
  std::fill(words.begin() + 2 * n * words_per_row,
            words.end(), 0x66666666u);
  std::vector<uint8_t> scales(3 * n, 127u);

  array shared = quantized_matmul(
      array(x_values.begin(), Shape{2, m, k}, float32),
      array(words.begin(), Shape{n, words_per_row}, uint32),
      array(scales.begin(), Shape{n, 1}, uint8),
      std::nullopt,
      true,
      std::nullopt,
      std::nullopt,
      "mxfp4",
      stream);
  CHECK_EQ(shared.shape(), Shape{2, m, n});
  check_constant_slices(readback(stream, shared), {32.0f, 64.0f}, m * n);

  array out = quantized_matmul(
      array(x_values.begin(), Shape{2, 1, m, k}, float32),
      array(words.begin(), Shape{1, 3, n, words_per_row}, uint32),
      array(scales.begin(), Shape{1, 3, n, 1}, uint8),
      std::nullopt,
      true,
      std::nullopt,
      std::nullopt,
      "mxfp4",
      stream);

  CHECK_EQ(out.shape(), Shape{2, 3, m, n});
  check_constant_slices(
      readback(stream, out), {32.0f, 64.0f, 128.0f, 64.0f, 128.0f, 256.0f},
      m * n);
}

TEST_CASE("fp reference path keeps batched weight offsets") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  constexpr int batch = 2;
  constexpr int m = 2;
  constexpr int k = 32;
  constexpr int n = 32;
  constexpr int words_per_row = k * 4 / 32;
  std::vector<float> x_values(batch * m * k, 1.0f);
  std::vector<uint32_t> words(batch * n * words_per_row);
  std::fill_n(words.begin(), n * words_per_row, 0x22222222u);
  std::fill(words.begin() + n * words_per_row, words.end(), 0x44444444u);
  std::vector<uint8_t> scales(batch * n, 127u);

  ReferenceKernelGate gate;
  array out = quantized_matmul(
      array(x_values.begin(), Shape{batch, m, k}, float32),
      fp_weights(batch, n, words_per_row, words),
      fp_scales(batch, n, 1, scales),
      std::nullopt,
      true,
      std::nullopt,
      std::nullopt,
      "mxfp4",
      stream);
  auto values = readback(stream, out);

  CHECK_EQ(out.shape(), Shape{batch, m, n});
  check_constant_slices(values, {32.0f, 64.0f}, m * n);
}

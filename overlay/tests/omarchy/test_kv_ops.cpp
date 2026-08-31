// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// KV-cache plumbing tests: Concatenate and SliceUpdate (None-reduce) on the
// Omarchy backend. Every op must route through the shared strided-copy
// engine; results are checked against exact values.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <iostream>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

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

TEST_CASE("Concatenate axis 0 and axis -1 of 2D fp32 arrays") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array b({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, float32);

  // Axis 0 with row-contiguous inputs uses the buffer-copy path.
  array rows = concatenate({a, b}, 0, stream);
  CHECK_EQ(rows.shape(), Shape{4, 2});
  check_values(rows, {1, 2, 3, 4, 5, 6, 7, 8}, stream);

  // Axis -1 must go through the strided-copy window path.
  array cols = concatenate({a, b}, -1, stream);
  CHECK_EQ(cols.shape(), Shape{2, 4});
  check_values(cols, {1, 2, 5, 6, 3, 4, 7, 8}, stream);
}

TEST_CASE("Concatenate 3D KV-cache shape along axis 1") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // prev tokens and new tokens, heads*dim laid out on the last axis.
  array prev({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
             {1, 2, 4},
             float32);
  array next({10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f},
             {1, 2, 4},
             float32);

  array cache = concatenate({prev, next}, 1, stream);
  CHECK_EQ(cache.shape(), Shape{1, 4, 4});
  check_values(
      cache,
      {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17},
      stream);
}

TEST_CASE("Concatenate fp16 KV blocks") {
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
  array prev = astype(
      array({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2}, float32), float16, stream);
  array next = astype(
      array({5.0f, 6.0f, 7.0f, 8.0f}, {1, 2, 2}, float32), float16, stream);

  array cache = concatenate({prev, next}, 1, stream);
  CHECK_EQ(cache.dtype(), float16);
  check_values(
      astype(cache, float32, stream), {1, 2, 3, 4, 5, 6, 7, 8}, stream, 1e-3);
}

TEST_CASE("SliceUpdate row window keeps surrounding data") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // KV-cache shape: write two new rows at positions 2..3 of 4.
  array src({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
             8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
            {1, 4, 4},
            float32);
  array upd({100.0f, 101.0f, 102.0f, 103.0f, 200.0f, 201.0f, 202.0f, 203.0f},
            {1, 2, 4},
            float32);

  array updated = slice_update(src, upd, Shape{0, 2, 0}, Shape{1, 4, 4}, stream);
  CHECK_EQ(updated.shape(), Shape{1, 4, 4});
  check_values(
      updated,
      {0, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103, 200, 201, 202, 203},
      stream);
}

TEST_CASE("KV cache growth: slice_update steps then concatenate when full") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array cache = zeros({1, 8, 4}, float32, stream);

  array k0({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {1, 2, 4},
           float32);
  array k1({9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
           {1, 2, 4},
           float32);

  cache = slice_update(cache, k0, Shape{0, 0, 0}, Shape{1, 2, 4}, stream);
  cache = slice_update(cache, k1, Shape{0, 2, 0}, Shape{1, 4, 4}, stream);
  check_values(
      cache,
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      stream);

  // Cache is half full; grow it and keep writing.
  array grown = concatenate({cache, zeros({1, 8, 4}, float32, stream)},
                            1,
                            stream);
  CHECK_EQ(grown.shape(), Shape{1, 16, 4});
  check_values(
      grown,
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      stream);

  array k2({17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
           {1, 2, 4},
           float32);
  array extended = slice_update(grown, k2, Shape{0, 8, 0}, Shape{1, 10, 4}, stream);
  check_values(
      extended,
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       17, 18, 19, 20, 21, 22, 23, 24,
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0},
      stream);
}

TEST_CASE("Unsupported SliceUpdate reduce mode reports a named error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array upd({10.0f, 20.0f}, {1, 2}, float32);

  std::string reduce_error = evaluation_error(
      slice_update_add(src, upd, Shape{0, 0}, Shape{1, 2}, stream));
  CHECK(reduce_error.find("[omarchy] SliceUpdate reduce") !=
        std::string::npos);
}

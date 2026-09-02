// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// KV-cache plumbing tests: Concatenate and SliceUpdate (None-reduce) on the
// Omarchy backend. Every op must route through the shared strided-copy
// engine; results are checked against exact values.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fast.h"
#include "mlx/io.h"
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

// Exact integer comparison against an evaluated int32 result.
void check_int32_values(
    array value,
    const std::vector<int32_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.dtype(), int32);
  REQUIRE_EQ(value.size(), expected.size());
  const int32_t* values = value.data<int32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

TEST_CASE("int32 to float32 and back round-trips exactly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // |value| <= 2^24 stays exact in float32.
  std::vector<int32_t> values = {
      0, 1, -1, 42, -42, 1000000, -999999, 16777216, -16777216};
  array ints(values.data(), Shape{static_cast<int>(values.size())}, int32);

  array floats = astype(ints, float32, stream);
  check_values(
      floats,
      {0.0f,
       1.0f,
       -1.0f,
       42.0f,
       -42.0f,
       1000000.0f,
       -999999.0f,
       16777216.0f,
       -16777216.0f},
      stream);

  array back = astype(floats, int32, stream);
  check_int32_values(back, values, stream);
}

TEST_CASE("float32 to int32 truncates toward zero") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Upstream astype float->int truncates: mlx/random.cpp pins
  // "-1.7 becomes -1" and the CPU kernel applies static_cast.
  array floats(
      {-1.7f,
       1.7f,
       -0.5f,
       0.5f,
       3.999f,
       -3.999f,
       2.0f,
       -2.0f,
       100.25f,
       -100.25f},
      {10},
      float32);

  array ints = astype(floats, int32, stream);
  check_int32_values(ints, {-1, 1, 0, 0, 3, -3, 2, -2, 100, -100}, stream);
}

TEST_CASE("scalar int32 astype to float32") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The mx.fast.rope composed fallback casts the int32 scalar offset to
  // float32 through a data_size-1 converting copy.
  array positive = astype(array(42, int32), float32, stream);
  check_values(positive, {42.0f}, stream);

  array negative = astype(array(-7, int32), float32, stream);
  check_values(negative, {-7.0f}, stream);
}

TEST_CASE("int32 and float16 casts convert through the 16-bit gates") {
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
  // int32 -> float16 -> int32 round-trips on exact f16 values.
  array ints({2048, -2048, 7, -7, 0}, {5}, int32);
  array halves = astype(ints, float16, stream);
  check_values(astype(halves, float32, stream), {2048, -2048, 7, -7, 0}, stream);
  check_int32_values(astype(halves, int32, stream), {2048, -2048, 7, -7, 0}, stream);

  // float16 -> int32 truncates toward zero.
  array halves_frac({1.5f, -1.5f, 2.75f, -2.75f}, {4}, float32);
  array truncated =
      astype(astype(halves_frac, float16, stream), int32, stream);
  check_int32_values(truncated, {1, -1, 2, -2}, stream);
}

TEST_CASE("int32 and bfloat16 casts convert through the 16-bit gates") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access || !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 storage features.");
    return;
  }
  // int32 -> bfloat16 -> int32 round-trips on exact bf16 values.
  array ints({16, -16, 256, -256, 3, -3}, {6}, int32);
  array brains = astype(ints, bfloat16, stream);
  check_int32_values(astype(brains, int32, stream), {16, -16, 256, -256, 3, -3}, stream);

  // bfloat16 -> int32 truncates the widened value toward zero.
  array floats_frac({2.75f, -2.75f}, {2}, float32);
  array truncated =
      astype(astype(floats_frac, bfloat16, stream), int32, stream);
  check_int32_values(truncated, {2, -2}, stream);
}

TEST_CASE("mx.fast.rope matches a host rotation on the composed fallback") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The composed fallback used to stop at the int32 scalar -> float32
  // offset cast, then at the half-split slice views (the named
  // "non-contiguous Multiply" blocker). Strided slice views now
  // materialize at eval, so the full rotation runs and is compared
  // against a host-computed reference.
  std::vector<float> x_values(2 * 1 * 4 * 12);
  for (size_t index = 0; index < x_values.size(); ++index) {
    x_values[index] = static_cast<float>(index);
  }
  array x(x_values.begin(), Shape{2, 1, 4, 12}, float32);

  array out = fast::rope(
      x,
      /*dims=*/8,
      /*traditional=*/false,
      /*base=*/10000.0f,
      /*scale=*/1.0f,
      /*offset=*/0,
      std::nullopt,
      stream);

  // Host reference: non-traditional half-split rotation of dims 0..7 and
  // passthrough of dims 8..11.
  const float inv_log_step = -std::log(10000.0f) / 4.0f;
  std::vector<float> expected(x_values.size(), 0.0f);
  for (int batch = 0; batch < 2; ++batch) {
    for (int token = 0; token < 4; ++token) {
      for (int i = 0; i < 4; ++i) {
        float theta =
            static_cast<float>(token) *
            std::exp(static_cast<float>(i) * inv_log_step);
        float cos_v = std::cos(theta);
        float sin_v = std::sin(theta);
        size_t row = (batch * 4 + token) * 12;
        float x1 = x_values[row + i];
        float x2 = x_values[row + 4 + i];
        expected[row + i] = x1 * cos_v - x2 * sin_v;
        expected[row + 4 + i] = x1 * sin_v + x2 * cos_v;
        expected[row + 8 + i] = x_values[row + 8 + i];
      }
    }
  }
  check_values(out, expected, stream, 1e-4);
}

TEST_CASE("safetensors load round-trips float32 through the gpu stream") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const std::string path =
      (std::filesystem::temp_directory_path() /
       "mlx_omarchy_kv_roundtrip_f32.safetensors")
          .string();

  // save_safetensors evals through the default stream; gpu_stream() has
  // already set the default device to the gpu.
  std::unordered_map<std::string, array> weights;
  weights.insert({"w", array({0.25f, -3.5f, 1000.0f, 1.0f}, {2, 2}, float32)});
  save_safetensors(path, std::move(weights));

  // The load pins an explicit gpu stream. The io layer resolves a default
  // stream to Device::cpu when a CPU backend exists (mlx-io-device.patch),
  // so the explicit stream keeps the round-trip on the gpu in every build.
  auto loaded = load_safetensors(path, stream);
  array out = loaded.first.at("w");
  out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(out.shape(), Shape{2, 2});
  const float* values = out.data<float>();
  CHECK_EQ(values[0], 0.25f);
  CHECK_EQ(values[1], -3.5f);
  CHECK_EQ(values[2], 1000.0f);
  CHECK_EQ(values[3], 1.0f);
  std::remove(path.c_str());
}

TEST_CASE("safetensors load round-trips bf16 bit patterns") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access || !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 storage features.");
    return;
  }
  const std::string path =
      (std::filesystem::temp_directory_path() /
       "mlx_omarchy_kv_roundtrip_bf16.safetensors")
          .string();

  array brains =
      astype(array({1.0f, 2.0f, 0.5f, -4.0f}, {4}, float32), bfloat16, stream);
  std::unordered_map<std::string, array> weights;
  weights.insert({"b", brains});
  save_safetensors(path, std::move(weights));

  auto loaded = load_safetensors(path, stream);
  array out = loaded.first.at("b");
  CHECK_EQ(out.dtype(), bfloat16);
  array widened = astype(out, float32, stream);
  widened.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(widened.size(), 4u);
  const float* values = widened.data<float>();
  CHECK_EQ(values[0], 1.0f);
  CHECK_EQ(values[1], 2.0f);
  CHECK_EQ(values[2], 0.5f);
  CHECK_EQ(values[3], -4.0f);
  std::remove(path.c_str());
}

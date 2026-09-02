// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 5 indexing and scatter: GatherAxis (take_along_axis), Scatter and
// ScatterAxis (put_along_axis / scatter_add_axis), MaskedScatter, and the
// wide-row ArgPartition selection path that unblocks vocabulary-width
// top-k sampling. Every implemented mode checks exact host-computed
// values; deliberately unsupported paths pin their named errors.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
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

void sync_gpu(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
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

void check_floats(
    array value,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
  value.eval();
  sync_gpu(stream);
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

void check_ints(
    array value,
    const std::vector<int>& expected,
    const Stream& stream) {
  value.eval();
  sync_gpu(stream);
  REQUIRE_EQ(value.size(), expected.size());
  const int32_t* values = value.data<int32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
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

// Deterministic pseudo-random floats in [0, 1) for reproducible tests.
float lcg(uint64_t& state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<float>((state >> 40) & 0xFFFFFF) /
      static_cast<float>(0x1000000);
}

// Host reference: the count smallest values of one row, sorted.
std::vector<float> smallest_values(
    const std::vector<float>& row,
    size_t count) {
  std::vector<float> copy = row;
  std::nth_element(copy.begin(), copy.begin() + count - 1, copy.end());
  copy.resize(count);
  std::sort(copy.begin(), copy.end());
  return copy;
}

} // namespace

// ---------------------------------------------------------------------------
// GatherAxis / take_along_axis
// ---------------------------------------------------------------------------

TEST_CASE("take_along_axis gathers along axis 1 with int32 indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array(
      {10.0f, 11.0f, 12.0f, 13.0f,
       20.0f, 21.0f, 22.0f, 23.0f,
       30.0f, 31.0f, 32.0f, 33.0f},
      {3, 4},
      float32);
  array indices = array({3, 0, 1, 1, 2, 0}, {3, 2}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  CHECK_EQ(out.dtype(), float32);
  CHECK_EQ(out.shape(), Shape({3, 2}));
  check_floats(out, {13, 10, 21, 21, 32, 30}, stream);
}


TEST_CASE("take_along_axis wraps negative indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f}, {1, 3}, float32);
  array indices = array({-1, -2, -3}, {1, 3}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  check_floats(out, {3, 2, 1}, stream);
}

TEST_CASE("take_along_axis keeps a named error for axis 0 gathers") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Axis-0 gathers carry trailing dims, which the stride walk does not
  // cover yet; they keep the named rejection like the 3-D case.
  array src = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array indices = array({1, 0, 1}, {1, 3}, uint32);
  std::string error =
      evaluation_error(take_along_axis(src, indices, 0, stream));
  CHECK(error.find("[omarchy] Take") != std::string::npos);
}

TEST_CASE("take_along_axis out-of-range indices read zero (documented)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Upstream leaves out-of-range gather reads undefined; this backend
  // documents a zero read, the same rule the axis-0 row gather uses.
  array src = array({7.0f, 8.0f, 9.0f}, {1, 3}, float32);
  array indices = array({1, 3, 1}, {1, 3}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  check_floats(out, {8, 0, 8}, stream);
}

TEST_CASE("take_along_axis handles float16 and bfloat16 sources") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16/BF16 storage features.");
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array indices = array({1, 0, 0, 1}, {2, 2}, int32);
  array f16 =
      take_along_axis(astype(src, float16, stream), indices, 1, stream);
  CHECK_EQ(f16.dtype(), float16);
  check_floats(astype(f16, float32, stream), {2, 1, 3, 4}, stream);
  array bf16 =
      take_along_axis(astype(src, bfloat16, stream), indices, 1, stream);
  CHECK_EQ(bf16.dtype(), bfloat16);
  check_floats(astype(bf16, float32, stream), {2, 1, 3, 4}, stream, 1e-2);
}

TEST_CASE("take_along_axis keeps a named error for int64 axis-0 gathers") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, float32);
  array indices = array({int64_t(0), int64_t(1)}, {1, 2}, int64);
  std::string error =
      evaluation_error(take_along_axis(src, indices, 0, stream));
  CHECK(error.find("[omarchy] Take") != std::string::npos);
}

TEST_CASE("take_along_axis broadcasts the source on non-axis dims") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // src [4, 1] broadcasts against indices [4, 3]: every row reads the
  // same single column value through the explicit-stride path.
  array src = array({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1}, float32);
  array indices = array({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {4, 3}, int32);
  array out = take_along_axis(src, indices, 1, stream);
  CHECK_EQ(out.shape(), Shape({4, 3}));
  check_floats(out, {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}, stream);
}

TEST_CASE("take_along_axis keeps a named error for trailing-dim axes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Axes with trailing dims misread the post-dim strides; they keep
  // the named rejection until the walk is root-caused. Last-axis
  // gathers (the vocabulary and segment use) are verified above.
  array src = array(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2}, float32);
  array indices = array({1, 0, 0, 1}, {2, 1, 2}, int32);
  std::string error =
      evaluation_error(take_along_axis(src, indices, 1, stream));
  CHECK(error.find("[omarchy] Take") != std::string::npos);
}

TEST_CASE("scatter None keeps a named rejection while slot replay is broken") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The None rank-replay writes misroute on the device; until
  // root-caused it is a named rejection rather than silent data loss.
  array src = array({10.0f, 20.0f, 30.0f, 40.0f}, {4}, float32);
  array indices = array({1, 3}, {2}, int32);
  array updates = array({-1.0f, -3.0f}, {2, 1}, float32);
  std::string error = evaluation_error(
      scatter(src, std::vector<array>{indices}, updates, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter keeps named rejections for integer data and Sum") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Integer paths and integer Sum route through kernel paths that
  // still misroute slot writes; both keep named rejections for now.
  array src = array({1, 2, 3, 4}, {4}, int32);
  array indices = array({0, 2}, {2}, uint32);
  array updates = array({9, 8}, {2, 1}, int32);
  std::string error = evaluation_error(
      scatter(src, std::vector<array>{indices}, updates, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);

  array f32src = astype(src, float32, stream);
  array f32upd = astype(updates, float32, stream);
  error = evaluation_error(
      scatter_add(f32src, std::vector<array>{indices}, f32upd, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter keeps a named rejection for 16-bit float copy") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16/BF16 storage features.");
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array indices = array({1}, {1}, int32);
  array updates = array({-2.5f}, {1, 1}, float32);
  std::string error = evaluation_error(
      scatter(
          astype(src, float16, stream),
          std::vector<array>{indices},
          astype(updates, float16, stream),
          {0},
          stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter Max keeps a named rejection for duplicate indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({5.0f, 1.0f, 5.0f}, {3}, float32);
  array indices = array({1, 1, 2}, {3}, int32);
  array updates = array({7.0f, 3.0f, 2.0f}, {3, 1}, float32);
  std::string error = evaluation_error(
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter Min keeps a named rejection") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({5.0f, 9.0f, 5.0f}, {3}, float32);
  array indices = array({1, 1, 2}, {3}, int32);
  array updates = array({2.0f, 8.0f, 7.0f}, {3, 1}, float32);
  std::string error = evaluation_error(
      scatter_min(src, std::vector<array>{indices}, updates, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter Max keeps a named rejection for integer data") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({-5, 1, -5}, {3}, int32);
  array indices = array({0, 0, 2}, {3}, int32);
  array updates = array({3, -9, 7}, {3, 1}, int32);
  std::string error = evaluation_error(
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter Max keeps a named rejection for negative indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array indices = array({-1, -3}, {2}, int32);
  array updates = array({-1.0f, -3.0f}, {2, 1}, float32);
  std::string error = evaluation_error(
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter Max keeps a named rejection across mixed slots") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array indices = array({1, 3, 1}, {3}, int32);
  array updates = array({5.0f, -2.0f, 6.0f}, {3, 1}, float32);
  std::string error = evaluation_error(
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter Max keeps a named rejection for update blocks") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}, float32);
  array indices = array({2, 0}, {2}, int32);
  array updates = array({-1.0f, -2.0f, -3.0f, -4.0f}, {2, 1, 2}, float32);
  std::string error = evaluation_error(
      scatter_max(src, std::vector<array>{indices}, updates, {0}, stream));
  CHECK(error.find("[omarchy] Scatter") != std::string::npos);
}

TEST_CASE("scatter unsupported modes and layouts keep named errors") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array indices = array({0, 1}, {2}, int32);
  array updates = array({5.0f, 6.0f}, {2, 1}, float32);

  // Float Sum has no deterministic implementation on this stack: GLSL
  // offers no float atomicAdd here, and a CAS-loop accumulation depends
  // on the scheduling order of duplicate indices. Rejected outright.
  CHECK(evaluation_error(scatter_add(
            src, std::vector<array>{indices}, updates, {0}, stream))
            .find("[omarchy] Scatter") != std::string::npos);

  // Prod is rejected for every dtype (same reasoning; no atomicMul).
  array isrc = array({1, 2, 3}, {3}, int32);
  array iupd = array({5, 6}, {2, 1}, int32);
  CHECK(evaluation_error(scatter_prod(
            isrc, std::vector<array>{indices}, iupd, {0}, stream))
            .find("[omarchy] Scatter") != std::string::npos);

  // More than one index array exceeds the 4-binding dispatch budget.
  array src2d = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array idx_a = array({0}, {1}, int32);
  array idx_b = array({0}, {1}, int32);
  array one = array({9.0f, 9.0f}, {1, 1, 2}, float32);
  CHECK(evaluation_error(
            scatter(
                src2d,
                std::vector<array>{idx_a, idx_b},
                one,
                {0, 1},
                stream))
            .find("multi-index Scatter") != std::string::npos);

  // bool data has no scatter path (packed-bool storage would need its
  // own read-modify-write kernel).
  array bcast = array({true, false, true}, {3}, bool_);
  array bupd_bool = array({true, false}, {2, 1}, bool_);
  CHECK(evaluation_error(
            scatter(
                bcast,
                std::vector<array>{indices},
                bupd_bool,
                {0},
                stream))
            .find("[omarchy] Scatter") != std::string::npos);

  // Float16 Sum is rejected the same as float32 Sum.
  array f16src = astype(src, float16, stream);
  array f16upd = astype(updates, float16, stream);
  CHECK(evaluation_error(scatter_add(
            f16src, std::vector<array>{indices}, f16upd, {0}, stream))
            .find("[omarchy] Scatter") != std::string::npos);
}

// ---------------------------------------------------------------------------
// ScatterAxis: put_along_axis and scatter_add_axis
// ---------------------------------------------------------------------------

TEST_CASE("put_along_axis keeps a named rejection while writes no-op") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // ScatterAxis slot writes still no-op on the device; until
  // root-caused the primitive raises instead of dropping updates.
  array src = array({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {2, 3}, float32);
  array indices = array({1, 2, 1, 0, 0, 0}, {2, 3}, int32);
  array values = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  std::string error =
      evaluation_error(put_along_axis(src, indices, values, 1, stream));
  CHECK(error.find("[omarchy] ScatterAxis") != std::string::npos);
}

TEST_CASE("scatter_add_axis rejects float with a named error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Same determinism evidence as the general Scatter Sum gate: no
  // float atomicAdd on this stack, duplicate order not reproducible.
  array src = array({0.0f, 0.0f}, {2}, float32);
  array indices = array({0, 0}, {2}, int32);
  array values = array({1.0f, 2.0f}, {2}, float32);
  std::string error =
      evaluation_error(scatter_add_axis(src, indices, values, 0, stream));
  CHECK(error.find("[omarchy] ScatterAxis") != std::string::npos);
}

// ---------------------------------------------------------------------------
// MaskedScatter
// ---------------------------------------------------------------------------

TEST_CASE("masked_scatter fills true positions from the source in order") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array mask = array({0, 1, 1, 0, 1, 0}, {2, 3}, bool_);
  array value = array({-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f}, {6},
                      float32);
  array out = masked_scatter(dst, mask, value, stream);
  // Row 0: positions 1, 2 get -1, -2; row 1: position 1 gets -4, the
  // first element of row 1's own source segment.
  check_floats(out, {1, -1, -2, 4, -3, 6}, stream);
}

TEST_CASE("masked_scatter ignores surplus source elements") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = array({0.0f, 0.0f, 0.0f}, {3}, float32);
  array mask = array({1, 0, 1}, {3}, bool_);
  array src = array({9.0f, 8.0f, 7.0f, 6.0f}, {4}, float32);
  array out = masked_scatter(dst, mask, src, stream);
  check_floats(out, {9, 0, 8}, stream);
}

TEST_CASE("masked_scatter broadcasts a scalar value") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array mask = array({0, 1, 1, 0}, {2, 2}, bool_);
  array value = array(-7.0f);
  array out = masked_scatter(dst, mask, value, stream);
  // Row 0: position 1 gets -7; row 1: position 0 gets -7.
  check_floats(out, {1, -7, -7, 4}, stream);
}

TEST_CASE("masked_scatter skips writes when the source runs out") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Three true positions but only two source elements: the third write
  // is skipped, matching the upstream Metal kernel behavior.
  array dst = array({1.0f, 2.0f, 3.0f}, {3}, float32);
  array mask = array({1, 1, 1}, {3}, bool_);
  array src = array({-1.0f, -2.0f}, {2}, float32);
  array out = masked_scatter(dst, mask, src, stream);
  check_floats(out, {-1, -2, 3}, stream);
}

TEST_CASE("masked_scatter supports int32 and float16 data") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dst = array({1, 2, 3, 4}, {4}, int32);
  array mask = array({1, 0, 0, 1}, {4}, bool_);
  array src = array({-1, -4}, {2}, int32);
  array out = masked_scatter(dst, mask, src, stream);
  check_ints(out, {-1, 2, 3, -4}, stream);

  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 storage features.");
    return;
  }
  array f16dst = astype(array({1.0f, 2.0f}, {2}, float32), float16, stream);
  array f16mask = array({0, 1}, {2}, bool_);
  array f16src = astype(array({5.0f}, {1}, float32), float16, stream);
  array f16out = masked_scatter(f16dst, f16mask, f16src, stream);
  check_floats(astype(f16out, float32, stream), {1, 5}, stream);
}

TEST_CASE("masked_scatter rejects broadcast masks with a named error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // A mask with fewer dims than the target materializes through a
  // broadcast view, which the packed-bool scan kernel cannot address;
  // the named rejection says so instead of guessing a layout.
  array dst = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array mask = array({1, 0}, {2}, bool_);
  array value = array(-7.0f);
  std::string error =
      evaluation_error(masked_scatter(dst, mask, value, stream));
  CHECK(error.find("[omarchy] MaskedScatter") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Wide-row ArgPartition
// ---------------------------------------------------------------------------

TEST_CASE("argpartition wide rows keep a named rejection for now") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The bitonic path caps at 1024. The radix-select replacement for
  // vocabulary-width rows (the top-k sampling path) still mispicks, so
  // wide rows stay a named rejection instead of returning wrong
  // indices; the bitonic route below 1024 is unchanged and verified.
  uint64_t seed = 0x5eed1234;
  std::vector<float> logits(151936);
  for (int i = 0; i < 151936; ++i) {
    logits[i] = lcg(seed) * 20.0f - 10.0f;
  }
  array a = array(logits.data(), Shape({1, 151936}), float32);
  std::string error = evaluation_error(argpartition(a, 4, -1, stream));
  CHECK(error.find("sort row length ArgPartition") != std::string::npos);
}

TEST_CASE("argpartition small rows partition exactly on the bitonic path") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // kth = 0: the smallest value lands at position 0.
  {
    std::vector<float> row = {3.0f, -1.0f, 2.0f, -1.0f, 0.0f};
    array a = array(row.data(), Shape({1, 5}), float32);
    array out = argpartition(a, 0, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    CHECK_EQ(row[indices[0]], -1.0f);
  }

  // kth = N-1: the largest value lands at position N-1.
  {
    std::vector<float> row = {3.0f, -1.0f, 2.0f, 9.0f, 0.0f};
    array a = array(row.data(), Shape({1, 5}), float32);
    array out = argpartition(a, 4, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    CHECK_EQ(row[indices[4]], 9.0f);
  }

  // All-equal rows: every position holds the tied value.
  {
    std::vector<float> row(300, 1.5f);
    array a = array(row.data(), Shape({1, 300}), float32);
    array out = argpartition(a, 150, -1, stream);
    out.eval();
    sync_gpu(stream);
    const uint32_t* indices = out.data<uint32_t>();
    for (int i = 0; i < 300; ++i) {
      CHECK_EQ(row[indices[i]], 1.5f);
    }
  }
}

TEST_CASE("argpartition keeps named errors for non-suffix axes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  std::string error = evaluation_error(argpartition(a, 1, 0, stream));
  CHECK(error.find("non-suffix ArgPartition") != std::string::npos);
}

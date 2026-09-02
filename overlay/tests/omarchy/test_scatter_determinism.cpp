// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Float scatter reductions, bool Scatter, and non-dense layouts.
//
// Verdict under test (upstream-grounded 2026-09-02): upstream's Metal
// backend scatters float32 through native float atomics and float16/
// bfloat16 through a packed CAS loop, so duplicate-index accumulation
// order is nondeterministic upstream and no ordering contract exists.
// This backend matches that: fp32 hardware atomicAdd
// (VK_EXT_shader_atomic_float, measured on llvmpipe and the M1
// Honeykrisp target), fp32 scratch accumulation for half types, and a
// deterministic byte RMW for bool. Tests use integer-valued updates
// wherever duplicates appear, so the expected totals are exact in any
// accumulation order; the one fractional-duplicate case pins the total
// within float tolerance instead of pretending the order is fixed.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <iostream>
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

// Bool outputs arrive as the packed word transport; read canonical 0/1
// bytes back through a uint32 view.
void check_bools(
    array value,
    const std::vector<int>& expected,
    const Stream& stream) {
  value.eval();
  sync_gpu(stream);
  REQUIRE_EQ(value.size(), expected.size());
  auto words = astype(value, uint32, stream);
  words.eval();
  sync_gpu(stream);
  const uint32_t* data = words.data<uint32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    uint32_t byte =
        (data[index >> 2] >> ((static_cast<uint32_t>(index) & 3u) * 8u)) &
        0xFFu;
    CHECK_EQ(byte, static_cast<uint32_t>(expected[index]));
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

} // namespace

// ---------------------------------------------------------------------------
// Float scatter_add (Sum): the embedding-gradient case
// ---------------------------------------------------------------------------

TEST_CASE("scatter_add float32 duplicate indices exact total") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({0.0f, 100.0f, 0.0f, 0.0f}, {4}, float32);
  array indices = array({1, 1, 1, 2}, {4}, int32);
  array updates = array({1.0f, 2.0f, 3.0f, 7.0f}, {4}, float32);
  // Integer-valued addends: any accumulation order yields exactly 106.
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {0.0f, 106.0f, 7.0f, 0.0f}, stream);
}

TEST_CASE("scatter_add float32 run twice is bitwise identical") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({8}, float32, stream);
  array indices = array({3, 3, 3, 3, 0, 0, 5, 5}, {8}, int32);
  array updates = array(
      {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 8.0f, 7.0f, 6.0f}, {8}, float32);
  array first = scatter_add(src, indices, updates, 0, stream);
  first.eval();
  sync_gpu(stream);
  array second = scatter_add(src, indices, updates, 0, stream);
  second.eval();
  sync_gpu(stream);
  const float* a = first.data<float>();
  const float* b = second.data<float>();
  for (size_t i = 0; i < 8; ++i) {
    // Exact addends commute bitwise regardless of device order.
    CHECK_EQ(a[i], b[i]);
  }
}

TEST_CASE("scatter_add float32 fractional duplicates hit total") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({3}, float32, stream);
  array indices = array({1, 1, 1}, {3}, int32);
  array updates = array({0.1f, 0.2f, 0.3f}, {3}, float32);
  // Order-dependent last bits are allowed (upstream Metal has the same
  // property); the total must match the host sum within tolerance.
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {0.0f, 0.6f, 0.0f}, stream, 1e-4);
}

TEST_CASE("scatter_add float32 update blocks with trailing dims") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // 2-D src, 1 index array of 2 slots, updates [2, 2, 3] (slot, row,
  // col): every slot writes a 2x3 block.
  array src = zeros({2, 3}, float32, stream);
  array indices = array({0, 1}, {2}, int32);
  array updates = array(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 20.0f, 30.0f, 40.0f,
       50.0f, 60.0f},
      {2, 2, 3},
      float32);
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(
      out,
      {1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60},
      stream);
}

TEST_CASE("scatter_add float16 and bfloat16 duplicate indices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  {
    array src = zeros({2}, float16, stream);
    array indices = array({1, 1, 1, 0}, {4}, int32);
    array updates = array({1.0f, 2.0f, 3.0f, 4.0f}, {4}, float16);
    array out = scatter_add(src, indices, updates, 0, stream);
    check_floats(astype(out, float32, stream), {4.0f, 6.0f}, stream);
  }
  {
    array src = zeros({2}, bfloat16, stream);
    array indices = array({1, 1, 1, 0}, {4}, int32);
    array updates = array({1.0f, 2.0f, 3.0f, 4.0f}, {4}, bfloat16);
    array out = scatter_add(src, indices, updates, 0, stream);
    check_floats(astype(out, float32, stream), {4.0f, 6.0f}, stream, 1e-2);
  }
}

TEST_CASE("scatter_add float32 broadcast updates materialize") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, 1.0f, 1.0f, 1.0f}, {4}, float32);
  array indices = array({0, 3}, {2}, int32);
  // Broadcast a row over the slot dim: a stride view whose data_size is
  // one quarter of its size, the lying-flag hazard the host must
  // detect through data_size.
  array row = array({5.0f, 6.0f, 7.0f, 8.0f}, {1, 4}, float32);
  array updates = broadcast_to(row, {2, 1, 4}, stream);
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {6.0f, 7.0f, 8.0f, 9.0f}, stream);
}

TEST_CASE("scatter_add float32 strided updates materialize") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 2}, float32, stream);
  // A transposed updates view: non-row-contiguous with full data_size.
  array dense = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                      {2, 2, 2},
                      float32);
  array updates = transpose(dense, {0, 2, 1}, stream);
  array indices = array({0, 1}, {2}, int32);
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {1, 3, 2, 4, 5, 7, 6, 8}, stream);
}

TEST_CASE("scatter_add float32 non-contiguous indices materialize") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({4}, float32, stream);
  array idx_dense = array({3, 1, 2, 0, 2, 2}, {2, 3}, int32);
  array indices = transpose(idx_dense, {1, 0}, stream);
  array updates = array(
      {10.0f, 10.0f, 10.0f, 20.0f, 20.0f, 20.0f}, {2, 3}, float32);
  // The transposed index view walks column-major: column 0 holds
  // (3, 2, 0) and column 1 holds (1, 2, 2), so with the row-of-updates
  // layout column j scatters its three updates of value (j+1)*10.
  array out = scatter_add(src, indices, updates, 0, stream);
  check_floats(out, {10.0f, 20.0f, 50.0f, 10.0f}, stream);
}

TEST_CASE("scatter_prod float32 duplicate indices exact product") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({2.0f, 1.0f}, {2}, float32);
  array indices = array({0, 0, 0, 1}, {4}, int32);
  array updates = array({2.0f, 3.0f, 4.0f, 5.0f}, {4}, float32);
  // Integer-valued factors: any multiplication order yields 2*24=48.
  array out = scatter_prod(src, indices, updates, 0, stream);
  check_floats(out, {48.0f, 5.0f}, stream);
}

TEST_CASE("scatter_max and scatter_min floats still key-select") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({1.0f, -5.0f}, {2}, float32);
  array indices = array({0, 0, 1, 1}, {4}, int32);
  array updates = array({4.0f, -2.0f, 3.0f, -9.0f}, {4}, float32);
  array mx = scatter_max(src, indices, updates, 0, stream);
  array mn = scatter_min(src, indices, updates, 0, stream);
  check_floats(mx, {4.0f, 3.0f}, stream);
  check_floats(mn, {-2.0f, -9.0f}, stream);
}

// ---------------------------------------------------------------------------
// Bool Scatter: deterministic packed-byte read-modify-write
// ---------------------------------------------------------------------------

TEST_CASE("scatter bool none duplicate indices last slot wins") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({false, true}, {2}, bool_);
  array indices = array({0, 0, 1}, {3}, int32);
  array updates = array({true, false, false}, {3}, bool_);
  array out = scatter(src, indices, updates, 0, stream);
  // Highest slot per target: target 0 -> slot 1 (false), target 1 ->
  // slot 2 (false).
  check_bools(out, {0, 0}, stream);
}

TEST_CASE("scatter bool add saturates like upstream") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({false, true}, {2}, bool_);
  array indices = array({0, 0, 1, 1, 1}, {5}, int32);
  array updates = array(
      {true, false, true, false, true}, {5}, bool_);
  array out = scatter_add(src, indices, updates, 0, stream);
  // Bool add is saturating: src OR updates.
  check_bools(out, {1, 1}, stream);
}

TEST_CASE("scatter bool prod and min and max") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({true, false, true, true}, {4}, bool_);
  array indices = array({0, 0, 1, 2, 3, 3}, {6}, int32);
  array updates = array(
      {true, false, true, false, false, true}, {6}, bool_);
  array prod = scatter_prod(src, indices, updates, 0, stream);
  // src AND updates: {true&true&false, false&true, true&false,
  // true&false&true}.
  check_bools(prod, {0, 0, 0, 0}, stream);
  array mx = scatter_max(src, indices, updates, 0, stream);
  check_bools(mx, {1, 1, 1, 1}, stream);
  array mn = scatter_min(src, indices, updates, 0, stream);
  check_bools(mn, {0, 0, 0, 0}, stream);
}

TEST_CASE("scatter bool across word lanes keeps siblings intact") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // 10 bools = 3 words; writes land in every lane of the middle word.
  array src = array(
      {false, false, false, false, false,
       false, false, false, false, false},
      {10},
      bool_);
  array indices = array({1, 4, 5, 6, 7, 8}, {6}, int32);
  array updates = array(
      {true, true, true, true, true, true}, {6}, bool_);
  array out = scatter_add(src, indices, updates, 0, stream);
  check_bools(out, {0, 1, 0, 0, 1, 1, 1, 1, 1, 0}, stream);
}

// ---------------------------------------------------------------------------
// ScatterAxis: float Sum and bool
// ---------------------------------------------------------------------------

TEST_CASE("scatter_add_axis float32 duplicate rows exact") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 3}, float32, stream);
  array indices = array({1, 1, 1, 0}, {4, 1}, int32);
  array updates = ones({4, 1, 3}, float32, stream);
  array out = scatter_add_axis(src, indices, updates, 0, stream);
  check_floats(
      out,
      {1, 1, 1, 3, 3, 3},
      stream);
}

TEST_CASE("scatter_add_axis float16 and bool") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  {
    array src = zeros({2, 2}, float16, stream);
    array indices = array({0, 0, 1}, {3, 1}, int32);
    array updates = array(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 1, 2}, float16);
    array out = scatter_add_axis(src, indices, updates, 0, stream);
    check_floats(
        astype(out, float32, stream),
        {4, 6, 5, 6},
        stream);
  }
  {
    array src = array({false, false, false, false}, {2, 2}, bool_);
    array indices = array({0, 0, 1}, {3, 1}, int32);
    array updates = array(
        {true, true, false, true, true, false}, {3, 1, 2}, bool_);
    array out = scatter_add_axis(src, indices, updates, 0, stream);
    check_bools(out, {1, 1, 1, 1}, stream);
  }
}

TEST_CASE("put_along_axis bool none across lanes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array(
      {true, true, true, true, true, true, true, true}, {2, 4},
      bool_,
      stream);
  array indices = array({1, 0, 1, 0}, {4, 1}, int32);
  array updates = array(
      {false, true, false, true, false, true, false, true}, {4, 1, 2},
      bool_,
      stream);
  array out = put_along_axis(src, indices, updates, 0, stream);
  check_bools(out, {0, 1, 0, 1, 1, 0, 1, 0}, stream);
}

// ---------------------------------------------------------------------------
// Multi-index float and bool
// ---------------------------------------------------------------------------

TEST_CASE("multi index float32 scatter add works") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = zeros({2, 3}, float32, stream);
  array idx0 = array({0, 1}, {2}, int32);
  array idx1 = array({2, 0}, {2}, int32);
  array updates = array({7.0f, 9.0f}, {2, 1}, float32);
  array out = scatter_add(src, {idx0, idx1}, updates, {0, 1}, stream);
  check_floats(out, {0, 0, 7, 9, 0}, stream);
}

TEST_CASE("multi index bool scatter add works") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = array({false, false, false, false, false, false}, {2, 3},
                    bool_);
  array idx0 = array({0, 1}, {2}, int32);
  array idx1 = array({2, 0}, {2}, int32);
  array updates = array({true, true}, {2, 1}, bool_);
  array out = scatter_add(src, {idx0, idx1}, updates, {0, 1}, stream);
  check_bools(out, {0, 0, 1, 1, 0, 0}, stream);
}

TEST_CASE("float scatter without atomic float keeps named error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& caps = omarchy::capability_report(0);
  if (caps.shader_atomic_float_add) {
    // The path is implemented on this device; nothing to pin.
    return;
  }
  array src = zeros({2}, float32, stream);
  array indices = array({0, 1}, {2}, int32);
  array updates = array({1.0f, 2.0f}, {2}, float32);
  array out = scatter_add(src, indices, updates, 0, stream);
  std::string error = evaluation_error(out);
  bool refused = !error.empty();
  CHECK(refused);
  bool named = error.find("atomic_float") != std::string::npos ||
      error.find("shaderBufferFloat32AtomicAdd") != std::string::npos;
  CHECK(named);
}

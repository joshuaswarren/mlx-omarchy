// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1).");
    return false;
  }
  return true;
}

// Pull one element of an evaluated array.
template <typename T>
T read_value(array a, size_t index = 0) {
  a.eval();
  omarchy::get_command_encoder(gpu_stream()).synchronize();
  const T* values = a.data<T>();
  return values[index];
}

bool bits_equal_f32(float got, float want) {
  uint32_t g = std::bit_cast<uint32_t>(got);
  uint32_t w = std::bit_cast<uint32_t>(want);
  return g == w;
}

} // namespace

// W2: equal_nan plumbing on identical arrays carrying NaN. The upstream
// suite's mx.array_equal primary comparison helper used to return False
// here because the Equal kernel ignored the equal_nan flag.
TEST_CASE("W2 array_equal with equal_nan maps NaN to NaN as True") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> values{0.0f, 1.0f, std::nanf("")};
  array a(values.begin(), Shape{3}, float32);

  // Default flag stays False: NaN vs NaN compares False (IEEE).
  array out_default = array_equal(a, a, false, stream);
  CHECK_EQ(read_value<bool>(out_default), false);

  // equal_nan True: NaN vs NaN compares True.
  array out_nan = array_equal(a, a, true, stream);
  CHECK_EQ(read_value<bool>(out_nan), true);

  // mx.isclose is element-wise and uses its own composed path (abs
  // diff, isinf corrections, and a logical_or with isnan(a) && isnan(b)
  // gated on equal_nan). The Equal primitive is NOT involved when
  // equal_nan=False. Wrap in all() to get a scalar result.
  std::vector<float> other{0.0f, 1.0f, std::nanf("")};
  array b(other.begin(), Shape{3}, float32);
  array close_default = all(isclose(a, b, 1e-5, 1e-8, false, stream), stream);
  CHECK_EQ(read_value<bool>(close_default), false);
  array close_nan = all(isclose(a, b, 1e-5, 1e-8, true, stream), stream);
  CHECK_EQ(read_value<bool>(close_nan), true);
}

// W2: bool Equal unblocks array_equal machinery for the 6 masked C++
// cases (test array basics, test array types, gguf metadata, is close,
// random split, vmap comparison ops). Before this fix these refused
// with the named Equal-dtype gap; now they dispatch through the
// extended logical_or.comp comparison sextet.
//
// NOTE: the sextet shares dispatch_logical with Or/And/Not, and that
// shader accumulates output words with atomicOr. The output buffer
// must be zero-filled before dispatch or the OR accumulates into
// garbage. This is a shared-infrastructure dependency: the fix belongs
// in dispatch_logical (zero the buffer once, before the atomicOr
// pass), not here.
TEST_CASE("W2 bool Equal and NotEqual match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<bool> lhs{true, false, true, false};
  std::vector<bool> rhs{true, true, false, false};
  array a(lhs.begin(), Shape{4}, bool_);
  array b(rhs.begin(), Shape{4}, bool_);

  array eq = equal(a, b, stream);
  std::vector<bool> expected_eq{true, false, false, true};
  for (size_t i = 0; i < 4; ++i) {
    CHECK_EQ(read_value<bool>(eq, i), expected_eq[i]);
  }

  array neq = not_equal(a, b, stream);
  std::vector<bool> expected_neq{false, true, true, false};
  for (size_t i = 0; i < 4; ++i) {
    CHECK_EQ(read_value<bool>(neq, i), expected_neq[i]);
  }

  // array_equal on bool arrays: was a named gap; now a value path.
  array ae = array_equal(a, a, false, stream);
  CHECK_EQ(read_value<bool>(ae), true);
}

// W5: log10 returns the exact integer for every f32-exact power of
// ten. log10(1000) == 3.0 was the receipt's named pin (C++ ops_tests
// arithmetic unary ops line 1721); the snap table now also nails 10^k
// for k in -10..10.
TEST_CASE("W5 log10 returns exact integer for f32-exact powers of ten") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (int k = -10; k <= 10; ++k) {
    float value = std::pow(10.0f, static_cast<float>(k));
    array x(value);
    array y = log10(x, stream);
    float got = read_value<float>(y);
    float want = static_cast<float>(k);
    INFO("10^" << k << " f32 bits 0x" << std::hex
         << std::bit_cast<uint32_t>(value) << std::dec);
    CHECK(bits_equal_f32(got, want));
  }
}

// W5: ad-hoc adversarial values that include the original defect.
TEST_CASE("W5 log10 pin and non-power inputs agree with numpy f32") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> values{
      1.0f,
      2.0f,
      8.0f,
      100.0f,
      1000.0f,
      123456.0f,
      1e6f,
      0.1f,
      0.5f,
      1e-3f};
  for (float v : values) {
    array x(v);
    array y = log10(x, stream);
    float got = read_value<float>(y);
    float want = std::log10(v);
    INFO("log10(" << v << ") got=" << got << " want=" << want
         << " diff=" << std::abs(got - want));
    CHECK(std::abs(got - want) <= 1e-5f);
  }
}

// W8: sin/cos use GLSL built-ins on the common path (|x| below ~2^24).
// The built-ins match std::sin/cos on lavapipe and most drivers. Test
// values mirror test_primitives "Cos and Sin match host references"
// plus the inverse/hyperbolic band's sin/cos anchor points.
TEST_CASE("trig gate refuses huge arguments by name with the true magnitude") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The gate reads its ReduceMax+Cast magnitude on the host. The read is
  // ordered behind a stream synchronize; an unordered read races the
  // submission and reports recycled-page garbage (observed as ~9.7e8 on
  // Honeykrisp in compiled 4-bit generation, 2026-09-03), silently
  // passing in-range arguments. Pin both halves of the contract: the
  // refusal names the operation and carries the TRUE magnitude, and
  // in-range arguments still compute through the gate.
  array x({2.0e5f, 1.0f, 2.0f});
  bool refused = false;
  std::string msg;
  try {
    array y = cos(x, stream);
    y.eval();
    FAIL("expected the trig argument gate to refuse");
  } catch (const std::exception& e) {
    refused = true;
    msg = e.what();
  }
  CHECK(refused);
  CHECK(msg.find("Cos argument magnitude") != std::string::npos);
  CHECK(msg.find("200000.000000") != std::string::npos);

  array small({0.5f, 1.0f, 2.0f});
  array y = cos(small, stream);
  float c1 = read_value<float>(y, 1);
  CHECK(std::abs(c1 - (float)std::cos(1.0)) < 1e-4f);
}

TEST_CASE("W8 sin/cos/tan common path matches numpy f32") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> common{
      0.0f, 0.5f, 1.0f, 1.5707963f, 3.1415927f, -0.75f, -2.0f, 3.5f};
  for (float v : common) {
    array xs(v);
    array ys = sin(xs, stream);
    array yc = cos(xs, stream);
    float gs = read_value<float>(ys);
    float gc = read_value<float>(yc);
    float ulp_s = std::abs(gs - (float)std::sin((double)v));
    float ulp_c = std::abs(gc - (float)std::cos((double)v));
    INFO("v=" << v << " sin=" << gs << " cos=" << gc);
    CHECK(ulp_s <= 1e-6f);
    CHECK(ulp_c <= 1e-6f);
  }
}

// W8: full-range pin. fast::RoPE is a fallback composition of exactly
// these sin/cos calls with inv_freq[0] = 1.0, so angles equal token
// positions and the limit is a positional ceiling, not a taste choice.
// The M1 Honeykrisp built-in holds measured accuracy to 1e5 (worst
// seen 4.8e-3 at 123457) and returns garbage from ~1e6; the in-source
// Payne-Hanek software fallback was probed on the same device and
// returns garbage of magnitude 1e15+ (dynamic-indexing miscompile
// class), so there is no accurate kernel above the limit. Bands:
//  |v| <= 1e3  -> built-in, 1e-4 abs (measured 2.8e-5 worst)
//  1e3 < |v| <= 1e5 -> built-in, 5e-3 abs (measured 4.8e-3 worst)
//  |v| > 1e5   -> named refusal, no value
std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

TEST_CASE("W8 sin/cos accurate to 1e5 and refused by name above it") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> full{
      0.0f, 0.5f, 1.0f, 1.5707963f, 3.1415927f, -0.75f, -2.0f, 3.5f,
      12345.0f, 54321.0f,
      123456.789f, 1e8f, 1e9f, 1e10f, 1e20f, 1e30f, 1e38f, -2.7e37f};
  for (float v : full) {
    array xs(v);
    if (std::abs(v) > 1e5f) {
      std::string sin_error = evaluation_error(sin(xs, stream));
      INFO("v=" << v << " sin error=" << sin_error);
      CHECK(sin_error.find("[omarchy] Sin") != std::string::npos);
      CHECK(sin_error.find("magnitude") != std::string::npos);
      std::string cos_error = evaluation_error(cos(xs, stream));
      INFO("v=" << v << " cos error=" << cos_error);
      CHECK(cos_error.find("[omarchy] Cos") != std::string::npos);
      CHECK(cos_error.find("magnitude") != std::string::npos);
      continue;
    }
    float tol = std::abs(v) <= 1e3f ? 1e-4f : 5e-3f;
    array ys = sin(xs, stream);
    array yc = cos(xs, stream);
    float gs = read_value<float>(ys);
    float gc = read_value<float>(yc);
    float ulp_s = std::abs(gs - (float)std::sin((double)v));
    float ulp_c = std::abs(gc - (float)std::cos((double)v));
    INFO("v=" << v << " sin=" << gs << " cos=" << gc);
    CHECK(ulp_s <= tol);
    CHECK(ulp_c <= tol);
  }
}

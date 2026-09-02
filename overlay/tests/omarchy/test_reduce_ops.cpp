// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 4 reduction and scan coverage: integer dtypes for Sum/Prod/Min/Max,
// Any/All, arbitrary-axis reduction, the full cumsum/cumprod/cummax/cummin
// flag matrix, and Hadamard. Every value test runs against a host reference
// or exact constants; the empty-reduction identities follow the upstream
// Limits<> seeds in mlx/backend/cpu/reduce.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
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

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

void sync(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
}

void check_values(
    array value,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    if (std::isinf(expected[index]) || std::isnan(expected[index]) ||
        std::isinf(values[index]) || std::isnan(values[index])) {
      // doctest::Approx cannot compare infinities; exact equality is the
      // right check for inf and NaN identities.
      CHECK_EQ(values[index], expected[index]);
    } else {
      CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
    }
  }
}

void check_int32_values(
    array value,
    const std::vector<int32_t>& expected,
    const Stream& stream) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.dtype(), int32);
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
  sync(stream);
  REQUIRE_EQ(value.dtype(), uint32);
  REQUIRE_EQ(value.size(), expected.size());
  const uint32_t* values = value.data<uint32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_bool_values(
    array value,
    const std::vector<bool>& expected,
    const Stream& stream) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.dtype(), bool_);
  REQUIRE_EQ(value.size(), expected.size());
  const bool* values = value.data<bool>();
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

// Host reference for one scan family: op 0 sum, 1 prod, 2 min, 3 max.
std::vector<float> host_scan(
    const std::vector<float>& in,
    size_t lines,
    size_t length,
    int op,
    bool reverse,
    bool inclusive) {
  std::vector<float> out(in.size());
  for (size_t line = 0; line < lines; ++line) {
    float acc = op == 0 ? 0.0f
        : op == 1     ? 1.0f
        : op == 2     ? std::numeric_limits<float>::infinity()
                      : -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < length; ++t) {
      size_t j = reverse ? length - 1 - t : t;
      float v = in[line * length + j];
      float next = op == 0   ? acc + v
          : op == 1          ? acc * v
          : op == 2          ? std::min(acc, v)
                             : std::max(acc, v);
      if (inclusive) {
        acc = next;
        out[line * length + j] = acc;
      } else {
        out[line * length + j] = acc;
        acc = next;
      }
    }
  }
  return out;
}

// Independent Hadamard reference: the transform is the Kronecker product
// H_m (x) W_2^k applied to each row, times the scale, where W_2 is the
// unnormalized [[1,1],[1,-1]].
std::vector<float> kron(
    const std::vector<float>& a,
    int am,
    int an,
    const std::vector<float>& b,
    int bm,
    int bn) {
  std::vector<float> out(static_cast<size_t>(am * bm * an * bn));
  for (int i = 0; i < am; ++i) {
    for (int j = 0; j < an; ++j) {
      for (int k = 0; k < bm; ++k) {
        for (int l = 0; l < bn; ++l) {
          out[static_cast<size_t>((i * bm + k) * (an * bn) + j * bn + l)] =
              a[static_cast<size_t>(i * an + j)] *
              b[static_cast<size_t>(k * bn + l)];
        }
      }
    }
  }
  return out;
}

std::vector<float> sylvester_power(int k) {
  std::vector<float> w{1.0f, 1.0f, 1.0f, -1.0f};
  std::vector<float> result{1.0f};
  int size = 1;
  for (int step = 0; step < k; ++step) {
    result = kron(result, size, size, w, 2, 2);
    size *= 2;
  }
  return result;
}

// The h12 matrix, parsed from the same string mlx/backend/common/hadamard.h
// embeds, so no hand transcription can drift. '+' is +1.
std::vector<float> hadamard_12() {
  static const char* rows[12] = {
      "+-++++++++++", "--+-+-+-+-+-", "+++-++----++", "+---+--+-++-",
      "+++++-++----", "+-+---+--+-+", "++--+++-++--", "+--++---+--+",
      "++----+++-++", "+--+-++---+-", "++++----+++-", "+-+--+-++---"};
  std::vector<float> matrix;
  for (int j = 0; j < 12; ++j) {
    for (int k = 0; k < 12; ++k) {
      matrix.push_back(rows[j][k] == '+' ? 1.0f : -1.0f);
    }
  }
  REQUIRE_EQ(matrix.size(), 144u);
  return matrix;
}

std::vector<float> hadamard_reference(
    const std::vector<float>& x,
    int rows,
    int n,
    int m,
    float scale) {
  std::vector<float> w = sylvester_power(n == 1 ? 0 : [&] {
    int k = 0;
    for (int v = n; v > 1; v /= 2) {
      ++k;
    }
    return k;
  }());
  std::vector<float> h;
  if (m == 1) {
    h = {1.0f};
  } else {
    h = hadamard_12();
  }
  std::vector<float> full = kron(h, m, m, w, n, n);
  std::vector<float> out(static_cast<size_t>(rows) * m * n);
  for (int row = 0; row < rows; ++row) {
    for (int i = 0; i < m * n; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < m * n; ++j) {
        sum += full[static_cast<size_t>(i * (m * n) + j)] *
            x[static_cast<size_t>(row) * m * n + j];
      }
      out[static_cast<size_t>(row) * m * n + i] = sum * scale;
    }
  }
  return out;
}

} // namespace

TEST_CASE("int32 Sum, Prod, Min, Max wrap exactly like the CPU contract") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Suffix reduction through the fast path stays exact for integers.
  array suffix(
      {1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}, int32);
  check_int32_values(
      sum(suffix, std::vector<int>{1, 2}, false, stream), {10, 26}, stream);
  check_int32_values(
      prod(suffix, std::vector<int>{1, 2}, false, stream), {24, 1680}, stream);
  check_int32_values(
      min(suffix, std::vector<int>{1, 2}, false, stream), {1, 5}, stream);
  check_int32_values(
      max(suffix, std::vector<int>{1, 2}, false, stream), {4, 8}, stream);

  // Overflow-adjacent sums: upstream accumulates int32 with wraparound.
  array edge({2147483647, 1}, {2}, int32);
  check_int32_values(
      sum(edge, std::vector<int>{0}, false, stream), {-2147483648}, stream);
  array triple({2147483647, 1, -2147483647 - 1}, {3}, int32);
  check_int32_values(
      sum(triple, std::vector<int>{0}, false, stream), {0}, stream);

  // Overflow-adjacent products: 2^16 * 2^16 wraps to 0.
  array square({65536, 65536}, {2}, int32);
  check_int32_values(
      prod(square, std::vector<int>{0}, false, stream), {0}, stream);
  array mixed({3, -5, 7}, {3}, int32);
  check_int32_values(
      prod(mixed, std::vector<int>{0}, false, stream), {-105}, stream);

  // Extremes participate in min and max.
  array extremes({0, -2147483647 - 1, 2147483647, 5}, {4}, int32);
  check_int32_values(
      min(extremes, std::vector<int>{0}, false, stream), {-2147483648},
      stream);
  check_int32_values(
      max(extremes, std::vector<int>{0}, false, stream), {2147483647},
      stream);
}

TEST_CASE("uint32 Sum, Prod, Min, Max order and wrap unsigned") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array values(
      {0u, 1u, 4294967295u, 2147483648u, 7u, 42u}, {2, 3}, uint32);

  check_uint32_values(
      sum(values, std::vector<int>{1}, false, stream),
      {0u, 2147483697u},
      stream);
  check_uint32_values(
      prod(values, std::vector<int>{1}, false, stream),
      {0u, 2147483648u * 7u * 42u % 4294967296u}, stream);
  check_uint32_values(
      min(values, std::vector<int>{1}, false, stream), {0u, 7u}, stream);
  check_uint32_values(
      max(values, std::vector<int>{1}, false, stream),
      {4294967295u, 2147483648u}, stream);

  array wrap({4294967295u, 1u}, {2}, uint32);
  check_uint32_values(
      sum(wrap, std::vector<int>{0}, false, stream), {0u}, stream);
  check_uint32_values(
      prod(wrap, std::vector<int>{0}, false, stream), {4294967295u}, stream);
}

TEST_CASE("integer reductions over leading and middle axes match host loops") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // (2, 3, 4) int32 with values 1..24.
  std::vector<int32_t> raw(24);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<int32_t>(i) + 1;
  }
  array grid(raw.begin(), Shape{2, 3, 4}, int32);

  // Axis 0 (leading): out[j][k] = sum_i (12*i + 4*j + k + 1).
  std::vector<int32_t> axis0;
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 4; ++k) {
      axis0.push_back(12 + 2 * (4 * j + k + 1));
    }
  }
  check_int32_values(
      sum(grid, std::vector<int>{0}, false, stream), axis0, stream);
  check_int32_values(
      sum(grid, std::vector<int>{0}, true, stream), axis0, stream);
  CHECK_EQ(
      sum(grid, std::vector<int>{0}, true, stream).shape(),
      Shape{1, 3, 4});

  // Axis 1 (middle): out[i][k] = sum_j (12*i + 4*j + k + 1).
  std::vector<int32_t> axis1;
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 4; ++k) {
      axis1.push_back(12 + 3 * (12 * i + k + 1));
    }
  }
  check_int32_values(
      sum(grid, std::vector<int>{1}, false, stream), axis1, stream);

  // Axis 2 stays exact through the general kernel for integers.
  std::vector<int32_t> axis2;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      int base = 12 * i + 4 * j;
      axis2.push_back(4 * base + 10);
    }
  }
  check_int32_values(
      sum(grid, std::vector<int>{2}, false, stream), axis2, stream);

  // uint32 over a leading axis.
  std::vector<uint32_t> uraw(24);
  for (size_t i = 0; i < uraw.size(); ++i) {
    uraw[i] = static_cast<uint32_t>(i) + 1;
  }
  array ugrid(uraw.begin(), Shape{2, 3, 4}, uint32);
  std::vector<uint32_t> uaxis0;
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 4; ++k) {
      uaxis0.push_back(static_cast<uint32_t>(12 + 2 * (4 * j + k + 1)));
    }
  }
  check_uint32_values(
      sum(ugrid, std::vector<int>{0}, false, stream), uaxis0, stream);
}

TEST_CASE("rank-4 float reductions hit every axis position and keepdims") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // (2, 3, 4, 5) with values 1..120.
  std::vector<float> raw(120);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<float>(i) + 1.0f;
  }
  array grid(raw.begin(), Shape{2, 3, 4, 5}, float32);
  auto at = [](int i, int j, int k, int l) {
    return static_cast<float>(((i * 3 + j) * 4 + k) * 5 + l + 1);
  };

  for (int axis = 0; axis < 4; ++axis) {
    std::vector<float> expected;
    int di[4] = {2, 3, 4, 5};
    di[axis] = 1;
    for (int i = 0; i < di[0]; ++i) {
      for (int j = 0; j < di[1]; ++j) {
        for (int k = 0; k < di[2]; ++k) {
          for (int l = 0; l < di[3]; ++l) {
            float total = 0.0f;
            int extent = axis == 0 ? 2 : axis == 1 ? 3 : axis == 2 ? 4 : 5;
            for (int t = 0; t < extent; ++t) {
              int r[4] = {i, j, k, l};
              r[axis] = t;
              total += at(r[0], r[1], r[2], r[3]);
            }
            expected.push_back(total);
          }
        }
      }
    }
    array reduced = sum(grid, std::vector<int>{axis}, false, stream);
    check_values(reduced, expected, stream);
    array kept = sum(grid, std::vector<int>{axis}, true, stream);
    Shape kept_shape = grid.shape();
    kept_shape[axis] = 1;
    CHECK_EQ(kept.shape(), kept_shape);
    check_values(kept, expected, stream);
  }

  // A non-suffix pair of axes in one pass.
  std::vector<float> pair_expected;
  for (int j = 0; j < 3; ++j) {
    for (int l = 0; l < 5; ++l) {
      float total = 0.0f;
      for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
          total += at(i, j, k, l);
        }
      }
      pair_expected.push_back(total);
    }
  }
  check_values(
      sum(grid, std::vector<int>{0, 2}, false, stream), pair_expected, stream);

  // Float Prod and Min over a leading axis route through the general
  // kernel too.
  std::vector<float> prod_expected;
  for (int j = 0; j < 3; ++j) {
    for (int l = 0; l < 5; ++l) {
      float total = 1.0f;
      for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
          total *= at(i, j, k, l);
        }
      }
      prod_expected.push_back(total);
    }
  }
  check_values(
      prod(grid, std::vector<int>{0, 2}, false, stream), prod_expected,
      stream);
}

TEST_CASE("float reductions over strided views match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f},
      {3, 4},
      float32);
  array flipped = transpose(base, stream);

  // Sum over axis 1 of the transpose (shape 4x3): one value per i, each
  // walking the physically strided column of the base.
  std::vector<float> column_sums;
  for (int i = 0; i < 4; ++i) {
    float total = 0.0f;
    for (int j = 0; j < 3; ++j) {
      total += static_cast<float>(j * 4 + i + 1);
    }
    column_sums.push_back(total);
  }
  check_values(
      sum(flipped, std::vector<int>{1}, false, stream), column_sums, stream);

  // A sliced view: rows 1..2, columns 1..3.
  array window = slice(base, {1, 1}, {3, 4}, {1, 1}, stream);
  std::vector<float> window_expected;
  for (int j = 1; j < 4; ++j) {
    float total = 0.0f;
    for (int i = 1; i < 3; ++i) {
      total += static_cast<float>(i * 4 + j + 1);
    }
    window_expected.push_back(total);
  }
  check_values(
      sum(window, std::vector<int>{0}, false, stream), window_expected,
      stream);
}

TEST_CASE("empty reductions return the upstream identity values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  array floats = zeros({2, 0, 3}, float32, stream);
  // Upstream throws in the op layer at construction time for min/max over
  // a size-0 axis (op_without_identity). Sum is 0 and prod is 1.
  auto construction_error = [](auto&& build) -> std::string {
    try {
      build();
    } catch (const std::exception& error) {
      return error.what();
    }
    return {};
  };
  CHECK(construction_error(
            [&] { return min(floats, std::vector<int>{1}, false, stream); })
            .find("Cannot min reduce") != std::string::npos);
  CHECK(construction_error(
            [&] { return max(floats, std::vector<int>{1}, false, stream); })
            .find("Cannot max reduce") != std::string::npos);

  // Integer identities: sum 0, prod 1.
  array ints = zeros({2, 0, 3}, int32, stream);
  check_int32_values(
      sum(ints, std::vector<int>{1}, false, stream),
      {0, 0, 0, 0, 0, 0}, stream);
  check_int32_values(
      prod(ints, std::vector<int>{1}, false, stream),
      {1, 1, 1, 1, 1, 1}, stream);
  CHECK(construction_error(
            [&] { return min(ints, std::vector<int>{1}, false, stream); })
            .find("Cannot min reduce") != std::string::npos);
  CHECK(construction_error(
            [&] { return max(ints, std::vector<int>{1}, false, stream); })
            .find("Cannot max reduce") != std::string::npos);
  array uints = zeros({2, 0, 3}, uint32, stream);
  CHECK(construction_error(
            [&] { return min(uints, std::vector<int>{1}, false, stream); })
            .find("Cannot min reduce") != std::string::npos);
  CHECK(construction_error(
            [&] { return max(uints, std::vector<int>{1}, false, stream); })
            .find("Cannot max reduce") != std::string::npos);

  // Any is false and all is true over an empty axis.
  array flags = zeros({2, 0, 3}, bool_, stream);
  check_bool_values(
      any(flags, std::vector<int>{1}, false, stream),
      {false, false, false, false, false, false}, stream);
  check_bool_values(
      all(flags, std::vector<int>{1}, false, stream),
      {true, true, true, true, true, true}, stream);

  // keepdims keeps the reduced axis at size 1.
  CHECK_EQ(sum(floats, std::vector<int>{1}, true, stream).shape(),
           Shape{2, 1, 3});
  check_values(
      sum(floats, std::vector<int>{1}, true, stream),
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, stream);
}

TEST_CASE("any and all reduce bool, integer, and float truthiness") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  std::vector<bool> flag_values = {true, false, true, false};
  array flags(flag_values.begin(), Shape{4}, bool_);
  check_bool_values(
      any(flags, std::vector<int>{0}, false, stream), {true}, stream);
  check_bool_values(
      all(flags, std::vector<int>{0}, false, stream), {false}, stream);
  std::vector<bool> none_values = {false, false};
  array none(none_values.begin(), Shape{2}, bool_);
  check_bool_values(
      any(none, std::vector<int>{0}, false, stream), {false}, stream);
  std::vector<bool> every_values = {true, true, true};
  array every(every_values.begin(), Shape{3}, bool_);
  check_bool_values(
      all(every, std::vector<int>{0}, false, stream), {true}, stream);

  // Non-word-multiple sizes exercise the partial-word bool write.
  std::vector<bool> five_values = {false, true, false, false, true};
  array five(five_values.begin(), Shape{5}, bool_);
  check_bool_values(
      any(five, std::vector<int>{0}, false, stream), {true}, stream);
  check_bool_values(
      all(five, std::vector<int>{0}, false, stream), {false}, stream);

  // Integer truthiness includes negative values.
  array ints({0, -3, 0}, {3}, int32);
  check_bool_values(
      any(ints, std::vector<int>{0}, false, stream), {true}, stream);
  check_bool_values(
      all(ints, std::vector<int>{0}, false, stream), {false}, stream);
  array uints({0u, 7u}, {2}, uint32);
  check_bool_values(
      any(uints, std::vector<int>{0}, false, stream), {true}, stream);

  // Float truthiness: -0.0 is zero, NaN is truthy.
  array zeros({0.0f, -0.0f}, {2}, float32);
  check_bool_values(
      any(zeros, std::vector<int>{0}, false, stream), {false}, stream);
  array nan_mixed({0.0f, std::nanf("")}, {2}, float32);
  check_bool_values(
      any(nan_mixed, std::vector<int>{0}, false, stream), {true}, stream);

  // Over a middle axis with keepdims.
  std::vector<bool> grid_values = {true, false, false, true};
  array grid(grid_values.begin(), Shape{2, 2}, bool_);
  check_bool_values(
      any(grid, std::vector<int>{1}, true, stream), {true, true}, stream);
}

TEST_CASE("float scans match host references across all flag combinations") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> row{3.0f, -1.0f, 2.0f, 0.5f, -2.0f, 4.0f};
  array x(row.begin(), Shape{6}, float32);
  const char* op_names[4] = {"sum", "prod", "min", "max"};

  for (int op = 0; op < 4; ++op) {
    for (int reverse = 0; reverse < 2; ++reverse) {
      for (int inclusive = 0; inclusive < 2; ++inclusive) {
        auto expected =
            host_scan(row, 1, 6, op, reverse != 0, inclusive != 0);
        array scanned = op == 0
            ? cumsum(x, 0, reverse != 0, inclusive != 0, stream)
            : op == 1
            ? cumprod(x, 0, reverse != 0, inclusive != 0, stream)
            : op == 2
            ? cummin(x, 0, reverse != 0, inclusive != 0, stream)
            : cummax(x, 0, reverse != 0, inclusive != 0, stream);
        INFO(
            "op=", op_names[op], " reverse=", reverse,
            " inclusive=", inclusive);
        check_values(scanned, expected, stream, 1e-6);
      }
    }
  }

  // Multi-row arrays scan independently along a middle axis.
  std::vector<float> grid_raw;
  for (int i = 0; i < 24; ++i) {
    grid_raw.push_back(
        static_cast<float>((static_cast<int>(i) * 7) % 11 - 5));
  }
  array grid(grid_raw.begin(), Shape{2, 3, 4}, float32);
  // Host reference in output memory order, exclusive variants writing the
  // accumulator before it absorbs the current element.
  std::vector<float> expected_sum(24);
  std::vector<float> expected_max(24);
  std::vector<float> expected_min(24);
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 4; ++k) {
      float sum_acc = 0.0f;
      // cumsum forward inclusive.
      for (int j = 0; j < 3; ++j) {
        sum_acc += grid_raw[(i * 3 + j) * 4 + k];
        expected_sum[static_cast<size_t>((i * 3 + j) * 4 + k)] = sum_acc;
      }
      // cummax reverse exclusive: running value covers later elements.
      float max_running = -std::numeric_limits<float>::infinity();
      for (int j = 2; j >= 0; --j) {
        size_t flat = static_cast<size_t>((i * 3 + j) * 4 + k);
        expected_max[flat] = max_running;
        max_running =
            std::max(max_running, grid_raw[(i * 3 + j) * 4 + k]);
      }
      // cummin forward exclusive: running value covers earlier elements.
      float min_running = std::numeric_limits<float>::infinity();
      for (int j = 0; j < 3; ++j) {
        size_t flat = static_cast<size_t>((i * 3 + j) * 4 + k);
        expected_min[flat] = min_running;
        min_running =
            std::min(min_running, grid_raw[(i * 3 + j) * 4 + k]);
      }
    }
  }
  check_values(cumsum(grid, 1, false, true, stream), expected_sum, stream, 1e-6);
  check_values(cummax(grid, 1, true, false, stream), expected_max, stream, 1e-6);
  check_values(cummin(grid, 1, false, false, stream), expected_min, stream, 1e-6);
}

TEST_CASE("int32 scans wrap and stay exact") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x({2147483647, 1, -5}, {3}, int32);
  check_int32_values(
      cumsum(x, 0, false, true, stream),
      {2147483647, -2147483648, 2147483643}, stream);
  check_int32_values(
      cumsum(x, 0, false, false, stream), {0, 2147483647, -2147483648},
      stream);
  array values({1, 2, 3}, {3}, int32);
  check_int32_values(
      cumsum(values, 0, true, false, stream), {5, 3, 0}, stream);
  check_int32_values(
      cumprod(values, 0, true, true, stream), {6, 6, 3}, stream);
}

TEST_CASE("scans walk strided views correctly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f},
      {3, 4},
      float32);
  array flipped = transpose(base, stream);

  // cumsum along axis 0 of the transpose walks the physically strided
  // columns of the base.
  std::vector<float> expected;
  float running[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      running[j] += static_cast<float>(j * 4 + i + 1);
      expected.push_back(running[j]);
    }
  }
  check_values(cumsum(flipped, 0, false, true, stream), expected, stream);

  // A sliced window scans its own values.
  array window = slice(base, {1, 1}, {3, 4}, {1, 1}, stream);
  std::vector<float> window_expected;
  for (int i = 1; i < 3; ++i) {
    float running = 0.0f;
    for (int j = 1; j < 4; ++j) {
      running += static_cast<float>(i * 4 + j + 1);
      window_expected.push_back(running);
    }
  }
  check_values(cumsum(window, 1, false, true, stream), window_expected,
               stream);
}

TEST_CASE("hadamard matches an independent Kronecker reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Sylvester sizes 2, 4, 8 with an explicit scale.
  for (int n : {2, 4, 8}) {
    std::vector<float> raw(static_cast<size_t>(n));
    for (size_t i = 0; i < raw.size(); ++i) {
      raw[i] = static_cast<float>(i) - static_cast<float>(n) / 2.0f;
    }
    array x(raw.begin(), Shape{n}, float32);
    auto expected = hadamard_reference(raw, 1, n, 1, 2.0f);
    check_values(hadamard_transform(x, 2.0f, stream), expected, stream, 1e-5);
  }

  // A 2-D batch transforms every row.
  std::vector<float> batch_raw(16);
  for (size_t i = 0; i < batch_raw.size(); ++i) {
    batch_raw[i] = static_cast<float>(i % 7) - 3.0f;
  }
  array batch(batch_raw.begin(), Shape{2, 8}, float32);
  auto batch_expected = hadamard_reference(batch_raw, 2, 8, 1, 2.0f);
  check_values(
      hadamard_transform(batch, 2.0f, stream), batch_expected, stream, 1e-5);

  // The default scale is 1/sqrt(N).
  std::vector<float> raw8{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  array x8(raw8.begin(), Shape{8}, float32);
  auto scaled_expected = hadamard_reference(
      raw8, 1, 8, 1, 1.0f / std::sqrt(8.0f));
  check_values(hadamard_transform(x8, std::nullopt, stream), scaled_expected,
               stream, 1e-5);

  // Power of two large enough to cross multiple workgroup dispatches.
  int n = 1024;
  std::vector<float> wide_raw(static_cast<size_t>(n));
  for (size_t i = 0; i < wide_raw.size(); ++i) {
    wide_raw[i] = static_cast<float>(i % 17) - 8.0f;
  }
  array wide(wide_raw.begin(), Shape{n}, float32);
  auto wide_expected = hadamard_reference(wide_raw, 1, n, 1, 1.0f);
  check_values(
      hadamard_transform(wide, 1.0f, stream), wide_expected, stream, 1e-4);
}

TEST_CASE("hadamard order-12 rows match the reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // N = 12: pure H_12 (n = 1).
  std::vector<float> raw12(12);
  for (size_t i = 0; i < raw12.size(); ++i) {
    raw12[i] = static_cast<float>(i) - 5.5f;
  }
  array x12(raw12.begin(), Shape{12}, float32);
  auto expected12 = hadamard_reference(raw12, 1, 1, 12, 1.0f);
  check_values(hadamard_transform(x12, 1.0f, stream), expected12, stream,
               1e-4);

  // N = 48 = 12 * 4: the mixed component transform.
  std::vector<float> raw48(48);
  for (size_t i = 0; i < raw48.size(); ++i) {
    raw48[i] = static_cast<float>(i % 9) - 4.0f;
  }
  array x48(raw48.begin(), Shape{48}, float32);
  auto expected48 = hadamard_reference(raw48, 1, 4, 12, 1.0f);
  check_values(hadamard_transform(x48, 1.0f, stream), expected48, stream,
               1e-4);
}

TEST_CASE("FP16 hadamard matches the reference within half precision") {
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
  std::vector<float> raw{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  array half = astype(array(raw.begin(), Shape{8}, float32), float16, stream);
  auto expected = hadamard_reference(raw, 1, 8, 1, 1.0f);
  check_values(
      astype(hadamard_transform(half, 1.0f, stream), float32, stream),
      expected,
      stream,
      1e-2);
}

TEST_CASE("out-of-scope dtypes and shapes keep their named errors") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // int64 reductions stay rejected.
  array wide({1, 2, 3}, {3}, int64);
  CHECK(evaluation_error(prod(wide, std::vector<int>{0}, false, stream))
            .find("Prod dtype") != std::string::npos);

  // bool Sum is not in the wave scope: the op layer promotes the output to
  // int32 but the primitive reads bool, so the dtype gate rejects it.
  std::vector<bool> flag_values = {true, false};
  array flags(flag_values.begin(), Shape{2}, bool_);
  CHECK(evaluation_error(sum(flags, std::vector<int>{0}, false, stream))
            .find("Sum dtype") != std::string::npos);

  // LogAddExp scans stay rejected.
  array x({1.0f, 2.0f}, {2}, float32);
  CHECK(evaluation_error(logcumsumexp(x, 0, false, true, stream))
            .find("Scan LogAddExp") != std::string::npos);

  // Hadamard only supports n = m*2^k for m in (1, 12, 20, 28).
  std::vector<float> bad_values(7, 1.0f);
  array bad(bad_values.begin(), Shape{7}, float32);
  CHECK(evaluation_error(hadamard_transform(bad, std::nullopt, stream))
            .find("Hadamard size") != std::string::npos);

  // Rank above the push-constant array cap stays a named error for the
  // integer general path.
  std::vector<int32_t> deep(32, 1);
  array grid5(deep.begin(), Shape{2, 2, 2, 2, 2}, int32);
  CHECK(evaluation_error(sum(grid5, std::vector<int>{0}, false, stream))
            .find("Sum rank above 4") != std::string::npos);
}

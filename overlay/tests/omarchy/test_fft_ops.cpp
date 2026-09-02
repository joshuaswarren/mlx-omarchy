// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 8 FFT coverage: complex-to-complex (fftn/ifftn), real-to-complex
// (rfftn) and complex-to-real (irfftn) over power-of-two lengths 2 to 2048,
// batched and on chosen axes of multi-dimensional arrays. Every value test
// compares against a naive O(n^2) DFT computed in double precision inside
// this file, plus analytic cases (delta, constant, single sinusoid),
// forward/inverse round-trips, Parseval's identity, and the n/2+1 output
// shape of the real variants.
//
// Tolerance: 2e-4 relative to the reference infinity norm, with a 1e-5
// absolute floor for near-zero references. The kernel computes twiddles on
// the fly in float32 (single turn around the circle per stage), so the
// error grows like eps * log2(n) with observed maxima around 1e-5 at
// n = 2048; 2e-4 leaves an order of magnitude of margin while still
// failing on a wrong bin, a missed 1/n scale, or a transposed axis, each
// of which lands orders of magnitude outside. Zero bins of analytic cases
// scale their tolerance with n for the same reason.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;
using cdouble = std::complex<double>;
using namespace mlx::core::fft;

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

// Naive O(n^2) DFT in double precision. Forward: X[k] = sum_j x[j]
// e^(-2*pi*i*j*k/n). Inverse carries the 1/n scale, matching the backward
// norm the CPU primitive applies per axis.
std::vector<cdouble> naive_dft(const std::vector<cdouble>& x, bool inverse) {
  size_t n = x.size();
  std::vector<cdouble> X(n);
  for (size_t k = 0; k < n; ++k) {
    cdouble sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      double angle = (inverse ? 2.0 : -2.0) * M_PI * double(j) * double(k) /
          double(n);
      sum += x[j] * std::polar(1.0, angle);
    }
    X[k] = inverse ? sum / double(n) : sum;
  }
  return X;
}

// Naive 2-D DFT over both axes of a row-major (rows, cols) array: one
// 1-D pass along rows, then one along columns. Order is irrelevant in
// exact arithmetic, which is the point of an honest reference.
std::vector<cdouble> naive_dft2(
    const std::vector<cdouble>& in,
    size_t rows,
    size_t cols,
    bool inverse) {
  std::vector<cdouble> tmp(in.size());
  for (size_t r = 0; r < rows; ++r) {
    std::vector<cdouble> row(in.begin() + r * cols, in.begin() + (r + 1) * cols);
    auto transformed = naive_dft(row, inverse);
    for (size_t c = 0; c < cols; ++c) {
      tmp[r * cols + c] = transformed[c];
    }
  }
  std::vector<cdouble> out(in.size());
  for (size_t c = 0; c < cols; ++c) {
    std::vector<cdouble> col(rows);
    for (size_t r = 0; r < rows; ++r) {
      col[r] = tmp[r * cols + c];
    }
    auto transformed = naive_dft(col, inverse);
    for (size_t r = 0; r < rows; ++r) {
      out[r * cols + c] = transformed[r];
    }
  }
  return out;
}

std::vector<cdouble> random_complex(size_t n, std::mt19937& gen) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<cdouble> x(n);
  for (auto& value : x) {
    value = {dist(gen), dist(gen)};
  }
  return x;
}

std::vector<double> random_real(size_t n, std::mt19937& gen) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(n);
  for (auto& value : x) {
    value = dist(gen);
  }
  return x;
}

array complex_array(const std::vector<cdouble>& v, Shape shape) {
  std::vector<complex64_t> host(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    host[i] = complex64_t(float(v[i].real()), float(v[i].imag()));
  }
  return array(host.begin(), std::move(shape), complex64);
}

array real_array(const std::vector<double>& v, Shape shape) {
  std::vector<float> host(v.begin(), v.end());
  return array(host.begin(), std::move(shape), float32);
}

std::vector<cdouble> read_complex(array a, const Stream& stream) {
  a.eval();
  sync(stream);
  const complex64_t* data = a.data<complex64_t>();
  std::vector<cdouble> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = {double(data[i].real()), double(data[i].imag())};
  }
  return out;
}


double inf_norm(const std::vector<cdouble>& v) {
  double norm = 0.0;
  for (const auto& value : v) {
    norm = std::max(norm, std::abs(value));
  }
  return norm;
}

double max_abs_diff(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref) {
  double diff = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    diff = std::max(diff, std::abs(got[i] - ref[i]));
  }
  return diff;
}

double max_real_diff(
    const std::vector<double>& got,
    const std::vector<double>& ref) {
  double diff = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    diff = std::max(diff, std::abs(got[i] - ref[i]));
  }
  return diff;
}

std::vector<double> read_real(array a, const Stream& stream) {
  a.eval();
  sync(stream);
  const float* data = a.data<float>();
  std::vector<double> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = double(data[i]);
  }
  return out;
}

void require_close(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref,
    double relative = 2e-4) {
  double tolerance = std::max(relative * inf_norm(ref), 1e-5);
  double diff = max_abs_diff(got, ref);
  REQUIRE_MESSAGE(
      diff <= tolerance,
      "max abs diff " << diff << " exceeds tolerance " << tolerance);
}

std::string evaluation_error(const std::function<void()>& build) {
  try {
    build();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

} // namespace

TEST_CASE("fftn delta and constant analytic cases") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t n = 16;

  // A delta transforms to a flat spectrum of ones.
  std::vector<cdouble> delta(n, {0.0, 0.0});
  delta[0] = {1.0, 0.0};
  auto flat = read_complex(fftn(complex_array(delta, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  std::vector<cdouble> ones(n, {1.0, 0.0});
  require_close(flat, ones, 1e-6);

  // A constant transforms to a single DC bin of n.
  std::vector<cdouble> constant(n, {1.0, 0.0});
  auto spectrum = read_complex(fftn(complex_array(constant, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  std::vector<cdouble> dc(n, {0.0, 0.0});
  dc[0] = {double(n), 0.0};
  require_close(spectrum, dc, 1e-6);
}

TEST_CASE("fftn single sinusoid lands in exactly one bin") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t n = 64;
  const size_t bin = 5;

  // exp(2*pi*i*k*j/n) transforms to n at bin k and zero elsewhere.
  std::vector<cdouble> tone(n);
  for (size_t j = 0; j < n; ++j) {
    tone[j] = std::polar(1.0, 2.0 * M_PI * double(bin) * double(j) / double(n));
  }
  auto spectrum = read_complex(fftn(complex_array(tone, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  for (size_t k = 0; k < n; ++k) {
    double expected = k == bin ? double(n) : 0.0;
    double tolerance = k == bin ? 0.01 * double(n) : 5e-3 * double(n);
    REQUIRE_MESSAGE(
        std::abs(spectrum[k] - cdouble(expected, 0.0)) <= tolerance,
        "bin " << k << " off: got " << spectrum[k]);
  }

  // A real cosine splits evenly between bins k and n-k.
  std::vector<cdouble> cosine(n);
  for (size_t j = 0; j < n; ++j) {
    cosine[j] = {std::cos(2.0 * M_PI * double(bin) * double(j) / double(n)), 0.0};
  }
  auto real_spectrum = read_complex(fftn(complex_array(cosine, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  REQUIRE_MESSAGE(std::abs(real_spectrum[bin] - cdouble(n / 2, 0.0)) <= 0.01 * n, "bin k wrong");
  REQUIRE_MESSAGE(
      std::abs(real_spectrum[n - bin] - cdouble(n / 2, 0.0)) <= 0.01 * n,
      "bin n-k wrong");
}

TEST_CASE("fftn matches naive DFT across length classes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(8);
  for (size_t n : {size_t(2), size_t(4), size_t(8), size_t(16), size_t(64),
                   size_t(256), size_t(1024), size_t(2048)}) {
    auto x = random_complex(n, gen);
    auto got = read_complex(fftn(complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream), stream);
    auto ref = naive_dft(x, false);
    require_close(got, ref);
  }
}

TEST_CASE("ifftn matches naive inverse including the 1/n scale") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(19);
  for (size_t n : {size_t(2), size_t(8), size_t(64), size_t(256), size_t(2048)}) {
    auto X = random_complex(n, gen);
    auto got = read_complex(ifftn(complex_array(X, Shape{int(n)}), FFTNorm::Backward, stream), stream);
    auto ref = naive_dft(X, true);
    require_close(got, ref);
  }
}

TEST_CASE("fftn then ifftn round-trips to the input") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(27);
  for (size_t n : {size_t(8), size_t(128), size_t(2048)}) {
    auto x = random_complex(n, gen);
    auto fwd = fftn(complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream);
    auto back = read_complex(ifftn(fwd, FFTNorm::Backward, stream), stream);
    require_close(back, x, 1e-4);
  }
}

TEST_CASE("Parseval identity: sum |x|^2 = (1/n) sum |X|^2") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(33);
  const size_t n = 64;
  auto x = random_complex(n, gen);
  auto X = read_complex(fftn(complex_array(x, Shape{int(n)}), FFTNorm::Backward, stream), stream);
  double time_energy = 0.0;
  for (const auto& value : x) {
    time_energy += std::norm(value);
  }
  double freq_energy = 0.0;
  for (const auto& value : X) {
    freq_energy += std::norm(value);
  }
  double expected = freq_energy / double(n);
  double tolerance = 1e-4 * std::max(1.0, expected);
  REQUIRE_MESSAGE(
      std::abs(time_energy - expected) <= tolerance,
      "Parseval violated: time " << time_energy << " vs freq/n " << expected);
}

TEST_CASE("multi-axis fftn over axes {0,1} matches naive 2-D reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(41);
  const size_t rows = 4;
  const size_t cols = 16;
  auto x = random_complex(rows * cols, gen);

  auto got = read_complex(
      fftn(complex_array(x, Shape{int(rows), int(cols)}), {0, 1}, FFTNorm::Backward, stream), stream);
  auto ref = naive_dft2(x, rows, cols, false);
  require_close(got, ref);

  // Axis order must not change the result.
  auto reordered = read_complex(
      fftn(complex_array(x, Shape{int(rows), int(cols)}), {1, 0}, FFTNorm::Backward, stream), stream);
  double order_diff = max_abs_diff(got, reordered);
  double order_tolerance = 1e-5 * std::max(1.0, inf_norm(ref));
  REQUIRE_MESSAGE(order_diff <= order_tolerance, "axis order changed the result");

  // Round-trip.
  auto back = read_complex(
      ifftn(fftn(complex_array(x, Shape{int(rows), int(cols)}), {0, 1}, FFTNorm::Backward, stream),
            {0, 1},
            FFTNorm::Backward,
            stream),
      stream);
  require_close(back, x, 1e-4);
}

TEST_CASE("fftn over a middle axis uses strided samples") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(47);
  const Shape shape{2, 8, 4};
  auto x = random_complex(2 * 8 * 4, gen);
  auto got = read_complex(fftn(complex_array(x, shape), {1}, FFTNorm::Backward, stream), stream);

  // Reference: for each fixed (i0, i2), naive DFT of the 8 samples down
  // the middle axis.
  for (size_t i0 = 0; i0 < 2; ++i0) {
    for (size_t i2 = 0; i2 < 4; ++i2) {
      std::vector<cdouble> column(8);
      for (size_t k = 0; k < 8; ++k) {
        column[k] = x[(i0 * 8 + k) * 4 + i2];
      }
      auto ref = naive_dft(column, false);
      for (size_t k = 0; k < 8; ++k) {
        double tolerance = std::max(2e-4 * std::abs(ref[k]), 1e-5);
        REQUIRE_MESSAGE(
            std::abs(got[(i0 * 8 + k) * 4 + i2] - ref[k]) <= tolerance,
            "middle-axis bin mismatch at (" << i0 << "," << k << "," << i2 << ")");
      }
    }
  }
}

TEST_CASE("batched last-axis fftn transforms each row independently") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(53);
  const size_t batch = 3;
  const size_t n = 16;
  auto x = random_complex(batch * n, gen);
  auto got = read_complex(fftn(complex_array(x, Shape{int(batch), int(n)}), {-1}, FFTNorm::Backward, stream), stream);
  for (size_t b = 0; b < batch; ++b) {
    std::vector<cdouble> row(x.begin() + b * n, x.begin() + (b + 1) * n);
    auto ref = naive_dft(row, false);
    std::vector<cdouble> got_row(got.begin() + b * n, got.begin() + (b + 1) * n);
    require_close(got_row, ref);
  }
}

TEST_CASE("rfftn returns n/2+1 bins matching the full DFT") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(59);
  for (size_t n : {size_t(8), size_t(64), size_t(256), size_t(2048)}) {
    auto x = random_real(n, gen);
    auto result = rfftn(real_array(x, Shape{int(n)}), FFTNorm::Backward, stream);
    REQUIRE_EQ(result.shape(0), int(n / 2 + 1));
    auto got = read_complex(result, stream);

    std::vector<cdouble> complex_input(n);
    for (size_t i = 0; i < n; ++i) {
      complex_input[i] = {x[i], 0.0};
    }
    auto full = naive_dft(complex_input, false);
    std::vector<cdouble> half(full.begin(), full.begin() + n / 2 + 1);
    require_close(got, half);

    // The dropped bins must equal the conjugates of the kept ones: the
    // half spectrum fully determines the transform of a real signal.
    for (size_t k = 1; k < n / 2; ++k) {
      REQUIRE_MESSAGE(
          std::abs(full[n - k] - std::conj(got[k])) <=
              std::max(2e-4 * std::abs(full[n - k]), 1e-5),
          "conjugate symmetry broken at bin " << k);
    }
  }
}

TEST_CASE("irfftn recovers the real signal from the half spectrum") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(61);
  for (size_t n : {size_t(8), size_t(64), size_t(256), size_t(2048)}) {
    auto x = random_real(n, gen);
    std::vector<cdouble> complex_input(n);
    for (size_t i = 0; i < n; ++i) {
      complex_input[i] = {x[i], 0.0};
    }
    auto full = naive_dft(complex_input, false);
    std::vector<cdouble> half(full.begin(), full.begin() + n / 2 + 1);

    auto got = read_real(irfftn(complex_array(half, Shape{int(n / 2 + 1)}), FFTNorm::Backward, stream), stream);
    REQUIRE_EQ(got.size(), n);
    double diff = max_real_diff(got, x);
    double tolerance = 1e-4 * std::max(1.0, inf_norm(half));
    REQUIRE_MESSAGE(diff <= tolerance, "irfft mismatch: " << diff);
  }
}

TEST_CASE("rfftn then irfftn round-trips a real signal") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(67);
  for (size_t n : {size_t(8), size_t(128), size_t(2048)}) {
    auto x = random_real(n, gen);
    auto fwd = rfftn(real_array(x, Shape{int(n)}), FFTNorm::Backward, stream);
    auto back = read_real(irfftn(fwd, FFTNorm::Backward, stream), stream);
    double diff = max_real_diff(back, x);
    REQUIRE_MESSAGE(
        diff <= 1e-4 * std::max(1.0, *std::max_element(x.begin(), x.end())),
        "round-trip mismatch: " << diff);
  }
}

TEST_CASE("rfftn over axes {0,1} keeps n/2+1 on the real axis") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(71);
  const size_t rows = 4;
  const size_t cols = 8;
  auto x = random_real(rows * cols, gen);

  auto result = rfftn(real_array(x, Shape{int(rows), int(cols)}), {0, 1}, FFTNorm::Backward, stream);
  REQUIRE_EQ(result.shape(0), int(rows));
  REQUIRE_EQ(result.shape(1), int(cols / 2 + 1));
  auto got = read_complex(result, stream);

  std::vector<cdouble> complex_input(rows * cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    complex_input[i] = {x[i], 0.0};
  }
  auto full = naive_dft2(complex_input, rows, cols, false);
  std::vector<cdouble> ref;
  for (size_t r = 0; r < rows; ++r) {
    ref.insert(ref.end(), full.begin() + r * cols, full.begin() + r * cols + cols / 2 + 1);
  }
  require_close(got, ref);

  // And the inverse returns the original signal.
  auto back = read_real(irfftn(result, {0, 1}, FFTNorm::Backward, stream), stream);
  double diff = max_real_diff(back, x);
  REQUIRE_MESSAGE(diff <= 1e-4 * std::max(1.0, inf_norm(ref)), "2-D irfft mismatch");
}

TEST_CASE("fft2 and ifft2 route through the same primitive") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(73);
  const size_t rows = 4;
  const size_t cols = 8;
  auto x = random_complex(rows * cols, gen);
  auto input = complex_array(x, Shape{int(rows), int(cols)});
  auto via_fft2 = read_complex(fft2(input, {0, 1}, FFTNorm::Backward, stream), stream);
  auto via_fftn = read_complex(fftn(input, {0, 1}, FFTNorm::Backward, stream), stream);
  require_close(via_fft2, via_fftn, 1e-6);

  auto spec = fftn(input, {0, 1}, FFTNorm::Backward, stream);
  auto back = read_complex(ifft2(spec, {0, 1}, FFTNorm::Backward, stream), stream);
}

TEST_CASE("non-power-of-two and oversized lengths keep the named error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(79);

  auto x12 = random_complex(12, gen);
  std::string length_error = evaluation_error([&] {
    auto y = fftn(complex_array(x12, Shape{12}), FFTNorm::Backward, stream);
    y.eval();
    sync(stream);
  });
  CHECK(length_error.find("[omarchy] FFT") != std::string::npos);
  CHECK(length_error.find("transform length 12") != std::string::npos);
  CHECK(length_error.find("Bluestein") != std::string::npos);

  auto x4096 = random_complex(4096, gen);
  std::string oversized_error = evaluation_error([&] {
    auto y = fftn(complex_array(x4096, Shape{4096}), FFTNorm::Backward, stream);
    y.eval();
    sync(stream);
  });
  CHECK(oversized_error.find("transform length 4096") != std::string::npos);

  auto r12 = random_real(12, gen);
  std::string rfft_error = evaluation_error([&] {
    auto y = rfftn(real_array(r12, Shape{12}), Shape{12}, {0}, FFTNorm::Backward, stream);
    y.eval();
    sync(stream);
  });
  CHECK(rfft_error.find("transform length 12") != std::string::npos);
}

TEST_CASE("explicit n pads with zeros or slices the input") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(83);

  // n larger than the input: the op layer zero-pads before the FFT.
  // (The complex pad route goes through SliceUpdate, which has no
  // complex64 strided-copy path; the float32 rfft route exercises the
  // same padded dense input into the FFT primitive.)
  auto x4 = random_real(4, gen);
  auto padded = read_complex(
      rfftn(real_array(x4, Shape{4}), Shape{8}, {0}, FFTNorm::Backward, stream),
      stream);
  REQUIRE_EQ(padded.size(), size_t(5));
  std::vector<cdouble> padded_ref(8, {0.0, 0.0});
  for (size_t i = 0; i < 4; ++i) {
    padded_ref[i] = {x4[i], 0.0};
  }
  auto padded_full = naive_dft(padded_ref, false);
  std::vector<cdouble> padded_half(padded_full.begin(), padded_full.begin() + 5);
  require_close(padded, padded_half);

  // n smaller than the input: the op layer slices, the eval layer
  // materializes the slice, and the FFT sees a dense float32 buffer.
  auto x8 = random_real(8, gen);
  auto truncated = read_complex(
      rfftn(real_array(x8, Shape{8}), Shape{4}, {0}, FFTNorm::Backward, stream),
      stream);
  REQUIRE_EQ(truncated.size(), size_t(3));
  std::vector<cdouble> trunc_input(4);
  for (size_t i = 0; i < 4; ++i) {
    trunc_input[i] = {x8[i], 0.0};
  }
  auto trunc_full = naive_dft(trunc_input, false);
  std::vector<cdouble> trunc_ref(trunc_full.begin(), trunc_full.begin() + 3);
  require_close(truncated, trunc_ref);
}

TEST_CASE("rfft on a non-trailing axis keeps its named error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(89);
  auto x = random_real(64, gen);

  std::string axis_error = evaluation_error([&] {
    auto y = rfftn(real_array(x, Shape{4, 4, 4}), {0, 1}, FFTNorm::Backward, stream);
    y.eval();
    sync(stream);
  });
  CHECK(axis_error.find("[omarchy] FFT") != std::string::npos);
  CHECK(axis_error.find("rfft on a non-trailing axis") != std::string::npos);
  // irfftn mirrors the same gate: axes {0,1} on a (4,5,4) half spectrum
  // (n = 8 on the middle axis) makes the real axis non-trailing.
  std::mt19937 gen_irfft(97);
  auto spectrum = random_complex(4 * 5 * 4, gen_irfft);
  std::string irfft_error = evaluation_error([&] {
    auto y = irfftn(complex_array(spectrum, Shape{4, 5, 4}), {0, 1}, FFTNorm::Backward, stream);
    y.eval();
    sync(stream);
  });
  CHECK(irfft_error.find("[omarchy] FFT") != std::string::npos);
  CHECK(irfft_error.find("irfft on a non-trailing axis") != std::string::npos);
}

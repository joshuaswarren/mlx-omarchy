// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Complex64Transport coverage: complex64 transport (allocation, scalar
// fill, same-dtype strided copies, the zero-copy view primitives),
// dtype casts in both directions against the upstream static_cast
// rules, Conjugate/Real/Imag on complex input, complex add/subtract/
// multiply/divide/negate, and the pad/concatenate/FFT paths that
// previously refused complex64 outright.
//
// Value policy: transport and casts compare bit-exact float32
// components against host references (a wrong value cannot hide, and
// every path here is pure data movement or an exact promotion).
// Multiply and divide compare against std::complex<double> references
// computed in this file at 1e-5 relative tolerance with a 1e-6
// absolute floor: float32 arithmetic on magnitudes <= 4 carries
// ~1e-7 relative error, so 1e-5 fails on a swapped component, a
// dropped conjugate, or a real/imag transposition, each of which
// lands orders of magnitude outside.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
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

std::vector<cdouble> random_complex(size_t n, std::mt19937& gen) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<cdouble> x(n);
  for (auto& value : x) {
    value = {dist(gen), dist(gen)};
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
  auto dense = contiguous(a);
  dense.eval();
  sync(stream);
  const complex64_t* data = dense.data<complex64_t>();
  std::vector<cdouble> out(dense.size());
  for (size_t i = 0; i < dense.size(); ++i) {
    out[i] = {double(data[i].real()), double(data[i].imag())};
  }
  return out;
}

std::vector<double> read_real(array a, const Stream& stream) {
  auto dense = contiguous(a);
  dense.eval();
  sync(stream);
  const float* data = dense.data<float>();
  return std::vector<double>(data, data + dense.size());
}

// Exact component comparison at float32 precision: transport carries
// float32 words unchanged, so the reference quantized through float32
// must equal the readback bit for bit.
void check_exact(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref) {
  REQUIRE_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    float want_re = float(ref[i].real());
    float want_im = float(ref[i].imag());
    INFO("index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") ref (", want_re, ", ", want_im, ")");
    CHECK_EQ(got[i].real(), want_re);
    CHECK_EQ(got[i].imag(), want_im);
  }
}

void check_exact_real(
    const std::vector<double>& got,
    const std::vector<double>& ref) {
  REQUIRE_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    float want = float(ref[i]);
    INFO("index ", i, " got ", got[i], " ref ", want);
    CHECK_EQ(got[i], want);
  }
}

void check_close(
    const std::vector<cdouble>& got,
    const std::vector<cdouble>& ref,
    double tol) {
  REQUIRE_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    double dr = std::abs(got[i].real() - ref[i].real());
    double di = std::abs(got[i].imag() - ref[i].imag());
    double scale_r = std::max(1.0, std::abs(ref[i].real()));
    double scale_i = std::max(1.0, std::abs(ref[i].imag()));
    INFO("index ", i, " got (", got[i].real(), ", ", got[i].imag(),
         ") ref (", ref[i].real(), ", ", ref[i].imag(), ")");
    CHECK(dr <= tol * scale_r + 1e-6);
    CHECK(di <= tol * scale_i + 1e-6);
  }
}

} // namespace

TEST_CASE("complex64 host construction and allocation round trip") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(1);
  auto ref = random_complex(37, gen);
  auto out = read_complex(complex_array(ref, Shape{37}), stream);
  check_exact(out, ref);
}

TEST_CASE("astype float32 to complex64 promotes with zero imaginary") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<double> ref{0.5, -1.25, 3.0, 0.0, -7.75};
  auto src = real_array(ref, Shape{5});
  auto dst = astype(src, complex64);
  CHECK_EQ(dst.dtype(), complex64);
  std::vector<cdouble> expect(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    expect[i] = {ref[i], 0.0};
  }
  check_exact(read_complex(dst, stream), expect);
}

TEST_CASE("astype complex64 to float32 keeps the real part") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(2);
  auto ref = random_complex(19, gen);
  auto src = complex_array(ref, Shape{19});
  auto dst = astype(src, float32);
  CHECK_EQ(dst.dtype(), float32);
  // Upstream complex64_t::operator float() returns real().
  std::vector<double> expect(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    expect[i] = ref[i].real();
  }
  check_exact_real(read_real(dst, stream), expect);
}

TEST_CASE("astype integer and bool sources promote to (x, 0)") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<int32_t> int_host{3, -4, 0};
  array ints(int_host.begin(), Shape{3});
  std::vector<cdouble> int_expect{{3.0, 0.0}, {-4.0, 0.0}, {0.0, 0.0}};
  check_exact(read_complex(astype(ints, complex64), stream), int_expect);

  std::vector<uint8_t> bool_host{1, 0, 1};
  array bools = array(bool_host.begin(), Shape{3}, bool_);
  std::vector<cdouble> bool_expect{{1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
  check_exact(read_complex(astype(bools, complex64), stream), bool_expect);
}

TEST_CASE("strided reshape materializes complex64 values") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // Transpose first, then reshape: the reshape cannot be a view, so it
  // routes through the strided copy engine's complex64 path.
  std::mt19937 gen(3);
  auto ref = random_complex(12, gen);
  auto src = complex_array(ref, Shape{3, 4});
  auto transposed = transpose(src);
  auto flat = reshape(transposed, Shape{12});
  std::vector<cdouble> expect(12);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      expect[c * 3 + r] = ref[r * 4 + c];
    }
  }
  check_exact(read_complex(flat, stream), expect);
}

TEST_CASE("transposed view reshaped to dense agrees with the reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(4);
  auto ref = random_complex(20, gen);
  auto src = complex_array(ref, Shape{4, 5});
  auto view = transpose(src);
  CHECK_EQ(view.shape()[0], 5);
  CHECK_EQ(view.shape()[1], 4);
  auto dense = reshape(view, Shape{20});
  std::vector<cdouble> expect(20);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 5; ++c) {
      expect[c * 4 + r] = ref[r * 5 + c];
    }
  }
  check_exact(read_complex(dense, stream), expect);
}

TEST_CASE("strided slice of a complex array materializes exactly") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(5);
  auto ref = random_complex(16, gen);
  auto src = complex_array(ref, Shape{16});
  auto sliced = slice(src, Shape{1}, Shape{16}, Shape{3});
  std::vector<cdouble> expect;
  for (int i = 1; i < 16; i += 3) {
    expect.push_back(ref[i]);
  }
  check_exact(read_complex(sliced, stream), expect);
}

TEST_CASE("broadcast view reshaped to dense materializes every element") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(6);
  auto row = random_complex(4, gen);
  auto src = complex_array(row, Shape{1, 4});
  auto wide = broadcast_to(src, Shape{3, 4});
  auto dense = reshape(wide, Shape{12});
  std::vector<cdouble> expect;
  for (int r = 0; r < 3; ++r) {
    expect.insert(expect.end(), row.begin(), row.end());
  }
  check_exact(read_complex(dense, stream), expect);
}

TEST_CASE("full and zeros fill complex64 scalars") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  auto filled = full(Shape{2, 3}, complex64_t(1.5f, -2.5f), complex64);
  std::vector<cdouble> fill_expect(6, {1.5, -2.5});
  check_exact(read_complex(filled, stream), fill_expect);

  auto zeroed = zeros(Shape{4}, complex64);
  std::vector<cdouble> zero_expect(4, {0.0, 0.0});
  check_exact(read_complex(zeroed, stream), zero_expect);
}

TEST_CASE("conjugate negates the imaginary component") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(7);
  auto ref = random_complex(15, gen);
  auto src = complex_array(ref, Shape{15});
  auto conj = conjugate(src);
  std::vector<cdouble> expect(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    expect[i] = {ref[i].real(), -ref[i].imag()};
  }
  check_exact(read_complex(conj, stream), expect);
}

TEST_CASE("real and imag extract their components") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(8);
  auto ref = random_complex(11, gen);
  auto src = complex_array(ref, Shape{11});
  std::vector<double> real_expect(ref.size());
  std::vector<double> imag_expect(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    real_expect[i] = ref[i].real();
    imag_expect[i] = ref[i].imag();
  }
  check_exact_real(read_real(real(src), stream), real_expect);
  check_exact_real(read_real(imag(src), stream), imag_expect);
}

TEST_CASE("complex add subtract and negate are componentwise") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(9);
  auto a_ref = random_complex(9, gen);
  auto b_ref = random_complex(9, gen);
  auto a = complex_array(a_ref, Shape{9});
  auto b = complex_array(b_ref, Shape{9});
  std::vector<cdouble> sum_expect(9), diff_expect(9), neg_expect(9);
  for (size_t i = 0; i < 9; ++i) {
    // The device computes in float32 on float32-quantized inputs, so
    // the reference must run the same single-precision arithmetic:
    // rounding a double-precision sum once more into float32 can sit
    // one ulp away from the correctly rounded float32 result.
    float ar = float(a_ref[i].real());
    float ai = float(a_ref[i].imag());
    float br = float(b_ref[i].real());
    float bi = float(b_ref[i].imag());
    sum_expect[i] = {double(ar + br), double(ai + bi)};
    diff_expect[i] = {double(ar - br), double(ai - bi)};
    neg_expect[i] = {double(-ar), double(-ai)};
  }
  check_exact(read_complex(a + b, stream), sum_expect);
  check_exact(read_complex(a - b, stream), diff_expect);
  check_exact(read_complex(negative(a), stream), neg_expect);
}

TEST_CASE("complex multiply matches a double reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(10);
  auto a_ref = random_complex(24, gen);
  auto b_ref = random_complex(24, gen);
  auto a = complex_array(a_ref, Shape{24});
  auto b = complex_array(b_ref, Shape{24});
  std::vector<cdouble> expect(24);
  for (size_t i = 0; i < 24; ++i) {
    expect[i] = a_ref[i] * b_ref[i];
  }
  check_close(read_complex(a * b, stream), expect, 1e-5);
}

TEST_CASE("complex divide matches a double reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(11);
  auto a_ref = random_complex(24, gen);
  auto b_ref = random_complex(24, gen);
  // Keep the reference magnitudes moderate so the float32 reciprocal
  // is well conditioned.
  for (auto& value : b_ref) {
    value = value * 0.5 + cdouble(0.6, -0.4);
  }
  auto a = complex_array(a_ref, Shape{24});
  auto b = complex_array(b_ref, Shape{24});
  std::vector<cdouble> expect(24);
  for (size_t i = 0; i < 24; ++i) {
    expect[i] = a_ref[i] / b_ref[i];
  }
  check_close(read_complex(a / b, stream), expect, 1e-5);
}

TEST_CASE("complex pad and concatenate transport") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(12);
  auto ref = random_complex(6, gen);
  auto src = complex_array(ref, Shape{6});
  auto padded = pad(src, std::vector<int>{0}, Shape{2}, Shape{1});
  std::vector<cdouble> pad_expect(9, {0.0, 0.0});
  for (size_t i = 0; i < ref.size(); ++i) {
    pad_expect[i + 2] = ref[i];
  }
  check_exact(read_complex(padded, stream), pad_expect);

  auto joined = concatenate({src, src});
  std::vector<cdouble> concat_expect(ref);
  concat_expect.insert(concat_expect.end(), ref.begin(), ref.end());
  check_exact(read_complex(joined, stream), concat_expect);
}

TEST_CASE("fft accepts a non-contiguous complex64 input") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // A (1, 8) row transposed to (8, 1) then reshaped to (8) forces the
  // strided complex64 materialization in front of the FFT pass.
  std::mt19937 gen(13);
  auto ref = random_complex(8, gen);
  auto src = complex_array(ref, Shape{1, 8});
  auto column = transpose(src);
  auto dense = reshape(column, Shape{8});
  auto spectrum = fftn(dense);
  std::vector<cdouble> expect(8, {0.0, 0.0});
  for (size_t k = 0; k < 8; ++k) {
    for (size_t j = 0; j < 8; ++j) {
      double angle = -2.0 * M_PI * double(j) * double(k) / 8.0;
      expect[k] += ref[j] * std::polar(1.0, angle);
    }
  }
  check_close(read_complex(spectrum, stream), expect, 1e-4);
}

TEST_CASE("complex square, logaddexp, and equality match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Square = z*z through the complex multiply kernel.
  array z({complex64_t{1, 2}, complex64_t{-1, 0.5f}, complex64_t{0, -3}}, {3});
  auto sq = read_complex(square(z, stream), stream);
  CHECK(sq[0].real() == doctest::Approx(-3.0));
  CHECK(sq[0].imag() == doctest::Approx(4.0));
  CHECK(sq[1].real() == doctest::Approx(0.75));
  CHECK(sq[1].imag() == doctest::Approx(-1.0));
  CHECK(sq[2].real() == doctest::Approx(-9.0));
  CHECK(sq[2].imag() == doctest::Approx(0.0).epsilon(1e-6));

  // Equality: componentwise exact, including equal_nan=false defaults.
  array e1({complex64_t{1, 2}, complex64_t{0, 0}}, {2});
  array e2({complex64_t{1, 2}, complex64_t{0, 1}}, {2});
  CHECK(all(equal(e1, e2, stream), stream).item<bool>() == false);
  CHECK(any(not_equal(e1, e2, stream), stream).item<bool>() == true);
  CHECK(all(not_equal(e1, e2, stream), stream).item<bool>() == false);
  CHECK(all(equal(e1, e1, stream), stream).item<bool>() == true);

  // LogAddExp on a vector pair against the host formula
  // max + log(1 + exp(min - max)) in complex arithmetic.
  array lv({complex64_t{1, 1}, complex64_t{2, 0}}, {2});
  array rv({complex64_t{1, 1}, complex64_t{1, 1}}, {2});
  auto la = read_complex(logaddexp(lv, rv, stream), stream);
  std::complex<double> expect0 =
      std::complex<double>(1, 1) + std::log(std::complex<double>(2, 0));
  std::complex<double> expect1 = std::complex<double>(2, 0) +
      std::log(1.0 + std::exp(std::complex<double>(-1, 1)));
  CHECK(la[0].real() == doctest::Approx(expect0.real()).epsilon(1e-5));
  CHECK(la[0].imag() == doctest::Approx(expect0.imag()).epsilon(1e-5));
  CHECK(la[1].real() == doctest::Approx(expect1.real()).epsilon(1e-5));
  CHECK(la[1].imag() == doctest::Approx(expect1.imag()).epsilon(1e-5));
}

TEST_CASE("fft of a small real signal is exact at quarter-turn twiddles") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // Upstream's fft tests compare with array_equal, so the radix-2
  // butterflies must produce exact results where the twiddles are
  // exact: cos(pi/2) evaluated directly is -4.4e-8, which once left
  // y[1].real at -1.99999988.
  array x({0.0f, 1.0f, 2.0f, 3.0f});
  array y = fft::fft(x, -1, FFTNorm::Backward, stream);
  std::vector<cdouble> expect{{6, 0}, {-2, 2}, {-2, 0}, {-2, -2}};
  check_exact(read_complex(y, stream), expect);
  array back = fft::ifft(y, -1, FFTNorm::Backward, stream);
  std::vector<cdouble> expect_back{{0, 0}, {1, 0}, {2, 0}, {3, 0}};
  check_exact(read_complex(back, stream), expect_back);
}

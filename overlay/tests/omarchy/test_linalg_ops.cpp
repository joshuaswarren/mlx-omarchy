// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Wave 7: linear algebra. Cholesky, Inverse, LUF, QRF, Eigh, and SVD
// valued against host double-precision references and reconstruction
// identities computed in this file (L L^T = A, A A^-1 = I, QR = A,
// A V = V diag(w), U diag(S) Vt = A), batched and wide/tall shapes
// included, plus named-error pins for the gated paths: non-PD Cholesky,
// singular Inverse/LUF, complex64, and the Eig gate. The eigvalsh,
// pinv, and svd value tests pin the 2026-09-02 fixes for the values-only
// identity clobber, the uninitialized V accumulator, and the Jacobi
// rotation skip that stalled pairs whose cosine rounded to 1.0f.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/device.h"
#include "mlx/linalg.h"
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

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

std::vector<float> readback_f32(const Stream& stream, array value) {
  value = astype(value, float32, stream);
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = value.data<float>();
  return std::vector<float>(data, data + value.size());
}

// Element check with a relative-plus-absolute tolerance against the host
// double reference.
void expect_close(
    const std::vector<float>& device,
    const std::vector<double>& expected,
    double tol,
    const std::string& label) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    double diff = std::abs(static_cast<double>(device[i]) - expected[i]);
    CHECK_MESSAGE(
        diff <= tol + tol * std::abs(expected[i]),
        label,
        " index ",
        i,
        ": device ",
        device[i],
        " vs host ",
        expected[i]);
  }
}

// doctest forbids operator&& inside CHECK expressions, so needle checks
// go through this helper.
bool contains_all(
    const std::string& text,
    std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (text.find(needle) == std::string::npos) {
      return false;
    }
  }
  return true;
}

void expect_close(
    const std::vector<float>& device,
    const std::vector<float>& expected,
    double tol,
    const std::string& label) {
  expect_close(
      device,
      std::vector<double>(expected.begin(), expected.end()),
      tol,
      label);
}

using HostMatrix = std::vector<double>;

HostMatrix host_matmul(
    const HostMatrix& a,
    const HostMatrix& b,
    int rows_a,
    int cols_a,
    int cols_b) {
  HostMatrix out(rows_a * cols_b, 0.0);
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      double sum = 0.0;
      for (int l = 0; l < cols_a; ++l) {
        sum += a[i * cols_a + l] * b[l * cols_b + j];
      }
      out[i * cols_b + j] = sum;
    }
  }
  return out;
}

HostMatrix host_transpose(const HostMatrix& a, int rows, int cols) {
  HostMatrix out(rows * cols, 0.0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out[j * rows + i] = a[i * cols + j];
    }
  }
  return out;
}

// Host Cholesky-Banachiewicz, lower triangular.
HostMatrix host_cholesky(const HostMatrix& a, int n) {
  HostMatrix l(n * n, 0.0);
  for (int j = 0; j < n; ++j) {
    double d = a[j * n + j];
    for (int k = 0; k < j; ++k) {
      d -= l[j * n + k] * l[j * n + k];
    }
    l[j * n + j] = std::sqrt(d);
    for (int i = j + 1; i < n; ++i) {
      double s = a[i * n + j];
      for (int k = 0; k < j; ++k) {
        s -= l[i * n + k] * l[j * n + k];
      }
      l[i * n + j] = s / l[j * n + j];
    }
  }
  return l;
}

// A well-conditioned SPD matrix: B^T B + n I from a fixed stream.
HostMatrix host_spd(std::mt19937& gen, int n) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  HostMatrix b(n * n);
  for (auto& value : b) {
    value = dist(gen);
  }
  HostMatrix bt = host_transpose(b, n, n);
  HostMatrix a = host_matmul(bt, b, n, n, n);
  for (int i = 0; i < n; ++i) {
    a[i * n + i] += n;
  }
  return a;
}

// Strictly diagonally dominant, so nonsingular by construction.
HostMatrix host_diag_dominant(std::mt19937& gen, int n) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  HostMatrix a(n * n);
  for (auto& value : a) {
    value = dist(gen);
  }
  for (int i = 0; i < n; ++i) {
    double row_abs = 0.0;
    for (int j = 0; j < n; ++j) {
      row_abs += std::abs(a[i * n + j]);
    }
    a[i * n + i] = row_abs + 1.0;
  }
  return a;
}

// Host cyclic Jacobi for a symmetric matrix in double precision: the
// independent eigenvalue reference for the device kernels. Ascending.
HostMatrix host_sym_eigvals(const HostMatrix& a, int n) {
  HostMatrix w = a;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        off += w[p * n + q] * w[p * n + q];
      }
    }
    if (off < 1e-26) {
      break;
    }
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        double apq = w[p * n + q];
        if (std::abs(apq) < 1e-15) {
          w[p * n + q] = 0.0;
          w[q * n + p] = 0.0;
          continue;
        }
        double tau = (w[q * n + q] - w[p * n + p]) / (2.0 * apq);
        double t = (tau >= 0.0)
            ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
            : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;
        for (int i = 0; i < n; ++i) {
          double x = w[i * n + p];
          double y = w[i * n + q];
          w[i * n + p] = c * x - s * y;
          w[i * n + q] = s * x + c * y;
        }
        for (int j = 0; j < n; ++j) {
          double x = w[p * n + j];
          double y = w[q * n + j];
          w[p * n + j] = c * x - s * y;
          w[q * n + j] = s * x + c * y;
        }
      }
    }
  }
  HostMatrix vals(n);
  for (int i = 0; i < n; ++i) {
    vals[i] = w[i * n + i];
  }
  std::sort(vals.begin(), vals.end());
  return vals;
}

HostMatrix host_identity(int n) {
  HostMatrix out(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    out[i * n + i] = 1.0;
  }
  return out;
}

std::vector<float> to_float(const HostMatrix& a) {
  return std::vector<float>(a.begin(), a.end());
}

array device_matrix(
    const Stream& stream,
    const std::vector<double>& values,
    Shape shape) {
  std::vector<float> flat(values.begin(), values.end());
  array out(flat.begin(), shape, float32);
  return astype(out, float32, stream);
}

HostMatrix reshape(
    const std::vector<float>& flat,
    size_t offset,
    int rows,
    int cols) {
  HostMatrix out(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    out[i] = static_cast<double>(flat[offset + i]);
  }
  return out;
}

} // namespace

TEST_CASE("cholesky lower matches host reference") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  for (int n : {4, 6}) {
    std::mt19937 gen(100 + n);
    HostMatrix a = host_spd(gen, n);
    HostMatrix host_l = host_cholesky(a, n);
    array device = device_matrix(stream, a, Shape{n, n});
    array factor = linalg::cholesky(device, /*upper=*/false, stream);
    expect_close(
        readback_f32(stream, factor),
        host_l,
        2e-5,
        "cholesky lower n=" + std::string(std::to_string(n)));
    // The opposite triangle is exactly zero.
    auto flat = readback_f32(stream, factor);
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        CHECK_EQ(flat[i * n + j], 0.0f);
      }
    }
  }
}

TEST_CASE("cholesky upper is the transposed factor") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(42);
  int n = 5;
  HostMatrix a = host_spd(gen, n);
  HostMatrix host_l = host_cholesky(a, n);
  HostMatrix host_u = host_transpose(host_l, n, n);
  array device = device_matrix(stream, a, Shape{n, n});
  array factor = linalg::cholesky(device, /*upper=*/true, stream);
  auto flat = readback_f32(stream, factor);
  expect_close(flat, host_u, 2e-5, "cholesky upper");
  // A = U^T U.
  HostMatrix u(flat.begin(), flat.end());
  HostMatrix ut = host_transpose(u, n, n);
  HostMatrix product = host_matmul(ut, u, n, n, n);
  expect_close(to_float(product), a, 1e-4, "cholesky upper reconstruction");
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      CHECK_EQ(flat[i * n + j], 0.0f);
    }
  }
}

TEST_CASE("cholesky batched") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(7);
  int n = 3;
  HostMatrix a0 = host_spd(gen, n);
  HostMatrix a1 = host_spd(gen, n);
  HostMatrix both(a0);
  both.insert(both.end(), a1.begin(), a1.end());
  array device = device_matrix(stream, both, Shape{2, n, n});
  array factor = linalg::cholesky(device, /*upper=*/false, stream);
  auto flat = readback_f32(stream, factor);
  REQUIRE_EQ(flat.size(), static_cast<size_t>(2 * n * n));
  expect_close(
      to_float(reshape(flat, 0, n, n)), host_cholesky(a0, n), 2e-5, "batch 0");
  expect_close(
      to_float(reshape(flat, n * n, n, n)),
      host_cholesky(a1, n),
      2e-5,
      "batch 1");
}

TEST_CASE("cholesky rejects a non-positive-definite matrix") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  HostMatrix a = {1.0, 2.0, 2.0, 1.0}; // eigenvalues -1 and 3
  array device = device_matrix(stream, a, Shape{2, 2});
  auto error = evaluation_error(linalg::cholesky(device, false, stream));
  CHECK_MESSAGE(
      contains_all(error, {"Cholesky", "positive definite"}),
      "unexpected error: ",
      error);
}

TEST_CASE("cholesky rejects float16 at the API layer") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(3);
  HostMatrix a = host_spd(gen, 3);
  array device = device_matrix(stream, a, Shape{3, 3});
  array half = astype(device, float16, stream);
  std::string error;
  try {
    linalg::cholesky(half, false, stream).eval();
  } catch (const std::exception& caught) {
    error = caught.what();
  }
  CHECK_MESSAGE(
      contains_all(error, {"[linalg::cholesky]", "float16"}),
      "unexpected error: ",
      error);
}

TEST_CASE("inverse general satisfies A A^-1 = I") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  for (int n : {3, 5}) {
    std::mt19937 gen(200 + n);
    HostMatrix a = host_diag_dominant(gen, n);
    array device = device_matrix(stream, a, Shape{n, n});
    array inv = linalg::inv(device, stream);
    auto flat = readback_f32(stream, inv);
    HostMatrix product =
        host_matmul(a, HostMatrix(flat.begin(), flat.end()), n, n, n);
    expect_close(
        to_float(product),
        host_identity(n),
        1e-4,
        "inverse n=" + std::string(std::to_string(n)));
  }
}

TEST_CASE("inverse batched") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(11);
  int n = 2;
  HostMatrix a0 = host_diag_dominant(gen, n);
  HostMatrix a1 = host_diag_dominant(gen, n);
  HostMatrix both(a0);
  both.insert(both.end(), a1.begin(), a1.end());
  array device = device_matrix(stream, both, Shape{2, n, n});
  array inv = linalg::inv(device, stream);
  auto flat = readback_f32(stream, inv);
  REQUIRE_EQ(flat.size(), static_cast<size_t>(2 * n * n));
  HostMatrix i0 =
      host_matmul(a0, reshape(flat, 0, n, n), n, n, n);
  HostMatrix i1 =
      host_matmul(a1, reshape(flat, n * n, n, n), n, n, n);
  expect_close(to_float(i0), host_identity(n), 1e-4, "batch 0 inverse");
  expect_close(to_float(i1), host_identity(n), 1e-4, "batch 1 inverse");
}

TEST_CASE("triangular inverse lower and upper") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int n = 4;
  std::mt19937 gen(31);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  HostMatrix t(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      t[i * n + j] = dist(gen);
    }
    t[i * n + i] += 2.0; // safely away from a zero diagonal
  }
  HostMatrix upper_t = host_transpose(t, n, n);
  for (bool upper : {false, true}) {
    const HostMatrix& src = upper ? upper_t : t;
    array device = device_matrix(stream, src, Shape{n, n});
    array inv = linalg::tri_inv(device, upper, stream);
    auto flat = readback_f32(stream, inv);
    HostMatrix product =
        host_matmul(src, HostMatrix(flat.begin(), flat.end()), n, n, n);
    expect_close(
        to_float(product),
        host_identity(n),
        1e-4,
        upper ? "tri upper inverse" : "tri lower inverse");
  }
}

TEST_CASE("inverse rejects a singular matrix") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  HostMatrix a = {1.0, 2.0, 2.0, 4.0}; // rank 1
  array device = device_matrix(stream, a, Shape{2, 2});
  auto error = evaluation_error(linalg::inv(device, stream));
  CHECK_MESSAGE(
      contains_all(error, {"Inverse", "singular"}),
      "unexpected error: ",
      error);
}

TEST_CASE("luf stays gated pending numeric verification") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(5);
  HostMatrix a(9);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto& value : a) {
    value = dist(gen);
  }
  array device = device_matrix(stream, a, Shape{3, 3});
  std::string error;
  try {
    linalg::lu_factor(device, stream).first.eval();
  } catch (const std::exception& caught) {
    error = caught.what();
  }
  CHECK_MESSAGE(
      contains_all(error, {"[LUF]", "gated"}),
      "unexpected error: ",
      error);
}

TEST_CASE("luf batched stays gated") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(23);
  int n = 3;
  HostMatrix a0 = host_diag_dominant(gen, n);
  HostMatrix a1 = host_diag_dominant(gen, n);
  HostMatrix both(a0);
  both.insert(both.end(), a1.begin(), a1.end());
  array device = device_matrix(stream, both, Shape{2, n, n});
  std::string error;
  try {
    linalg::lu_factor(device, stream).first.eval();
  } catch (const std::exception& caught) {
    error = caught.what();
  }
  CHECK_MESSAGE(
      contains_all(error, {"[LUF]", "gated"}),
      "unexpected error: ",
      error);
}

TEST_CASE("luf gate fires before the singular check") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  HostMatrix a = {0.0, 0.0, 1.0, 2.0};
  array device = device_matrix(stream, a, Shape{2, 2});
  std::string error;
  try {
    linalg::lu_factor(device, stream).first.eval();
  } catch (const std::exception& caught) {
    error = caught.what();
  }
  CHECK_MESSAGE(
      contains_all(error, {"[LUF]", "gated"}),
      "unexpected error: ",
      error);
}

TEST_CASE("qr reconstruction tall square and wide") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(9);
  struct Case {
    int m;
    int n;
  };
  for (auto [m, n] : {Case{4, 3}, Case{3, 3}, Case{3, 4}}) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    HostMatrix a(m * n);
    for (auto& value : a) {
      value = dist(gen);
    }
    array device = device_matrix(stream, a, Shape{m, n});
    auto [q, r] = linalg::qr(device, stream);
    int k = std::min(m, n);
    auto q_flat = readback_f32(stream, q);
    auto r_flat = readback_f32(stream, r);
    REQUIRE_EQ(q_flat.size(), static_cast<size_t>(m * k));
    REQUIRE_EQ(r_flat.size(), static_cast<size_t>(k * n));
    // Q^T Q = I.
    HostMatrix qt = host_transpose(
        HostMatrix(q_flat.begin(), q_flat.end()), m, k);
    HostMatrix qtq = host_matmul(qt, HostMatrix(q_flat.begin(), q_flat.end()), k, m, k);
    expect_close(
        to_float(qtq),
        host_identity(k),
        1e-4,
        "q^t q m=" + std::string(std::to_string(m)) + " n=" +
            std::to_string(n));
    // R is upper trapezoid.
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < std::min(i, n); ++j) {
        CHECK_MESSAGE(
            std::abs(r_flat[i * n + j]) < 1e-5,
            "R lower entry nonzero at ",
            i,
            ",",
            j);
      }
    }
    // Q R = A.
    HostMatrix product =
        host_matmul(
            HostMatrix(q_flat.begin(), q_flat.end()),
            HostMatrix(r_flat.begin(), r_flat.end()),
            m,
            k,
            n);
    expect_close(
        to_float(product),
        a,
        1e-4,
        "qr reconstruction m=" + std::string(std::to_string(m)) + " n=" +
            std::to_string(n));
  }
}

TEST_CASE("qr batched stays gated") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(13);
  int m = 3;
  int n = 2;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  HostMatrix a(2 * m * n);
  for (auto& value : a) {
    value = dist(gen);
  }
  array device = device_matrix(stream, a, Shape{2, m, n});
  std::string error;
  try {
    auto parts = linalg::qr(device, stream);
    parts.first.eval();
  } catch (const std::exception& caught) {
    error = caught.what();
  }
  CHECK_MESSAGE(
      contains_all(error, {"[QRF]", "gated"}),
      "unexpected error: ",
      error);
}

TEST_CASE("eigh known spectrum and reconstruction") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
  HostMatrix a = {2.0, 1.0, 1.0, 2.0};
  array device = device_matrix(stream, a, Shape{2, 2});
  auto [w, v] = linalg::eigh(device, "L", stream);
  auto w_flat = readback_f32(stream, w);
  REQUIRE_EQ(w_flat.size(), static_cast<size_t>(2));
  CHECK(doctest::Approx(w_flat[0]).epsilon(1e-5) == 1.0f);
  CHECK(doctest::Approx(w_flat[1]).epsilon(1e-5) == 3.0f);
  auto v_flat = readback_f32(stream, v);
  HostMatrix vm(v_flat.begin(), v_flat.end());
  HostMatrix av = host_matmul(a, vm, 2, 2, 2);
  HostMatrix vd = host_matmul(vm, HostMatrix{w_flat[0], 0.0, 0.0, w_flat[1]}, 2, 2, 2);
  expect_close(to_float(av), vd, 1e-4, "A V = V diag(w)");
}

TEST_CASE("eigh random 4x4 matches the host Jacobi spectrum") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(77);
  int n = 4;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  HostMatrix full(n * n);
  for (auto& value : full) {
    value = dist(gen);
  }
  HostMatrix symmetric = host_matmul(full, host_transpose(full, n, n), n, n, n);
  array device = device_matrix(stream, symmetric, Shape{n, n});
  // Regression: this shape once tripped the 60-sweep cap because pairs
  // whose cosine rounded to 1.0f skipped their rotation while still far
  // above the pin threshold, stalling between the two thresholds. The
  // convergence now holds, so the spectrum is held to the host Jacobi
  // reference instead of pinning a refusal.
  auto [w, v] = linalg::eigh(device, "L", stream);
  auto w_flat = readback_f32(stream, w);
  auto v_flat = readback_f32(stream, v);
  REQUIRE_EQ(w_flat.size(), static_cast<size_t>(n));
  expect_close(w_flat, host_sym_eigvals(symmetric, n), 1e-4, "eigh spectrum");
  HostMatrix vm = reshape(v_flat, 0, n, n);
  HostMatrix av = host_matmul(symmetric, vm, n, n, n);
  HostMatrix vd(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      vd[i * n + j] = vm[i * n + j] * w_flat[j];
    }
  }
  expect_close(to_float(av), to_float(vd), 1e-4, "A V = V diag(w)");
  HostMatrix vtv = host_matmul(host_transpose(vm, n, n), vm, n, n, n);
  expect_close(to_float(vtv), to_float(host_identity(n)), 1e-4, "Vt V = I");
}

TEST_CASE("eigh batched and values-only") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(21);
  int n = 3;
  HostMatrix a0 = host_spd(gen, n);
  HostMatrix a1 = host_spd(gen, n);
  HostMatrix both(a0);
  both.insert(both.end(), a1.begin(), a1.end());
  array device = device_matrix(stream, both, Shape{2, n, n});
  auto w = linalg::eigvalsh(device, "L", stream);
  auto w_flat = readback_f32(stream, w);
  REQUIRE_EQ(w_flat.size(), static_cast<size_t>(2 * n));
  // Value-checked per batch against the host Jacobi reference, not just
  // positivity: this case once passed with the all-ones spectrum produced
  // by the values-only clobber bug, which positivity and sortedness
  // cannot detect.
  HostMatrix ref0 = host_sym_eigvals(a0, n);
  HostMatrix ref1 = host_sym_eigvals(a1, n);
  for (int i = 0; i < n; ++i) {
    CHECK_MESSAGE(
        std::abs(w_flat[i] - static_cast<float>(ref0[i])) <=
                1e-4 + 1e-4 * std::abs(ref0[i]),
        "batch 0 eigenvalue ",
        i,
        ": device ",
        w_flat[i],
        " vs host ",
        ref0[i]);
    CHECK_MESSAGE(
        std::abs(w_flat[n + i] - static_cast<float>(ref1[i])) <=
                1e-4 + 1e-4 * std::abs(ref1[i]),
        "batch 1 eigenvalue ",
        i,
        ": device ",
        w_flat[n + i],
        " vs host ",
        ref1[i]);
  }
  for (int i = 1; i < n; ++i) {
    CHECK(w_flat[i - 1] <= w_flat[i] + 1e-5f);
  }
}

TEST_CASE("eigh rejects complex64 input") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  array value = astype(
      array({1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2}, float32), float32, stream);
  // Complex storage has no Omarchy transport at all, so the converting
  // copy refuses before linalg is reached; either named refusal is the
  // contract, never a silent value.
  array complex_input = astype(value, complex64, stream);
  auto error =
      evaluation_error(linalg::eigh(complex_input, "L", stream).first);
  CHECK_MESSAGE(
      contains_all(error, {"not implemented", "complex64"}),
      "unexpected error: ",
      error);
}

TEST_CASE("svd full factors reconstruct a random tall 4x3") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(15);
  int m = 4;
  int n = 3;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  HostMatrix a(m * n);
  for (auto& value : a) {
    value = dist(gen);
  }
  array device = device_matrix(stream, a, Shape{m, n});
  // Regression: one-sided Jacobi once stalled on shapes like this because
  // pairs whose cosine rounded to 1.0f never applied their rotation, so
  // the sweep cap refused by name. With convergence fixed the factors are
  // held to reconstruction and the host singular-value reference.
  auto outs = linalg::svd(device, true, stream);
  auto u_flat = readback_f32(stream, outs[0]);
  auto s_flat = readback_f32(stream, outs[1]);
  auto vt_flat = readback_f32(stream, outs[2]);
  REQUIRE_EQ(u_flat.size(), static_cast<size_t>(m * m));
  REQUIRE_EQ(s_flat.size(), static_cast<size_t>(n));
  REQUIRE_EQ(vt_flat.size(), static_cast<size_t>(n * n));
  // Host reference: singular values are the square roots of the
  // eigenvalues of A^T A, descending.
  HostMatrix at = host_transpose(a, m, n);
  HostMatrix ata = host_matmul(at, a, n, m, n);
  HostMatrix eig = host_sym_eigvals(ata, n);
  HostMatrix ref_s(n);
  for (int i = 0; i < n; ++i) {
    ref_s[i] = std::sqrt(eig[n - 1 - i]);
    CHECK_MESSAGE(
        std::abs(s_flat[i] - static_cast<float>(ref_s[i])) <=
                1e-4 + 1e-4 * ref_s[i],
        "singular value ",
        i,
        ": device ",
        s_flat[i],
        " vs host ",
        ref_s[i]);
  }
  // U diag(S) Vt == A on the returned factors.
  HostMatrix u = reshape(u_flat, 0, m, m);
  HostMatrix vt = reshape(vt_flat, 0, n, n);
  HostMatrix us(m * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      us[i * n + j] = u[i * m + j] * s_flat[j];
    }
  }
  HostMatrix product = host_matmul(us, vt, m, n, n);
  expect_close(to_float(product), to_float(a), 1e-4, "U diag(S) Vt = A");
  HostMatrix utu = host_matmul(host_transpose(u, m, m), u, m, m, m);
  expect_close(to_float(utu), to_float(host_identity(m)), 1e-4, "Ut U = I");
  HostMatrix vvt = host_matmul(host_transpose(vt, n, n), vt, n, n, n);
  expect_close(to_float(vvt), to_float(host_identity(n)), 1e-4, "Vt^T Vt = I");
}

TEST_CASE("svd values only matches the host singular values") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(17);
  int m = 4;
  int n = 3;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  HostMatrix a(m * n);
  for (auto& value : a) {
    value = dist(gen);
  }
  array device = device_matrix(stream, a, Shape{m, n});
  auto s_flat = readback_f32(stream, linalg::svd(device, /*compute_uv=*/false, stream).at(0));
  REQUIRE_EQ(s_flat.size(), static_cast<size_t>(n));
  HostMatrix at = host_transpose(a, m, n);
  HostMatrix ata = host_matmul(at, a, n, m, n);
  HostMatrix eig = host_sym_eigvals(ata, n);
  for (int i = 0; i < n; ++i) {
    double ref = std::sqrt(eig[n - 1 - i]);
    CHECK_MESSAGE(
        std::abs(s_flat[i] - static_cast<float>(ref)) <= 1e-4 + 1e-4 * ref,
        "singular value ",
        i,
        ": device ",
        s_flat[i],
        " vs host ",
        ref);
  }
}

TEST_CASE("svd rejects complex64 and float16") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  array real = array({1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2}, float32);
  // Complex storage has no Omarchy transport; the converting copy
  // refuses first, and the backend gate behind it would say the same.
  array complex_input = astype(real, complex64, stream);
  auto error = evaluation_error(linalg::svd(complex_input, true, stream)[0]);
  CHECK_MESSAGE(
      contains_all(error, {"not implemented", "complex64"}),
      "unexpected error: ",
      error);
  // The API dtype check refuses float16 before the backend is reached,
  // matching the upstream CPU contract; the throw happens at the call.
  array half = astype(real, float16, stream);
  std::string half_error;
  try {
    linalg::svd(half, true, stream);
  } catch (const std::exception& caught) {
    half_error = caught.what();
  }
  CHECK_MESSAGE(
      contains_all(half_error, {"[linalg::svd]", "float16"}),
      "unexpected error: ",
      half_error);
}

TEST_CASE("eig stays gated with the named error") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::mt19937 gen(19);
  int n = 3;
  HostMatrix a = host_diag_dominant(gen, n);
  array device = device_matrix(stream, a, Shape{n, n});
  auto values = linalg::eigvals(device, stream);
  auto error = evaluation_error(values);
  CHECK_MESSAGE(
      contains_all(error, {"[omarchy] Eig is not implemented", "No CPU fallback"}),
      "unexpected error: ",
      error);
}

TEST_CASE("eigvalsh values-only matches the analytic spectrum") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // [[4, 1], [1, 3]]: eigenvalues (7 +/- sqrt(5)) / 2 exactly. Regression:
  // the values-only Eigh path once bound the work buffer into the vectors
  // slot, whose identity init clobbered the matrix and returned [1, 1].
  HostMatrix a = {4.0, 1.0, 1.0, 3.0};
  array device = device_matrix(stream, a, Shape{2, 2});
  auto w = linalg::eigvalsh(device, "L", stream);
  auto w_flat = readback_f32(stream, w);
  REQUIRE_EQ(w_flat.size(), static_cast<size_t>(2));
  const double lo = (7.0 - std::sqrt(5.0)) / 2.0;
  const double hi = (7.0 + std::sqrt(5.0)) / 2.0;
  expect_close(w_flat, HostMatrix{lo, hi}, 1e-5, "eigvalsh analytic spectrum");
}

TEST_CASE("pinv square equals the analytic inverse") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // [[4, 1], [2, 3]] is invertible: pinv must equal the exact inverse
  // [[3/10, -1/10], [-1/5, 2/5]]. Regression: the SVD sweep once
  // accumulated V into an uninitialized buffer, and pinv returned zeros
  // or garbage while S and U looked healthy.
  HostMatrix a = {4.0, 1.0, 2.0, 3.0};
  array device = device_matrix(stream, a, Shape{2, 2});
  auto p = linalg::pinv(device, stream);
  auto p_flat = readback_f32(stream, p);
  REQUIRE_EQ(p_flat.size(), static_cast<size_t>(4));
  expect_close(
      p_flat, HostMatrix{0.3, -0.1, -0.2, 0.4}, 1e-5, "pinv square analytic");
}

TEST_CASE("pinv rectangular satisfies the Moore-Penrose identity") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // [[1, 2], [3, 4], [5, 6]] has the exact pseudo-inverse
  // (A^T A)^-1 A^T = [[-4/3, -1/3, 2/3], [13/12, 1/3, -5/12]].
  HostMatrix a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  array device = device_matrix(stream, a, Shape{3, 2});
  auto p = linalg::pinv(device, stream);
  auto p_flat = readback_f32(stream, p);
  REQUIRE_EQ(p_flat.size(), static_cast<size_t>(6));
  expect_close(
      p_flat,
      HostMatrix{-4.0 / 3.0,
       -1.0 / 3.0,
       2.0 / 3.0,
       13.0 / 12.0,
       1.0 / 3.0,
       -5.0 / 12.0},
      1e-5,
      "pinv rectangular analytic");
  // Penrose identities from the device values against the host reference.
  HostMatrix pm = reshape(p_flat, 0, 2, 3);
  HostMatrix at = host_transpose(a, 3, 2);
  HostMatrix apa = host_matmul(host_matmul(a, pm, 3, 2, 3), a, 3, 3, 2);
  HostMatrix pap = host_matmul(host_matmul(pm, a, 2, 3, 2), pm, 2, 2, 3);
  expect_close(to_float(apa), to_float(a), 1e-4, "A pinv(A) A = A");
  expect_close(to_float(pap), to_float(pm), 1e-4, "pinv(A) A pinv(A) = pinv(A)");
}

TEST_CASE("svd full factors reconstruct the matrix") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // Regression: Vt once came back all zeros (uninitialized V accumulator),
  // so U diag(S) Vt silently missed A by exactly A.
  HostMatrix a = {4.0, 1.0, 2.0, 3.0};
  array device = device_matrix(stream, a, Shape{2, 2});
  auto outs = linalg::svd(device, true, stream);
  auto u_flat = readback_f32(stream, outs[0]);
  auto s_flat = readback_f32(stream, outs[1]);
  auto vt_flat = readback_f32(stream, outs[2]);
  REQUIRE_EQ(u_flat.size(), static_cast<size_t>(4));
  REQUIRE_EQ(s_flat.size(), static_cast<size_t>(2));
  REQUIRE_EQ(vt_flat.size(), static_cast<size_t>(4));
  // Analytic singular values sqrt(15 +/- 5 sqrt(5)), descending.
  const double s_hi = std::sqrt(15.0 + 5.0 * std::sqrt(5.0));
  const double s_lo = std::sqrt(15.0 - 5.0 * std::sqrt(5.0));
  expect_close(s_flat, HostMatrix{s_hi, s_lo}, 1e-5, "svd analytic singular values");
  // The zero-Vt regression is caught by plain reconstruction: with Vt = 0
  // the product misses A by max(A) = 4, far above tolerance.
  HostMatrix u = reshape(u_flat, 0, 2, 2);
  HostMatrix vt = reshape(vt_flat, 0, 2, 2);
  HostMatrix sm = {s_flat[0], 0.0, 0.0, s_flat[1]};
  HostMatrix udm = host_matmul(u, sm, 2, 2, 2);
  HostMatrix product = host_matmul(udm, vt, 2, 2, 2);
  expect_close(to_float(product), to_float(a), 1e-4, "U diag(S) Vt = A");
  // Orthonormality of the returned factors.
  HostMatrix utu = host_matmul(
      host_transpose(u, 2, 2), u, 2, 2, 2);
  expect_close(to_float(utu), to_float(host_identity(2)), 1e-4, "Ut U = I");
  HostMatrix vvt = host_matmul(
      host_transpose(vt, 2, 2), vt, 2, 2, 2);
  expect_close(to_float(vvt), to_float(host_identity(2)), 1e-4, "Vt^T Vt = I");
}

TEST_CASE("cholesky_inv matches the analytic inverse") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // inv([[4, 1], [1, 3]]) = [[3, -1], [-1, 4]] / 11.
  HostMatrix a = {4.0, 1.0, 1.0, 3.0};
  array device = device_matrix(stream, a, Shape{2, 2});
  auto l = linalg::cholesky(device, false, stream);
  auto ci = linalg::cholesky_inv(l, false, stream);
  auto ci_flat = readback_f32(stream, ci);
  REQUIRE_EQ(ci_flat.size(), static_cast<size_t>(4));
  expect_close(
      ci_flat,
      HostMatrix{3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0},
      1e-5,
      "cholesky_inv analytic inverse");
}


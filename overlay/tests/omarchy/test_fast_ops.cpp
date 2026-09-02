// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 9 fast-op tests: fused RMSNorm/LayerNorm forward and VJP kernels, the
// cross-entropy VJP, FP8 (E4M3) conversion, and the composed-path anchors.
// Every value test names its reference:
//   - "finite differences": central differences through the real device
//     kernel, with the perturbed math evaluated in double on the host,
//   - "composed formula": the upstream mlx/fast.cpp fallback algebra rebuilt
//     from core mlx ops in the test (the graph the composed fallback ran
//     before these primitives went native),
//   - "host math": the closed-form math computed in double in this file,
//   - "upstream bit algorithm": the ConvertFP8 CPU bit twiddling, scalarized
//     in this file.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

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

std::string caught_message(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

void require_close(
    const std::vector<float>& got,
    const std::vector<double>& want,
    double tolerance,
    const std::string& what) {
  REQUIRE_EQ(got.size(), want.size());
  for (size_t index = 0; index < want.size(); ++index) {
    double diff = std::abs(static_cast<double>(got[index]) - want[index]);
    CHECK_MESSAGE(
        diff <= tolerance,
        what,
        " element ",
        index,
        ": got ",
        got[index],
        " want ",
        want[index]);
  }
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

// Deterministic pseudo-random values in [-1, 1).
std::vector<float> pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 20000u) / 10000.0) -
        1.0f);
  }
  return values;
}

std::vector<double> widen(const std::vector<float>& values) {
  return std::vector<double>(values.begin(), values.end());
}

// ---- host math references (double) ----

struct RowStats {
  double norm;
  double mu;
};

RowStats host_row_stats(
    const std::vector<double>& x,
    size_t row,
    size_t cols,
    double eps,
    bool layer) {
  double sum = 0.0;
  double sum_sq = 0.0;
  for (size_t col = 0; col < cols; ++col) {
    double value = x[row * cols + col];
    sum += value;
    sum_sq += value * value;
  }
  double mu = sum / cols;
  RowStats stats;
  stats.mu = mu;
  double mean_square = sum_sq / cols;
  stats.norm =
      1.0 / std::sqrt((layer ? (mean_square - mu * mu) : mean_square) + eps);
  return stats;
}

std::vector<double> host_norm(
    const std::vector<double>& x,
    const std::vector<double>& w,
    const std::vector<double>& b,
    size_t rows,
    size_t cols,
    double eps,
    bool layer) {
  std::vector<double> out(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    auto stats = host_row_stats(x, row, cols, eps, layer);
    for (size_t col = 0; col < cols; ++col) {
      double xc =
          layer ? (x[row * cols + col] - stats.mu) : x[row * cols + col];
      out[row * cols + col] =
          xc * stats.norm * w[col] + (layer ? b[col] : 0.0);
    }
  }
  return out;
}

// The upstream mlx/fast.cpp VJP fallback algebra, host math.
std::vector<double> host_norm_vjp_dx(
    const std::vector<double>& x,
    const std::vector<double>& w,
    const std::vector<double>& g,
    size_t rows,
    size_t cols,
    double eps,
    bool layer) {
  std::vector<double> dx(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    auto stats = host_row_stats(x, row, cols, eps, layer);
    double n3 = stats.norm * stats.norm * stats.norm;
    double sum_wg = 0.0;
    double sum_wgxc = 0.0;
    for (size_t col = 0; col < cols; ++col) {
      size_t index = row * cols + col;
      double wg = w[col] * g[index];
      double xc = layer ? (x[index] - stats.mu) : x[index];
      sum_wg += wg;
      sum_wgxc += wg * xc;
    }
    for (size_t col = 0; col < cols; ++col) {
      size_t index = row * cols + col;
      double wg = w[col] * g[index];
      double xc = layer ? (x[index] - stats.mu) : x[index];
      if (layer) {
        dx[index] =
            (wg - sum_wg / cols) * stats.norm - xc * (sum_wgxc / cols) * n3;
      } else {
        dx[index] = wg * stats.norm - x[index] * (sum_wgxc / cols) * n3;
      }
    }
  }
  return dx;
}

std::vector<double> host_norm_vjp_dw(
    const std::vector<double>& x,
    const std::vector<double>& g,
    size_t rows,
    size_t cols,
    double eps,
    bool layer) {
  std::vector<double> dw(cols, 0.0);
  for (size_t row = 0; row < rows; ++row) {
    auto stats = host_row_stats(x, row, cols, eps, layer);
    for (size_t col = 0; col < cols; ++col) {
      size_t index = row * cols + col;
      double xc = layer ? (x[index] - stats.mu) : x[index];
      dw[col] += g[index] * xc * stats.norm;
    }
  }
  return dw;
}

std::vector<double> host_norm_vjp_db(
    const std::vector<double>& g, size_t rows, size_t cols) {
  std::vector<double> db(cols, 0.0);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      db[col] += g[row * cols + col];
    }
  }
  return db;
}

// Upstream CrossEntropy::vjp fallback algebra, host math.
std::vector<double> host_cross_entropy_vjp(
    const std::vector<double>& x,
    const std::vector<int>& y,
    const std::vector<double>& g,
    size_t rows,
    size_t cols) {
  std::vector<double> gx(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    double max_value = x[row * cols];
    for (size_t col = 1; col < cols; ++col) {
      max_value = std::max(max_value, x[row * cols + col]);
    }
    double sum_exp = 0.0;
    for (size_t col = 0; col < cols; ++col) {
      sum_exp += std::exp(x[row * cols + col] - max_value);
    }
    double lse = max_value + std::log(sum_exp);
    for (size_t col = 0; col < cols; ++col) {
      double p = std::exp(x[row * cols + col] - lse);
      double onehot = (static_cast<int>(col) == y[row]) ? 1.0 : 0.0;
      gx[row * cols + col] = g[row] * (p - onehot);
    }
  }
  return gx;
}

// ---- finite differences through the real device kernel ----

double central_difference(
    const std::vector<double>& base,
    size_t element,
    double h,
    const std::function<double(const std::vector<double>&)>& objective) {
  std::vector<double> plus = base;
  std::vector<double> minus = base;
  plus[element] += h;
  minus[element] -= h;
  return (objective(plus) - objective(minus)) / (2.0 * h);
}

// ---- FP8 host reference: the upstream bit algorithm, scalarized ----

uint8_t host_to_fp8(double value) {
  float f = static_cast<float>(value);
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t fp8_max = 543u << 21u;
  uint32_t denorm_mask = 141u << 23u;
  uint32_t sign = bits & 0x80000000u;
  bits ^= sign;
  float low_input;
  std::memcpy(&low_input, &bits, sizeof(low_input));
  float denorm_bias;
  std::memcpy(&denorm_bias, &denorm_mask, sizeof(denorm_bias));
  float low_sum = low_input + denorm_bias;
  uint32_t f_bits_low;
  std::memcpy(&f_bits_low, &low_sum, sizeof(f_bits_low));
  uint32_t result_low = (f_bits_low - denorm_mask) & 0xFFu;
  uint32_t mant_odd = (bits >> 20u) & 1u;
  uint32_t f_bits_high = bits + (((7u - 127u) << 23u) + 0x7FFFFu);
  f_bits_high += mant_odd;
  uint32_t result_high = (f_bits_high >> 20u) & 0xFFu;
  uint32_t result = (bits < (121u << 23u)) ? result_low : result_high;
  result = (bits >= fp8_max) ? 0x7Eu : result;
  return static_cast<uint8_t>(result | (sign >> 24u));
}

} // namespace

// ---- forward kernels against host math ----

TEST_CASE("native RMSNorm matches host math on f32 rows") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 3;
  const size_t cols = 8;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 7);
  auto w_data = pattern(cols, 11);

  array x = array(x_data.begin(), Shape{int(rows), int(cols)}, float32);
  array w = array(w_data.begin(), Shape{int(cols)}, float32);
  auto got = flat(fast::rms_norm(x, w, eps, stream), stream);

  std::vector<double> x_host = widen(x_data);
  std::vector<double> w_host = widen(w_data);
  std::vector<double> w_broad(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      w_broad[row * cols + col] = w_host[col];
    }
  }
  std::vector<double> zeros(rows * cols, 0.0);
  auto want = host_norm(x_host, w_broad, zeros, rows, cols, eps, false);
  require_close(got, want, 1e-5, "rms_norm weighted");

  // Weightless form: upstream passes a scalar 1.0 weight.
  auto got_bare = flat(fast::rms_norm(x, std::nullopt, eps, stream), stream);
  std::vector<double> ones(rows * cols, 1.0);
  auto want_bare = host_norm(x_host, ones, zeros, rows, cols, eps, false);
  require_close(got_bare, want_bare, 1e-5, "rms_norm weightless");
}

TEST_CASE("native LayerNorm matches host math on f32 rows") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 4;
  const size_t cols = 6;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 13);
  auto w_data = pattern(cols, 17);
  auto b_data = pattern(cols, 19);

  array x = array(x_data.begin(), Shape{int(rows), int(cols)}, float32);
  array w = array(w_data.begin(), Shape{int(cols)}, float32);
  array b = array(b_data.begin(), Shape{int(cols)}, float32);
  auto got = flat(fast::layer_norm(x, w, b, eps, stream), stream);

  std::vector<double> x_host = widen(x_data);
  std::vector<double> w_host = widen(w_data);
  std::vector<double> b_host = widen(b_data);
  std::vector<double> w_broad(rows * cols);
  std::vector<double> b_broad(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      w_broad[row * cols + col] = w_host[col];
      b_broad[row * cols + col] = b_host[col];
    }
  }
  auto want = host_norm(x_host, w_broad, b_broad, rows, cols, eps, true);
  require_close(got, want, 1e-5, "layer_norm weighted");

  // Weightless and biasless (upstream passes scalar 1 and scalar 0).
  auto got_bare =
      flat(fast::layer_norm(x, std::nullopt, std::nullopt, eps, stream), stream);
  std::vector<double> ones(rows * cols, 1.0);
  std::vector<double> zeros(rows * cols, 0.0);
  auto want_bare = host_norm(x_host, ones, zeros, rows, cols, eps, true);
  require_close(got_bare, want_bare, 1e-5, "layer_norm bare");
}

TEST_CASE("fused norm forward on f16 and bf16 storage") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 2;
  const size_t cols = 8;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 23);
  std::vector<double> x_host = widen(x_data);
  std::vector<double> ones(rows * cols, 1.0);
  std::vector<double> zeros(rows * cols, 0.0);
  auto want = host_norm(x_host, ones, zeros, rows, cols, eps, false);

  array x16 = astype(
      array(x_data.begin(), Shape{int(rows), int(cols)}, float32),
      float16,
      stream);
  auto got16 = flat(fast::rms_norm(x16, std::nullopt, eps, stream), stream);
  require_close(got16, want, 2e-2, "rms_norm f16");

  array xbf = astype(
      array(x_data.begin(), Shape{int(rows), int(cols)}, float32),
      bfloat16,
      stream);
  auto gotbf = flat(fast::rms_norm(xbf, std::nullopt, eps, stream), stream);
  require_close(gotbf, want, 5e-2, "rms_norm bf16");
}

TEST_CASE("RMSNormVJP matches finite differences and the composed formula") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 2;
  const size_t cols = 4;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 29);
  auto w_data = pattern(cols, 31);
  auto c_data = pattern(rows * cols, 37);

  array x = array(x_data.begin(), Shape{int(rows), int(cols)}, float32);
  array w = array(w_data.begin(), Shape{int(cols)}, float32);
  array cot = array(c_data.begin(), Shape{int(rows), int(cols)}, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        fast::rms_norm(inputs[0], inputs[1], eps, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{x, w}, {cot});
  auto got_dx = flat(grads[0], stream);
  auto got_dw = flat(grads[1], stream);

  std::vector<double> x_host = widen(x_data);
  std::vector<double> w_host = widen(w_data);
  std::vector<double> c_host = widen(c_data);
  auto objective_x = [&](const std::vector<double>& point) {
    auto out = host_norm(point, w_host, std::vector<double>(w_host.size(), 0.0),
                         rows, cols, eps, false);
    double dot = 0.0;
    for (size_t index = 0; index < out.size(); ++index) {
      dot += out[index] * c_host[index];
    }
    return dot;
  };
  auto objective_w = [&](const std::vector<double>& point) {
    auto out = host_norm(x_host, point, std::vector<double>(point.size(), 0.0),
                         rows, cols, eps, false);
    double dot = 0.0;
    for (size_t index = 0; index < out.size(); ++index) {
      dot += out[index] * c_host[index];
    }
    return dot;
  };
  double h = 1e-2;
  std::vector<double> fd_dx(x_host.size());
  for (size_t index = 0; index < x_host.size(); ++index) {
    fd_dx[index] = central_difference(x_host, index, h, objective_x);
  }
  std::vector<double> fd_dw(w_host.size());
  for (size_t index = 0; index < w_host.size(); ++index) {
    fd_dw[index] = central_difference(w_host, index, h, objective_w);
  }
  // Reference 1: finite differences through the real kernel.
  require_close(got_dx, fd_dx, 2e-2, "rms vjp dx finite difference");
  require_close(got_dw, fd_dw, 2e-2, "rms vjp dw finite difference");

  // Reference 2: the composed fallback algebra rebuilt from core ops,
  // exactly as mlx/fast.cpp RMSNorm::vjp writes it.
  auto n = rsqrt(
      add(mean(square(x, stream), -1, true, stream),
          array(eps, float32),
          stream),
      stream);
  auto n3 = power(n, array(3.0f, float32), stream);
  auto gw = multiply(cot, w, stream);
  auto t = mean(multiply(gw, x, stream), -1, true, stream);
  auto composed_dx = subtract(
      multiply(gw, n, stream),
      multiply(multiply(x, t, stream), n3, stream),
      stream);
  auto composed_dw = sum(
      multiply(cot, multiply(x, n, stream), stream), 0, false, stream);
  require_close(
      got_dx,
      widen(flat(composed_dx, stream)),
      1e-5,
      "rms vjp dx composed formula");
  require_close(
      got_dw,
      widen(flat(composed_dw, stream)),
      1e-5,
      "rms vjp dw composed formula");

  // Reference 3: the closed-form host algebra.
  require_close(
      got_dx,
      host_norm_vjp_dx(x_host, w_host, c_host, rows, cols, eps, false),
      1e-5,
      "rms vjp dx host math");
  require_close(
      got_dw,
      host_norm_vjp_dw(x_host, c_host, rows, cols, eps, false),
      1e-5,
      "rms vjp dw host math");
}

TEST_CASE("LayerNormVJP matches finite differences and the composed formula") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 2;
  const size_t cols = 4;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 41);
  auto w_data = pattern(cols, 43);
  auto b_data = pattern(cols, 47);
  auto c_data = pattern(rows * cols, 53);

  array x = array(x_data.begin(), Shape{int(rows), int(cols)}, float32);
  array w = array(w_data.begin(), Shape{int(cols)}, float32);
  array b = array(b_data.begin(), Shape{int(cols)}, float32);
  array cot = array(c_data.begin(), Shape{int(rows), int(cols)}, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        fast::layer_norm(inputs[0], inputs[1], inputs[2], eps, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{x, w, b}, {cot});
  auto got_dx = flat(grads[0], stream);
  auto got_dw = flat(grads[1], stream);
  auto got_db = flat(grads[2], stream);

  std::vector<double> x_host = widen(x_data);
  std::vector<double> w_host = widen(w_data);
  std::vector<double> b_host = widen(b_data);
  std::vector<double> c_host = widen(c_data);
  auto dot_with_c = [&](const std::vector<double>& out) {
    double dot = 0.0;
    for (size_t index = 0; index < out.size(); ++index) {
      dot += out[index] * c_host[index];
    }
    return dot;
  };
  auto objective_x = [&](const std::vector<double>& point) {
    return dot_with_c(host_norm(point, w_host, b_host, rows, cols, eps, true));
  };
  auto objective_w = [&](const std::vector<double>& point) {
    return dot_with_c(host_norm(x_host, point, b_host, rows, cols, eps, true));
  };
  auto objective_b = [&](const std::vector<double>& point) {
    return dot_with_c(host_norm(x_host, w_host, point, rows, cols, eps, true));
  };
  double h = 1e-2;
  std::vector<double> fd_dx(x_host.size());
  for (size_t index = 0; index < x_host.size(); ++index) {
    fd_dx[index] = central_difference(x_host, index, h, objective_x);
  }
  std::vector<double> fd_dw(w_host.size());
  for (size_t index = 0; index < w_host.size(); ++index) {
    fd_dw[index] = central_difference(w_host, index, h, objective_w);
  }
  std::vector<double> fd_db(b_host.size());
  for (size_t index = 0; index < b_host.size(); ++index) {
    fd_db[index] = central_difference(b_host, index, h, objective_b);
  }
  require_close(got_dx, fd_dx, 2e-2, "layer vjp dx finite difference");
  require_close(got_dw, fd_dw, 2e-2, "layer vjp dw finite difference");
  require_close(got_db, fd_db, 2e-2, "layer vjp db finite difference");

  // Composed fallback algebra, exactly as mlx/fast.cpp LayerNorm::vjp
  // writes it.
  auto norm_count = number_of_elements(x, {-1}, true, float32, stream);
  auto sumx = sum(x, -1, true, stream);
  auto sumx2 = sum(square(x, stream), -1, true, stream);
  auto mu = multiply(sumx, norm_count, stream);
  auto mu2 = multiply(sumx2, norm_count, stream);
  auto var = subtract(mu2, square(mu, stream), stream);
  auto n = rsqrt(add(var, array(eps, float32), stream), stream);
  auto n3 = power(n, array(3.0f, float32), stream);
  auto x_c = subtract(x, mu, stream);
  auto wg = multiply(w, cot, stream);
  auto sumwg = multiply(sum(wg, -1, true, stream), norm_count, stream);
  auto sumwgxc = multiply(
      sum(multiply(wg, x_c, stream), -1, true, stream), norm_count, stream);
  auto t1 = multiply(multiply(x_c, sumwgxc, stream), n3, stream);
  auto t2 = multiply(subtract(wg, sumwg, stream), n, stream);
  auto composed_dx = subtract(t2, t1, stream);
  auto composed_dw = sum(
      multiply(cot, multiply(x_c, n, stream), stream), 0, false, stream);
  auto composed_db = sum(cot, 0, false, stream);
  require_close(
      got_dx,
      widen(flat(composed_dx, stream)),
      1e-5,
      "layer vjp dx composed formula");
  require_close(
      got_dw,
      widen(flat(composed_dw, stream)),
      1e-5,
      "layer vjp dw composed formula");
  require_close(
      got_db,
      widen(flat(composed_db, stream)),
      1e-5,
      "layer vjp db composed formula");

  require_close(
      got_dx,
      host_norm_vjp_dx(x_host, w_host, c_host, rows, cols, eps, true),
      1e-5,
      "layer vjp dx host math");
  require_close(
      got_dw,
      host_norm_vjp_dw(x_host, c_host, rows, cols, eps, true),
      1e-5,
      "layer vjp dw host math");
  require_close(
      got_db,
      host_norm_vjp_db(c_host, rows, cols),
      1e-5,
      "layer vjp db host math");
}

TEST_CASE("cross_entropy forward matches host math on the composed path") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 3;
  const size_t cols = 7;
  auto logits_data = pattern(rows * cols, 67);
  std::vector<int> targets{0, 3, 6};

  array logits =
      array(logits_data.begin(), Shape{int(rows), int(cols)}, float32);
  array targets_arr = array(targets.begin(), Shape{int(rows)}, int32);
  auto got = flat(fast::cross_entropy(logits, targets_arr, stream), stream);

  // Reference: host math lse - score.
  std::vector<double> want(rows);
  std::vector<double> logit_host = widen(logits_data);
  for (size_t row = 0; row < rows; ++row) {
    double max_value = logit_host[row * cols];
    for (size_t col = 1; col < cols; ++col) {
      max_value = std::max(max_value, logit_host[row * cols + col]);
    }
    double sum_exp = 0.0;
    for (size_t col = 0; col < cols; ++col) {
      sum_exp += std::exp(logit_host[row * cols + col] - max_value);
    }
    want[row] = max_value + std::log(sum_exp) -
        logit_host[row * cols + size_t(targets[row])];
  }
  require_close(got, want, 1e-5, "cross_entropy composed forward");
}

TEST_CASE("CrossEntropyVJP matches finite differences and the composed formula") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 2;
  const size_t cols = 5;
  auto logits_data = pattern(rows * cols, 71);
  std::vector<int> targets{1, 4};
  auto cot_data = pattern(rows, 73);
  array logits =
      array(logits_data.begin(), Shape{int(rows), int(cols)}, float32);
  array targets_arr = array(targets.begin(), Shape{int(rows)}, int32);
  array cot = array(cot_data.begin(), Shape{int(rows)}, float32);
  // Targets are constants in training: capture them so the VJP runs with
  // respect to the logits only, with an arbitrary cotangent.
  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        fast::cross_entropy(inputs[0], targets_arr, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{logits}, {cot});
  auto got = flat(grads[0], stream);

  std::vector<double> logit_host = widen(logits_data);
  std::vector<double> cot_host = widen(cot_data);
  auto objective = [&](const std::vector<double>& point) {
    // Objective: dot(loss, cotangent), loss_row = lse_row - x_row[y_row].
    double total = 0.0;
    for (size_t row = 0; row < rows; ++row) {
      double max_value = point[row * cols];
      for (size_t col = 1; col < cols; ++col) {
        max_value = std::max(max_value, point[row * cols + col]);
      }
      double sum_exp = 0.0;
      for (size_t col = 0; col < cols; ++col) {
        sum_exp += std::exp(point[row * cols + col] - max_value);
      }
      total += cot_host[row] *
          (max_value + std::log(sum_exp) -
           point[row * cols + size_t(targets[row])]);
    }
    return total;
  };
  double h = 1e-2;
  std::vector<double> fd(logits_data.size());
  for (size_t index = 0; index < logits_data.size(); ++index) {
    fd[index] = central_difference(logit_host, index, h, objective);
  }
  require_close(got, fd, 2e-2, "cross entropy vjp finite difference");

  // Composed formula from upstream CrossEntropy::vjp: g * (p - onehot).
  auto score = squeeze(
      take_along_axis(
          logits, expand_dims(targets_arr, -1, stream), -1, stream),
      -1,
      stream);
  auto lse = add(fast::cross_entropy(logits, targets_arr, stream),
                 score,
                 stream);
  auto p = exp(subtract(logits, expand_dims(lse, -1, stream), stream), stream);
  Shape class_shape{1, int(cols)};
  auto onehot = astype(
      equal(
          expand_dims(targets_arr, -1, stream),
          reshape(arange(0, int(cols), int32, stream), class_shape, stream),
          stream),
      float32,
      stream);
  auto composed = multiply(
      expand_dims(cot, -1, stream), subtract(p, onehot, stream), stream);
  require_close(
      got,
      widen(flat(composed, stream)),
      1e-5,
      "cross entropy vjp composed formula");

  require_close(
      got,
      host_cross_entropy_vjp(logit_host, targets, cot_host, rows, cols),
      1e-5,
      "cross entropy vjp host math");
}

TEST_CASE("scaled_dot_product_attention backward matches finite differences") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Composed anchor: fast::ScaledDotProductAttentionVJP keeps use_fallback
  // true, so autograd runs the composed fallback graph over the same
  // Elementwise, Matmul, and Softmax kernels the forward uses. Reference:
  // finite differences of the attention output, host math.
  const int head_dim = 4;
  const float scale = 1.0f / std::sqrt(float(head_dim));
  auto q_data = pattern(2 * 1 * 2 * head_dim, 79);
  auto k_data = pattern(2 * 1 * 2 * head_dim, 83);
  auto v_data = pattern(2 * 1 * 2 * head_dim, 89);
  auto cot_data = pattern(2 * 1 * 2 * head_dim, 97);

  Shape shape{2, 1, 2, head_dim};
  array q = array(q_data.begin(), shape, float32);
  array k = array(k_data.begin(), shape, float32);
  array v = array(v_data.begin(), shape, float32);
  array cot = array(cot_data.begin(), shape, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        fast::scaled_dot_product_attention(
            inputs[0], inputs[1], inputs[2], scale, "", {}, std::nullopt, false, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{q, k, v}, {cot});
  auto got_dq = flat(grads[0], stream);

  std::vector<double> q_host = widen(q_data);
  std::vector<double> k_host = widen(k_data);
  std::vector<double> v_host = widen(v_data);
  std::vector<double> cot_host = widen(cot_data);
  auto attention = [&](const std::vector<double>& qq) {
    std::vector<double> out(2 * 1 * 2 * head_dim);
    for (int batch = 0; batch < 2; ++batch) {
      for (int q_pos = 0; q_pos < 2; ++q_pos) {
        double max_score = -1e30;
        std::vector<double> scores(2);
        for (int k_pos = 0; k_pos < 2; ++k_pos) {
          double dot = 0.0;
          for (int dim = 0; dim < head_dim; ++dim) {
            dot += qq[((batch * 2 + q_pos) * head_dim) + dim] *
                k_host[((batch * 2 + k_pos) * head_dim) + dim];
          }
          scores[k_pos] = dot * scale;
          max_score = std::max(max_score, scores[k_pos]);
        }
        double sum_exp = 0.0;
        for (auto& score : scores) {
          score = std::exp(score - max_score);
          sum_exp += score;
        }
        for (int dim = 0; dim < head_dim; ++dim) {
          double acc = 0.0;
          for (int k_pos = 0; k_pos < 2; ++k_pos) {
            acc += scores[k_pos] / sum_exp *
                v_host[((batch * 2 + k_pos) * head_dim) + dim];
          }
          out[(batch * 2 + q_pos) * head_dim + dim] = acc;
        }
      }
    }
    return out;
  };
  auto objective_q = [&](const std::vector<double>& point) {
    auto out = attention(point);
    double dot = 0.0;
    for (size_t index = 0; index < out.size(); ++index) {
      dot += out[index] * cot_host[index];
    }
    return dot;
  };
  double h = 1e-2;
  std::vector<double> fd_dq(q_host.size());
  for (size_t index = 0; index < q_host.size(); ++index) {
    fd_dq[index] = central_difference(q_host, index, h, objective_q);
  }
  require_close(got_dq, fd_dq, 2e-2, "sdpa dq finite difference");
}

TEST_CASE("fp8 conversion matches the upstream bit algorithm") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Known E4M3 patterns: zero, denorm floor 2^-9, one, min normal 2^-6,
  // max 448, signs, non-trivial mantissas.
  std::vector<double> probe{
      0.0,
      1.0,
      -1.0,
      448.0,
      -448.0,
      0.5,
      2.0,
      3.5,
      0.001953125,
      7.0,
      0.015625,
      264.0};
  std::vector<float> probe_f(probe.begin(), probe.end());
  array x = array(probe_f.begin(), Shape{int(probe.size())}, float32);

  array encoded = to_fp8(x, stream);
  encoded.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const uint8_t* bytes = encoded.data<uint8_t>();
  for (size_t index = 0; index < probe.size(); ++index) {
    CHECK_EQ(int(bytes[index]), int(host_to_fp8(probe[index])));
  }
  // Pinpoints: one, the saturation ceiling, its negation, denorm floor.
  CHECK_EQ(int(host_to_fp8(1.0)), 0x38);
  CHECK_EQ(int(host_to_fp8(448.0)), 0x7E);
  CHECK_EQ(int(host_to_fp8(-448.0)), 0xFE);
  CHECK_EQ(int(host_to_fp8(0.001953125)), 0x01);

  array decoded = from_fp8(encoded, float32, stream);
  auto round_trip = flat(decoded, stream);
  for (size_t index = 0; index < probe.size(); ++index) {
    double quantum = std::max(1.0, std::abs(probe[index])) * 0.0625;
    CHECK(std::abs(double(round_trip[index]) - probe[index]) <= quantum + 1e-6);
  }

  // f16 and bf16 storages carry the same E4M3 payload.
  array x16 = astype(x, float16, stream);
  array encoded16 = to_fp8(x16, stream);
  encoded16.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const uint8_t* bytes16 = encoded16.data<uint8_t>();
  for (size_t index = 0; index < probe.size(); ++index) {
    CHECK_EQ(int(bytes16[index]), int(host_to_fp8(probe[index])));
  }
  array xbf = astype(x, bfloat16, stream);
  array encodedbf = to_fp8(xbf, stream);
  encodedbf.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const uint8_t* bytesbf = encodedbf.data<uint8_t>();
  for (size_t index = 0; index < probe.size(); ++index) {
    CHECK_EQ(int(bytesbf[index]), int(host_to_fp8(probe[index])));
  }
  // Narrow decodes must reproduce the f32 decode payload exactly: every
  // E4M3 value is exact in f16 and bf16.
  auto decoded16 = flat(from_fp8(encoded16, float16, stream), stream);
  require_close(decoded16, widen(round_trip), 1e-6, "fp8 decode f16");
  auto decodedbf = flat(from_fp8(encodedbf, bfloat16, stream), stream);
  require_close(decodedbf, widen(round_trip), 1e-6, "fp8 decode bf16");
}

TEST_CASE("non-contiguous norm inputs raise the named error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base = arange(0, 16, float32, stream);
  array transposed =
      transpose(reshape(base, Shape{4, 4}, stream), {1, 0}, stream);
  array weight = ones({4}, float32, stream);

  auto rms_message = caught_message([&] {
    fast::rms_norm(transposed, weight, 1e-5f, stream).eval();
  });
  CHECK(rms_message.find("[omarchy]") != std::string::npos);
  CHECK(rms_message.find("non-contiguous") != std::string::npos);

  auto layer_message = caught_message([&] {
    fast::layer_norm(transposed, weight, weight, 1e-5f, stream).eval();
  });
  CHECK(layer_message.find("non-contiguous") != std::string::npos);

  // A weight whose last axis does not match the row length is refused by
  // the upstream op validation before the backend gate can run.
  auto weight_message = caught_message([&] {
    fast::rms_norm(
        reshape(base, Shape{4, 4}, stream),
        ones({3}, float32, stream),
        1e-5f,
        stream)
        .eval();
  });
  CHECK(weight_message.find("[rms_norm]") != std::string::npos);
  CHECK(weight_message.find("same size") != std::string::npos);

  // The backend keeps its own parameter gate for direct primitive
  // construction, which bypasses upstream validation.
  array mismatched = array(
      Shape{4, 4},
      float32,
      std::make_shared<fast::RMSNorm>(
          stream,
          [](const std::vector<array>& inputs) {
            return std::vector<array>{inputs[0]};
          },
          1e-5f),
      {reshape(base, Shape{4, 4}, stream), ones({3}, float32, stream)});
  auto gate_message = caught_message([&] { mismatched.eval(); });
  CHECK(gate_message.find("parameter shape") != std::string::npos);
  CHECK(gate_message.find("[omarchy]") != std::string::npos);
}

TEST_CASE("CustomKernel reports the Metal-source incompatibility") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array input = arange(0, 8, float32, stream);
  auto primitive = std::make_shared<fast::CustomKernel>(
      stream,
      "custom_kernel_probe",
      std::string("kernel void custom_kernel_probe() {}"),
      std::tuple<int, int, int>{1, 1, 1},
      std::tuple<int, int, int>{1, 1, 1},
      std::vector<std::tuple<bool, bool, bool>>{},
      false,
      std::nullopt,
      std::vector<fast::ScalarArg>{},
      false,
      0);
  array out(Shape{8}, float32, primitive, {input});
  auto message = caught_message([&] { out.eval(); });
  CHECK(message.find("fast::CustomKernel") != std::string::npos);
  CHECK(message.find("Metal") != std::string::npos);
  CHECK(message.find("no silent CPU fallback") != std::string::npos);
}
